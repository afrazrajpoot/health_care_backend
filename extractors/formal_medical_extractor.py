"""
FormalMedicalReportExtractor - Enhanced Extractor for Comprehensive Medical Reports
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier

logger = logging.getLogger("document_ai")


class FormalMedicalReportExtractor:
    """
    Enhanced Formal Medical Report extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different medical specialties
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, Endoscopy, Genetics, Discharge Summaries
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.report_type_patterns = {
            'surgery': re.compile(r'\b(surgery|surgical|pre[- ]?op|post[- ]?op|operative|procedure)\b', re.IGNORECASE),
            'anesthesia': re.compile(r'\b(anesthesia|anesthetic|sedation|airway|intubation)\b', re.IGNORECASE),
            'emg': re.compile(r'\b(EMG|NCS|electromyography|nerve conduction|needle exam)\b', re.IGNORECASE),
            'pathology': re.compile(r'\b(pathology|biopsy|specimen|histology|microscopic)\b', re.IGNORECASE),
            'cardiology': re.compile(r'\b(cardiology|EKG|ECG|echocardiogram|stress test|holter)\b', re.IGNORECASE),
            'sleep': re.compile(r'\b(sleep study|polysomnography|PSG|apnea|hypopnea)\b', re.IGNORECASE),
            'endoscopy': re.compile(r'\b(endoscopy|colonoscopy|EGD|gastroscopy|bronchoscopy)\b', re.IGNORECASE),
            'genetics': re.compile(r'\b(genetic|mutation|variant|DNA|RNA|chromosome)\b', re.IGNORECASE),
            'discharge': re.compile(r'\b(discharge|admission|hospital course|disposition)\b', re.IGNORECASE)
        }
        
        # Medical procedure patterns
        self.procedure_patterns = {
            'cpt_codes': re.compile(r'\bCPT[:\s]*(\d{4,5})', re.IGNORECASE),
            'icd_codes': re.compile(r'\b(ICD[-]?10[:\s]*([A-Z]\d{2,})|([A-Z]\d{2,}))', re.IGNORECASE),
            'medications': re.compile(r'\b(\d+\s*(mg|mcg|g|ml)\s*[\w\s]+\s*(PO|IV|IM|SC|QD|BID|TID|QID|PRN))', re.IGNORECASE)
        }
        
        logger.info("✅ FormalMedicalReportExtractor initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract Formal Medical Report data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Surgery, Anesthesia, EMG, Pathology, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING FORMAL MEDICAL REPORT EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Auto-detect specific report type if not specified
        detected_type = self._detect_report_type(text, doc_type)
        logger.info(f"📋 Report Type: {detected_type} (original: {doc_type})")
        
        # Log context guidance if available
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Critical Locations: {list(critical_locations.keys())}")
        else:
            logger.warning("⚠️ No context analysis provided - proceeding without guidance")
        
        # Check document size
        text_length = len(text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Extract with FULL CONTEXT and contextual guidance
        raw_result = self._extract_full_context_with_guidance(
            text=text,
            doc_type=detected_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )

        # Stage 2: Build long summary from ALL raw data
        long_summary = self._build_comprehensive_long_summary(raw_result, detected_type, fallback_date)

        # Stage 3: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)

        logger.info("=" * 80)
        logger.info("✅ FORMAL MEDICAL REPORT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_report_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific medical report type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for report_type, pattern in self.report_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[report_type] = len(matches)
        
        # Boost scores for procedure-specific terminology
        if self.procedure_patterns['cpt_codes'].search(text):
            for report_type in ['surgery', 'procedure']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 2
        
        if self.procedure_patterns['icd_codes'].search(text):
            for report_type in ['pathology', 'discharge']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].upper().replace('_', ' ')
                logger.info(f"🔍 Auto-detected report type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect report type, using: {original_type}")
        return original_type or "MEDICAL_REPORT"

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        """
        logger.info("🔍 Processing ENTIRE medical report in single context window with guidance...")
        
        # Extract guidance from context analysis
        focus_sections = []
        critical_locations = {}
        ambiguities = []
        
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical documentation specialist analyzing a COMPLETE {doc_type} report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE medical report at once, allowing you to:
- Understand the complete clinical picture from history to conclusions
- Connect pre-procedure assessments with intraoperative findings and post-procedure outcomes
- Identify relationships between clinical indications, procedures performed, and results
- Provide comprehensive extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate medical information
   - DO NOT fill in "typical" or "common" medical values
   - DO NOT use medical knowledge to "complete" incomplete information
   
2. **PROCEDURES & FINDINGS - EXACT WORDING ONLY**
   - Extract procedures using EXACT wording from report
   - Extract findings verbatim - do not interpret or rephrase
   - For pathology: extract microscopic descriptions EXACTLY as stated
   - For lab values: extract numbers and units EXACTLY as written
   
3. **MEDICATIONS & DOSES - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY medications explicitly listed in medication sections
   - Include dosages, routes, frequencies ONLY if explicitly stated
   - DO NOT extract medications mentioned as examples or comparisons
   
4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
5. **MEDICAL CODING - EXACT REFERENCES ONLY**
   - Extract CPT/ICD codes ONLY if explicitly listed in the report
   - DO NOT assign codes based on procedure descriptions
   - Include code descriptors ONLY if provided

EXTRACTION FOCUS - 8 CRITICAL MEDICAL REPORT CATEGORIES:

I. REPORT IDENTITY & CONTEXT
- Report type, dates, identification numbers
- Facility and department information
- All healthcare providers involved

II. PATIENT CLINICAL CONTEXT
- Patient demographics and identifiers
- Clinical history and presenting symptoms
- Pre-existing conditions and risk factors
- Indications for procedure/study

III. PROCEDURE/TEST DETAILS (CORE CONTENT)
- Procedure/test name and type
- Anatomical locations and specific sites
- Technique/methodology used
- Duration and technical details
- Specimen information (for pathology)

IV. INTRAOPERATIVE/INTRA-PROCEDURAL FINDINGS
- Detailed findings during procedure
- Anatomical observations
- Complications or unexpected findings
- Blood loss, fluids, vital signs

V. SPECIMEN/PATHOLOGY DETAILS (if applicable)
- Specimen descriptions and labeling
- Gross examination findings
- Microscopic examination details
- Special stains and results

VI. RESULTS & INTERPRETATIONS
- Test results with values and units
- Physician interpretations and conclusions
- Diagnostic impressions
- Correlation with clinical information

VII. MEDICATIONS & ANESTHESIA (if applicable)
- Anesthetic agents and techniques
- Medications administered during procedure
- Dosages, routes, and timing
- Anesthesia complications

VIII. FOLLOW-UP & RECOMMENDATIONS
- Post-procedure instructions
- Medication prescriptions
- Follow-up scheduling
- Additional testing recommendations

⚠️ FINAL REMINDER:
- If information is NOT in the report, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate medical information
- PROCEDURE DETAILS: Use exact wording from report
- It is BETTER to have empty fields than incorrect medical information

Now analyze this COMPLETE {doc_type} medical report and extract ALL relevant information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} MEDICAL REPORT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical medical details:

{{
  "report_identity": {{
    "report_type": "{doc_type}",
    "report_date": "",
    "procedure_date": "",
    "accession_number": "",
    "medical_record_number": "",
    "facility_name": "",
    "department": ""
  }},
  
  "patient_information": {{
    "patient_name": "",
    "patient_dob": "",
    "patient_age": "",
    "patient_gender": "",
    "medical_record_number": "",
    "allergies": "",
    "clinical_history": ""
  }},
  
  "providers": {{
    "ordering_physician": {{
      "name": "",
      "specialty": "",
      "npi": ""
    }},
    "performing_physician": {{
      "name": "",
      "specialty": "",
      "credentials": "",
      "role": "Performing/Attending Physician"
    }},
    "assistant_physicians": [],
    "anesthesiologist": {{
      "name": "",
      "credentials": ""
    }},
    "referring_physician": {{
      "name": "",
      "specialty": ""
    }}
  }},
  
  "clinical_context": {{
    "indications": "",
    "preoperative_diagnosis": "",
    "postoperative_diagnosis": "",
    "clinical_history_summary": "",
    "relevant_medical_history": []
  }},
  
  "procedure_details": {{
    "procedure_name": "",
    "procedure_type": "",
    "cpt_codes": [],
    "icd_codes": [],
    "anatomical_sites": [],
    "laterality": "",
    "technique_used": "",
    "procedure_duration": "",
    "anesthesia_type": "",
    "specimens_collected": []
  }},
  
  "intraoperative_findings": {{
    "description": "",
    "key_observations": [],
    "complications": [],
    "blood_loss": "",
    "fluids_administered": "",
    "vital_signs_stability": ""
  }},
  
  "pathology_findings": {{
    "gross_description": "",
    "microscopic_description": "",
    "special_stains": [],
    "pathological_diagnosis": "",
    "tumor_characteristics": {{
      "size": "",
      "margins": "",
      "grade": "",
      "stage": ""
    }}
  }},
  
  "test_results": {{
    "results_summary": "",
    "key_measurements": [],
    "normal_ranges": "",
    "abnormal_findings": [],
    "interpretation": ""
  }},
  
  "medications_anesthesia": {{
    "preoperative_medications": [],
    "anesthetic_agents": [],
    "intraoperative_medications": [],
    "postoperative_medications": [],
    "medication_allergies": ""
  }},
  
  "conclusions_recommendations": {{
    "final_diagnosis": "",
    "clinical_impressions": "",
    "recommendations": [],
    "follow_up_plan": "",
    "prescriptions": []
  }},
  
  "critical_findings": []
}}

⚠️ CRITICAL MEDICAL REMINDERS:
1. For "procedure_details": Extract EXACT procedure names from report
   - Include anatomical specifics ONLY if explicitly stated
   - Include CPT/ICD codes ONLY if explicitly listed

2. For "pathology_findings": Extract microscopic descriptions VERBATIM
   - Do not interpret or summarize pathological findings
   - Include tumor characteristics ONLY if explicitly measured/stated

3. For "test_results": Extract values and units EXACTLY as written
   - Include reference ranges ONLY if provided
   - Do not interpret abnormal vs normal - extract values only

4. For "medications_anesthesia": Extract ONLY medications explicitly administered
   - Include dosages and routes ONLY if explicitly stated
   - Do not include medications mentioned in history or recommendations

5. For "critical_findings": Include only clinically significant findings
   - Malignancies or positive cancer diagnoses
   - Critical abnormal lab values
   - Significant complications
   - Life-threatening conditions
""")

        # Build context guidance summary
        context_guidance_text = f"""
REPORT TYPE: {doc_type}
FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- Procedure Details: {critical_locations.get('procedure_location', 'Search entire document')}
- Results/Findings: {critical_locations.get('results_location', 'Search entire document')}
- Conclusions: {critical_locations.get('conclusions_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context medical report extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "context_guidance": context_guidance_text
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context medical report extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char medical report")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context medical report extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Medical report exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large reports")
            
            return self._get_fallback_result(doc_type, fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted medical report data.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted medical data...")
        
        sections = []
        
        # Section 1: REPORT OVERVIEW
        sections.append("📋 MEDICAL REPORT OVERVIEW")
        sections.append("-" * 50)
        
        report_identity = raw_data.get("report_identity", {})
        overview_lines = [
            f"Report Type: {report_identity.get('report_type', doc_type)}",
            f"Report Date: {report_identity.get('report_date', fallback_date)}",
            f"Procedure Date: {report_identity.get('procedure_date', 'Not specified')}",
            f"Accession Number: {report_identity.get('accession_number', 'Not specified')}",
            f"Facility: {report_identity.get('facility_name', 'Not specified')}",
            f"Department: {report_identity.get('department', 'Not specified')}"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PATIENT INFORMATION
        sections.append("\n👤 PATIENT INFORMATION")
        sections.append("-" * 50)
        
        patient_info = raw_data.get("patient_information", {})
        patient_lines = [
            f"Name: {patient_info.get('patient_name', 'Not specified')}",
            f"DOB: {patient_info.get('patient_dob', 'Not specified')}",
            f"Age: {patient_info.get('patient_age', 'Not specified')}",
            f"Gender: {patient_info.get('patient_gender', 'Not specified')}",
            f"Allergies: {patient_info.get('allergies', 'Not specified')}",
            f"MRN: {patient_info.get('medical_record_number', 'Not specified')}"
        ]
        sections.append("\n".join(patient_lines))
        
        # Section 3: PROVIDERS
        sections.append("\n👨‍⚕️ HEALTHCARE PROVIDERS")
        sections.append("-" * 50)
        
        providers = raw_data.get("providers", {})
        provider_lines = []
        
        # Performing physician
        performing_md = providers.get("performing_physician", {})
        if performing_md.get("name"):
            provider_lines.append(f"Performing Physician: {performing_md['name']}")
            if performing_md.get("specialty"):
                provider_lines.append(f"  Specialty: {performing_md['specialty']}")
        
        # Ordering physician
        ordering_md = providers.get("ordering_physician", {})
        if ordering_md.get("name"):
            provider_lines.append(f"Ordering Physician: {ordering_md['name']}")
            if ordering_md.get("specialty"):
                provider_lines.append(f"  Specialty: {ordering_md['specialty']}")
        
        # Anesthesiologist
        anesthesiologist = providers.get("anesthesiologist", {})
        if anesthesiologist.get("name"):
            provider_lines.append(f"Anesthesiologist: {anesthesiologist['name']}")
        
        sections.append("\n".join(provider_lines) if provider_lines else "No provider information extracted")
        
        # Section 4: CLINICAL CONTEXT
        sections.append("\n🏥 CLINICAL CONTEXT")
        sections.append("-" * 50)
        
        clinical_context = raw_data.get("clinical_context", {})
        context_lines = []
        
        if clinical_context.get("indications"):
            context_lines.append(f"Indications: {clinical_context['indications']}")
        
        if clinical_context.get("preoperative_diagnosis"):
            context_lines.append(f"Preoperative Diagnosis: {clinical_context['preoperative_diagnosis']}")
        
        if clinical_context.get("postoperative_diagnosis"):
            context_lines.append(f"Postoperative Diagnosis: {clinical_context['postoperative_diagnosis']}")
        
        if clinical_context.get("clinical_history_summary"):
            context_lines.append(f"Clinical History: {clinical_context['clinical_history_summary']}")
        
        sections.append("\n".join(context_lines) if context_lines else "No clinical context extracted")
        
        # Section 5: PROCEDURE DETAILS
        sections.append("\n🔧 PROCEDURE DETAILS")
        sections.append("-" * 50)
        
        procedure_details = raw_data.get("procedure_details", {})
        procedure_lines = []
        
        if procedure_details.get("procedure_name"):
            procedure_lines.append(f"Procedure: {procedure_details['procedure_name']}")
        
        if procedure_details.get("procedure_type"):
            procedure_lines.append(f"Type: {procedure_details['procedure_type']}")
        
        anatomical_sites = procedure_details.get("anatomical_sites", [])
        if anatomical_sites:
            procedure_lines.append(f"Anatomical Sites: {', '.join(anatomical_sites[:3])}")
        
        if procedure_details.get("laterality"):
            procedure_lines.append(f"Laterality: {procedure_details['laterality']}")
        
        if procedure_details.get("anesthesia_type"):
            procedure_lines.append(f"Anesthesia: {procedure_details['anesthesia_type']}")
        
        if procedure_details.get("procedure_duration"):
            procedure_lines.append(f"Duration: {procedure_details['procedure_duration']}")
        
        # CPT Codes
        cpt_codes = procedure_details.get("cpt_codes", [])
        if cpt_codes:
            procedure_lines.append(f"CPT Codes: {', '.join(cpt_codes[:3])}")
        
        sections.append("\n".join(procedure_lines) if procedure_lines else "No procedure details extracted")
        
        # Section 6: FINDINGS & RESULTS
        sections.append("\n🔍 FINDINGS & RESULTS")
        sections.append("-" * 50)
        
        findings_lines = []
        
        # Intraoperative findings
        intraop_findings = raw_data.get("intraoperative_findings", {})
        if intraop_findings.get("description"):
            findings_lines.append(f"Intraoperative Findings: {intraop_findings['description']}")
        
        # Pathology findings
        pathology_findings = raw_data.get("pathology_findings", {})
        if pathology_findings.get("pathological_diagnosis"):
            findings_lines.append(f"Pathological Diagnosis: {pathology_findings['pathological_diagnosis']}")
        
        if pathology_findings.get("microscopic_description"):
            # Truncate long microscopic descriptions
            micro_desc = pathology_findings['microscopic_description']
            if len(micro_desc) > 200:
                micro_desc = micro_desc[:197] + "..."
            findings_lines.append(f"Microscopic: {micro_desc}")
        
        # Test results
        test_results = raw_data.get("test_results", {})
        if test_results.get("results_summary"):
            findings_lines.append(f"Results Summary: {test_results['results_summary']}")
        
        if test_results.get("interpretation"):
            findings_lines.append(f"Interpretation: {test_results['interpretation']}")
        
        sections.append("\n".join(findings_lines) if findings_lines else "No findings/results extracted")
        
        # Section 7: MEDICATIONS & ANESTHESIA
        sections.append("\n💊 MEDICATIONS & ANESTHESIA")
        sections.append("-" * 50)
        
        meds_anesthesia = raw_data.get("medications_anesthesia", {})
        med_lines = []
        
        # Anesthetic agents
        anesthetic_agents = meds_anesthesia.get("anesthetic_agents", [])
        if anesthetic_agents:
            med_lines.append("Anesthetic Agents:")
            for agent in anesthetic_agents[:5]:
                if isinstance(agent, dict):
                    agent_name = agent.get("name", "")
                    agent_dose = agent.get("dose", "")
                    if agent_name:
                        if agent_dose:
                            med_lines.append(f"  • {agent_name} - {agent_dose}")
                        else:
                            med_lines.append(f"  • {agent_name}")
                elif agent:
                    med_lines.append(f"  • {agent}")
        
        # Intraoperative medications
        intraop_meds = meds_anesthesia.get("intraoperative_medications", [])
        if intraop_meds:
            med_lines.append("\nIntraoperative Medications:")
            for med in intraop_meds[:5]:
                if isinstance(med, dict):
                    med_name = med.get("name", "")
                    med_dose = med.get("dose", "")
                    if med_name:
                        if med_dose:
                            med_lines.append(f"  • {med_name} - {med_dose}")
                        else:
                            med_lines.append(f"  • {med_name}")
                elif med:
                    med_lines.append(f"  • {med}")
        
        sections.append("\n".join(med_lines) if med_lines else "No medication/anesthesia details extracted")
        
        # Section 8: CONCLUSIONS & RECOMMENDATIONS
        sections.append("\n🎯 CONCLUSIONS & RECOMMENDATIONS")
        sections.append("-" * 50)
        
        conclusions = raw_data.get("conclusions_recommendations", {})
        conclusion_lines = []
        
        if conclusions.get("final_diagnosis"):
            conclusion_lines.append(f"Final Diagnosis: {conclusions['final_diagnosis']}")
        
        if conclusions.get("clinical_impressions"):
            conclusion_lines.append(f"Clinical Impressions: {conclusions['clinical_impressions']}")
        
        recommendations = conclusions.get("recommendations", [])
        if recommendations:
            conclusion_lines.append("\nRecommendations:")
            for rec in recommendations[:5]:
                if isinstance(rec, dict):
                    rec_desc = rec.get("recommendation", "")
                    if rec_desc:
                        conclusion_lines.append(f"  • {rec_desc}")
                elif rec:
                    conclusion_lines.append(f"  • {rec}")
        
        follow_up = conclusions.get("follow_up_plan", "")
        if follow_up:
            conclusion_lines.append(f"\nFollow-up Plan: {follow_up}")
        
        sections.append("\n".join(conclusion_lines) if conclusion_lines else "No conclusions/recommendations extracted")
        
        # Section 9: CRITICAL FINDINGS
        sections.append("\n🚨 CRITICAL FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            for finding in critical_findings[:8]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("finding", "")
                    finding_priority = finding.get("priority", "")
                    if finding_desc:
                        if finding_priority:
                            sections.append(f"• [{finding_priority}] {finding_desc}")
                        else:
                            sections.append(f"• {finding_desc}")
                elif finding:
                    sections.append(f"• {finding}")
        else:
            # Check for critical findings in other sections
            critical_items = []
            
            # Check pathology for malignancies
            path_dx = pathology_findings.get("pathological_diagnosis", "").lower()
            if any(term in path_dx for term in ['malignant', 'carcinoma', 'cancer', 'neoplasm']):
                critical_items.append("Malignancy identified in pathology")
            
            # Check for significant complications
            complications = intraop_findings.get("complications", [])
            if complications:
                critical_items.append(f"Procedure complications: {len(complications)} noted")
            
            if critical_items:
                for item in critical_items:
                    sections.append(f"• {item}")
            else:
                sections.append("No critical findings explicitly listed")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Medical report long summary built: {len(long_summary)} characters")
        
        return long_summary
    def _clean_pipes_from_summary(self, short_summary: str) -> str:
            """
            Clean empty pipes from short summary to avoid consecutive pipes or trailing pipes.
            
            Args:
                short_summary: The pipe-delimited short summary string
                
            Returns:
                Cleaned summary with proper pipe formatting
            """
            if not short_summary or '|' not in short_summary:
                return short_summary
            
            # Split by pipe and clean each part
            parts = short_summary.split('|')
            cleaned_parts = []
            
            for part in parts:
                # Remove whitespace and check if part has meaningful content
                stripped_part = part.strip()
                # Keep part if it has actual content (not just empty or whitespace)
                if stripped_part:
                    cleaned_parts.append(stripped_part)
            
            # Join back with pipes - only include parts with actual content
            cleaned_summary = ' . '.join(cleaned_parts)
            
            logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
            return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word structured medical summary.
        Zero hallucinations, pipe-delimited, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word medical structured summary...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a medical-report summarization specialist.

    TASK:
    Create a concise, factual summary of a medical report using ONLY information explicitly present in the long summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:
    [Report Title] | [Author/Physician or The person who signed the report] | [Date] | [Body parts] | [Diagnosis] | [Medication] | [MMI Status] | [Key Action Items] | [Work Status] | [Recommendation] | [Critical Finding] | Urgent Next Steps

    3. DO NOT fabricate or infer missing data — simply SKIP fields that do not exist.
    4. Use ONLY information explicitly found in the long summary.
    5. Output must be a SINGLE LINE (no line breaks).
    6. Content priority:
    - report title
    - author name
    - date
    - affected body parts
    - primary diagnosis
    - medications (if present)
    - MMI status (if present)
    - work status (if present)
    - key recommendation(s) (if present)
    - one critical finding (if present)
    - urgent next steps (if present)
    - follow-up plan (if present)

    7. ABSOLUTE NO:
    - assumptions
    - clinical interpretation
    - invented medications
    - invented dates
    - narrative sentences

    8. If a field is missing, SKIP IT—do NOT write "None" or "Not provided" and simply leave the field empty also donot use | for this field as if 2 fileds are missing then it shows ||

    Your final output must be 30–60 words and MUST follow the exact pipe-delimited format above.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    MEDICAL REPORT LONG SUMMARY:

    {long_summary}

    Produce a 30–60 word structured medical summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Apply pipe cleaning function
            summary = self._clean_pipes_from_summary(summary)

            # Validate 30–60 word range
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Medical summary out of range ({wc} words). Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous output contained {wc} words. Rewrite it to be **between 30 and 60 words**, keeping all factual content, maintaining the pipe-delimited format, and adding NO invented details."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                summary = self._clean_pipes_from_summary(summary)  # Clean pipes again after regeneration

            logger.info(f"✅ Medical summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Medical summary generation failed: {e}")
            return "Summary unavailable due to processing error."

  
    def _create_medical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback medical summary directly from long summary"""
        
        # Extract key medical information using regex patterns
        patterns = {
            'procedure': r'Procedure:\s*([^\n]+)',
            'physician': r'Performing Physician:\s*([^\n]+)',
            'diagnosis': r'Final Diagnosis:\s*([^\n]+)',
            'findings': r'Intraoperative Findings:\s*([^\n]+)',
            'pathology': r'Pathological Diagnosis:\s*([^\n]+)',
            'recommendations': r'Recommendations:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type
        parts.append(f"{doc_type} report")
        
        if 'procedure' in extracted:
            parts.append(f"Procedure: {extracted['procedure']}")
        
        # Add physician context
        if 'physician' in extracted:
            parts.append(f"by {extracted['physician']}")
        
        # Add findings
        if 'findings' in extracted:
            first_finding = extracted['findings'][:80] + "..." if len(extracted['findings']) > 80 else extracted['findings']
            parts.append(f"Findings: {first_finding}")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"Diagnosis: {extracted['diagnosis']}")
        elif 'pathology' in extracted:
            parts.append(f"Pathology: {extracted['pathology']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with standard medical follow-up", "following established protocols", "with routine clinical monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used medical fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, doc_type: str, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for medical reports"""
        return {
            "report_identity": {
                "report_type": doc_type,
                "report_date": fallback_date,
                "procedure_date": "",
                "accession_number": "",
                "medical_record_number": "",
                "facility_name": "",
                "department": ""
            },
            "patient_information": {
                "patient_name": "",
                "patient_dob": "",
                "patient_age": "",
                "patient_gender": "",
                "medical_record_number": "",
                "allergies": "",
                "clinical_history": ""
            },
            "providers": {
                "ordering_physician": {
                    "name": "",
                    "specialty": "",
                    "npi": ""
                },
                "performing_physician": {
                    "name": "",
                    "specialty": "",
                    "credentials": "",
                    "role": "Performing/Attending Physician"
                },
                "assistant_physicians": [],
                "anesthesiologist": {
                    "name": "",
                    "credentials": ""
                },
                "referring_physician": {
                    "name": "",
                    "specialty": ""
                }
            },
            "clinical_context": {
                "indications": "",
                "preoperative_diagnosis": "",
                "postoperative_diagnosis": "",
                "clinical_history_summary": "",
                "relevant_medical_history": []
            },
            "procedure_details": {
                "procedure_name": "",
                "procedure_type": "",
                "cpt_codes": [],
                "icd_codes": [],
                "anatomical_sites": [],
                "laterality": "",
                "technique_used": "",
                "procedure_duration": "",
                "anesthesia_type": "",
                "specimens_collected": []
            },
            "intraoperative_findings": {
                "description": "",
                "key_observations": [],
                "complications": [],
                "blood_loss": "",
                "fluids_administered": "",
                "vital_signs_stability": ""
            },
            "pathology_findings": {
                "gross_description": "",
                "microscopic_description": "",
                "special_stains": [],
                "pathological_diagnosis": "",
                "tumor_characteristics": {
                    "size": "",
                    "margins": "",
                    "grade": "",
                    "stage": ""
                }
            },
            "test_results": {
                "results_summary": "",
                "key_measurements": [],
                "normal_ranges": "",
                "abnormal_findings": [],
                "interpretation": ""
            },
            "medications_anesthesia": {
                "preoperative_medications": [],
                "anesthetic_agents": [],
                "intraoperative_medications": [],
                "postoperative_medications": [],
                "medication_allergies": ""
            },
            "conclusions_recommendations": {
                "final_diagnosis": "",
                "clinical_impressions": "",
                "recommendations": [],
                "follow_up_plan": "",
                "prescriptions": []
            },
            "critical_findings": []
        }