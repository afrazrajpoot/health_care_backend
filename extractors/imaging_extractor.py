"""
Imaging Reports Enhanced Extractor - 6 Critical Imaging Fields Focused

Optimized for MRI, X-ray, CT-scan, and other imaging modalities
Full-context processing with anti-hallucination rules
NO assumptions, NO self-additions, ONLY explicit information from report
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
from utils.doctor_detector import DoctorDetector
from utils.document_context_analyzer import DocumentContextAnalyzer

logger = logging.getLogger("document_ai")


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - 6-FIELD IMAGING FOCUS (Header, Clinical Data, Technique, Key Findings, Impression, Recommendations)
    - Context-aware extraction using DocumentContextAnalyzer
    - ZERO tolerance for hallucination, assumptions, or self-additions
    - Only extracts EXPLICITLY STATED information from imaging reports
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        self.context_analyzer = DocumentContextAnalyzer(llm)
        
        # Pre-compile regex patterns for imaging specific content
        self.imaging_patterns = {
            'modality': re.compile(r'\b(MRI|CT|X-RAY|XRAY|ULTRASOUND|US|MAMMOGRAM|PET|SPECT|DEXA)\b', re.IGNORECASE),
            'body_part': re.compile(r'\b(shoulder|knee|spine|wrist|hip|ankle|elbow|hand|foot|brain|chest|abdomen|pelvis|lumbar|cervical|thoracic)\b', re.IGNORECASE),
            'contrast': re.compile(r'\b(with|without)\s+contrast\b', re.IGNORECASE),
            'finding_severity': re.compile(r'\b(mild|moderate|severe|minimal|marked|advanced|subtle|questionable|probable|likely)\b', re.IGNORECASE)
        }
        
        logger.info("✅ ImagingExtractorChained initialized (6-Field Imaging Focus)")

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
        Extract imaging data with FULL CONTEXT and 6-field focus.
        Returns dictionary with long_summary and short_summary like QME extractor.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (MRI, CT, X-ray, Ultrasound, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
        """
        
        logger.info("=" * 80)
        logger.info("📊 STARTING IMAGING EXTRACTION (6-FIELD FOCUS + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze document context using DocumentContextAnalyzer
            if not context_analysis:
                logger.info("🔍 Analyzing document context for guidance...")
                context_analysis = self.context_analyzer.analyze(text, doc_type)
            
            logger.info("🎯 Context Guidance Received:")
            logger.info(f"   Document Type: {context_analysis.get('document_type', 'Unknown')}")
            logger.info(f"   Confidence: {context_analysis.get('confidence', 'medium')}")
            logger.info(f"   Key Sections: {context_analysis.get('key_sections', [])}")
            
            # Step 2: Extract raw data with full context and zone-aware radiologist detection
            raw_data = self._extract_raw_data(text, doc_type, fallback_date, context_analysis, page_zones)
            
            # Step 3: Build comprehensive long summary from ALL raw data
            long_summary = self._build_comprehensive_long_summary(raw_data, doc_type, fallback_date)
            
            # Step 4: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context imaging extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted imaging data from complete {len(text):,} char document")
            
            logger.info("=" * 80)
            logger.info("✅ IMAGING EXTRACTION COMPLETE (6-FIELD FOCUS)")
            logger.info("=" * 80)
            
            # Return dictionary with both summaries like QME extractor
            return {
                "long_summary": long_summary,
                "short_summary": short_summary
            }
        
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            # Return fallback result structure
            return {
                "long_summary": f"Imaging extraction failed: {str(e)}",
                "short_summary": "Imaging summary not available"
            }

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str, context_analysis: Dict, page_zones: Optional[Dict] = None) -> Dict:
        """Extract raw imaging data using LLM with full context and robust zone-aware radiologist detection"""
        # Detect radiologist using zone-aware logic
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        radiologist_name = detection_result.get("doctor_name")
        radiologist_confidence = detection_result.get("confidence")
        if not radiologist_name:
            radiologist_name = context_analysis.get("identified_professionals", {}).get("primary_provider", "")
        
        # Build comprehensive context guidance for LLM
        context_str = f"""
DOCUMENT CONTEXT ANALYSIS (from DocumentContextAnalyzer):
- Document Type: {context_analysis.get('document_type', 'Unknown')}
- Confidence Level: {context_analysis.get('confidence', 'medium')}
- Key Sections: {', '.join(context_analysis.get('key_sections', ['Findings', 'Impression']))}
- Critical Keywords: {', '.join(context_analysis.get('critical_keywords', [])[:10])}

RADIOLOGIST: {radiologist_name or 'Extract from document'}

CRITICAL EXTRACTION RULES FOR IMAGING:
1. Extract ONLY explicitly stated findings - NO assumptions
2. Include measurements ONLY if explicitly stated in report
3. For work/treatment implications: Use EXACT radiologist wording
4. Empty fields are BETTER than guessed fields
5. Do NOT upgrade severity (mild→moderate) or add typical findings
"""
        
        # Build system prompt with 6-field imaging focus
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert radiological report specialist analyzing a COMPLETE imaging report.

PRIMARY PURPOSE: Extract the 6 critical imaging fields for accurate medical documentation.

CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If NOT explicitly mentioned in report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
   - DO NOT add typical findings, standard protocols, or common measurements
   - DO NOT use radiological training to "complete" incomplete information

Examples of INCORRECT extractions:
   ❌ Report says "mass in right upper lobe" → DO NOT extract size if not stated
   ❌ Report mentions "findings suggest tear" → DO NOT extract "rotator cuff tear" (upgrades certainty)
   ❌ Report shows "mild degenerative changes" → DO NOT upgrade to "moderate" findings
   ❌ Report says "no acute fracture" → DO NOT list fracture as a finding

2. **FINDINGS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY findings explicitly described in FINDINGS or IMPRESSION sections
   - Include measurements ONLY if explicitly stated (not estimated)
   - Include severity qualifiers EXACTLY as stated (mild/moderate/severe)
   - DO NOT extract "possible" or "rule out" as confirmed findings
   - DO NOT add normal variants unless explicitly highlighted as findings

3. **TECHNICAL DETAILS - EXACT WORDING**
   - Contrast status: Use EXACT report wording (with/without/not specified)
   - Body part: Include specific anatomical location and laterality
   - Protocol: Extract only if explicitly named in technique section
   - Quality: Extract only if explicitly assessed in report

4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return empty field than to guess
   - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""
   - DO NOT assume standard protocols or typical measurements

5. **RADIOLOGIST'S EXACT LANGUAGE**
   - Use EXACT terminology from report
   - Preserve certainty qualifiers: "suspicious", "likely", "consistent with", "probable"
   - DO NOT remove nuanced descriptions: "subtle", "questionable", "mild enhancement"
   - DO NOT interpret radiological significance beyond what's stated

6 CRITICAL IMAGING FIELDS:

FIELD 1: HEADER & CONTEXT (Report Identity & Date)
FIELD 2: CLINICAL DATA/INDICATION (Reason for the Study)
FIELD 3: TECHNIQUE/PRIOR STUDIES (Methodology & Comparison)
FIELD 4: KEY FINDINGS - POSITIVE/NEGATIVE (Evidence of Pathology)
FIELD 5: IMPRESSION/CONCLUSION (Radiologist's Final Diagnosis)
FIELD 6: RECOMMENDATIONS/FOLLOW-UP (Actionable Next Steps)

Now analyze this COMPLETE imaging report and extract 6 critical fields:
""")
        
        # Build user prompt
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE IMAGING REPORT TEXT:

{full_document_text}

Extract into STRUCTURED JSON focusing on 6 CRITICAL IMAGING FIELDS:

{{
  "field_1_header_context": {{
    "imaging_center": "",
    "exam_date": "",
    "exam_type": "",
    "patient_name": "",
    "patient_dob": "",
    "referring_physician": "",
    "radiologist": {{
      "name": "{primary_radiologist}",
      "credentials": "",
      "specialty": "Radiology"
    }}
  }},
  
  "field_2_clinical_data": {{
    "clinical_indication": "",
    "clinical_history": "",
    "specific_clinical_questions": "",
    "chief_complaint": ""
  }},
  
  "field_3_technique_prior": {{
    "study_type": "{doc_type}",
    "body_part_imaged": "",
    "laterality": "",
    "contrast_used": "",
    "contrast_type": "",
    "prior_studies_available": "",
    "prior_study_dates": [],
    "technical_quality": "",
    "limitations": ""
  }},
  
  "field_4_key_findings": {{
    "primary_finding": {{
      "description": "",
      "location": "",
      "size": "",
      "characteristics": "",
      "acuity": "acute|chronic|chronic_on_acute",
      "significance": ""
    }},
    "secondary_findings": [
      {{
        "description": "",
        "location": "",
        "significance": ""
      }}
    ],
    "normal_findings": [
      ""
    ]
  }},
  
  "field_5_impression_conclusion": {{
    "overall_impression": "",
    "primary_diagnosis": "",
    "differential_diagnoses": [],
    "clinical_correlation_statement": "",
    "final_diagnostic_statement": ""
  }},
  
  "field_6_recommendations_followup": {{
    "follow_up_recommended": "",
    "follow_up_modality": "",
    "follow_up_timing": "",
    "clinical_correlation_needed": "",
    "specialist_consultation": ""
  }}
}}

⚠️ MANDATORY EXTRACTION RULES:
1. Field 1: Extract EXACT dates and names from report
2. Field 2: Use EXACT clinical indication wording from report
3. Field 3: Contrast status must be EXPLICIT (with/without/not stated)
4. Field 4: Size/measurements ONLY if explicitly stated in report
5. Field 5: Use RADIOLOGIST'S EXACT impression language
6. Field 6: Include recommendations ONLY if explicitly stated
7. EMPTY FIELDS ARE ACCEPTABLE - Better than guessed information
8. NO assumptions, NO additions, NO upgrades of severity
""")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])
        
        # Create chain
        chain = prompt | self.llm | self.parser
        
        logger.info(f"📄 Document size: {len(text):,} chars (~{len(text) // 4:,} tokens)")
        logger.info("🔍 Processing ENTIRE imaging report in single context window with 6-field focus...")
        logger.info("🤖 Invoking LLM for full-context imaging extraction...")
        
        # Invoke LLM
        try:
            raw_data = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_str,
                "doc_type": doc_type,
                "primary_radiologist": radiologist_name or ""
            })
            
            logger.info("✅ Extracted imaging data from complete document")
            return raw_data
        
        except Exception as e:
            logger.error(f"❌ LLM extraction failed: {str(e)}")
            return self._get_fallback_result(fallback_date, doc_type)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted raw data with detailed headings.
        Similar to QME extractor structure.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted imaging data...")
        
        sections = []
        
        # Section 1: IMAGING OVERVIEW
        sections.append("📋 IMAGING OVERVIEW")
        sections.append("-" * 50)
        
        header_context = raw_data.get("field_1_header_context", {})
        radiologist = header_context.get("radiologist", {})
        
        radiologist_name = radiologist.get("name", "")
        exam_date = header_context.get("exam_date", fallback_date)
        exam_type = header_context.get("exam_type", doc_type)
        imaging_center = header_context.get("imaging_center", "")
        referring_physician = header_context.get("referring_physician", "")
        
        overview_lines = [
            f"Document Type: {doc_type}",
            f"Exam Date: {exam_date}",
            f"Exam Type: {exam_type}",
            f"Radiologist: {radiologist_name}",
            f"Imaging Center: {imaging_center}" if imaging_center else "Imaging Center: Not specified",
            f"Referring Physician: {referring_physician}" if referring_physician else "Referring Physician: Not specified"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PATIENT INFORMATION
        sections.append("\n👤 PATIENT INFORMATION")
        sections.append("-" * 50)
        
        patient_lines = [
            f"Name: {header_context.get('patient_name', 'Not specified')}",
            f"Date of Birth: {header_context.get('patient_dob', 'Not specified')}"
        ]
        sections.append("\n".join(patient_lines))
        
        # Section 3: CLINICAL INDICATION
        sections.append("\n🎯 CLINICAL INDICATION")
        sections.append("-" * 50)
        
        clinical_data = raw_data.get("field_2_clinical_data", {})
        clinical_lines = [
            f"Clinical Indication: {clinical_data.get('clinical_indication', 'Not specified')}",
            f"Clinical History: {clinical_data.get('clinical_history', 'Not specified')}",
            f"Chief Complaint: {clinical_data.get('chief_complaint', 'Not specified')}",
            f"Specific Questions: {clinical_data.get('specific_clinical_questions', 'Not specified')}"
        ]
        sections.append("\n".join(clinical_lines))
        
        # Section 4: TECHNICAL DETAILS
        sections.append("\n🔧 TECHNICAL DETAILS")
        sections.append("-" * 50)
        
        technique = raw_data.get("field_3_technique_prior", {})
        technique_lines = [
            f"Study Type: {technique.get('study_type', doc_type)}",
            f"Body Part Imaged: {technique.get('body_part_imaged', 'Not specified')}",
            f"Laterality: {technique.get('laterality', 'Not specified')}",
            f"Contrast Used: {technique.get('contrast_used', 'Not specified')}",
            f"Contrast Type: {technique.get('contrast_type', 'Not specified')}",
            f"Prior Studies Available: {technique.get('prior_studies_available', 'Not specified')}",
            f"Technical Quality: {technique.get('technical_quality', 'Not specified')}",
            f"Limitations: {technique.get('limitations', 'None specified')}"
        ]
        sections.append("\n".join(technique_lines))
        
        # Section 5: KEY FINDINGS (MOST IMPORTANT)
        sections.append("\n📊 KEY FINDINGS")
        sections.append("-" * 50)
        
        key_findings = raw_data.get("field_4_key_findings", {})
        findings_lines = []
        
        # Primary finding
        primary_finding = key_findings.get("primary_finding", {})
        if isinstance(primary_finding, dict):
            primary_desc = primary_finding.get("description", "")
            if primary_desc:
                findings_lines.append("Primary Finding:")
                findings_lines.append(f"  • Description: {primary_desc}")
                
                location = primary_finding.get("location", "")
                if location:
                    findings_lines.append(f"  • Location: {location}")
                
                size = primary_finding.get("size", "")
                if size:
                    findings_lines.append(f"  • Size: {size}")
                
                characteristics = primary_finding.get("characteristics", "")
                if characteristics:
                    findings_lines.append(f"  • Characteristics: {characteristics}")
                
                acuity = primary_finding.get("acuity", "")
                if acuity:
                    findings_lines.append(f"  • Acuity: {acuity}")
        
        # Secondary findings
        secondary_findings = key_findings.get("secondary_findings", [])
        if secondary_findings:
            findings_lines.append("\nSecondary Findings:")
            for finding in secondary_findings[:5]:  # Limit to 5 secondary findings
                if isinstance(finding, dict):
                    finding_desc = finding.get("description", "")
                    finding_location = finding.get("location", "")
                    if finding_desc:
                        if finding_location:
                            findings_lines.append(f"  • {finding_location}: {finding_desc}")
                        else:
                            findings_lines.append(f"  • {finding_desc}")
        
        # Normal findings
        normal_findings = key_findings.get("normal_findings", [])
        if normal_findings:
            findings_lines.append("\nNormal Findings:")
            for normal in normal_findings[:5]:  # Limit to 5 normal findings
                if normal and str(normal).strip():
                    findings_lines.append(f"  • {normal}")
        
        sections.append("\n".join(findings_lines) if findings_lines else "No significant findings extracted")
        
        # Section 6: IMPRESSION & CONCLUSION
        sections.append("\n💡 IMPRESSION & CONCLUSION")
        sections.append("-" * 50)
        
        impression = raw_data.get("field_5_impression_conclusion", {})
        impression_lines = []
        
        overall_impression = impression.get("overall_impression", "")
        if overall_impression:
            impression_lines.append(f"Overall Impression: {overall_impression}")
        
        primary_diagnosis = impression.get("primary_diagnosis", "")
        if primary_diagnosis:
            impression_lines.append(f"Primary Diagnosis: {primary_diagnosis}")
        
        final_diagnostic = impression.get("final_diagnostic_statement", "")
        if final_diagnostic:
            impression_lines.append(f"Final Diagnostic Statement: {final_diagnostic}")
        
        # Differential diagnoses
        differentials = impression.get("differential_diagnoses", [])
        if differentials:
            impression_lines.append("\nDifferential Diagnoses:")
            for dx in differentials[:3]:  # Limit to 3 differentials
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", "")
                    if dx_name:
                        impression_lines.append(f"  • {dx_name}")
                elif dx and str(dx).strip():
                    impression_lines.append(f"  • {dx}")
        
        clinical_correlation = impression.get("clinical_correlation_statement", "")
        if clinical_correlation:
            impression_lines.append(f"\nClinical Correlation: {clinical_correlation}")
        
        sections.append("\n".join(impression_lines) if impression_lines else "No impression/conclusion extracted")
        
        # Section 7: RECOMMENDATIONS & FOLLOW-UP
        sections.append("\n📋 RECOMMENDATIONS & FOLLOW-UP")
        sections.append("-" * 50)
        
        recommendations = raw_data.get("field_6_recommendations_followup", {})
        rec_lines = []
        
        follow_up = recommendations.get("follow_up_recommended", "")
        if follow_up:
            rec_lines.append(f"Follow-up Recommended: {follow_up}")
        
        follow_up_modality = recommendations.get("follow_up_modality", "")
        if follow_up_modality:
            rec_lines.append(f"Follow-up Modality: {follow_up_modality}")
        
        follow_up_timing = recommendations.get("follow_up_timing", "")
        if follow_up_timing:
            rec_lines.append(f"Follow-up Timing: {follow_up_timing}")
        
        clinical_correlation_needed = recommendations.get("clinical_correlation_needed", "")
        if clinical_correlation_needed:
            rec_lines.append(f"Clinical Correlation Needed: {clinical_correlation_needed}")
        
        specialist_consultation = recommendations.get("specialist_consultation", "")
        if specialist_consultation:
            rec_lines.append(f"Specialist Consultation: {specialist_consultation}")
        
        sections.append("\n".join(rec_lines) if rec_lines else "No specific recommendations provided")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Long summary built: {len(long_summary)} characters")
        
        return long_summary

    def _generate_short_summary_from_long_summary(self, long_summary: str) -> str:
        """
        Generate a precise 30–60 word structured imaging summary.
        Zero hallucinations. Pipe-delimited. Skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word structured imaging summary...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a radiology-report summarization specialist.

    TASK:
    Produce a concise structured summary of an imaging report using ONLY details explicitly present in the long summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:

    PR-2 Progress Report | [Author/Physician or The person who signed the report] | [Visit Date] | [Affected Body Parts] | [Impression] | [Medications] | [Primary Diagnosis] | [Work Status] | [Restrictions] | [Treatment Progress] | [Authorization Requests] | [Follow-up Plan] | [Critical Finding]

    3. DO NOT generate narrative sentences.
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
    - impression (if present)
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
    IMAGING REPORT LONG SUMMARY:

    {long_summary}

    Create a strict 30–60 word imaging summary using the required pipe-delimited format.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()

            # Validate 30–60 word requirement
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Imaging summary word count out of range: {wc} words. Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior output was {wc} words. Rewrite it to be between 30–60 words, preserving only factual content, keeping the exact pipe format, and adding NO fabricated details."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())

            return summary

        except Exception as e:
            logger.error(f"❌ Imaging summary generation failed: {e}")
            return "Summary unavailable due to processing error."

    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract radiologist information
        radiologist_match = re.search(r'Radiologist:\s*([^\n]+)', long_summary)
        radiologist = radiologist_match.group(1).strip() if radiologist_match else "Radiologist"
        
        # Extract key information using regex patterns
        patterns = {
            'modality': r'Exam Type:\s*([^\n]+)',
            'body_part': r'Body Part Imaged:\s*([^\n]+)',
            'indication': r'Clinical Indication:\s*([^\n]+)',
            'findings': r'Primary Finding:(.*?)(?:\n\n|\n[A-Z]|$)',
            'impression': r'Overall Impression:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with modality and body part
        if 'modality' in extracted and 'body_part' in extracted:
            parts.append(f"{extracted['modality']} {extracted['body_part']}")
        elif 'modality' in extracted:
            parts.append(f"{extracted['modality']} study")
        
        # Add radiologist
        parts.append(f"by {radiologist}")
        
        # Add indication
        if 'indication' in extracted:
            parts.append(f"for {extracted['indication'][:60]}")
        
        # Add findings
        if 'findings' in extracted:
            # Take first line of findings
            first_finding = extracted['findings'].split('\n')[0].replace('•', '').replace('Description:', '').strip()[:80]
            if first_finding:
                parts.append(f"Findings: {first_finding}")
        
        # Add impression
        if 'impression' in extracted:
            parts.append(f"Impression: {extracted['impression'][:80]}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["comprehensive radiological evaluation", "with diagnostic interpretation", "and clinical implications"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary

    def _safe_str(self, value, default="") -> str:
        """Convert any value to string safely - MUST return STRING"""
        try:
            if value is None:
                return str(default)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return str(value)
            if isinstance(value, list):
                flat = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        flat.append(str(item))
                    elif item:
                        flat.append(str(item))
                return ", ".join(flat) if flat else str(default)
            return str(value)
        except Exception as e:
            logger.error(f"Error in _safe_str: {str(e)}")
            return str(default)

    def _get_fallback_result(self, fallback_date: str, doc_type: str) -> Dict:
        """Return fallback result structure matching 6-field imaging structure"""
        return {
            "field_1_header_context": {
                "imaging_center": "",
                "exam_date": fallback_date,
                "exam_type": doc_type,
                "patient_name": "",
                "patient_dob": "",
                "referring_physician": "",
                "radiologist": {
                    "name": "",
                    "credentials": "",
                    "specialty": "Radiology"
                }
            },
            "field_2_clinical_data": {
                "clinical_indication": "",
                "clinical_history": "",
                "specific_clinical_questions": "",
                "chief_complaint": ""
            },
            "field_3_technique_prior": {
                "study_type": doc_type,
                "body_part_imaged": "",
                "laterality": "",
                "contrast_used": "",
                "contrast_type": "",
                "prior_studies_available": "",
                "prior_study_dates": [],
                "technical_quality": "",
                "limitations": ""
            },
            "field_4_key_findings": {
                "primary_finding": {
                    "description": "",
                    "location": "",
                    "size": "",
                    "characteristics": "",
                    "acuity": "",
                    "significance": ""
                },
                "secondary_findings": [],
                "normal_findings": []
            },
            "field_5_impression_conclusion": {
                "overall_impression": "",
                "primary_diagnosis": "",
                "differential_diagnoses": [],
                "clinical_correlation_statement": "",
                "final_diagnostic_statement": ""
            },
            "field_6_recommendations_followup": {
                "follow_up_recommended": "",
                "follow_up_modality": "",
                "follow_up_timing": "",
                "clinical_correlation_needed": "",
                "specialist_consultation": ""
            }
        }