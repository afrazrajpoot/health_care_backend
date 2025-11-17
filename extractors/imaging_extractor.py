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
    ) -> ExtractionResult:
        """
        Extract imaging data with FULL CONTEXT and 6-field focus.
        
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
            
            # Step 3: Build initial result
            result = self._build_initial_result(raw_data, doc_type, fallback_date, context_analysis)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context imaging extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted imaging data from complete {len(text):,} char document")
            
            logger.info("=" * 80)
            logger.info("✅ IMAGING EXTRACTION COMPLETE (6-FIELD FOCUS)")
            logger.info("=" * 80)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            raise

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

═══════════════════════════════════════════════════════════════════════
FIELD 1: HEADER & CONTEXT (Report Identity & Date)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Imaging Center/Facility
- Date of Exam (EXACT date from report)
- Type of Exam (e.g., "MRI Lumbar Spine without contrast")
- Patient Name and DOB
- Referring Physician Name
- Radiologist Name and Credentials

Why Critical: Establishes authenticity, timeliness, and clinical authority

═══════════════════════════════════════════════════════════════════════
FIELD 2: CLINICAL DATA/INDICATION (Reason for the Study)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Clinical Indication (why study was ordered - EXACT wording)
- Relevant Clinical History (pertinent history from report)
- Specific Clinical Questions (if stated)
- Chief Complaint or Symptom prompting study

Why Critical: Links imaging findings to clinical context and validates medical necessity

Example:
✅ CORRECT: "Indication: Left shoulder pain, history of work injury. History: Patient reports ongoing pain with lifting activities"
❌ WRONG: "Indicate: Shoulder evaluation" (incomplete/changed wording)

═══════════════════════════════════════════════════════════════════════
FIELD 3: TECHNIQUE/PRIOR STUDIES (Methodology & Comparison)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Type of Study and Specific Protocol
- Use of Contrast (WITH or WITHOUT - EXACT)
- Body Part Imaged and Laterality
- Prior Studies Available for Comparison (yes/no and date if stated)
- Technical Quality Assessment
- Any Limitations Noted

Why Critical: Understanding technique affects interpretation validity; prior comparisons show progression/regression

Examples:
✅ CORRECT: "MRI right knee with and without contrast. No prior studies available for comparison"
✅ CORRECT: "CT chest with IV contrast - adequate study, no artifacts"
❌ WRONG: "Study was performed" (doesn't specify modality, body part, or contrast)

═══════════════════════════════════════════════════════════════════════
FIELD 4: KEY FINDINGS - POSITIVE/NEGATIVE (Evidence of Pathology)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Primary Abnormality (most clinically significant finding)
- Location and Size (ONLY if explicitly measured)
- Specific Characteristics (e.g., disc herniation type, fracture pattern)
- Acute vs Chronic Assessment (if stated)
- Secondary Findings
- Normal Findings (what is NOT present/abnormal)

Why Critical: Objective evidence that supports or refutes diagnosis and treatment decisions

CRITICAL ANTI-HALLUCINATION RULES FOR FINDINGS:

1. SIZE/MEASUREMENTS: Include ONLY if explicitly stated
   ✅ CORRECT: Report states "3 cm disc herniation at L5-S1" → Extract: "3 cm disc herniation at L5-S1"
   ❌ WRONG: Report states "disc herniation at L5-S1" → DO NOT add: "3 cm" (size not stated)
   ✅ CORRECT: Extract measurement field as EMPTY if not mentioned

2. SEVERITY: Use EXACT qualifiers from report
   ✅ CORRECT: Report states "mild degenerative changes" → Extract: "mild degenerative changes"
   ❌ WRONG: Report states "mild degenerative changes" → DO NOT extract: "moderate changes"

3. POSITIVE vs NEGATIVE: Extract exactly as stated
   ✅ CORRECT: Report states "no acute fracture" → Mark as: negative finding
   ❌ WRONG: Report states "no acute fracture" → DO NOT list as: "fracture present"

4. CHARACTERISTIC FINDINGS: Use exact radiological language
   ✅ CORRECT: "Anterior disc bulge compressing anterior thecal sac"
   ❌ WRONG: "Disc compression" (loses specific detail)

Examples of CORRECT Field 4 Extraction:
- Primary Finding: "Acute L5-S1 disc herniation"
- Location: "L5-S1 intervertebral space"
- Size: "12mm"
- Characteristics: "Subligamentous, compressing anterior thecal sac"
- Acute/Chronic: "Acute"
- Secondary Findings: ["Mild degenerative changes L4-5", "Small anterior osteophytes"]
- Normal Findings: ["No fracture", "Normal alignment", "Patent neural foramina"]

═══════════════════════════════════════════════════════════════════════
FIELD 5: IMPRESSION/CONCLUSION (Radiologist's Final Diagnosis)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Overall Impression (summary statement from report)
- Primary Diagnosis (main finding/diagnosis)
- Differential Diagnoses (if provided by radiologist)
- Clinical Correlation Statement
- Final Diagnostic Statement (e.g., "Features of acute L5-S1 disc herniation")

Why Critical: This is the DEFINITIVE clinical conclusion; most critical for treating physician

EXACT WORDING RULE:
- Extract radiologist's EXACT language from impression section
- Preserve all qualifying language: "features of", "consistent with", "likely"
- DO NOT simplify or interpret radiologist's conclusion

Examples:
✅ CORRECT: Radiologist states "Features consistent with full-thickness rotator cuff tear"
   Extract: "Features consistent with full-thickness rotator cuff tear"
❌ WRONG: Extract only "rotator cuff tear" (removes "features consistent with" qualifier)

═══════════════════════════════════════════════════════════════════════
FIELD 6: RECOMMENDATIONS/FOLLOW-UP (Actionable Next Steps)
═══════════════════════════════════════════════════════════════════════

Critical Elements:
- Specific Follow-up Recommended (EXACT wording)
- Follow-up Modality if Different (e.g., "suggest CT for better characterization")
- Follow-up Timing if Stated (e.g., "6 month follow-up ultrasound")
- Clinical Correlation Request
- Any Specific Clinical Actions (e.g., "Orthopedic consultation recommended")

Why Critical: Guides treating physician on next steps in care

CRITICAL RULE: Extract ONLY recommendations explicitly stated
❌ WRONG: Radiologist doesn't recommend follow-up → DO NOT suggest "routine follow-up"
✅ CORRECT: Leave recommendations empty if none stated

Example:
✅ CORRECT: Report states "Follow-up MRI in 3 months if clinically indicated"
   Extract: "Follow-up MRI in 3 months if clinically indicated"
❌ WRONG: Report doesn't mention follow-up → DO NOT extract "Routine follow-up recommended"

═══════════════════════════════════════════════════════════════════════

EXTRACTION STRATEGY:

Priority flow:
1. WHO, WHAT, WHEN: (Field 1) Radiologist performed (Modality) on (Date)
2. WHY: Exam was ordered for (Field 2 - Clinical Indication)
3. THE FINDINGS: What was actually seen (Field 4 - Key Findings)
4. THE DIAGNOSIS: What it means (Field 5 - Impression)
5. NEXT STEPS: What to do about it (Field 6 - Recommendations)

⚠️ FINAL CRITICAL REMINDER:
- If information is NOT EXPLICITLY in report → return EMPTY ("" or [])
- NEVER assume, infer, extrapolate, or use medical knowledge to fill gaps
- Findings ONLY: Extract what is explicitly described
- It is BETTER to have empty fields than INCORRECT information
- Do NOT upgrade severity, add typical findings, or interpret beyond radiologist's statement

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

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str, context_analysis: Dict = None) -> ExtractionResult:
        """Build initial result from extracted imaging data"""
        
        if context_analysis is None:
            context_analysis = {}
        
        logger.info("🔨 Building initial imaging extraction result...")
        
        try:
            # Extract from 6-field structure
            header_context = raw_data.get("field_1_header_context", {})
            radiologist = header_context.get("radiologist", {})
            key_findings = raw_data.get("field_4_key_findings", {})
            
            # Build comprehensive imaging summary
            summary_line = self._build_imaging_narrative_summary(raw_data, doc_type, fallback_date)
            
            # CRITICAL: Ensure summary_line is STRING
            if not isinstance(summary_line, str):
                summary_line = str(summary_line) if summary_line else "Imaging summary not available"
            
            result = ExtractionResult(
                document_type=doc_type,
                document_date=header_context.get("exam_date", fallback_date),
                summary_line=summary_line,  # MUST be STRING
                examiner_name=radiologist.get("name", ""),
                specialty=radiologist.get("specialty", "Radiology"),
                body_parts=[header_context.get("body_part_imaged", "")] if header_context.get("body_part_imaged") else [],
                raw_data=raw_data,
            )
            
            logger.info(f"✅ Initial imaging result built (Radiologist: {result.examiner_name})")
            return result
        
        except Exception as e:
            logger.error(f"❌ Error building initial result: {str(e)}")
            raise

    def _build_imaging_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive narrative summary for imaging reports.
        
        Format: WHO performed WHAT on WHEN for WHY resulting in DIAGNOSIS with RECOMMENDATIONS
        """
        
        try:
            # Extract all data from 6-field structure
            header_context = data.get("field_1_header_context", {})
            clinical_data = data.get("field_2_clinical_data", {})
            technique = data.get("field_3_technique_prior", {})
            key_findings = data.get("field_4_key_findings", {})
            impression = data.get("field_5_impression_conclusion", {})
            recommendations = data.get("field_6_recommendations_followup", {})
            
            # Build narrative sections
            narrative_parts = []
            
            # Section 0: RADIOLOGIST & DATE CONTEXT
            radiologist = header_context.get("radiologist", {})
            radiologist_name = self._safe_str(radiologist.get("name", ""))
            exam_date = self._safe_str(header_context.get("exam_date", fallback_date))
            
            if radiologist_name:
                context_line = f"Radiologist: {radiologist_name.strip()} on {exam_date if exam_date else fallback_date}"
                narrative_parts.append(context_line)
            
            # Section 1: STUDY IDENTIFICATION & TECHNIQUE
            study_text = self._build_study_narrative(header_context, technique, doc_type)
            if study_text and isinstance(study_text, str) and study_text.strip():
                narrative_parts.append(f"**Study:** {study_text.strip()}")
            
            # Section 2: CLINICAL INDICATION (WHY study was done)
            indication_text = self._safe_str(clinical_data.get("clinical_indication", ""))
            if indication_text and indication_text.strip():
                narrative_parts.append(f"**Indication:** {indication_text.strip()}")
            
            # Section 3: KEY FINDINGS (WHAT was found - MOST IMPORTANT)
            findings_text = self._build_findings_narrative(key_findings)
            if findings_text and isinstance(findings_text, str) and findings_text.strip():
                narrative_parts.append(f"**Findings:** {findings_text.strip()}")
            
            # Section 4: IMPRESSION & DIAGNOSIS (WHAT IT MEANS)
            impression_text = self._build_impression_narrative(impression)
            if impression_text and isinstance(impression_text, str) and impression_text.strip():
                narrative_parts.append(f"**Impression:** {impression_text.strip()}")
            
            # Section 5: RECOMMENDATIONS & FOLLOW-UP (NEXT STEPS)
            recommendations_text = self._build_recommendations_narrative(recommendations)
            if recommendations_text and isinstance(recommendations_text, str) and recommendations_text.strip():
                narrative_parts.append(f"**Recommendations:** {recommendations_text.strip()}")
            
            # Filter and join
            valid_parts = [str(part) for part in narrative_parts if part and isinstance(part, str) and part.strip()]
            full_narrative = "\n\n".join(valid_parts)
            
            logger.info(f"📝 Imaging narrative summary generated: {len(full_narrative)} characters")
            return full_narrative if full_narrative and isinstance(full_narrative, str) else "Imaging summary not available"
        
        except Exception as e:
            logger.error(f"❌ Error building imaging narrative: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _build_study_narrative(self, header_context: Dict, technique: Dict, doc_type: str) -> str:
        """Build study identification and technique narrative"""
        try:
            study_parts = []
            
            # Modality and body part
            study_type = self._safe_str(technique.get("study_type", doc_type))
            body_part = self._safe_str(technique.get("body_part_imaged", ""))
            laterality = self._safe_str(technique.get("laterality", ""))
            
            if study_type:
                study_str = study_type.strip()
                if body_part:
                    study_str += f" {body_part.strip()}"
                if laterality and laterality.strip():
                    study_str += f" ({laterality.strip()})"
                study_parts.append(study_str)
            
            # Contrast status (CRITICAL - exact wording)
            contrast = self._safe_str(technique.get("contrast_used", ""))
            if contrast and contrast.strip():
                study_parts.append(contrast.strip())
            
            # Technical quality
            quality = self._safe_str(technique.get("technical_quality", ""))
            if quality and quality.strip() and quality.strip().lower() != "diagnostic":
                study_parts.append(f"Quality: {quality.strip()}")
            
            # Limitations
            limitations = self._safe_str(technique.get("limitations", ""))
            if limitations and limitations.strip():
                study_parts.append(f"Limitations: {limitations.strip()}")
            
            return " - ".join(study_parts) if study_parts else f"{doc_type} study"
        except Exception as e:
            logger.error(f"Error in _build_study_narrative: {str(e)}")
            return ""

    def _build_findings_narrative(self, key_findings: Dict) -> str:
        """Build key findings narrative - CRITICAL SECTION"""
        try:
            findings_items = []
            
            # Primary finding (highest priority - MUST include)
            primary = key_findings.get("primary_finding", {})
            if isinstance(primary, dict):
                primary_desc = self._safe_str(primary.get("description", ""))
                if primary_desc and primary_desc.strip():
                    # Add complete primary finding with details
                    primary_text = primary_desc.strip()
                    
                    location = self._safe_str(primary.get("location", ""))
                    if location and location.strip():
                        primary_text += f" at {location.strip()}"
                    
                    size = self._safe_str(primary.get("size", ""))
                    if size and size.strip():
                        primary_text += f" ({size.strip()})"
                    
                    findings_items.append(primary_text)
            
            # Secondary findings (important but not critical)
            secondary = key_findings.get("secondary_findings", [])
            if secondary and isinstance(secondary, list):
                for finding in secondary[:3]:  # Top 3 secondary findings
                    if isinstance(finding, dict):
                        finding_desc = self._safe_str(finding.get("description", ""))
                        if finding_desc and finding_desc.strip():
                            findings_items.append(finding_desc.strip())
            
            # Normal findings (what is NOT present - important for ruling out)
            normal = key_findings.get("normal_findings", [])
            if normal and isinstance(normal, list):
                normal_items = [self._safe_str(n).strip() for n in normal if n and self._safe_str(n).strip()]
                if normal_items:
                    findings_items.append(f"Normal: {'; '.join(normal_items[:2])}")
            
            return "; ".join(findings_items) if findings_items else "No significant abnormalities noted"
        except Exception as e:
            logger.error(f"Error in _build_findings_narrative: {str(e)}")
            return ""

    def _build_impression_narrative(self, impression: Dict) -> str:
        """Build impression and diagnostic conclusion narrative"""
        try:
            impression_parts = []
            
            # Primary diagnosis (most important)
            primary_dx = self._safe_str(impression.get("primary_diagnosis", ""))
            if primary_dx and primary_dx.strip():
                impression_parts.append(primary_dx.strip())
            
            # Overall impression (radiologist's summary)
            overall = self._safe_str(impression.get("overall_impression", ""))
            if overall and overall.strip() and overall.strip() != primary_dx:
                impression_parts.append(overall.strip())
            
            # Final diagnostic statement (radiologist's definitive conclusion)
            final_dx = self._safe_str(impression.get("final_diagnostic_statement", ""))
            if final_dx and final_dx.strip() and final_dx.strip() not in impression_parts:
                impression_parts.append(final_dx.strip())
            
            # Differential diagnoses if provided (helps guide treatment)
            differentials = impression.get("differential_diagnoses", [])
            if differentials and isinstance(differentials, list):
                diff_list = [self._safe_str(d).strip() for d in differentials[:2] if d]
                if diff_list:
                    impression_parts.append(f"Differential: {', '.join(diff_list)}")
            
            return "; ".join(impression_parts) if impression_parts else "Impression not specified"
        except Exception as e:
            logger.error(f"Error in _build_impression_narrative: {str(e)}")
            return ""

    def _build_recommendations_narrative(self, recommendations: Dict) -> str:
        """Build recommendations and follow-up narrative"""
        try:
            rec_items = []
            
            # Follow-up recommendations (what radiologist recommends)
            follow_up = self._safe_str(recommendations.get("follow_up_recommended", ""))
            if follow_up and follow_up.strip():
                rec_items.append(follow_up.strip())
            
            # Follow-up timing (when to follow up)
            timing = self._safe_str(recommendations.get("follow_up_timing", ""))
            if timing and timing.strip():
                rec_items.append(f"Timeline: {timing.strip()}")
            
            # Clinical correlation needed
            correlation = self._safe_str(recommendations.get("clinical_correlation_needed", ""))
            if correlation and correlation.strip():
                rec_items.append(f"Correlation: {correlation.strip()}")
            
            # Specialist consultation
            specialist = self._safe_str(recommendations.get("specialist_consultation", ""))
            if specialist and specialist.strip():
                rec_items.append(f"Consultation: {specialist.strip()}")
            
            return "; ".join(rec_items) if rec_items else ""
        except Exception as e:
            logger.error(f"Error in _build_recommendations_narrative: {str(e)}")
            return ""

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