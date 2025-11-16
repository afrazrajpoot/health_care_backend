"""
QME/AME/IME Enhanced Extractor - Full Context with Context-Awareness
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class QMEExtractorChained:
    """
    Enhanced QME extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Uses DocumentContextAnalyzer guidance = Context-aware extraction
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        
        logger.info("✅ QMEExtractorChained initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,  # NEW: from DocumentContextAnalyzer
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract QME data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer (CRITICAL)
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING QME EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Log context guidance if available
        if context_analysis:
            primary_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Primary Physician: {primary_physician.get('name', 'Unknown')}")
            logger.info(f"   Confidence: {primary_physician.get('confidence', 'Unknown')}")
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
            doc_type=doc_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )

        # Log medicine/medications extraction
        meds = None
        if 'current_medications' in raw_result:
            meds = raw_result['current_medications']
        elif 'medications' in raw_result:
            meds = raw_result['medications']
        if meds:
            logger.info(f"✅ Extracted medications: {meds}")
        else:
            logger.warning("⚠️ No 'current_medications' or 'medications' found in QME extraction result.")

        # Stage 2: Override physician if context identified one with high confidence
        if context_analysis:
            context_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            if context_physician.get("name") and context_physician.get("confidence") in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified physician: {context_physician.get('name')}")
                raw_result["qme_physician_name"] = context_physician.get("name")

        # Stage 3: Fallback to DoctorDetector if no physician identified
        if not raw_result.get("qme_physician_name"):
            logger.info("🔍 No physician from context/extraction, using DoctorDetector...")
            examiner_name = self._detect_examiner(text, page_zones)
            raw_result["qme_physician_name"] = examiner_name

        # Stage 4: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)

        # Stage 5: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)

        logger.info("=" * 80)
        logger.info("✅ QME EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        return final_result

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        This mimics Gemini's approach of processing the entire document at once.
        """
        logger.info("🔍 Processing ENTIRE document in single context window with guidance...")
        
        # Extract guidance from context analysis
        primary_physician = ""
        focus_sections = []
        critical_locations = {}
        physician_reasoning = ""
        ambiguities = []
        
        if context_analysis:
            phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            primary_physician = phys_analysis.get("name", "")
            physician_reasoning = phys_analysis.get("reasoning", "")
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE QME/AME/IME report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing you to:
- Understand the complete case narrative from start to finish
- Connect findings across all sections (history → examination → conclusions)
- Identify relationships between symptoms, diagnoses, and recommendations
- Provide comprehensive, context-aware extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate information
   - DO NOT fill in "typical" or "common" values
   - DO NOT use medical knowledge to "complete" incomplete information
   
   Examples:
   ✅ CORRECT: If document says "Patient takes Gabapentin 300mg TID", extract: {{"name": "Gabapentin", "dose": "300mg TID"}}
   ❌ WRONG: If document says "Patient takes Gabapentin", DO NOT extract: {{"name": "Gabapentin", "dose": "300mg TID"}} (dose not stated)
   ✅ CORRECT: Extract: {{"name": "Gabapentin", "dose": ""}} (dose field empty)

2. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY medications explicitly listed in the "Current Medications" or "Medications" section
   - Include dosage ONLY if explicitly stated
   - DO NOT extract:
     * Medications mentioned as discontinued
     * Medications mentioned in past medical history
     * Medications recommended for future use (put those in future_medications)
     * Medications you "think" the patient should be taking
   
   Examples:
   ✅ CORRECT: Document states "Current Medications: Gabapentin 300mg TID, Meloxicam 15mg daily"
   Extract: {{"current_medications": [{{"name": "Gabapentin", "dose": "300mg TID"}}, {{"name": "Meloxicam", "dose": "15mg daily"}}]}}
   
   ❌ WRONG: Document states "Patient previously took Oxycodone but discontinued 6 months ago"
   DO NOT extract Oxycodone in current_medications
   
   ❌ WRONG: Document states "Consider adding Amitriptyline for sleep"
   DO NOT extract Amitriptyline in current_medications (put in future_medications)

3. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
   Examples:
   ✅ CORRECT: If no pain score mentioned, return: "pain_score_current": ""
   ❌ WRONG: Return: "pain_score_current": "Not mentioned" (use empty string instead)

4. **EXACT QUOTES FOR CRITICAL FIELDS**
   - For MMI status, WPI, Work Restrictions: use EXACT wording from document
   - DO NOT paraphrase or interpret
   - If exact value not found, return empty
   
   Examples:
   ✅ CORRECT: Document says "Patient has reached MMI as of 10/15/2024"
   Extract: "mmi_status": {{"status": "Patient has reached MMI as of 10/15/2024"}}
   
   ❌ WRONG: Document says "Patient improving with treatment"
   DO NOT extract: "mmi_status": {{"status": "Not at MMI"}} (this is inference, not stated)

5. **NO CLINICAL ASSUMPTIONS**
   - DO NOT assume typical dosages, frequencies, or durations
   - DO NOT assume standard procedures or treatments
   - DO NOT assume body parts if not explicitly stated
   
   Examples:
   ❌ WRONG: Document mentions "knee injection"
   DO NOT assume: "corticosteroid injection" (steroid type not stated)
   ✅ CORRECT: Extract: "knee injection" (exact wording)

6. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
   Before returning your extraction, verify:
   □ Every medication has a direct quote in the document
   □ Every diagnosis is explicitly stated (not inferred from symptoms)
   □ Every recommendation is directly from "Recommendations" or "Plan" section
   □ No fields are filled with "typical" or "standard" values
   □ Empty fields are truly empty (not "Not mentioned" or "Unknown")

EXTRACTION FOCUS - 6 CRITICAL MEDICAL-LEGAL CATEGORIES:

I. CORE IDENTITY
- Patient name, age, DOB
- Date of Injury (DOI) - often in history section
- Report date - check header and conclusion
- QME Physician: **USE THE PRIMARY PHYSICIAN IDENTIFIED IN CONTEXT GUIDANCE ABOVE**
  * This is the REPORT AUTHOR, not treating physicians mentioned in history
  * Reasoning: {physician_reasoning}

II. DIAGNOSIS
- Primary diagnosis(es) - synthesize from examination findings AND conclusion
- ICD-10 codes if mentioned anywhere in document
- Affected body part(s) - consolidate all mentions throughout document

III. CLINICAL STATUS
- Past surgeries - scan entire history section for surgical history
- Current chief complaint - patient's own words from subjective section
- Pain score (current/max on 0-10 scale) - look in subjective complaints
- Objective findings:✅ Extraction complete
  * ROM limitations - from physical examination section
  * Gait abnormalities - from observation/ambulation section
  * Positive tests - from clinical tests section (e.g., Hawkins, Neer, McMurray)
  * Effusion/swelling - from inspection/palpation findings

IV. MEDICATIONS ⚠️ CRITICAL - ZERO ASSUMPTIONS
- Current medications - from medication list or current medications section
- **ONLY extract medications EXPLICITLY listed as "current" or "taking"**
- **DO NOT extract discontinued, past, or recommended future medications**
- Categorize into: narcotics/opioids, nerve pain meds, anti-inflammatories, other
- Include dosages ONLY if explicitly stated (e.g., "Gabapentin 300mg TID")
- If dosage not stated, leave dose field empty
- Focus on CURRENT medications, not historical discontinued meds

Example extraction:
Document states: "Current Medications: 1. Gabapentin 300mg three times daily, 2. Meloxicam 15mg once daily, 3. Tramadol 50mg as needed for pain. Patient discontinued Oxycodone 3 months ago."

✅ CORRECT extraction:
{{
  "medications": {{
    "current_medications": [
      {{"name": "Gabapentin", "dose": "300mg TID", "purpose": "nerve pain"}},
      {{"name": "Meloxicam", "dose": "15mg daily", "purpose": "anti-inflammatory"}},
      {{"name": "Tramadol", "dose": "50mg PRN", "purpose": "pain"}}
    ]
  }}
}}

❌ WRONG - DO NOT include:
- Oxycodone (discontinued)
- Any medications not explicitly listed

V. MEDICAL-LEGAL CONCLUSIONS (MOST CRITICAL - HIGHEST PRIORITY)
**FOCUS ON THESE SECTIONS:** {focus_sections}
**CRITICAL LOCATIONS:** {critical_locations}

- MMI/P&S Status:
  * Look for explicit statement (e.g., "Patient has reached MMI as of [date]")
  * If MMI deferred, extract SPECIFIC REASON (e.g., "pending MRI results", "awaiting surgical outcome")
  * Location hint: {mmi_location}

- WPI (Whole Person Impairment):
  * Look for percentage WITH body part (e.g., "15% WPI to left shoulder")
  * Include method used (e.g., "per AMA Guides 5th Edition")
  * If WPI deferred, extract SPECIFIC REASON
  * Location hint: {wpi_location}


VI. ACTIONABLE RECOMMENDATIONS (SECOND HIGHEST PRIORITY)
**These are critical for immediate clinical action:**

- Future treatment: Be SPECIFIC
  * Surgeries: Include procedure name and body part (e.g., "total knee arthroplasty")
  * Injections: Include type and location (e.g., "ESI C5-6", "corticosteroid injection R shoulder")
  * Therapy: Include type and frequency (e.g., "PT 2x/week for 6 weeks")
  * Diagnostics: Include test type and body part (e.g., "MRI L-spine without contrast")

- Work restrictions: Extract EXACT functional limitations
  * Be specific: "no lifting >10 lbs" not "modified duty"
  * Include positional restrictions: "no overhead reaching", "no kneeling/squatting"
  * Include duration if stated: "restrictions for 8 weeks"
  * Location hint: {work_restrictions_location}

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- MEDICATIONS: Only extract what is explicitly listed as current
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE QME report and extract ALL relevant information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE QME/AME/IME DOCUMENT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical details:

{{
  "patient_information": {{
    "patient_name": "",
    "patient_age": "",
    "patient_dob": "",
    "date_of_injury": "",
    "claim_number": "",
    "employer": ""
  }},
  
  "report_metadata": {{
    "report_title": "",
    "report_date": "",
    "evaluation_date": "",
    "report_type": "QME/AME/IME"
  }},
  
  "physicians": {{
    "qme_physician": {{
      "name": "{primary_physician}",
      "specialty": "",
      "credentials": "",
      "role": "Evaluating Physician/QME/AME"
    }},
    "treating_physicians": [],
    "consulting_physicians": [],
    "referring_source": {{
      "name": "",
      "type": ""
    }}
  }},
  
  "diagnosis": {{
    "primary_diagnoses": [],
    "secondary_diagnoses": [],
    "historical_conditions": []
  }},
  
  "clinical_status": {{
    "chief_complaint": "",
    "pain_scores": {{
      "current": "",
      "maximum": "",
      "location": ""
    }},
    "functional_limitations": [],
    "past_surgeries": [],
    "objective_findings": {{
      "rom_limitations": "",
      "gait": "",
      "positive_tests": "",
      "other_findings": ""
    }}
  }},
  
  "medications": {{
    "current_medications": [],
    "future_medications": []
  }},
  
  "treatment_history": {{
    "past_treatments": [],
    "current_treatments": []
  }},
  
  "medical_legal_conclusions": {{
    "mmi_status": {{
      "status": "",
      "reason": "",
      "reasoning": ""
    }},
    "wpi_impairment": {{
      "total_wpi": "",
      "breakdown": [],
      "reasoning": ""
    }},
  }},
  
  "work_status": {{
    "current_status": "",
    "work_restrictions": [],
    "prognosis_for_return_to_work": ""
  }},
  
  "recommendations": {{
    "diagnostic_tests": [],
    "interventional_procedures": [],
    "specialist_referrals": [],
    "therapy": [],
    "future_surgical_needs": []
  }},
  
  "critical_findings": []
}}

⚠️ CRITICAL REMINDERS:
1. For "work_restrictions": Extract EXACT wording from document
   - If document says "no lifting", extract: "no lifting" (NOT "no lifting >10 lbs")
   - If document says "no standing", extract: "no standing" (NOT "no prolonged standing >15 min")
   - DO NOT add weight limits, time limits, or specifics not stated

2. For "current_medications": Extract ONLY from "Current Medications" section
   - Include dosage ONLY if explicitly stated
   - DO NOT extract discontinued medications
   - DO NOT extract recommended future medications (use future_medications for those)

3. For "critical_findings": Include MAIN actionable points only (max 5-8 items)
   - Focus on: MMI status, required procedures, required QMEs, important diagnostic tests
   - DO NOT include minor details or routine follow-ups
""")

        # Build context guidance summary
        context_guidance_text = f"""
PRIMARY PHYSICIAN (Report Author): {primary_physician or 'Not identified in context'}
REASONING: {physician_reasoning or 'See document for identification'}

FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- MMI Status: {critical_locations.get('mmi_location', 'Search entire document')}
- WPI Percentage: {critical_locations.get('wpi_location', 'Search entire document')}
- Work Restrictions: {critical_locations.get('work_restrictions_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_guidance_text,
                "primary_physician": primary_physician or "Extract from document",
                "physician_reasoning": physician_reasoning or "Use credentials and signature section",
                "focus_sections": ', '.join(focus_sections) if focus_sections else "All sections",
                "critical_locations": str(critical_locations),
                "mmi_location": critical_locations.get('mmi_location', 'Search document'),
                "wpi_location": critical_locations.get('wpi_location', 'Search document'),
                "work_restrictions_location": critical_locations.get('work_restrictions_location', 'Search document'),
                "ambiguities": str(ambiguities)
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char document")
            
            # Log extracted categories
            # self._log_extracted_categories(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(fallback_date)

    def _detect_examiner(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Fallback: Detect QME/AME examiner using DoctorDetector"""
        logger.info("🔍 Fallback: Running DoctorDetector for QME physician...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ QME Physician detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid QME physician found")
            return ""

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result from extracted data"""
        logger.info("🔨 Building initial extraction result...")
        
        # Extract core identity
        cat1 = raw_data.get("category_1_core_identity", {})
        cat2 = raw_data.get("category_2_diagnosis", {})
        cat5 = raw_data.get("category_5_medical_legal_conclusions", {})
        
        # Build summary
        summary_line = self._build_comprehensive_narrative_summary(raw_data, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cat1.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=raw_data.get("qme_physician_name", ""),
            specialty=cat1.get("qme_specialty", ""),
            body_parts=cat2.get("affected_body_parts", []),
            medications=raw_data.get("medications"),
            raw_data=raw_data,
        )
        
        logger.info(f"✅ Initial result built (Physician: {result.examiner_name})")
        return result

    def _build_comprehensive_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive narrative summary matching Gemini's style.
        
        Gemini style: "Diagnosis: [conditions]. Work Status: [status]. Treatment/Critical Findings: [actions]."
        """
        
        # Extract all data
        patient_info = data.get("patient_information", {})
        physicians = data.get("physicians", {})
        diagnosis = data.get("diagnosis", {})
        medical_legal = data.get("medical_legal_conclusions", {})
        work_status = data.get("work_status", {})
        recommendations = data.get("recommendations", {})
        critical_findings = data.get("critical_findings", [])
        
        # Fallback to old structure if new structure not found
        if not diagnosis and data.get("category_2_diagnosis"):
            # Using old 6-category structure
            return self._build_medical_legal_summary_legacy(data, doc_type, fallback_date)
        
        # Helper function for safe string conversion
        def safe_str(value, default=""):
            if not value:
                return default
            if isinstance(value, list):
                flat = []
                for item in value:
                    if isinstance(item, list):
                        flat.extend([str(x) for x in item if x])
                    elif item:
                        flat.append(str(item))
                return ", ".join(flat) if flat else default
            return str(value)
        
        # Build narrative sections
        narrative_parts = []
        
        # Section 1: DIAGNOSIS
        diagnosis_text = self._build_diagnosis_narrative(diagnosis)
        if diagnosis_text:
            narrative_parts.append(f"**Diagnosis:** {diagnosis_text}")
        
        # Section 2: WORK STATUS & MMI
        work_mmi_text = self._build_work_mmi_narrative(work_status, medical_legal)
        if work_mmi_text:
            narrative_parts.append(f"**Work Status:** {work_mmi_text}")
        
        # Section 3: TREATMENT/CRITICAL FINDINGS (Most important)
        medications = data.get("medications", {})
        # Section 3: TREATMENT/CRITICAL FINDINGS (Most important) - NOW includes medications
        treatment_critical_text = self._build_treatment_critical_narrative(
            recommendations, 
            critical_findings, 
            medical_legal,
            medications  # PASS medications to include in summary
        )
        if treatment_critical_text:
            narrative_parts.append(f"**Treatment/Critical Findings:** {treatment_critical_text}")
        
        # Section 4: PHYSICIAN & DATE (contextual info)
        qme_physician = physicians.get("qme_physician", {})
        physician_name = safe_str(qme_physician.get("name", ""))
        specialty = safe_str(qme_physician.get("specialty", ""))
        report_date = data.get("report_metadata", {}).get("report_date", fallback_date)
        
        if physician_name:
            context_line = f"QME by {physician_name}"
            if specialty:
                context_line += f" ({self._abbreviate_specialty(specialty)})"
            context_line += f" on {report_date}"
            narrative_parts.insert(0, context_line)  # Add at beginning
        
        # Join with proper formatting
        full_narrative = "\n\n".join(narrative_parts)
        
        logger.info(f"📝 Narrative summary generated: {len(full_narrative)} characters")
        return full_narrative

    def _build_diagnosis_narrative(self, diagnosis: Dict) -> str:
        """Build diagnosis narrative section"""
        dx_parts = []
        
        # Primary diagnoses
        primary = diagnosis.get("primary_diagnoses", [])
        if primary and isinstance(primary, list):
            primary_text = []
            for dx in primary:
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", "")
                    body_part = dx.get("body_part", "")
                    if dx_name:
                        if body_part:
                            primary_text.append(f"{body_part} {dx_name}")
                        else:
                            primary_text.append(dx_name)
                elif dx:
                    primary_text.append(str(dx))
            
            if primary_text:
                dx_parts.extend(primary_text)
        
        # Secondary/comorbid conditions
        secondary = diagnosis.get("secondary_diagnoses", [])
        if secondary and isinstance(secondary, list):
            secondary_text = []
            for dx in secondary:
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", "")
                    if dx_name:
                        secondary_text.append(dx_name)
                elif dx:
                    secondary_text.append(str(dx))
            
            if secondary_text:
                dx_parts.extend(secondary_text)
        
        # Historical conditions
        historical = diagnosis.get("historical_conditions", [])
        if historical and isinstance(historical, list):
            dx_parts.extend([str(h) for h in historical if h])
        
        return ", ".join(dx_parts) if dx_parts else "Diagnosis not fully extracted"

    def _build_work_mmi_narrative(self, work_status: Dict, medical_legal: Dict) -> str:
        """Build work status and MMI narrative"""
        parts = []
        
        # Current work status
        current_status = work_status.get("current_status", "")
        if isinstance(current_status, dict):
            current_status = current_status.get("status", "")
        if current_status:
            parts.append(str(current_status))
        
        # MMI status
        mmi_data = medical_legal.get("mmi_status", {})
        if isinstance(mmi_data, dict):
            mmi_status = mmi_data.get("status", "")
        else:
            mmi_status = str(mmi_data) if mmi_data else ""
        
        if mmi_status:
            parts.append(str(mmi_status).lower())
        
        return ", ".join(parts) if parts else "Work status not specified"

    def _build_treatment_critical_narrative(self, recommendations: Dict, critical_findings: list, medical_legal: Dict, medications: Dict = None) -> str:
      """Build treatment and critical findings narrative (most important section) - includes medications"""
      treatment_items = []
      
      # 1. MEDICATIONS (if any) - FIRST PRIORITY for summary
      if medications:
          current_meds = medications.get("current_medications", [])
          if current_meds and isinstance(current_meds, list):
              med_names = []
              for med in current_meds[:5]:  # Max 5 medications in summary
                  if isinstance(med, dict):
                      med_name = med.get("name", "")
                      med_dose = med.get("dose", "")
                      if med_name:
                          if med_dose:
                              med_names.append(f"{med_name} {med_dose}")
                          else:
                              med_names.append(med_name)
                  elif med:
                      med_names.append(str(med))
              
              if med_names:
                  treatment_items.insert(0, f"Current medications: {', '.join(med_names)}")
      
      # 2. Interventional procedures (high priority)
      procedures = recommendations.get("interventional_procedures", [])
      if procedures and isinstance(procedures, list):
          for proc in procedures[:3]:  # Max 3 procedures
              if isinstance(proc, dict):
                  proc_name = proc.get("procedure", "")
                  body_part = proc.get("body_part", "")
                  if proc_name:
                      if body_part:
                          treatment_items.append(f"{proc_name} for {body_part}")
                      else:
                          treatment_items.append(proc_name)
              elif proc:
                  treatment_items.append(str(proc))
      
      # 3. Diagnostic tests
      tests = recommendations.get("diagnostic_tests", [])
      if tests and isinstance(tests, list):
          for test in tests[:3]:  # Max 3 tests
              if isinstance(test, dict):
                  test_name = test.get("test", "")
                  reason = test.get("reason", "")
                  if test_name:
                      if reason:
                          treatment_items.append(f"{test_name} for {reason}")
                      else:
                          treatment_items.append(test_name)
              elif test:
                  treatment_items.append(str(test))
      
      # 4. Specialist referrals (QMEs)
      referrals = recommendations.get("specialist_referrals", [])
      if referrals and isinstance(referrals, list):
          for ref in referrals[:3]:  # Max 3 referrals
              if isinstance(ref, dict):
                  specialty = ref.get("specialty", "")
                  reason = ref.get("reason", "")
                  if specialty:
                      if reason:
                          treatment_items.append(f"{specialty} QME for {reason}")
                      else:
                          treatment_items.append(f"{specialty} QME")
              elif ref:
                  treatment_items.append(str(ref))
      
      # 5. Therapy
      therapy = recommendations.get("therapy", [])
      if therapy and isinstance(therapy, list):
          for tx in therapy[:2]:  # Max 2 therapy types
              if isinstance(tx, dict):
                  tx_type = tx.get("type", "")
                  body_part = tx.get("body_part", "")
                  if tx_type:
                      if body_part:
                          treatment_items.append(f"{tx_type} for {body_part}")
                      else:
                          treatment_items.append(tx_type)
              elif tx:
                  treatment_items.append(str(tx))
      
      # 6. Work restrictions (from work_status, NOT recommendations)
      # This ensures we get the EXACT wording from the extracted work_status
      
      return "; ".join(treatment_items[:10]) if treatment_items else "No specific recommendations extracted"

    def _build_medical_legal_summary_legacy(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Legacy summary builder for old 6-category structure.
        Kept for backward compatibility.
        """
        cat1 = data.get("category_1_core_identity", {})
        cat2 = data.get("category_2_diagnosis", {})
        cat5 = data.get("category_5_medical_legal_conclusions", {})
        cat6 = data.get("category_6_actionable_recommendations", {})
        
        parts = []
        
        # Date and physician
        date = cat1.get("report_date", fallback_date)
        physician = data.get("qme_physician_name", "")
        specialty = cat1.get("qme_specialty", "")
        
        parts.append(f"{date}: {doc_type}")
        if physician:
            if specialty:
                parts.append(f"by {physician} ({self._abbreviate_specialty(specialty)})")
            else:
                parts.append(f"by {physician}")
        
        # Body parts and diagnoses
        body_parts = cat2.get("affected_body_parts", [])
        diagnoses = cat2.get("primary_diagnoses", [])
        
        # Flatten lists
        def flatten_list(items):
            flat = []
            for item in items:
                if isinstance(item, list):
                    flat.extend([str(x) for x in item if x])
                elif item:
                    flat.append(str(item))
            return flat
        
        if body_parts:
            body_parts = flatten_list(body_parts)[:2]
            parts.append(f"for {', '.join(body_parts)}")
        
        if diagnoses:
            diagnoses = flatten_list(diagnoses)[:2]
            parts.append(f"= Dx: {', '.join(diagnoses)}")
        
        # Medical-legal conclusions
        mmi = str(cat5.get("mmi_status", "")) if cat5.get("mmi_status") else ""
        wpi = str(cat5.get("wpi_percentage", "")) if cat5.get("wpi_percentage") else ""
        
        conclusions = []
        if mmi:
            conclusions.append(mmi)
        if wpi:
            conclusions.append(f"WPI: {wpi}")
        
        if conclusions:
            parts.append(f"| {'; '.join(conclusions)}")
        
        summary = " ".join(parts)
        words = summary.split()
        # if len(words) > 170:
        #     summary = " ".join(words[:170]) + "..."
        
        return summary

    def _abbreviate_specialty(self, specialty: str) -> str:
        """Abbreviate medical specialties"""
        abbreviations = {
            "Orthopedic Surgery": "Ortho",
            "Orthopedics": "Ortho",
            "Neurology": "Neuro",
            "Pain Management": "Pain",
            "Psychiatry": "Psych",
            "Psychology": "Psych",
            "Physical Medicine & Rehabilitation": "PM&R",
            "Physical Medicine and Rehabilitation": "PM&R",
            "Internal Medicine": "IM",
            "Occupational Medicine": "Occ Med",
        }
        return abbreviations.get(specialty, specialty[:12])
    
    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure"""
        return {
            "category_1_core_identity": {
                "patient_name": "",
                "patient_age": "",
                "patient_dob": "",
                "date_of_injury": "",
                "report_date": fallback_date,
                "qme_physician_name": "",
                "qme_specialty": ""
            },
            "category_2_diagnosis": {
                "primary_diagnoses": [],
                "icd_codes": [],
                "affected_body_parts": []
            },
            "category_3_clinical_status": {
                "past_surgeries": [],
                "current_chief_complaint": "",
                "pain_score_current": "",
                "pain_score_max": "",
                "objective_findings": {
                    "rom_limitations": "",
                    "gait_abnormalities": "",
                    "positive_tests": "",
                    "effusion_swelling": "",
                    "other_objective": ""
                }
            },
            "category_4_medications": {
                "narcotics_opioids": [],
                "nerve_pain_meds": [],
                "anti_inflammatories": [],
                "other_long_term_meds": []
            },
            "category_5_medical_legal_conclusions": {
                "mmi_status": "",
                "mmi_deferred_reason": "",
                "wpi_percentage": "",
                "wpi_deferred_reason": "",
            },
            "category_6_actionable_recommendations": {
                "future_surgery": "",
                "future_injections": "",
                "future_therapy": "",
                "future_diagnostics": "",
                "work_status": "",
                "work_restrictions_specific": []
            }
        }
