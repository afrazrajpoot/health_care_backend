"""
QME/AME/IME Enhanced Extractor - Full Context with Context-Awareness
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
    ) -> Dict:
        """
        Extract QME data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer (CRITICAL)
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
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

        # Stage 4: Build long summary from ALL raw data
        long_summary = self._build_comprehensive_long_summary(raw_result, doc_type, fallback_date)

        # Stage 5: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary)

        logger.info("=" * 80)
        logger.info("✅ QME EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

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
- Objective findings:
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

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted raw data with detailed headings.
        
        Args:
            raw_data: Complete extracted data from LLM
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date
            
        Returns:
            String with detailed long summary organized by sections
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted data...")
        
        sections = []
        
        # Section 1: REPORT OVERVIEW
        sections.append("📋 REPORT OVERVIEW")
        sections.append("-" * 50)
        
        # Physician and date info
        physicians = raw_data.get("physicians", {})
        qme_physician = physicians.get("qme_physician", {})
        report_metadata = raw_data.get("report_metadata", {})
        
        physician_name = qme_physician.get("name", raw_data.get("qme_physician_name", ""))
        specialty = qme_physician.get("specialty", "")
        report_date = report_metadata.get("report_date", fallback_date)
        report_type = report_metadata.get("report_type", doc_type)
        
        overview_lines = [
            f"Document Type: {report_type}",
            f"Report Date: {report_date}",
            f"Evaluating Physician: {physician_name}",
            f"Specialty: {specialty}" if specialty else "Specialty: Not specified"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PATIENT INFORMATION
        sections.append("\n👤 PATIENT INFORMATION")
        sections.append("-" * 50)
        
        patient_info = raw_data.get("patient_information", {})
        patient_lines = [
            f"Name: {patient_info.get('patient_name', 'Not specified')}",
            f"Age: {patient_info.get('patient_age', 'Not specified')}",
            f"Date of Birth: {patient_info.get('patient_dob', 'Not specified')}",
            f"Date of Injury: {patient_info.get('date_of_injury', 'Not specified')}",
            f"Claim Number: {patient_info.get('claim_number', 'Not specified')}",
            f"Employer: {patient_info.get('employer', 'Not specified')}"
        ]
        sections.append("\n".join(patient_lines))
        
        # Section 3: DIAGNOSIS - FIXED: Ensure diagnoses are properly extracted
        sections.append("\n🏥 DIAGNOSIS")
        sections.append("-" * 50)
        
        diagnosis = raw_data.get("diagnosis", {})
        diagnosis_lines = []
        
        # Primary diagnoses - FIXED: Handle both list and dict formats
        primary_dx = diagnosis.get("primary_diagnoses", [])
        if primary_dx:
            diagnosis_lines.append("Primary Diagnoses:")
            for dx in primary_dx[:5]:  # Limit to 5 primary diagnoses
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", dx.get("name", ""))
                    body_part = dx.get("body_part", "")
                    if dx_name and dx_name.strip():
                        if body_part and body_part.strip():
                            diagnosis_lines.append(f"  • {body_part}: {dx_name}")
                        else:
                            diagnosis_lines.append(f"  • {dx_name}")
                elif dx and str(dx).strip():
                    diagnosis_lines.append(f"  • {dx}")
        
        # Secondary diagnoses
        secondary_dx = diagnosis.get("secondary_diagnoses", [])
        if secondary_dx:
            diagnosis_lines.append("\nSecondary/Comorbid Conditions:")
            for dx in secondary_dx[:3]:  # Limit to 3 secondary
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", dx.get("name", ""))
                    if dx_name and dx_name.strip():
                        diagnosis_lines.append(f"  • {dx_name}")
                elif dx and str(dx).strip():
                    diagnosis_lines.append(f"  • {dx}")
        
        # Historical conditions
        historical = diagnosis.get("historical_conditions", [])
        if historical:
            diagnosis_lines.append("\nHistorical Conditions:")
            for condition in historical[:3]:  # Limit to 3 historical
                if isinstance(condition, dict):
                    cond_name = condition.get("condition", condition.get("name", ""))
                    if cond_name and cond_name.strip():
                        diagnosis_lines.append(f"  • {cond_name}")
                elif condition and str(condition).strip():
                    diagnosis_lines.append(f"  • {condition}")
        
        # If no diagnoses found, check for alternative fields
        if not diagnosis_lines:
            # Check if there are any diagnoses in other fields
            if diagnosis.get("primary_diagnosis"):
                diagnosis_lines.append(f"Primary Diagnosis: {diagnosis.get('primary_diagnosis')}")
            elif raw_data.get("clinical_status", {}).get("diagnosis"):
                diagnosis_lines.append(f"Diagnosis: {raw_data['clinical_status']['diagnosis']}")
        
        sections.append("\n".join(diagnosis_lines) if diagnosis_lines else "No diagnoses extracted")
        
        # Section 4: CLINICAL STATUS
        sections.append("\n🔬 CLINICAL STATUS")
        sections.append("-" * 50)
        
        clinical_status = raw_data.get("clinical_status", {})
        clinical_lines = []
        
        # Chief complaint
        chief_complaint = clinical_status.get("chief_complaint", "")
        if chief_complaint:
            clinical_lines.append(f"Chief Complaint: {chief_complaint}")
        
        # Pain scores
        pain_scores = clinical_status.get("pain_scores", {})
        current_pain = pain_scores.get("current", "")
        max_pain = pain_scores.get("maximum", "")
        pain_location = pain_scores.get("location", "")
        
        if current_pain or max_pain:
            pain_info = []
            if current_pain:
                pain_info.append(f"Current: {current_pain}/10")
            if max_pain:
                pain_info.append(f"Maximum: {max_pain}/10")
            if pain_location:
                pain_info.append(f"Location: {pain_location}")
            clinical_lines.append(f"Pain Scores: {', '.join(pain_info)}")
        
        # Functional limitations
        functional_limitations = clinical_status.get("functional_limitations", [])
        if functional_limitations:
            clinical_lines.append("\nFunctional Limitations:")
            for limitation in functional_limitations[:5]:
                if isinstance(limitation, dict):
                    desc = limitation.get("description", "")
                    if desc:
                        clinical_lines.append(f"  • {desc}")
                elif limitation:
                    clinical_lines.append(f"  • {limitation}")
        
        # Past surgeries
        past_surgeries = clinical_status.get("past_surgeries", [])
        if past_surgeries:
            clinical_lines.append("\nPast Surgeries:")
            for surgery in past_surgeries[:5]:
                if isinstance(surgery, dict):
                    procedure = surgery.get("procedure", "")
                    date = surgery.get("date", "")
                    if procedure:
                        if date:
                            clinical_lines.append(f"  • {procedure} ({date})")
                        else:
                            clinical_lines.append(f"  • {procedure}")
                elif surgery:
                    clinical_lines.append(f"  • {surgery}")
        
        # Objective findings
        objective_findings = clinical_status.get("objective_findings", {})
        if objective_findings:
            clinical_lines.append("\nObjective Findings:")
            for key, value in objective_findings.items():
                if value and value not in ["", "Not specified"]:
                    clinical_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        sections.append("\n".join(clinical_lines) if clinical_lines else "No clinical status details extracted")
        
        # Section 5: MEDICATIONS
        sections.append("\n💊 MEDICATIONS")
        sections.append("-" * 50)
        
        medications = raw_data.get("medications", {})
        medication_lines = []
        
        # Current medications
        current_meds = medications.get("current_medications", [])
        if current_meds:
            medication_lines.append("Current Medications:")
            for med in current_meds[:10]:  # Limit to 10 medications
                if isinstance(med, dict):
                    med_name = med.get("name", "")
                    med_dose = med.get("dose", "")
                    med_purpose = med.get("purpose", "")
                    
                    if med_name:
                        med_info = med_name
                        if med_dose:
                            med_info += f" - {med_dose}"
                        if med_purpose:
                            med_info += f" ({med_purpose})"
                        medication_lines.append(f"  • {med_info}")
                elif med:
                    medication_lines.append(f"  • {med}")
        else:
            medication_lines.append("No current medications listed")
        
        # Future medications
        future_meds = medications.get("future_medications", [])
        if future_meds:
            medication_lines.append("\nRecommended Future Medications:")
            for med in future_meds[:5]:
                if isinstance(med, dict):
                    med_name = med.get("name", "")
                    med_reason = med.get("reason", "")
                    if med_name:
                        if med_reason:
                            medication_lines.append(f"  • {med_name} - {med_reason}")
                        else:
                            medication_lines.append(f"  • {med_name}")
                elif med:
                    medication_lines.append(f"  • {med}")
        
        sections.append("\n".join(medication_lines))
        
        # Section 6: MEDICAL-LEGAL CONCLUSIONS
        sections.append("\n⚖️ MEDICAL-LEGAL CONCLUSIONS")
        sections.append("-" * 50)
        
        medical_legal = raw_data.get("medical_legal_conclusions", {})
        legal_lines = []
        
        # MMI Status
        mmi_status = medical_legal.get("mmi_status", {})
        if isinstance(mmi_status, dict):
            mmi_stat = mmi_status.get("status", "")
            mmi_reason = mmi_status.get("reason", "")
            if mmi_stat:
                legal_lines.append(f"MMI Status: {mmi_stat}")
                if mmi_reason:
                    legal_lines.append(f"MMI Reason: {mmi_reason}")
        elif mmi_status:
            legal_lines.append(f"MMI Status: {mmi_status}")
        
        # WPI Impairment
        wpi_impairment = medical_legal.get("wpi_impairment", {})
        if isinstance(wpi_impairment, dict):
            total_wpi = wpi_impairment.get("total_wpi", "")
            wpi_breakdown = wpi_impairment.get("breakdown", [])
            if total_wpi:
                legal_lines.append(f"Whole Person Impairment (WPI): {total_wpi}%")
            if wpi_breakdown:
                legal_lines.append("WPI Breakdown:")
                for item in wpi_breakdown[:3]:
                    if isinstance(item, dict):
                        body_part = item.get("body_part", "")
                        percentage = item.get("percentage", "")
                        if body_part and percentage:
                            legal_lines.append(f"  • {body_part}: {percentage}%")
                    elif item:
                        legal_lines.append(f"  • {item}")
        
        sections.append("\n".join(legal_lines) if legal_lines else "No medical-legal conclusions extracted")
        
        # Section 7: WORK STATUS
        sections.append("\n💼 WORK STATUS")
        sections.append("-" * 50)
        
        work_status = raw_data.get("work_status", {})
        work_lines = []
        
        current_status = work_status.get("current_status", "")
        if current_status:
            work_lines.append(f"Current Work Status: {current_status}")
        
        work_restrictions = work_status.get("work_restrictions", [])
        if work_restrictions:
            work_lines.append("\nWork Restrictions:")
            for restriction in work_restrictions[:10]:
                if isinstance(restriction, dict):
                    desc = restriction.get("description", "")
                    duration = restriction.get("duration", "")
                    if desc:
                        if duration:
                            work_lines.append(f"  • {desc} ({duration})")
                        else:
                            work_lines.append(f"  • {desc}")
                elif restriction:
                    work_lines.append(f"  • {restriction}")
        
        prognosis = work_status.get("prognosis_for_return_to_work", "")
        if prognosis:
            work_lines.append(f"\nPrognosis for Return to Work: {prognosis}")
        
        sections.append("\n".join(work_lines) if work_lines else "No work status information extracted")
        
        # Section 8: RECOMMENDATIONS
        sections.append("\n🎯 RECOMMENDATIONS")
        sections.append("-" * 50)
        
        recommendations = raw_data.get("recommendations", {})
        rec_lines = []
        
        # Diagnostic tests
        diagnostic_tests = recommendations.get("diagnostic_tests", [])
        if diagnostic_tests:
            rec_lines.append("Diagnostic Tests Recommended:")
            for test in diagnostic_tests[:5]:
                if isinstance(test, dict):
                    test_name = test.get("test", "")
                    test_reason = test.get("reason", "")
                    if test_name:
                        if test_reason:
                            rec_lines.append(f"  • {test_name} - {test_reason}")
                        else:
                            rec_lines.append(f"  • {test_name}")
                elif test:
                    rec_lines.append(f"  • {test}")
        
        # Interventional procedures
        procedures = recommendations.get("interventional_procedures", [])
        if procedures:
            rec_lines.append("\nInterventional Procedures:")
            for proc in procedures[:5]:
                if isinstance(proc, dict):
                    proc_name = proc.get("procedure", "")
                    proc_location = proc.get("body_part", "")
                    if proc_name:
                        if proc_location:
                            rec_lines.append(f"  • {proc_name} - {proc_location}")
                        else:
                            rec_lines.append(f"  • {proc_name}")
                elif proc:
                    rec_lines.append(f"  • {proc}")
        
        # Specialist referrals
        referrals = recommendations.get("specialist_referrals", [])
        if referrals:
            rec_lines.append("\nSpecialist Referrals:")
            for ref in referrals[:3]:
                if isinstance(ref, dict):
                    specialty = ref.get("specialty", "")
                    ref_reason = ref.get("reason", "")
                    if specialty:
                        if ref_reason:
                            rec_lines.append(f"  • {specialty} - {ref_reason}")
                        else:
                            rec_lines.append(f"  • {specialty}")
                elif ref:
                    rec_lines.append(f"  • {ref}")
        
        # Therapy
        therapy = recommendations.get("therapy", [])
        if therapy:
            rec_lines.append("\nTherapy Recommendations:")
            for tx in therapy[:3]:
                if isinstance(tx, dict):
                    tx_type = tx.get("type", "")
                    tx_frequency = tx.get("frequency", "")
                    tx_duration = tx.get("duration", "")
                    if tx_type:
                        tx_info = tx_type
                        if tx_frequency:
                            tx_info += f" - {tx_frequency}"
                        if tx_duration:
                            tx_info += f" for {tx_duration}"
                        rec_lines.append(f"  • {tx_info}")
                elif tx:
                    rec_lines.append(f"  • {tx}")
        
        # Future surgical needs
        future_surgery = recommendations.get("future_surgical_needs", [])
        if future_surgery:
            rec_lines.append("\nFuture Surgical Needs:")
            for surgery in future_surgery[:3]:
                if isinstance(surgery, dict):
                    surg_name = surgery.get("procedure", "")
                    surg_reason = surgery.get("reason", "")
                    if surg_name:
                        if surg_reason:
                            rec_lines.append(f"  • {surg_name} - {surg_reason}")
                        else:
                            rec_lines.append(f"  • {surg_name}")
                elif surgery:
                    rec_lines.append(f"  • {surgery}")
        
        sections.append("\n".join(rec_lines) if rec_lines else "No specific recommendations extracted")
        
        # Section 9: CRITICAL FINDINGS
        sections.append("\n🚨 CRITICAL FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            for finding in critical_findings[:8]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("description", "")
                    finding_priority = finding.get("priority", "")
                    if finding_desc:
                        if finding_priority:
                            sections.append(f"• [{finding_priority}] {finding_desc}")
                        else:
                            sections.append(f"• {finding_desc}")
                elif finding:
                    sections.append(f"• {finding}")
        else:
            sections.append("No critical findings explicitly listed")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Long summary built: {len(long_summary)} characters")
        
        return long_summary

    def _generate_short_summary_from_long_summary(self, long_summary: str) -> str:
        """
        Generate a precise 30–60 word, pipe-delimited actionable summary in key-value format.
        No hallucination, no assumptions. Missing fields are omitted.
        """

        logger.info("🎯 Generating 30–60 word actionable short summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a medical-legal extraction specialist.

    Your task: Generate a short, highly actionable summary from a VERIFIED long medical summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be between **30 and 60 words** (min 30, max 60).
    2. Format MUST be EXACTLY:

    [Report Title] | [Author] | Date:[value] | Body Parts:[value] | Diagnosis:[value] | Medication:[value] | MMI Status:[value] | Work Status:[value] | Recommendation:[value] | Critical Finding:[value]

    3. DO NOT fabricate or infer missing data — simply SKIP entire key-value pairs that do not exist.
    4. Use ONLY information explicitly found in the long summary.
    5. Output must be a SINGLE LINE (no line breaks).
    6. Content priority:
    - report title (without "Report Title:" key)
    - author name (without "Author:" key)  
    - date
    - affected body parts
    - primary diagnosis
    - medications (if present)
    - MMI status (if present)
    - work status (if present)
    - key recommendation(s)
    - one critical finding

    7. ABSOLUTE NO:
    - assumptions
    - clinical interpretation
    - invented medications
    - invented dates
    - narrative sentences

    8. If a field is missing, SKIP THE ENTIRE KEY-VALUE PAIR—do NOT include empty key-value pairs.

    Your final output must be 30–60 words and MUST follow the exact format above.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    LONG SUMMARY:

    {long_summary}

    Now produce the 30–60 word single-line summary following the strict rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        # Single attempt only — model should follow instructions reliably
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})
            summary = response.content.strip()

            # Clean whitespace only (no pipe cleaning)
            summary = re.sub(r'\s+', ' ', summary).strip()

            # Word count check
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Summary outside word limit ({wc} words). Attempting auto-fix.")

                # Let the model fix it in one retry
                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary had {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy and format. DO NOT add invented data."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])
                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r'\s+', ' ', fixed.content.strip())
                logger.info(f"🔧 Fixed summary word count: {len(summary.split())} words")

            logger.info(f"✅ Final short summary: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return "Summary unavailable due to processing error."
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
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary with physician inclusion"""
        
        # Extract physician information
        physician_match = re.search(r'Evaluating Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'diagnosis': r'Primary Diagnoses:(.*?)(?:\n\n|\n[A-Z]|$)',
            'clinical': r'Clinical Status:(.*?)(?:\n\n|\n[A-Z]|$)',
            'mmi': r'MMI Status:\s*([^\n]+)',
            'restrictions': r'Work Restrictions:(.*?)(?:\n\n|\n[A-Z]|$)',
            'medications': r'Current Medications:(.*?)(?:\n\n|\n[A-Z]|$)',
            'recommendations': r'Recommendations:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary with physician
        parts = []
        
        # Start with physician
        parts.append(f"{physician} evaluated")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            first_dx = extracted['diagnosis'].split('\n')[0].replace('•', '').strip()[:100]
            parts.append(f"for {first_dx}")
        
        # Add clinical findings
        if 'clinical' in extracted:
            clinical_lines = extracted['clinical'].split('\n')
            pain_match = re.search(r'Pain Scores:\s*([^\n]+)', extracted['clinical'])
            if pain_match:
                parts.append(f"with {pain_match.group(1).strip()}")
        
        # Add medications
        if 'medications' in extracted and "No current medications" not in extracted['medications']:
            meds_line = extracted['medications'].split('\n')[0].replace('•', '').strip()[:80]
            parts.append(f"Medications: {meds_line}")
        
        # Add treatment
        if 'recommendations' in extracted:
            first_rec = extracted['recommendations'].split('\n')[0].replace('•', '').strip()[:80]
            parts.append(f"Treatment: {first_rec}")
        
        # Add work status
        if 'mmi' in extracted:
            parts.append(f"MMI: {extracted['mmi'][:50]}")
        if 'restrictions' in extracted:
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:60]
            parts.append(f"Restrictions: {first_restrict}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["comprehensive medical management", "with ongoing evaluation", "and progress monitoring"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure"""
        return {
            "patient_information": {
                "patient_name": "",
                "patient_age": "",
                "patient_dob": "",
                "date_of_injury": "",
                "claim_number": "",
                "employer": ""
            },
            "report_metadata": {
                "report_title": "",
                "report_date": fallback_date,
                "evaluation_date": "",
                "report_type": "QME/AME/IME"
            },
            "physicians": {
                "qme_physician": {
                    "name": "",
                    "specialty": "",
                    "credentials": "",
                    "role": "Evaluating Physician/QME/AME"
                },
                "treating_physicians": [],
                "consulting_physicians": [],
                "referring_source": {
                    "name": "",
                    "type": ""
                }
            },
            "diagnosis": {
                "primary_diagnoses": [],
                "secondary_diagnoses": [],
                "historical_conditions": []
            },
            "clinical_status": {
                "chief_complaint": "",
                "pain_scores": {
                    "current": "",
                    "maximum": "",
                    "location": ""
                },
                "functional_limitations": [],
                "past_surgeries": [],
                "objective_findings": {
                    "rom_limitations": "",
                    "gait": "",
                    "positive_tests": "",
                    "other_findings": ""
                }
            },
            "medications": {
                "current_medications": [],
                "future_medications": []
            },
            "treatment_history": {
                "past_treatments": [],
                "current_treatments": []
            },
            "medical_legal_conclusions": {
                "mmi_status": {
                    "status": "",
                    "reason": "",
                    "reasoning": ""
                },
                "wpi_impairment": {
                    "total_wpi": "",
                    "breakdown": [],
                    "reasoning": ""
                },
            },
            "work_status": {
                "current_status": "",
                "work_restrictions": [],
                "prognosis_for_return_to_work": ""
            },
            "recommendations": {
                "diagnostic_tests": [],
                "interventional_procedures": [],
                "specialist_referrals": [],
                "therapy": [],
                "future_surgical_needs": []
            },
            "critical_findings": []
        }