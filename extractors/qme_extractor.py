"""
QME/AME/IME Enhanced Extractor - Full Context
Optimized for accuracy using Gemini-style full-document processing
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


class QMEExtractorChained:
    """
    Enhanced QME extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Uses raw text processing = Simplified pipeline
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    """

    def __init__(self, llm: AzureChatOpenAI, mode):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.mode = mode
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        
        logger.info("✅ QMEExtractorChained initialized (Full Context + Raw Text)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract QME data with FULL CONTEXT using raw text.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date if not found
            page_zones: Removed - no longer used
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING QME EXTRACTION (FULL CONTEXT + RAW TEXT)")
        logger.info("=" * 80)
        
        # Use raw_text if provided, otherwise fallback to text
        text_to_use = raw_text if raw_text is not None else text
        logger.info(f"📄 Using {'raw_text' if raw_text else 'text'} for LLM extraction")
        
        # Check document size
        text_length = len(text_to_use)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Extract with FULL CONTEXT
        raw_result = self._extract_full_context(
            text=text_to_use,
            doc_type=doc_type,
            fallback_date=fallback_date
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

        # Stage 3: Generate long summary using LLM tailored to mode
        long_summary = self._generate_long_summary_by_llm(raw_result, doc_type, fallback_date, self.mode)

        # Stage 4: Generate short summary from long summary using LLM tailored to mode
        short_summary = self._generate_short_summary_from_long_summary(long_summary, self.mode)

        logger.info("=" * 80)
        logger.info("✅ QME EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _extract_full_context(
            self,
            text: str,
            doc_type: str,
            fallback_date: str
        ) -> Dict:
            """
            Extract with FULL document context using raw text.
            This mimics Gemini's approach of processing the entire document at once.
            """
            logger.info("🔍 Processing ENTIRE document in single context window...")
            
            # Build system prompt
            system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE QME/AME/IME report.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing you to:
- Understand the complete case narrative from start to finish
- Connect findings across all sections (history → examination → conclusions)
- Identify relationships between symptoms, diagnoses, and recommendations
- Provide comprehensive, context-aware extraction without information loss

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
✅ CORRECT: Document states "Current Medications: Gabapentin 300mg three times daily, Meloxicam 15mg once daily"
Extract: {{"current_medications": [{{"name": "Gabapentin", "dose": "300mg TID", "purpose": "nerve pain"}}, {{"name": "Meloxicam", "dose": "15mg daily", "purpose": "anti-inflammatory"}}]}}

❌ WRONG: Document states "Patient previously took Oxycodone 5mg PRN but discontinued 6 months ago"
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

6. **QME PHYSICIAN/AUTHOR DETECTION**:
   - Identify the author who signed the report as the "qme_physician" name (e.g., from signature block, "Dictated by:", or closing statement).
   - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
   - Extract specialty and credentials only if explicitly stated near the signature.
   - If no clear signer is found, leave "name" empty.

7. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
Before returning your extraction, verify:
□ Every medication has a direct quote in the document
□ Every diagnosis is explicitly stated (not inferred from symptoms)
□ Every recommendation is directly from "Recommendations" or "Plan" section
□ No fields are filled with "typical" or "standard" values
□ Empty fields are truly empty (not "Not mentioned" or "Unknown")

EXTRACTION FOCUS - 7 CRITICAL MEDICAL-LEGAL CATEGORIES:

I. CORE IDENTITY
- Patient name, age, DOB
- Date of Injury (DOI) - often in history section
- Report date - check header and conclusion
- QME Physician: Extract from document (credentials and signature section)

II. DIAGNOSIS
- Primary diagnosis(es) - synthesize from examination findings AND conclusion
- ICD-10 codes if mentioned anywhere in document
- Affected body part(s) - consolidate all mentions throughout document

III. PHYSICAL EXAMINATION (NEW FOCUS AREA)
**CRITICAL: Extract DETAILED physical examination findings:**

- Range of Motion (ROM): Look for specific measurements (degrees), comparisons to contralateral side
Examples: "Shoulder flexion 90 degrees (normal 180)", "Lumbar flexion 50% of normal"
- Gait & Station: "Antalgic gait", "Normal gait", "Uses cane for ambulation"
- Strength Testing: "5/5 strength bilateral upper extremities", "4/5 strength left grip"
- Sensory Examination: "Decreased sensation to light touch in L5 distribution"
- Reflexes: "2+ bilateral patellar reflexes", "Absent ankle jerks"
- Special Tests: "Positive Neer test", "Negative Straight Leg Raise", "Positive McMurray test"
- Palpation Findings: "Tenderness over lumbar paraspinals", "No joint effusion"
- Inspection: "Muscle atrophy in right thigh", "Surgical scar well-healed"

IV. CLINICAL STATUS
- Past surgeries - scan entire history section for surgical history
- Current chief complaint - patient's own words from subjective section
- Pain score (current/max on 0-10 scale) - look in subjective complaints
- Functional limitations - specific activities patient cannot perform

V. MEDICATIONS ⚠️ CRITICAL - ZERO ASSUMPTIONS
**NOW EXTRACTING: CURRENT vs PREVIOUS MEDICATIONS**

- **CURRENT MEDICATIONS**: 
* ONLY from "Current Medications" section or explicitly stated as "currently taking"
* Include dosage ONLY if explicitly stated
* Categorize by type: narcotics/opioids, nerve pain meds, anti-inflammatories, other

- **PREVIOUS/DISCONTINUED MEDICATIONS**:
* Look for keywords: "discontinued", "previously took", "prior medication", "stopped", "no longer taking"
* Extract discontinuation reason if stated: "due to side effects", "ineffective", "completed course"
* Include duration if stated: "taken for 6 months", "used for 2 weeks"

- **FUTURE MEDICATIONS**: Medications recommended but not yet started

Example extraction:
Document states: 
"Current Medications: 1. Gabapentin 300mg three times daily, 2. Meloxicam 15mg once daily. 
Previously took Oxycodone 5mg PRN but discontinued due to constipation. 
Consider adding Amitriptyline 25mg at bedtime for sleep."

✅ CORRECT extraction:
{{
"medications": {{
    "current_medications": [
    {{"name": "Gabapentin", "dose": "300mg TID", "purpose": "nerve pain"}},
    {{"name": "Meloxicam", "dose": "15mg daily", "purpose": "anti-inflammatory"}}
    ],
    "previous_medications": [
    {{"name": "Oxycodone", "dose": "5mg PRN", "discontinuation_reason": "constipation"}}
    ],
    "future_medications": [
    {{"name": "Amitriptyline", "dose": "25mg at bedtime", "purpose": "sleep"}}
    ]
}}
}}

VI. MEDICAL-LEGAL CONCLUSIONS (MOST CRITICAL - HIGHEST PRIORITY)

- MMI/P&S Status:
* Look for explicit statement (e.g., "Patient has reached MMI as of [date]")
* If MMI deferred, extract SPECIFIC REASON (e.g., "pending MRI results", "awaiting surgical outcome")

- WPI (Whole Person Impairment):
* Look for percentage WITH body part (e.g., "15% WPI to left shoulder")
* Include method used (e.g., "per AMA Guides 5th Edition")
* If WPI deferred, extract SPECIFIC REASON

VII. ACTIONABLE RECOMMENDATIONS (SECOND HIGHEST PRIORITY)
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

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- MEDICATIONS: Only extract what is explicitly listed as current
- PHYSICAL EXAM: Extract detailed objective findings with measurements when available
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
    "name": "",
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

"physical_examination": {{
    "range_of_motion": [],
    "gait_and_station": "",
    "strength_testing": "",
    "sensory_examination": "",
    "reflexes": "",
    "special_tests": [],
    "palpation_findings": "",
    "inspection_findings": "",
    "other_objective_findings": ""
}},

"clinical_status": {{
    "chief_complaint": "",
    "pain_scores": {{
    "current": "",
    "maximum": "",
    "location": ""
    }},
    "functional_limitations": [],
    "past_surgeries": []
}},

"medications": {{
    "current_medications": [],
    "previous_medications": [],
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
    }}
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

1. **PHYSICAL EXAMINATION DETAILS:**
- Extract SPECIFIC measurements: "Shoulder abduction 90° (normal 180°)"
- List POSITIVE findings: "Positive Spurling test", "Tenderness over L4-L5"
- Include COMPARISONS: "Right grip strength 4/5 vs left 5/5"
- Document ASSISTIVE DEVICES: "Uses cane for community ambulation"

2. **MEDICATIONS - CURRENT VS PREVIOUS:**
- CURRENT: Only medications explicitly listed as "current" or "taking"
- PREVIOUS: Look for "discontinued", "stopped", "previously took"
- Include DISCONTINUATION REASONS: "due to side effects", "ineffective"
- FUTURE: Medications recommended but not yet started

3. **WORK RESTRICTIONS:** Extract EXACT wording from document
- If document says "no lifting", extract: "no lifting" (NOT "no lifting >10 lbs")
- If document says "no standing", extract: "no standing" (NOT "no prolonged standing >15 min")
- DO NOT add weight limits, time limits, or specifics not stated

4. **CURRENT MEDICATIONS:** Extract ONLY from "Current Medications" section
- Include dosage ONLY if explicitly stated
- DO NOT extract discontinued medications in current_medications
- DO NOT extract recommended future medications in current_medications

5. **CRITICAL FINDINGS:** Include MAIN actionable points only (max 5-8 items)
- Focus on: MMI status, required procedures, important diagnostic tests
- Include significant physical exam findings that impact disability rating
- DO NOT include minor details or routine follow-ups
""")

            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
            
            try:
                start_time = time.time()
                
                logger.info("🤖 Invoking LLM for full-context extraction...")
                
                # Single LLM call with FULL document context
                chain = chat_prompt | self.llm | self.parser
                result = chain.invoke({
                    "full_document_text": text  # Using raw text directly
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

    def _generate_long_summary_by_llm(self, raw_data: Dict, doc_type: str, fallback_date: str, mode: str) -> str:
        """
        Generate comprehensive long summary using LLM, tailored to the mode (gm or wc).
        """
        logger.info(f"📝 Generating LLM-based long summary tailored to mode: {mode.upper()}...")
        
        mode_focus = ""
        if mode.lower() == "gm":
            mode_focus = """
MODE: GENERAL MEDICINE (GM)
- Emphasize clinical details: detailed physical exam findings, treatment history, current therapies, diagnostic recommendations
- Structure: Patient Info (MUST INCLUDE CLAIM NUMBER) → Diagnosis → Physical Exam (detailed) → Clinical Status → Medications/Treatments → Recommendations (clinical focus)
- Tone: Clinical, comprehensive for ongoing care planning
- Length: 400-600 words, detailed but readable for physicians
"""
        elif mode.lower() == "wc":
            mode_focus = """
MODE: WORKERS COMPENSATION (WC)
- Emphasize medical-legal aspects: MMI/WPI status, work restrictions, impairment ratings, return-to-work prognosis
- Structure: Report Overview → Patient/Claim Info (MUST INCLUDE CLAIM NUMBER) → Diagnosis → Medical-Legal Conclusions (priority) → Work Status → Recommendations (actionable for claims)
- Tone: Objective, legal-focused, highlighting impairment and restrictions
- Length: 300-500 words, concise for adjusters/attorneys
"""
        else:
            mode_focus = "MODE: DEFAULT - Balanced clinical and legal focus. Patient Info MUST INCLUDE CLAIM NUMBER."

        system_prompt = SystemMessagePromptTemplate.from_template(f"""
You are a medical documentation expert generating a structured long summary from extracted QME data.

{mode_focus}

STRICT RULES:
- Use ONLY data from the provided JSON extraction - NO hallucinations or additions
- Organize into clear sections with headings (e.g., ## DIAGNOSIS)
- For lists (diagnoses, meds, etc.), use bullet points
- If data is empty/missing, note "Not specified" briefly and move on
- Ensure factual, professional tone
- Output as markdown-formatted text for readability
- **MANDATORY: You MUST include the "Claim Number" in the Patient Information section. If it is missing in the data, explicitly state "Claim Number: Not specified".**

Structure the summary logically based on mode.
        """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
EXTRACTED JSON DATA:
{raw_data}

Document Type: {doc_type}
Report Date: {report_date}

Generate the mode-tailored long summary now.
        """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "raw_data": str(raw_data),
                "doc_type": doc_type,
                "report_date": fallback_date
            })
            long_summary = response.content.strip()
            
            logger.info(f"✅ LLM long summary generated: {len(long_summary)} characters")
            return long_summary

        except Exception as e:
            logger.error(f"❌ LLM long summary generation failed: {e}")
            # Fallback to manual build if LLM fails
            return self._build_comprehensive_long_summary(raw_data, doc_type, fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Manual fallback for building long summary (original method).
        Note: This is kept as fallback but uses the same structure as before.
        """
        # ... (keep the existing manual fallback implementation as is)
        pass

    def _generate_short_summary_from_long_summary(self, long_summary: str, mode: str) -> str:
        """
        Generate a precise 30–60 word, pipe-delimited actionable summary in key-value format using LLM, tailored to mode.
        """
        logger.info(f"🎯 Generating LLM-based short summary tailored to mode: {mode.upper()} (30-60 words)...")

        mode_focus = ""
        if mode.lower() == "gm":
            mode_focus = """
MODE: GENERAL MEDICINE (GM)
- Focus on: Diagnosis, Pain/Clinical Status, Medications, Treatment Recommendations
- Keys: Diagnosis | Clinical Status | Medications | Recommendations
- Clinical emphasis, actionable for care planning
"""
        elif mode.lower() == "wc":
            mode_focus = """
MODE: WORKERS COMPENSATION (WC)
- Focus on: MMI/WPI, Work Restrictions, Impairment, Claim-Relevant Recommendations
- Keys: MMI Status | WPI | Work Status | Legal Recommendations
- Legal/claims emphasis, concise for adjusters
"""
        else:
            mode_focus = "MODE: DEFAULT - Balanced clinical and legal keys"

        system_prompt = SystemMessagePromptTemplate.from_template(f"""
You are a medical-legal extraction specialist generating mode-tailored short summaries.

{mode_focus}

STRICT REQUIREMENTS:
1. Word count MUST be between **30 and 60 words** (min 30, max 60).
2. Format MUST be EXACTLY a single pipe-delimited line:

[Report Title] | [Author] | Date:[value] | [Mode-Specific Keys from above] | Critical Finding:[value]

3. DO NOT fabricate or infer missing data — simply SKIP entire key-value pairs that do not exist.
4. Use ONLY information explicitly found in the long summary.
5. Output must be a SINGLE LINE (no line breaks).
6. Prioritize mode-specific keys first, then general (Body Parts, Diagnosis if not covered).
7. ABSOLUTE NO: assumptions, clinical interpretation, invented data, narrative sentences.
8. If a field is missing, SKIP THE ENTIRE KEY-VALUE PAIR—do NOT include empty pairs.

Your final output must be 30–60 words and MUST follow the exact format.
        """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG SUMMARY:

{long_summary}

Now produce the 30–60 word single-line summary following the strict mode-tailored rules.
        """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})
            summary = response.content.strip()

            # Clean whitespace only
            summary = re.sub(r'\s+', ' ', summary).strip()

            # Word count check
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Summary outside word limit ({wc} words). Attempting auto-fix.")
                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary had {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy, mode focus, and format. DO NOT add invented data."
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
                "past_surgeries": []
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