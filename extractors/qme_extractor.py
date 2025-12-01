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
        
        # ✅ ONLY 2 LLM CALLS:
        # 1. Generate long summary directly from text using your existing context and prompts
        long_summary = self._extract_full_context(
            text=text_to_use,
            doc_type=doc_type,
            fallback_date=fallback_date
        )

        # 2. Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, self.mode)

        logger.info("=" * 80)
        logger.info("✅ QME EXTRACTION COMPLETE (2 LLM CALLS ONLY)")
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
        ) -> str:
            """
            Generate long summary directly from document text using your EXACT same context and prompts.
            This now returns the long summary string directly instead of JSON.
            """
            logger.info("🔍 Processing ENTIRE document to generate long summary...")
            
            # Build system prompt - YOUR EXACT SAME PROMPT
            system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert medical-legal documentation specialist analyzing a COMPLETE QME/AME/IME report.

    CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
    You are seeing the ENTIRE document at once, allowing you to:
    - Understand the complete case narrative from start to finish
    - Connect findings across all sections (history → examination → conclusions)
    - Identify relationships between symptoms, diagnoses, and recommendations
    - Provide comprehensive, context-aware extraction without information loss

    ⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY) (donot include in output, for LLM use only):

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

    ⚠️ FINAL REMINDER (donot include in output, for LLM use only):
    - If information is NOT in the document, return EMPTY ("" or [])
    - NEVER assume, infer, or extrapolate
    - MEDICATIONS: Only extract what is explicitly listed as current
    - PHYSICAL EXAM: Extract detailed objective findings with measurements when available
    - It is BETTER to have empty fields than incorrect information

    **NOW GENERATE A COMPREHENSIVE LONG SUMMARY IN MARKDOWN FORMAT**

    Instead of returning JSON, generate a comprehensive long summary with the following structure:

    ## PATIENT INFORMATION
    - **Name:** [extracted name]
    - **Date of Birth:** [extracted DOB] 
    - **Claim Number:** [extracted claim number]
    - **Date of Injury:** [extracted DOI]
    - **Employer:** [extracted employer]
    ## REPORT DETAILS
    - **Report Type:** [QME/AME/IME]
    - **Report Date:** [extracted date]
    - **Evaluating Physician:** [extracted physician name and credentials]
    Author:
    hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
    • Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]
    
   ## All Doctors Involved:
    • [list all extracted doctors with names and titles]
    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
    - Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
    - Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
    - Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
    - Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
    - If no doctors found, leave list empty []. 
    
    ━━━ CLAIM NUMBER EXTRACTION PATTERNS ━━━
    CRITICAL: Scan the ENTIRE document mainly (header, footer, cc: lines, letterhead) for claim numbers.

    Common claim number patterns (case-insensitive) and make sure to extract EXACTLY as written and must be claim number not just random numbers (like chart numbers, or id numbers) that look similar:
    - "[Claim #XXXXXXXXX]" or "[Claim #XXXXX-XXX]"
    - "Claim Number: XXXXXXXXX" or "Claim #: XXXXXXXXX"
    - "Claim: XXXXXXXXX" or "Claim #XXXXXXXXX"
    - "WC Claim: XXXXXXXXX" or "Workers Comp Claim: XXXXXXXXX"
    - "Policy/Claim: XXXXXXXXX"
    - In "cc:" lines: "Broadspire [Claim #XXXXXXXXX]"
    - In subject lines or reference fields: "Claim #XXXXXXX"
                                                                      
    ## DIAGNOSIS
    - Primary diagnoses with affected body parts
    - ICD-10 codes if available

    ## PHYSICAL EXAMINATION FINDINGS
    - Detailed range of motion measurements
    - Strength testing results
    - Special test findings
    - Other objective findings

    ## CLINICAL STATUS
    - Chief complaint in patient's own words
    - Pain scores (current/maximum)
    - Functional limitations
    - Past surgical history

    ## MEDICATIONS
    - **Current Medications:** [List ALL current medications with dosages and purposes if specified]
    - **Previous Medications:** [List previously discontinued medications with reasons if stated]
    - **Future Medications:** [List medications recommended for future use]
    - **If no medications mentioned:** State "None explicitly stated" for each category

    ## MEDICAL-LEGAL CONCLUSIONS
    - MMI status and date
    - WPI impairment ratings
    - Work restrictions and limitations

    ## RECOMMENDATIONS
    - Diagnostic tests needed
    - Treatments and therapies
    - Specialist referrals
    - Follow-up requirements

    Use markdown formatting with clear headings and bullet points. Be comprehensive but concise.
    """)
            user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE QME/AME/IME DOCUMENT TEXT:

{full_document_text}

Document Type: {doc_type}
Report Date: {fallback_date}

Generate the comprehensive long summary now following all the rules above.
""")

            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
            
            try:
                start_time = time.time()
                
                logger.info("🤖 Invoking LLM for long summary generation...")
                
                # Single LLM call to generate long summary directly
                chain = chat_prompt | self.llm
                response = chain.invoke({
                    "full_document_text": text,
                    "doc_type": doc_type, 
                    "fallback_date": fallback_date
                })
                
                long_summary = response.content.strip()
                end_time = time.time()
                processing_time = end_time - start_time
                
                logger.info(f"⚡ Long summary generated in {processing_time:.2f}s")
                logger.info(f"✅ Generated long summary: {len(long_summary):,} chars")
                
                return long_summary
                
            except Exception as e:
                logger.error(f"❌ Long summary generation failed: {e}", exc_info=True)
                
                # Check if context length exceeded
                if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                    logger.error("❌ Document exceeds GPT-4o 128K context window")
                
                return self._get_fallback_summary(doc_type, fallback_date)

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
    - **ONLY include, critical, or clinically significant findings**.
    - **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
    
    Make sure to follow this EXACT format:
    [Report Title] | [Author] | Date:[value] | Body Parts:[value] | Diagnosis:[value] | Physical Examination:[value (only critical findings or abnormalities else skip)] | Vital Signs:[value (only critical vital signs else skip)] | Treatment Plan:[value] | Critical Finding:[value (only abnormalities) if applicable] | Follow-up:[value] | Recommendations:[value] | Work Status:[value] | MMI Status:[value] | WPI:[value] | Medications:[value]

    3. DO NOT fabricate or infer missing data — simply SKIP entire key-value pairs that do not exist.
    4. Use ONLY information explicitly found in the long summary.
    5. Output must be a SINGLE LINE (no line breaks).
    6. Prioritize mode-specific keys first, then general (Body Parts, Diagnosis).
    7. ABSOLUTELY NO: assumptions, invented data, narrative sentences.
    8. If a field is missing, SKIP THE ENTIRE KEY-VALUE PAIR—do NOT include empty pairs OR placeholders like:
    - "not included"
    - "not listed"
    - "no abnormalities detected"
    - "not provided"
    - "not discussed"

    ABSOLUTELY FORBIDDEN:
- Normal findings (ignore them entirely)
- assumptions, interpretations, invented medications, or inferred diagnoses
- placeholder text or "Not provided"
- narrative writing
- duplicate pipes or empty pipe fields (e.g., "||")
- any patient details (patient name, DOB, ID)

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

            # 🚀 Extra Cleanup: Remove any "not included / not provided / no X" pairs
            forbidden_phrases = [
                "not included", "not listed", "not provided", "not discussed",
                "no abnormalities detected", "no critical findings", "none", "N/A"
            ]

            segments = [seg.strip() for seg in summary.split("|")]
            cleaned_segments = []

            for seg in segments:
                if any(bad in seg.lower() for bad in forbidden_phrases):
                    continue  # skip this key-value entirely
                cleaned_segments.append(seg)

            summary = " | ".join(cleaned_segments).strip()

            # Word count check
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Summary outside word limit ({wc} words). Attempting auto-fix.")
                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary had {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy, mode focus, exact format, and skipping missing keys."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])
                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r'\s+', ' ', fixed.content.strip())

                # Remove placeholders after fix as well
                segments = [seg.strip() for seg in summary.split("|")]
                cleaned_segments = []
                for seg in segments:
                    if any(bad in seg.lower() for bad in forbidden_phrases):
                        continue
                    cleaned_segments.append(seg)
                summary = " | ".join(cleaned_segments).strip()

                logger.info(f"🔧 Fixed summary word count: {len(summary.split())} words")

            logger.info(f"✅ Final short summary: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return "Summary unavailable due to processing error."

    def _get_fallback_summary(self, doc_type: str, fallback_date: str) -> str:
        """Fallback summary when processing fails"""
        return f"""## {doc_type} Report
**Report Date:** {fallback_date}

**Note:** Comprehensive analysis unavailable due to processing limitations. 
Please review the original document for complete details."""