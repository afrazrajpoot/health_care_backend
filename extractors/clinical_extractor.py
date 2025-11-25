"""
ClinicalNoteExtractor - Enhanced Extractor for Clinical Progress Notes and Therapy Reports
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
Version: 1.3 - Strict Anti-Hallucination for Signatures
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


class ClinicalNoteExtractor:
    """
    Enhanced Clinical Note extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different clinical specialties
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Progress Notes, PT/OT/Chiro/Acupuncture, Pain Management, Psychiatry, Nursing Notes
    - Direct LLM generation for long summary (removes intermediate extraction)
    - Extracts patient details and signature author for long summary (strict extraction from sign block only)
    - Short summary focuses only on critical findings and abnormal actions (no patient details), includes author if explicitly signed
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.note_type_patterns = {
            'progress_note': re.compile(r'\b(progress note|follow[- ]?up|office visit|clinic note)\b', re.IGNORECASE),
            'physical_therapy': re.compile(r'\b(physical therapy|PT|therapeutic exercise|range of motion|ROM)\b', re.IGNORECASE),
            'occupational_therapy': re.compile(r'\b(occupational therapy|OT|ADL|activities of daily living|functional capacity)\b', re.IGNORECASE),
            'chiropractic': re.compile(r'\b(chiropractic|chiropractor|adjustment|manipulation|subluxation)\b', re.IGNORECASE),
            'acupuncture': re.compile(r'\b(acupuncture|needle|meridian|qi|energy flow)\b', re.IGNORECASE),
            'pain_management': re.compile(r'\b(pain management|pain clinic|chronic pain|pain scale|analgesic)\b', re.IGNORECASE),
            'psychiatry': re.compile(r'\b(psychiatry|psychiatric|mental status|affect|mood|psychotropic)\b', re.IGNORECASE),
            'psychology': re.compile(r'\b(psychology|psychological|therapy session|counseling|behavioral)\b', re.IGNORECASE),
            'nursing': re.compile(r'\b(nursing note|nurse visit|vital signs|nursing assessment|medication administration)\b', re.IGNORECASE)
        }
        
        # Clinical assessment patterns
        self.assessment_patterns = {
            'pain_scale': re.compile(r'\b(pain|discomfort)\s*(scale|level|score)?\s*[:\-]?\s*(\d+/10|\d+\s*out\s*of\s*10)', re.IGNORECASE),
            'rom_measurements': re.compile(r'\b(ROM|range of motion)\s*[:\-]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'functional_status': re.compile(r'\b(able to|unable to|independent|assist|assistance|with help)\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'treatment_codes': re.compile(r'\b(CPT[:\s]*(\d{4,5})|(9716[01234]|9753[05]|9775[05]))', re.IGNORECASE)
        }
        
        # Patterns for patient details
        self.patient_patterns = {
            'name': re.compile(r'\b(patient name|name|mr\.?\s*mrs\.?\s*ms\.?\s*)\s*[:\-]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
            'dob': re.compile(r'\b(dob|date of birth|birthdate)\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', re.IGNORECASE)
        }
        
        # Enhanced patterns for signature (more comprehensive, distinguish physical/electronic)
        self.signature_patterns = {
            'physical_author': re.compile(r'(?i)(?:handwritten|wet|physical signature|ink signature)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:,?\s*(?:MD|DO|PA|NP|RN|PT|OT|DPT|DC|PhD|etc\.?))?)', re.DOTALL),
            'electronic_author': re.compile(r'(?i)(?:electronically signed|e-signature|digital signature|/s/|typed signature)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:,?\s*(?:MD|DO|PA|NP|RN|PT|OT|DPT|DC|PhD|etc\.?))?)', re.DOTALL),
            'sign_block': re.compile(r'(?i)(signature|sign off|attestation|certification|approval)\s*(?:section|block)?[:\-]?\s*(.*?)(?=\n{2,}|\Z)', re.DOTALL)
        }
        
        logger.info("✅ ClinicalNoteExtractor v1.3 initialized (Full Context + Strict Anti-Hallucination for Signatures)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
     
    ) -> Dict:
        """
        Extract Clinical Note data with FULL CONTEXT.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Progress Note, PT, OT, Chiro, Acupuncture, Pain Management, Psychiatry, Nursing)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING CLINICAL NOTE EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        # Auto-detect specific note type if not specified
        detected_type = self._detect_note_type(text, doc_type)
        logger.info(f"📋 Clinical Note Type: {detected_type} (original: {doc_type})")
        
        # Check document size
        text_length = len(text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Directly generate long summary with FULL CONTEXT (no intermediate extraction)
        long_summary = self._generate_long_summary_direct(
            text=text,
            doc_type=detected_type,
            fallback_date=fallback_date
        )

        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)

        logger.info("=" * 80)
        logger.info("✅ CLINICAL NOTE EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_note_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific clinical note type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for note_type, pattern in self.note_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[note_type] = len(matches)
        
        # Boost scores for treatment-specific terminology
        if self.assessment_patterns['treatment_codes'].search(text):
            for note_type in ['physical_therapy', 'occupational_therapy', 'chiropractic']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 2
        
        if self.assessment_patterns['pain_scale'].search(text):
            for note_type in ['pain_management', 'progress_note']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected note type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect note type, using: {original_type}")
        return original_type or "Clinical Note"

    def _generate_long_summary_direct(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Enhanced for strict extraction of physical/electronic signature author only from explicit sign blocks.
        """
        logger.info("🔍 Processing ENTIRE clinical note in single context window for direct long summary...")
        
        # Enhanced System Prompt with stricter rules for signature extraction (physical/electronic) - Anti-Hallucination Focus
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert clinical documentation specialist analyzing a COMPLETE {doc_type}.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE clinical note at once, allowing you to:
- Understand the complete clinical encounter from subjective complaints to treatment plan
- Track progress across multiple visits and treatment sessions
- Identify patterns in symptoms, functional limitations, and treatment response
- Provide comprehensive extraction without information loss
- Extract patient details from demographics/header sections
- Extract signature author STRICTLY from explicit sign block/signature section at the end (e.g., "Electronically signed by", "Signature:", "Attested by"). Distinguish physical (handwritten/wet) vs electronic (e-signature/digital).

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY - ABSOLUTE FOR SIGNATURES):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the note, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate clinical information
   - DO NOT fill in "typical" or "common" clinical values
   - DO NOT use clinical knowledge to "complete" incomplete information
   
2. **SUBJECTIVE COMPLAINTS - PATIENT'S EXACT WORDS**
   - Extract patient complaints using EXACT wording from note
   - DO NOT interpret or rephrase patient statements
   - Include pain descriptions, functional limitations, and concerns verbatim
   
3. **OBJECTIVE FINDINGS - MEASURED VALUES ONLY**
   - Extract ROM measurements, strength grades, pain scales ONLY if explicitly stated
   - Include units and specific values EXACTLY as written
   - DO NOT calculate or estimate ranges
   
4. **TREATMENTS & MODALITIES - SPECIFIC DETAILS ONLY**
   - Extract treatment techniques, modalities, exercises ONLY if explicitly listed
   - Include durations, frequencies, parameters ONLY if specified
   - DO NOT add typical treatment protocols
   
5. **ASSESSMENT & PLAN - CLINICIAN'S EXACT WORDING**
   - Extract clinical assessment using EXACT phrasing from note
   - Include treatment plan details verbatim
   - DO NOT interpret clinical reasoning

6. **PATIENT DETAILS - FROM DEMOGRAPHICS/HEADER**
   - Extract patient name, DOB, ID ONLY if explicitly stated in patient info section (e.g., "## PATIENT INFORMATION")
   - Use exact formatting from note
   - Do NOT extract from narrative text

7. **SIGNATURE AUTHOR - STRICTLY FROM SIGN BLOCK ONLY (PHYSICAL/ELECTRONIC) - NO HALLUCINATIONS**
   - Extract author name and credentials ONLY if there is an EXPLICIT signature section with signing language (e.g., "Electronically signed by [Name]", "Handwritten by [Name]", "Signed by [Name], MD", "Attested by [Name] with e-signature", "Signature: /s/ [Name]").
   - Distinguish: Physical (e.g., "Handwritten by Dr. X", "Wet signature by Dr. X") vs Electronic (e.g., "Electronically signed by Dr. Y", "/s/ Dr. Y", "Digital signature by Dr. Y").
   - List separately if both present; use exact names/titles from the document.
   - CRITICAL: DO NOT extract ANY name as signer if there is NO explicit signing phrase. Provider names, treating physicians, or mentioned authors are NOT signers unless the sign block explicitly says they signed.
   - Examples of INVALID extraction (DO NOT DO THIS):
     - If note says "Provider: Joshua T. Ritter, D.C." but no "signed by" -> OMIT entirely, do not assume electronic.
     - If only "Dictated by Joshua" but no signature block -> OMIT.
     - If "Report prepared by Joshua" without signing language -> OMIT.
   - If no explicit sign block with signing language found, leave COMPLETELY EMPTY (no "Not explicitly signed" - just omit the bullets).
   - Scan the ENTIRE document, but prioritize the end for sign blocks.

EXTRACTION FOCUS - 10 CRITICAL CLINICAL NOTE CATEGORIES:

I. NOTE IDENTITY & ENCOUNTER CONTEXT
- Note type, dates, encounter information
- Provider details and credentials (treating/radiologist/referring)

II. PATIENT INFORMATION
- Patient name, DOB, other demographics from explicit header sections

III. SUBJECTIVE FINDINGS (PATIENT'S PERSPECTIVE)
- Chief complaint and history of present illness
- Pain characteristics: location, intensity, quality, timing
- Functional limitations and impact on daily activities
- Patient's goals and expectations
- Relevant medical and social history

IV. OBJECTIVE EXAMINATION FINDINGS (CLINICIAN'S OBSERVATIONS)
- Vital signs and general appearance
- Physical examination findings:
  * Range of motion (ROM) measurements with specific degrees
  * Manual muscle testing grades (0-5)
  * Palpation findings and tender points
  * Special tests and orthopedic assessments
  * Neurological examination findings
- Functional capacity assessments
- Observation of movement patterns and gait

V. TREATMENT PROVIDED (SESSION-SPECIFIC)
- Specific techniques and modalities used:
  * Manual therapy techniques
  * Therapeutic exercises prescribed
  * Modalities applied (heat, ice, electrical stimulation, etc.)
  * Acupuncture points or chiropractic adjustments
- Treatment parameters: duration, intensity, frequency
- Patient response to treatment
- Any adverse reactions or complications

VI. ASSESSMENT & CLINICAL IMPRESSION
- Clinical assessment and diagnosis
- Progress since last visit
- Changes in functional status
- Barriers to recovery
- Prognosis and expected outcomes

VII. TREATMENT PLAN & GOALS
- Short-term and long-term goals
- Specific plan for next visit
- Home exercise program details
- Frequency and duration of continued care
- Referrals or consultations needed

VIII. WORK STATUS & FUNCTIONAL CAPACITY
- Current work restrictions
- Functional limitations
- Ability to perform job duties
- Expected return to work timeline

IX. SIGNATURE & AUTHOR (STRICT PHYSICAL/ELECTRONIC - NO ASSUMPTIONS)
- Signed by author from explicit sign block only, distinguishing physical/electronic - OMIT if not explicit

X. OUTCOME MEASURES & PROGRESS TRACKING
- Standardized outcome measures (ODI, NDI, DASH, etc.)
- Pain scale ratings (0-10)
- Functional improvement metrics
- Patient satisfaction measures

⚠️ FINAL REMINDER:
- If information is NOT in the note, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate clinical information - ESPECIALLY FOR SIGNATURES
- PAIN SCALES: Extract exact numbers (e.g., "6/10") not descriptions
- ROM MEASUREMENTS: Extract exact degrees, not ranges
- SIGNATURE: ONLY from explicit sign block with signing language; distinguish physical vs electronic; OMIT if no explicit signing phrase; provider names are NOT signers unless specified with "signed by"
- It is BETTER to have empty fields than incorrect clinical information - EMPTY SIGNATURE > WRONG SIGNER

Now analyze this COMPLETE {doc_type} and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # User Prompt updated for physical/electronic distinction
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no visit date found):

👤 PATIENT INFORMATION
--------------------------------------------------
Name: [extracted from explicit demographics/header, e.g., "## PATIENT INFORMATION"]
DOB: [extracted]
Other Details: [extracted, e.g., ID, gender, claim number, DOI]

📋 CLINICAL ENCOUNTER OVERVIEW
--------------------------------------------------
Note Type: {doc_type}
Visit Date: [extracted or {fallback_date}]
Visit Type: [extracted]
Duration: [extracted]
Facility: [extracted]

👨‍⚕️ PROVIDER INFORMATION
--------------------------------------------------
Treating Provider: [name, e.g., Radiologist or Referring]
  Credentials: [extracted]
  Specialty: [extracted]

🗣️ SUBJECTIVE FINDINGS
--------------------------------------------------
Chief Complaint: [extracted]
Pain: [location, intensity, quality]
Functional Limitations:
• [list up to 5, exact wording]

🔍 OBJECTIVE EXAMINATION
--------------------------------------------------
Range of Motion:
• [list up to 5, with body part, motion, degrees or qualitative if no degrees]
Manual Muscle Testing:
• [list up to 3, with muscle and grade/5]
Special Tests:
• [list up to 3, with results]

💆 TREATMENT PROVIDED
--------------------------------------------------
Treatment Techniques:
• [list up to 5]
Therapeutic Exercises:
• [list up to 5]
Modalities Used:
• [list up to 3]

🏥 CLINICAL ASSESSMENT
--------------------------------------------------
Assessment: [extracted]
Progress: [extracted]
Clinical Impression: [extracted]
Prognosis: [extracted]

🎯 TREATMENT PLAN
--------------------------------------------------
Short-term Goals:
• [list up to 3]
Home Exercise Program:
• [list up to 3]
Frequency/Duration: [extracted]
Next Appointment: [extracted]

💼 WORK STATUS
--------------------------------------------------
Current Status: [extracted]
Work Restrictions:
• [list up to 5, exact wording]
Functional Capacity: [extracted]

📊 OUTCOME MEASURES
--------------------------------------------------
Pain Scale: [extracted, e.g., 6/10]
Functional Scores:
• [list up to 3, with measure and value]

✍️ SIGNATURE & AUTHOR
--------------------------------------------------
Signer/Author:
• Physical Signature: [extracted name/title ONLY if explicit physical signing phrase present; otherwise omit]
• Electronic Signature: [extracted name/title ONLY if explicit electronic signing phrase present; otherwise omit]

🚨 CRITICAL CLINICAL FINDINGS
--------------------------------------------------
• [list up to 8 most significant items]

⚠️ CRITICAL CLINICAL REMINDERS:
1. For "range_of_motion": Extract EXACT measurements with degrees; use qualitative (e.g., "mildly reduced") if no numbers
   - Include body part, motion, and specific degrees/qualitative
   - Example: "Cervical flexion: mildly reduced with pain"

2. For "pain_scale": Extract EXACT numerical values (0-10 scale)
   - Example: "6/10" not "moderate pain"
   - Include location if specified

3. For "treatment_techniques": Extract SPECIFIC techniques used
   - Include parameters if specified (e.g., "US 1.5 W/cm² for 8 minutes")
   - Do not include techniques mentioned as options or for future use

4. For "work_restrictions": Extract EXACT limitations stated
   - Include weight limits, positional restrictions, duration
   - Example: "no lifting >10 lbs" not "light duty"

5. For "critical_clinical_findings": Include clinically significant changes
   - Worsening symptoms or functional decline
   - New neurological findings
   - Significant progress or setbacks
   - Compliance issues affecting treatment

6. For patient details: Extract ONLY from explicit demographics sections (e.g., "## PATIENT INFORMATION")
   - Do not infer from narrative or provider sections

7. For signature author: Extract STRICTLY from sign block/signature section (usually at end), distinguishing physical vs electronic
   - MUST have explicit signing language (e.g., "Electronically signed by [Name]", "Handwritten by [Name]")
   - Radiologist/Referring Physician is NOT a signer unless in sign block with signing phrase
   - If absent or no signing language, OMIT the entire Signer/Author section - do not add "Not explicitly signed"
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context clinical note long summary generation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Direct clinical note long summary generation completed in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char clinical note")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct clinical note long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Clinical note exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large notes")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"
    
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
        cleaned_summary = ' | '.join(cleaned_parts)
        
        logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
        return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word clinical note summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        Focuses ONLY on critical findings and abnormal actions (no patient details), includes author ONLY if explicitly signed with type.
        Starts with Doc Type first.
        """

        logger.info("🎯 Generating 30–60 word clinical structured summary (critical findings & abnormal actions only, strict author with type, Doc Type first)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a clinical documentation specialist.

    TASK:
    Create a concise, factual clinical summary using ONLY information explicitly present in the long summary.
    Focus EXCLUSIVELY on critical findings and abnormal actions. NO patient details (name, DOB, etc.).
    Include the signing author ONLY if explicitly stated in "SIGNATURE & AUTHOR" section as signed, with type (Physical/Electronic).

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:

    Doc Type:[value] | Author:[value] | Critical Findings:[value] | Abnormal Actions:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - NEVER include empty fields. If a field is missing (e.g., no explicit signer), SKIP that key and remove its pipe.
    - NEVER fabricate: no invented findings, actions, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.
    - Key-value format: Key:[value] (values as comma-separated lists if multiple)
    - Doc Type: Always include the document type as the first field (e.g., "Doc Type: Progress Note")
    - Author: ONLY from "SIGNATURE & AUTHOR" if explicitly signed with signing language (e.g., "Electronic Signature: Dr. Smith" -> "Author: Dr. Smith (Electronic)"); skip if omitted or no signing phrase. CRITICAL: Do not assume signing from provider names alone.
    - Critical Findings: Significant clinical changes, worsening symptoms, new abnormalities, progress setbacks
    - Abnormal Actions: Unexpected treatment responses, adverse reactions, compliance issues, abnormal exam results

    CONTENT PRIORITY (only if provided in the long summary, no patient info):
    1. Doc Type: Always "{doc_type}"
    2. Author ONLY if explicit signature with type and signing language (e.g., "Physical Signature: John Doe, MD" -> "Author: John Doe, MD (Physical)"); OMIT if no explicit signing
    3. Critical clinical findings from "CRITICAL CLINICAL FINDINGS" section
    4. Abnormal objective findings (e.g., reduced ROM, low strength)
    5. Adverse treatment reactions
    6. Functional declines or barriers
    7. Compliance or abnormal patient responses

    ABSOLUTELY FORBIDDEN:
    - Patient details (name, DOB, demographics)
    - Using provider/radiologist as author unless explicitly signed with type and signing phrase (e.g., NO: "Provider: Joshua" -> do not make "Author: Joshua (Electronic)")
    - assumptions, interpretations, or inferred issues
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")

    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style as string text.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    CLINICAL LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured clinical summary following ALL rules, starting with Doc Type, including author ONLY if explicitly signed with type and signing language, focusing only on critical findings and abnormal actions.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })
            summary = response.content.strip()

            # Normalize whitespace only - no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()

            # Validate word count
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Clinical summary out of range ({wc} words). Attempting auto-fix...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior summary contained {wc} words. Rewrite it to be between 30 and 60 words. "
                        "DO NOT add fabricated details or patient info. Preserve all factual elements including author ONLY if explicit with type and signing language, Doc Type first. "
                        "Maintain key-value pipe-delimited format: Doc Type:[value] | Author:[value] | Critical Findings:[value] | Abnormal Actions:[value] (skip missing keys)"
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after auto-fix

            logger.info(f"✅ Clinical summary generated: {len(summary.split())} words")
            return self._clean_pipes_from_summary(summary)

        except Exception as e:
            logger.error(f"❌ Clinical summary generation failed: {e}")
            return "Summary unavailable due to processing error."

    def _create_clinical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback clinical summary directly from long summary"""
        
        # Extract key clinical information using regex patterns
        patterns = {
            'provider': r'Treating Provider:\s*([^\n]+)',
            'pain': r'Pain Scale:\s*([^\n]+)',
            'complaint': r'Chief Complaint:\s*([^\n]+)',
            'rom': r'Range of Motion:(.*?)(?:\n\n|\n[A-Z]|$)',
            'treatment': r'Treatment Techniques:(.*?)(?:\n\n|\n[A-Z]|$)',
            'assessment': r'Assessment:\s*([^\n]+)',
            'plan': r'Frequency/Duration:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} visit")
        
        # Add provider context
        if 'provider' in extracted:
            parts.append(f"by {extracted['provider']}")
        
        # Add chief complaint and pain
        if 'complaint' in extracted:
            complaint = extracted['complaint'][:60] + "..." if len(extracted['complaint']) > 60 else extracted['complaint']
            parts.append(f"for {complaint}")
        
        if 'pain' in extracted:
            parts.append(f"Pain: {extracted['pain']}")
        
        # Add key findings
        if 'rom' in extracted:
            first_rom = extracted['rom'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Findings: {first_rom}")
        
        # Add treatment
        if 'treatment' in extracted:
            first_treatment = extracted['treatment'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Treatment: {first_treatment}")
        
        # Add assessment and plan
        if 'assessment' in extracted:
            assessment = extracted['assessment'][:60] + "..." if len(extracted['assessment']) > 60 else extracted['assessment']
            parts.append(f"Assessment: {assessment}")
        
        if 'plan' in extracted:
            parts.append(f"Plan: {extracted['plan']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with continued clinical management", "following treatment protocols", "with ongoing progress monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used clinical fallback summary: {len(summary.split())} words")
        return summary