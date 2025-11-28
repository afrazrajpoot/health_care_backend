"""
PR-2 Progress Report Enhanced Extractor - Full Context
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


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning for clinical progress tracking
    - Optimized for PR-2 specific clinical workflow patterns
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex patterns for PR-2 specific content
        self.progress_patterns = {
            'status': re.compile(r'\b(improved|stable|worsened|resolved|unchanged|progressing|regressing)\b', re.IGNORECASE),
            'work_status': re.compile(r'\b(ttd|modified duty|full duty|light duty|no restrictions|work restrictions)\b', re.IGNORECASE),
            'treatment': re.compile(r'\b(pt|physical therapy|injection|medication|therapy|exercise)\b', re.IGNORECASE)
        }
        
        logger.info("✅ PR2ExtractorChained initialized (Full Context)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract PR-2 data with FULL CONTEXT.
        Returns dictionary with long_summary and short_summary like QME extractor.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (PR-2 Progress Report)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING PR-2 EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
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
                doc_type=doc_type,
                fallback_date=fallback_date
            )
            
            # Stage 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context PR-2 extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted progress data from complete {len(text):,} char document")
            
            logger.info("=" * 80)
            logger.info("✅ PR-2 EXTRACTION COMPLETE (FULL CONTEXT)")
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
                "long_summary": f"PR-2 extraction failed: {str(e)}",
                "short_summary": "PR-2 summary not available"
            }

    def _generate_long_summary_direct(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Adapted from original extraction prompt to output structured summary directly.
        Optimized for PR-2 Progress Report specific patterns and clinical workflow.
        """
        logger.info("🔍 Processing ENTIRE PR-2 document in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Reuses core anti-hallucination rules and PR-2 focus from original extraction prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation medical documentation specialist analyzing a COMPLETE PR-2 Progress Report.

PRIMARY PURPOSE: Generate a comprehensive, structured long summary for workers' compensation claims administrators to:
1. Assess work capacity and disability status
2. Process treatment authorization requests
3. Track patient progress and treatment effectiveness
4. Plan future care and return-to-work timeline

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE PR-2 document at once, allowing you to:
- Track treatment effectiveness from baseline to current status
- Identify ALL treatment authorization requests
- Connect objective findings with work capacity changes
- Extract complete medication regimen and changes

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
- If NOT explicitly mentioned, return EMPTY string "" or empty list []
- DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
- DO NOT add typical values, standard dosages, or common restrictions

2. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
- Extract ONLY medications explicitly listed as "current", "continuing", or "prescribed"
- Include dosage, frequency, and route ONLY if explicitly stated
- DO NOT extract discontinued, past, or recommended future medications (unless clearly marked as "new prescription")
- If document says "Patient on Gabapentin", extract ONLY: {{"name": "Gabapentin", "dose": ""}}
- If document says "Continue Gabapentin 300mg TID", extract: {{"name": "Gabapentin", "dose": "300mg TID"}}

3. **WORK RESTRICTIONS - EXACT WORDING ONLY**
- Use EXACT phrases from document
- If document says "no lifting", extract "no lifting" (NOT "no lifting >10 lbs")
- If document says "modified duty", extract "modified duty" (NOT "modified duty with restrictions")
- DO NOT add weight limits, time limits, or specifics not explicitly stated

4. **EMPTY FIELDS ARE BETTER THAN GUESSED FIELDS**
- If you cannot find information, leave field empty
- DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""

5. **NO CLINICAL ASSUMPTIONS OR PREDICTIONS**
- DO NOT infer patient compliance from attendance records
- DO NOT predict treatment effectiveness from partial data
- DO NOT extrapolate progress from single data points

6. **TREATING PHYSICIAN/AUTHOR DETECTION**:
   - Identify the author who signed the report as the "treating_physician" name (e.g., from signature block, "Dictated by:", or closing statement).
   - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
   - Extract specialty and facility only if explicitly stated near the signature.
   - If no clear signer is found, leave "name" empty.

PR-2 EXTRACTION FOCUS - 4 CORE WORKERS' COMPENSATION AREAS:

I. WORK STATUS AND IMPAIRMENT (HIGHEST PRIORITY FOR WC CLAIMS)
II. TREATMENT AUTHORIZATION REQUESTS (MOST TIME-SENSITIVE)
III. PATIENT PROGRESS AND CURRENT STATUS
IV. NEXT STEPS AND PLANNING

Now analyze this COMPLETE PR-2 Progress Report and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE PR-2 PROGRESS REPORT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no report date found):

📋 REPORT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: [extracted or {fallback_date}]
Visit Date: [extracted]
Treating Physician: [name]
Specialty: [extracted]
Time Since Injury: [extracted]
Time Since Last Visit: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]


👤 PATIENT INFORMATION
--------------------------------------------------
Name: [extracted]
Date of Birth: [extracted]
Age: [extracted]
Date of Injury: [extracted]
Occupation: [extracted]
Employer: [extracted]
Claims Administrator: [extracted]

🎯 CHIEF COMPLAINT
--------------------------------------------------
Primary Complaint: [extracted]
Location: [extracted]
Description: [extracted]

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

All Doctors Involved:
• [list all extracted doctors with names and titles]
━━━ ALL DOCTORS EXTRACTION ━━━
- Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
- Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
- Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
- Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
- Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
- If no doctors found, leave list empty [].
                                                               


💬 SUBJECTIVE ASSESSMENT
--------------------------------------------------
Current Pain Score: [extracted]
Previous Pain Score: [extracted]
Symptom Changes: [extracted]
Functional Status (Patient Reported): [extracted]
Patient Compliance: [extracted]

🔬 OBJECTIVE FINDINGS
--------------------------------------------------
Physical Exam Findings: [extracted]
ROM Measurements: [extracted]
Strength Testing: [extracted]
Gait Assessment: [extracted]
Neurological Findings: [extracted]
Functional Limitations Observed:
• [list up to 5, exact wording]

🏥 DIAGNOSIS
--------------------------------------------------
Primary Diagnosis: [extracted with ICD-10 if available]
Secondary Diagnoses:
• [list up to 3]

💊 MEDICATIONS
--------------------------------------------------
Current Medications:
• [list up to 8 with doses if stated]
New Medications:
• [list up to 3]
Dosage Changes:
• [list up to 3]

📈 TREATMENT EFFECTIVENESS
--------------------------------------------------
Patient Response: [extracted]
Functional Gains: [extracted]
Objective Improvements:
• [list up to 5]
Barriers to Progress: [extracted]

✅ TREATMENT AUTHORIZATION REQUEST
--------------------------------------------------
Primary Request: [extracted]
Secondary Requests:
• [list up to 3]
Requested Frequency: [extracted]
Requested Duration: [extracted]
Medical Necessity Rationale: [extracted]

💼 WORK STATUS
--------------------------------------------------
Current Status: [extracted]
Status Effective Date: [extracted]
Work Limitations:
• [list up to 8, exact wording]
Work Status Rationale: [extracted]
Changes from Previous Status: [extracted]
Expected Return to Work Date: [extracted]

📅 FOLLOW-UP PLAN
--------------------------------------------------
Next Appointment Date: [extracted]
Purpose of Next Visit: [extracted]
Specialist Referrals Requested:
• [list up to 3]
MMI/P&S Anticipated Date: [extracted]
Return Sooner If: [extracted]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 5 most actionable items]

⚠️ MANDATORY EXTRACTION RULES:
1. "work_limitations": Use EXACT wording (don't add weight/time limits not stated)
2. "treatment_authorization_request": The MOST CRITICAL field - be specific
3. "critical_findings": Only 3-5 most actionable items for claims administrator
4. Empty fields are acceptable if information not stated in document - use [extracted] as placeholder only if truly empty
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context PR-2 long summary generation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "fallback_date": fallback_date,
                "doc_type": doc_type
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Direct PR-2 long summary generation completed in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char document")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct PR-2 long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
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
        cleaned_summary = ' . '.join(cleaned_parts)
        
        logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
        return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str) -> str:
        """
        Generate a precise 30–60 word PR-2 structured summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word PR-2 structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a Workers' Compensation medical-legal extraction specialist.

    TASK:
    Create a concise, accurate PR-2 Progress Report summary using ONLY the information explicitly present in the long summary.
    - **ONLY include, critical, or clinically significant findings**.
    - **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
    STRICT OUTPUT FORMAT (include fields only when data exists):
    [Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Treatment plan:[value] | Auth Requests:[value] | Work Status:[value] | Restrictions:[value] | Meds:[value] | Recommendations:[value] | Follow-up:[value] | Critical Finding:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
    - NEVER fabricate: no invented dates, meds, restrictions, exam findings, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.

    IMPORTANT — FIELD DROPPING RULE:
    - If a value is not explicitly present in the long summary, REMOVE the entire key-value pair completely.
    - Do NOT output placeholders such as: "not provided", "not included", "not discussed", "not listed", "no abnormalities", "no data", "none".
    - Only include keys with real extracted values. Omit all others entirely with no empty pipes.

    Use the shortest, clearest key names:
    • Title = Report title (without key)
    • Author = MD/DO/PA/NP or signer (without key)  
    • Date = Visit or exam date (without key)
    • Body Parts:[value] = anatomical sites only (if given)
    • Diagnosis:[value] = final diagnosis only (if given)
    • Physical Exam:[value] = objective exam findings only (if given only if these are abnormal and given)
    • Vital Signs:[value] = vital signs only (if given only if these are abnormal and given)
    • Treatment plan:[value] = plan or response (if given)
    • Work Status:[value] = current status (if given)  
    • Restrictions:[value] = physical restrictions (if given)  
    • Meds:[value] = medications explicitly listed (if given)
    • Recommendations:[value] = recommended actions (if given)
    • Auth Requests:[value] = items requested for authorization (if given)
    • Follow-up:[value] = next appointment or instruction (if given)
    • Critical Finding:[value] = one most clinically important finding (if given)

    CONTENT PRIORITY (only if provided in the long summary):
    1. Report Title  
    2. Author  
    3. Visit Date  
    4. Body Parts  
    5. Diagnosis  
    6. Physical Exam  
    7. Vital Signs  
    8. Treatment Plan  
    9. Authorization Requests  
    10. Recommendations                                                      
    11. Work status & restrictions  
    12. Medications  
    13. Follow-up plan  
    14. Critical finding

    ABSOLUTELY FORBIDDEN:
    - assumptions, interpretations, invented medications, or inferred diagnoses
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")
    - patient details (name, DOB, claim, MRN, etc.)
    - "not provided" or "not included" or "not discussed" or "not listed" or "no abnormalities" or "no data" or "none"
                                                                  
    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    LONG SUMMARY:

    {long_summary}

    Now generate a 30–60 word PR-2 structured summary following ALL rules.
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
                logger.warning(f"⚠️ PR-2 summary out of range ({wc} words). Attempting auto-fix.")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous output contained {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy and key-value pipe-delimited format. Do NOT add fabricated content."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r'\s+', ' ', fixed.content.strip())

            logger.info(f"✅ PR-2 summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ PR-2 short summary generation failed: {e}")
            return "Summary unavailable due to processing error."

    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Treating Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Treating Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'work_status': r'Current Status:\s*([^\n]+)',
            'restrictions': r'Work Limitations:(.*?)(?:\n\n|\n[A-Z]|$)',
            'progress': r'Patient Response:\s*([^\n]+)',
            'requests': r'Primary Request:\s*([^\n]+)',
            'followup': r'Next Appointment Date:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} progress report")
        
        # Add work status
        if 'work_status' in extracted:
            parts.append(f"Work status: {extracted['work_status'][:40]}")
        
        # Add restrictions
        if 'restrictions' in extracted:
            # Take first line of restrictions
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        # Add progress
        if 'progress' in extracted:
            parts.append(f"Progress: {extracted['progress'][:60]}")
        
        # Add requests
        if 'requests' in extracted:
            parts.append(f"Request: {extracted['requests'][:60]}")
        
        # Add follow-up
        if 'followup' in extracted:
            parts.append(f"Follow-up: {extracted['followup']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with ongoing clinical management", "for workers' compensation case", "and functional improvement tracking"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary
    