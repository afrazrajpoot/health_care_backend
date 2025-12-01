"""
Specialist Consult Enhanced Extractor - Full Context
Optimized for accuracy using Gemini-style full-document processing
8 CRITICAL FIELDS FOCUSED (NO HALLUCINATION, ONLY EXPLICIT INFORMATION)
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


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning for specialist recommendation evaluation
    - 8-FIELD FOCUSED extraction for zero hallucination
    - Optimized for Workers' Compensation consultation analysis
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)

    def extract(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        Extract specialist consult report with full context processing. 
        Returns dictionary with long_summary and short_summary like QME extractor.
        """
        logger.info("=" * 80)
        logger.info("👨‍⚕️ STARTING CONSULT EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        start_time = time.time()
        
        try:
            # Step 1: Directly generate long summary with full context (no intermediate extraction)
            long_summary = self._generate_long_summary_direct(text, doc_type, fallback_date)

            # Step 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)

            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context consultation extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted consultation data from complete {len(text):,} char document")
            logger.info("=" * 80)
            logger.info("✅ CONSULT EXTRACTION COMPLETE (FULL CONTEXT)")
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
                "long_summary": f"Consultation extraction failed: {str(e)}",
                "short_summary": "Consultation summary not available"
            }

    def _generate_long_summary_direct(self, text: str, doc_type: str, fallback_date: str) -> str:
        """
        Directly generate comprehensive long summary with full document context using LLM.
        Adapted from original extraction prompt to output structured summary directly.
        """
        logger.info("🔍 Processing ENTIRE consultation report in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Reuses core anti-hallucination rules and consultation focus from original extraction prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation consultation report specialist analyzing a COMPLETE Specialist Consultation Report.

PRIMARY PURPOSE: Generate a comprehensive, structured long summary of the 8 critical fields that define current medical status, diagnostic findings, and treatment plan changes for claims administration.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE consultation report at once, allowing you to:
- Understand complete referral context and clinical questions
- Correlate subjective complaints with objective examination findings
- Assess specialist's diagnostic reasoning and treatment justifications
- Identify ALL treatment recommendations with medical necessity rationale
- Extract complete work capacity and restriction changes

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY) (donot include in output, for LLM use only):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If NOT explicitly mentioned, return EMPTY string "" or empty list []
   - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
   - DO NOT add typical findings, standard recommendations, or common restrictions

2. **DIAGNOSES - INCLUDE ALL WITH EXACT TERMINOLOGY**
   - Extract PRIMARY diagnosis with ICD-10 code if stated
   - Extract ALL SECONDARY diagnoses - DO NOT miss any
   - Include diagnostic certainty: "probable", "confirmed", "consistent with", "rule out"
   - Include causation statements if explicitly stated

3. **WORK RESTRICTIONS - EXACT WORDING ONLY**
   - Use EXACT phrases from document
   - If "no overhead work" stated, extract "no overhead work" (NOT "no overhead reaching")
   - If "lifting restrictions" stated WITHOUT specifics, extract "lifting restrictions" (NOT "no lifting >10 lbs")

4. **RECOMMENDATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY recommendations explicitly stated in Assessment/Plan sections
   - DO NOT extract treatments "considered but not recommended"
   - Include ALL explicitly recommended procedures, injections, medications, therapy
   - DO NOT add standard care not mentioned

5. **EMPTY FIELDS ARE BETTER THAN GUESSED FIELDS**
   - Leave fields empty if information not found
   - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""

6. **CONSULTING PHYSICIAN/AUTHOR DETECTION**:
   - Identify the author who signed the report as the "consulting_physician" name (e.g., from signature block, "Dictated by:", or closing statement).
   - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
   - Extract specialty, credentials, and facility only if explicitly stated near the signature.
   - If no clear signer is found, leave "name" empty.

CONSULTATION EXTRACTION FOCUS - 8 CRITICAL FIELDS FOR CLAIMS ADMINISTRATION:

FIELD 1: HEADER & CONTEXT - Report Identity & Authority
FIELD 2: CHIEF COMPLAINT - Patient's Primary Issue
FIELD 3: DIAGNOSIS & ASSESSMENT - Medical Conclusion (HIGHEST PRIORITY)
FIELD 4: HISTORY OF PRESENT ILLNESS - Symptoms & Severity Context
FIELD 5: PRIOR TREATMENT & EFFICACY - Failure of Conservative Care
FIELD 6: OBJECTIVE FINDINGS - Verifiable Evidence (Exam & Imaging)
FIELD 7: PLAN - RECOMMENDED TREATMENTS (MOST CRITICAL FOR AUTHORIZATION)
FIELD 8: WORK STATUS & IMPAIRMENT - Legal/Administrative Status

Now analyze this COMPLETE specialist consultation report and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")
        
        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE SPECIALIST CONSULTATION REPORT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no consultation date found):

📋 CONSULTATION OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Consultation Date: [extracted or {fallback_date}]
Consulting Physician: [name]
Specialty: [extracted]
Referring Physician: [extracted]

👤 PATIENT INFORMATION
--------------------------------------------------
Name: [extracted]
Date of Birth: [extracted]
Date of Injury: [extracted]
Claim Number: [extracted]
                                                               
All Doctors Involved:
• [list all extracted doctors with names and titles]
━━━ ALL DOCTORS EXTRACTION ━━━
- Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
- Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
- Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
- Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
- Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
- If no doctors found, leave list empty [].

🎯 CHIEF COMPLAINT
--------------------------------------------------
Primary Complaint: [extracted]
Location: [extracted]
Duration: [extracted]
Radiation Pattern: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]


🏥 DIAGNOSIS & ASSESSMENT
--------------------------------------------------
Primary Diagnosis: [extracted with ICD-10 if available] ([certainty])
Secondary Diagnoses:
• [list up to 5]
Causation: [extracted]

🔬 CLINICAL HISTORY & SYMPTOMS
--------------------------------------------------
Pain Quality: [extracted]
Pain Location: [extracted]
Radiation: [extracted]
Aggravating Factors:
• [list up to 5, exact wording]
Alleviating Factors:
• [list up to 5, exact wording]

💊 PRIOR TREATMENT & EFFICACY
--------------------------------------------------
Prior Treatments Received:
• [list up to 8 with durations if stated]
Level of Relief:
• [physical_therapy: extracted]
• [medications: extracted]
• [injections: extracted]
• [chiropractic: extracted]
Treatment Failure Statement: [extracted]

📊 OBJECTIVE FINDINGS
--------------------------------------------------
Physical Examination:
• [ROM measurements: extracted]
• [Strength Testing: extracted]
• [Special Tests Positive: list up to 5]
• [Palpation Findings: extracted]
• [Inspection Findings: extracted]
Imaging Review:
• [MRI Findings: extracted]
• [X-ray Findings: extracted]
• [CT Findings: extracted]
• [Correlation with Symptoms: extracted]

🎯 TREATMENT RECOMMENDATIONS
--------------------------------------------------
Injections Requested:
• [list up to 5 with locations]
Procedures Requested:
• [list up to 5 with reasons]
Surgery Recommended:
• [list up to 3 with urgency]
Diagnostics Ordered:
• [list up to 5 with reasons]
Medication Changes:
  New Medications:
    • [list up to 5 with doses]
  Dosage Adjustments:
    • [list up to 3]
Therapy Recommendations:
• [therapy_type - frequency for duration]
  Focus Areas:
    • [list up to 3]

💼 WORK STATUS & IMPAIRMENT
--------------------------------------------------
Current Work Status: [extracted]
Work Restrictions:
• [list up to 10, exact wording with durations if stated]
Restriction Duration: [extracted]
Return to Work Plan: [extracted]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 8 most actionable items for claims administration]

⚠️ MANDATORY EXTRACTION RULES (donot include in output, for LLM use only):
1. Field 3 (Diagnosis): Use EXACT diagnostic terminology, include ALL secondary diagnoses
2. Field 5 (Prior Treatment): Include quantified relief ("50% improvement", "no relief")
3. Field 6 (Objective): Extract EXACT measurements and imaging findings with anatomical specificity
4. Field 7 (Plan): Extract ONLY explicitly recommended treatments - THE MOST CRITICAL
5. Field 8 (Work Restrictions): Use EXACT wording (don't add weight/time limits not stated)
6. Empty fields are acceptable if not stated in report - use "Not specified" only if truly empty
""")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])
        
        logger.info(f"📄 Document size: {len(text):,} chars (~{len(text) // 4:,} tokens)")
        logger.info("🔍 Processing ENTIRE consultation report in single context window...")
        logger.info("🤖 Invoking LLM for direct full-context consultation long summary generation...")
        
        # Invoke LLM
        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "fallback_date": fallback_date,
                "doc_type": doc_type
            })
            
            long_summary = result.content.strip()
            
            logger.info("✅ Generated consultation long summary from complete document")
            return long_summary
        
        except Exception as e:
            logger.error(f"❌ Direct LLM generation failed: {str(e)}")
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
        Generate a precise 30–60 word consultation summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word consultation structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal consultation specialist.

TASK:
Create a concise, factual consultation summary using ONLY information explicitly stated in the long summary.
- **ONLY include, critical, or clinically significant findings**.
- **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Key Findings:[value] | Medication:[value] | Treatments:[value] | Clinical Assessment:[value] | Plan:[value] | MMI Status:[value] | Work Status:[value] | Critical Finding:[value]

KEY RULES:
- ONLY include abnormal or critical findings for these fields (physical exam, vital signs).
- If a value is missing or not extractable, omit the ENTIRE key-value pair.
- NEVER output empty fields or placeholder text.
- NEVER fabricate dates, meds, findings, or recommendations.
- NO narrative sentences; use short factual fragments.
- First three fields (Report Title, Author, Date) appear without keys.
- All other fields use key-value format: Key:[value].
- Key Findings and Physical Exam details include **only abnormalities or critical observations**.
- Medications, Treatments, Plan, MMI Status, Work Status included only if explicitly stated in the summary.

CONTENT PRIORITY (only if critical and provided):
1. Report Title  
2. Author  
3. Date  
4. Body Parts  
5. Diagnosis  
6. Physical Exam (abnormal only if present)  
7. Vital Signs (abnormal only if present)  
8. Key objective findings (abnormal only)  
9. Treatment plan (given explicit only)
10. Medications (given explicit only)  
11. Recommendations (if explicitly stated) or Plan (if explicitly or stated)  
12. MMI status (if stated)  
13. Work status (if stated)  
14. Critical finding

ABSOLUTELY FORBIDDEN:
- Normal findings (ignore entirely for physical exam and vital signs)
- Assumptions, interpretations, inferred diagnoses
- Patient details (name, DOB, claim, MRN, etc.)
- Narrative writing
- Placeholder text or "Not provided"
- Duplicate or empty pipes (||)

Your final output MUST be between 30–60 words, single-line, pipe-delimited, and contain ONLY explicitly provided abnormal or critical information.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
    CONSULTATION LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured consultation summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "long_summary": long_summary
            })
            summary = response.content.strip()

            # Normalize whitespace only - no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()

            # Validate word count
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Consultation summary out of range ({wc} words). Fixing...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary contained {wc} words. Rewrite it to be **between 30 and 60** words. "
                        "Do NOT add fabricated data. Preserve all factual elements. Maintain the key-value pipe-delimited format: [Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after fix

            logger.info(f"✅ Consultation summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Consultation summary generation failed: {e}")
            return "Summary unavailable due to processing error."
 
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Consulting Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Consulting Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'diagnosis': r'Primary Diagnosis:\s*([^\n]+)',
            'complaint': r'Primary Complaint:\s*([^\n]+)',
            'recommendations': r'TREATMENT RECOMMENDATIONS(.*?)(?:\n\n|\n[A-Z]|$)',
            'restrictions': r'Work Restrictions:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} consultation")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"for {extracted['diagnosis'][:80]}")
        elif 'complaint' in extracted:
            parts.append(f"for {extracted['complaint'][:80]}")
        
        # Add recommendations
        if 'recommendations' in extracted:
            # Take first line of recommendations
            first_rec = extracted['recommendations'].split('\n')[0].replace('•', '').strip()[:60]
            if first_rec:
                parts.append(f"Recommendations: {first_rec}")
        
        # Add work restrictions
        if 'restrictions' in extracted:
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with follow-up planned", "for ongoing management", "and progress evaluation"] 
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
                # For dicts, convert to string representation
                return str(value)
            if isinstance(value, list):
                # For lists, join with comma
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