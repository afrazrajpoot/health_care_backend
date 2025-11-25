"""
AdministrativeExtractor - Enhanced Extractor for Administrative and Legal Documents
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


class AdministrativeExtractor:
    """
    Enhanced Administrative Document extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different administrative document types
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Attorney Letters, NCM Notes, Employer Reports, Disability Forms, Legal Correspondence
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.admin_type_patterns = {
            'attorney': re.compile(r'\b(attorney|lawyer|counsel|legal|subpoena|deposition)\b', re.IGNORECASE),
            'ncm': re.compile(r'\b(ncm|nurse case manager|case management|utilization review)\b', re.IGNORECASE),
            'employer': re.compile(r'\b(employer|supervisor|incident report|work injury|workers comp)\b', re.IGNORECASE),
            'disability': re.compile(r'\b(disability|claim form|dwc|workers compensation|benefits)\b', re.IGNORECASE),
            'job_analysis': re.compile(r'\b(job analysis|physical demands|work capacity|vocational)\b', re.IGNORECASE),
            'medication': re.compile(r'\b(medication administration|mar|pharmacy|prescription log)\b', re.IGNORECASE),
            'pharmacy': re.compile(r'\b(pharmacy|prescription|drug|medication list|pharmacist)\b', re.IGNORECASE),
            'telemedicine': re.compile(r'\b(telemedicine|telehealth|virtual visit|remote consultation)\b', re.IGNORECASE),
            'legal': re.compile(r'\b(legal correspondence|demand letter|settlement|liability)\b', re.IGNORECASE)
        }
        
        # Administrative patterns
        self.admin_patterns = {
            'claim_numbers': re.compile(r'\b(claim\s*(?:number|no|#)?[:\s]*([A-Z0-9\-]+))', re.IGNORECASE),
            'dates': re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[-]\d{2}[-]\d{2})\b'),
            'deadlines': re.compile(r'\b(deadline|due date|response required by|must respond by)\s*[:]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'contact_info': re.compile(r'\b(phone|tel|fax|email|@)\s*[:\-]?\s*([^\s,]+)', re.IGNORECASE)
        }
        
        logger.info("✅ AdministrativeExtractor initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
    
    ) -> Dict:
        """
        Extract Administrative Document data with FULL CONTEXT.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Attorney, NCM, Employer, Disability, etc.)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING ADMINISTRATIVE DOCUMENT EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        # Auto-detect specific administrative type if not specified
        detected_type = self._detect_admin_type(text, doc_type)
        logger.info(f"📋 Administrative Type: {detected_type} (original: {doc_type})")
        
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
        logger.info("✅ ADMINISTRATIVE DOCUMENT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_admin_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific administrative document type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for admin_type, pattern in self.admin_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[admin_type] = len(matches)
        
        # Boost scores for administrative-specific terminology
        if self.admin_patterns['claim_numbers'].search(text):
            for admin_type in ['disability', 'ncm', 'employer']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 2
        
        if self.admin_patterns['deadlines'].search(text):
            for admin_type in ['attorney', 'legal']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected administrative type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect administrative type, using: {original_type}")
        return original_type or "Administrative Document"

    def _generate_long_summary_direct(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Adapted from original extraction prompt to output structured summary directly.
        """
        logger.info("🔍 Processing ENTIRE administrative document in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Reuses core anti-hallucination rules and administrative focus from original extraction prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert administrative and legal documentation specialist analyzing a COMPLETE {doc_type}.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE administrative document at once, allowing you to:
- Understand the complete administrative or legal context from headers to signatures
- Track deadlines, requirements, and action items throughout the document
- Identify key parties, contact information, and procedural requirements
- Provide comprehensive extraction without information loss

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate administrative information
   - DO NOT fill in "typical" or "common" administrative values
   - DO NOT use legal knowledge to "complete" incomplete information
   
2. **DATES & DEADLINES - EXACT WORDING ONLY**
   - Extract deadlines and dates using EXACT wording from document
   - Include timeframes and response requirements verbatim
   - DO NOT interpret or calculate dates
   
3. **CONTACT INFORMATION - VERBATIM EXTRACTION**
   - Extract phone numbers, emails, addresses EXACTLY as written
   - DO NOT format or normalize contact information
   - Include titles and roles ONLY if explicitly stated
   
4. **LEGAL & ADMINISTRATIVE TERMS - PRECISE EXTRACTION**
   - Extract legal demands, requirements, and conditions verbatim
   - Include specific claim numbers, case numbers, reference numbers
   - DO NOT interpret legal language or implications
   
5. **ACTION ITEMS & REQUIREMENTS - SPECIFIC DETAILS ONLY**
   - Extract required actions, documentation, responses ONLY if explicitly listed
   - Include quantities, formats, submission methods ONLY if specified
   - DO NOT add typical administrative procedures

EXTRACTION FOCUS - 8 CRITICAL ADMINISTRATIVE CATEGORIES:

I. DOCUMENT IDENTITY & CONTEXT
- Document type, dates, identification numbers
- Purpose and subject of the document
- All reference numbers and case identifiers

II. PARTIES INVOLVED
- All individuals and organizations mentioned
- Roles and relationships (sender, recipient, cc'd parties)
- Complete contact information for all parties
- Legal representation details if applicable
- Patient details (name, DOB, ID) if explicitly stated
- Report author/signer (name, title, signature details) if explicitly stated

III. KEY DATES & DEADLINES (MOST CRITICAL)
- Document date, effective dates, response deadlines
- Hearing dates, appointment dates, follow-up dates
- Time-sensitive requirements and cutoffs
- Statute of limitations if mentioned

IV. ADMINISTRATIVE CONTENT & SUBJECT
- Primary purpose and subject matter
- Summary of key points or issues addressed
- Background context and relevant history
- Specific incidents or events described

V. ACTION ITEMS & REQUIREMENTS
- Required responses and submissions
- Documentation or evidence required
- Specific actions to be taken by parties
- Compliance requirements and conditions

VI. LEGAL & PROCEDURAL ELEMENTS
- Legal demands, offers, or positions
- Procedural requirements and next steps
- Rights, obligations, and responsibilities
- Consequences of non-compliance if stated

VII. MEDICAL & CLAIM INFORMATION (if applicable)
- Claim numbers and case references
- Medical treatment authorizations or denials
- Work status and disability information
- Medication and treatment details

VIII. CONTACT & FOLLOW-UP PROCEDURES
- Submission methods and addresses
- Contact persons and departments
- Required response formats
- Follow-up procedures

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate administrative information
- DEADLINES: Use exact wording from document
- CONTACT INFO: Extract verbatim without formatting
- It is BETTER to have empty fields than incorrect administrative information

Now analyze this COMPLETE {doc_type} and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no document date found):

📋 ADMINISTRATIVE DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Document Date: [extracted or {fallback_date}]
Subject: [extracted]
Purpose: [extracted]
Document ID: [extracted]

👥 PARTIES INVOLVED
--------------------------------------------------
Patient Details: [name, DOB, ID if extracted]
From: [sender name]
  Organization: [extracted]
  Title: [extracted]
To: [recipient name]
  Organization: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit]
Legal Representation: [attorney name]
  Firm: [extracted]

📅 KEY DATES & DEADLINES
--------------------------------------------------
Response Deadline: [extracted]
Hearing Date: [extracted]
Appointment Date: [extracted]
Time-Sensitive Requirements:
• [list up to 3, exact wording]

📄 ADMINISTRATIVE CONTENT
--------------------------------------------------
Primary Subject: [extracted]
Key Points: [extracted]
Current Status: [extracted]
Incident Details: [extracted, truncate if >200 chars]

✅ ACTION ITEMS & REQUIREMENTS
--------------------------------------------------
Required Responses:
• [list up to 5]
Documentation Required:
• [list up to 5]
Specific Actions:
• [list up to 5]

⚖️ LEGAL & PROCEDURAL ELEMENTS
--------------------------------------------------
Legal Demands:
• [list up to 3]
Next Steps:
• [list up to 3]
Consequences of Non-Compliance: [extracted]

🏥 MEDICAL & CLAIM INFORMATION
--------------------------------------------------
Claim Number: [extracted]
Case Number: [extracted]
Work Status: [extracted]
Disability Information: [extracted]
Treatment Authorizations:
• [list up to 3]

📞 CONTACT & FOLLOW-UP
--------------------------------------------------
Contact Person: [extracted]
Phone: [extracted]
Email: [extracted]
Submission Address: [extracted]
Response Format: [extracted]

🚨 CRITICAL ADMINISTRATIVE FINDINGS
--------------------------------------------------
• [list up to 8 most actionable/time-sensitive items]

⚠️ CRITICAL ADMINISTRATIVE REMINDERS:
1. For "key_dates_deadlines": Extract EXACT date wording from document
   - Include phrases like "within 30 days", "by close of business", etc.
   - Do not interpret or calculate actual dates

2. For "action_items_requirements": Extract ONLY explicitly stated requirements
   - Include specific documentation, forms, evidence required
   - Do not include implied or typical requirements

3. For "legal_procedural_elements": Extract legal language VERBATIM
   - Include exact demands, offers, procedural requirements
   - Do not interpret legal implications

4. For "contact_follow_up": Extract contact information EXACTLY as written
   - Include phone numbers, emails, addresses without formatting
   - Preserve original spacing and punctuation

5. For "critical_administrative_findings": Include time-sensitive items only
   - Response deadlines
   - Legal deadlines
   - Critical compliance requirements
   - Urgent action items
   - If none explicit, derive from deadlines/actions (but prioritize explicit)

6. For "patient_details": Extract ONLY if explicitly stated (name, DOB, patient ID)
   - Do not infer patient information

7. For "report_author_signer": Extract signer/author name, title, and signature details verbatim from end of document or signature block
   - Look for phrases like "Signed by", "Prepared by", or signature lines
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context administrative document long summary generation...")
            
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
            
            logger.info(f"⚡ Direct administrative document long summary generation completed in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char administrative document")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct administrative document long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Administrative document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word administrative summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        Includes Author (report signer) but excludes patient details.
        """

        logger.info("🎯 Generating 30–60 word administrative structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an administrative and legal-document extraction specialist.

TASK:
Create a concise, accurate administrative summary using ONLY information explicitly present in the long summary.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Document Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Medication:[value] | MMI Status:[value] | Work Status:[value] | Restrictions:[value] | Action Items:[value] | Critical Finding:[value] | Follow-up:[value]

NEW KEY RULES (IMPORTANT):
- **ONLY include abnormalities or pathological findings. Skip normal findings entirely.**
- **If a value is not extracted, omit the ENTIRE key-value pair.**
- **Never output an empty key, an empty value, or placeholders.**
- **No duplicate pipes, no empty pipes (no '||').**

FORMAT & RULES:
- MUST be **30–60 words**.
- MUST be **ONE LINE**, pipe-delimited, no line breaks.
- First three fields (Document Title, Author, Date) appear without keys.
- All other fields use key-value format: Key:[value].
- DO NOT include patient details (name, DOB, ID).
- NEVER fabricate any information or infer abnormalities.

CONTENT PRIORITY (ONLY IF ABNORMAL AND PRESENT IN THE SUMMARY):
1. Document Title
2. Author
3. Date
4. Abnormal body parts or injury locations
5. Abnormal diagnoses
6. Medications (only if explicitly listed)
7. MMI status (only if explicitly stated)
8. Work status & restrictions (only if abnormal)
9. Action items (only if they indicate issues)
10. Critical findings
11. Follow-up requirements

ABSOLUTELY FORBIDDEN:
- Normal findings (ignore them entirely)
- assumptions, interpretations, invented medications, or inferred diagnoses
- placeholder text or "Not provided"
- narrative writing
- duplicate pipes or empty pipe fields (e.g., "||")
- any patient details (patient name, DOB, ID)

Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
""")


        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG ADMINISTRATIVE SUMMARY:

{long_summary}

Now produce a 30–60 word administrative structured summary following ALL rules.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary, "doc_type": doc_type})
            summary = response.content.strip()

            # Clean formatting - only whitespace, no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()

            # Word count validation
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Administrative summary out of bounds ({wc} words). Auto-correcting...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior output contained {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while keeping all facts accurate and key-value pipe-delimited format. DO NOT add fabricated details. Maintain format: [Document Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc. DO NOT include patient details."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])
                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after correction

            logger.info(f"✅ Administrative summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Administrative summary generation failed: {e}")
            return "Summary unavailable due to processing error."
    
    
    def _create_admin_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback administrative summary directly from long summary"""
        
        # Extract key administrative information using regex patterns
        patterns = {
            'sender': r'From:\s*([^\n]+)',
            'recipient': r'To:\s*([^\n]+)',
            'deadline': r'Response Deadline:\s*([^\n]+)',
            'subject': r'Subject:\s*([^\n]+)',
            'actions': r'Required Responses:(.*?)(?:\n\n|\n[A-Z]|$)',
            'contact': r'Contact Person:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} document")
        
        # Add sender/recipient context
        if 'sender' in extracted:
            parts.append(f"from {extracted['sender']}")
        
        if 'recipient' in extracted:
            parts.append(f"to {extracted['recipient']}")
        
        # Add subject
        if 'subject' in extracted:
            subject = extracted['subject'][:60] + "..." if len(extracted['subject']) > 60 else extracted['subject']
            parts.append(f"regarding {subject}")
        
        # Add deadline
        if 'deadline' in extracted:
            parts.append(f"Deadline: {extracted['deadline']}")
        
        # Add contact
        if 'contact' in extracted:
            parts.append(f"Contact: {extracted['contact']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with specified submission requirements", "following established procedures", "requiring timely response"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used administrative fallback summary: {len(summary.split())} words")
        return summary