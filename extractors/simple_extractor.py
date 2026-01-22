"""
Simplified Four-LLM Chain Extractor with Conditional Routing
Version: 4.5 - Structured Long Summary with Pydantic Validation
"""
import logging
import json
import re
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch

from utils.summary_helpers import ensure_date_and_author, clean_long_summary
from helpers.short_summary_generator import generate_structured_short_summary
from helpers.long_summary_generator import format_bullet_summary_to_json, format_long_summary_to_text
from models.long_summary_models import (
    UniversalLongSummary,
    DoctorInfo,
    format_universal_long_summary,
    create_fallback_long_summary
)


logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Smart four-LLM chain extractor with conditional routing:
    1. First LLM: Extract key findings and critical data
    2. Conditional Check: Determine if document has medical content
    3. Second LLM: Generate medical long summary OR Fourth LLM: Generate administrative summary
    4. Third LLM: Generate short summary from long summary
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.long_summary_parser = PydanticOutputParser(pydantic_object=UniversalLongSummary)
        logger.info("✅ SimpleExtractor v4.5 initialized with Pydantic Long Summary Validation")
    
    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Extract document data with FULL CONTEXT using raw text.
        
        Args:
            text: Complete document text (layout-preserved)
            raw_text: Summarized original context from Document AI
            doc_type: Document type
            fallback_date: Fallback date if not found
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING SIMPLE EXTRACTION (FULL CONTEXT + RAW TEXT)")
        logger.info("=" * 80)
        
        # Check document size
        text_length = len(raw_text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        try:
            # STEP 1: Format the summarizer output (raw_text) into structured long summary
            # This is a FORMATTING task only - no new content generation
            formatted_json = format_bullet_summary_to_json(
                bullet_summary=raw_text,
                llm=self.llm,
                document_type=doc_type
            )
            long_summary = format_long_summary_to_text(formatted_json)
            
            # STEP 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
            long_summary = clean_long_summary(long_summary)
            
            # STEP 2: Generate structured short summary from raw_text (primary context)
            short_summary = self._generate_short_summary_from_long_summary(raw_text, doc_type)
            
            logger.info("=" * 80)
            logger.info("✅ SIMPLE EXTRACTION COMPLETE (2 LLM CALLS ONLY)")
            logger.info("=" * 80)
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "content_type": "universal"
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _generate_long_summary_direct(self, text: str, raw_text: str, doc_type: str, fallback_date: str) -> str:
        """
        Generate long summary with PRIORITIZED context hierarchy:
        1. PRIMARY SOURCE: raw_text (accurate Document AI summarized context)
        2. SUPPLEMENTARY: text (full OCR extraction for missing details only)
        
        This ensures accurate context preservation while capturing all necessary details.
        Uses Pydantic validation for consistent, structured output without hallucination.
        """
        logger.info("🔍 Processing document with DUAL-CONTEXT approach + Pydantic validation...")
        
        # Get format instructions from Pydantic parser
        format_instructions = self.long_summary_parser.get_format_instructions()
        
        # Build system prompt with CLEAR PRIORITY INSTRUCTIONS
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a universal medical and administrative document summarization expert analyzing a COMPLETE document.

🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
**YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
- NEVER generate, infer, assume, or fabricate ANY information
- If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
- An incomplete summary is 100x better than a fabricated one
- Every single piece of information in your output MUST be traceable to the source text

🎯 CRITICAL CONTEXT HIERARCHY:

You are provided with TWO versions of the document:

1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
   - This is the MOST ACCURATE, context-aware summary generated by Google's Document AI foundation model
   - It has been intelligently processed to preserve CRITICAL CONTEXT
   - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
   - This contains the CORRECT interpretations, accurate findings, and proper context
   - **ALWAYS PRIORITIZE information from this source**

2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
   - This is the complete OCR text extraction (may have formatting noise, OCR artifacts)
   - Use ONLY to fill in SPECIFIC DETAILS that may be missing from the accurate context
   - Examples of acceptable supplementary use:
       * Exact medication dosages if not in primary source
       * Specific claim numbers or identifiers
       * Additional doctor names mentioned
       * Precise dates or measurements
   - **DO NOT let this override the context from the primary source**

⚠️ STRICT ANTI-HALLUCINATION RULES:

1. **ZERO FABRICATION TOLERANCE**:
   - If a field (e.g., DOB, Claim Number, Diagnosis) is NOT in either source → LEAVE IT BLANK or OMIT
   - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
   - NEVER fill in "standard" or "typical" values - only actual extracted values

2. **CONTEXT PRIORITY ENFORCEMENT**:
   - When both sources provide information about the SAME finding:
     ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
     ❌ NEVER override with potentially inaccurate full text version

3. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** - Empty if not mentioned
4. **NO ASSUMPTIONS** - Do not infer or add typical values
5. **ADAPT STRUCTURE** - Use medical sections for clinical content, administrative for non-clinical
6. **EMPTY FIELDS BETTER THAN GUESSES** - Omit sections if no data
7. **SIGNATURE EXTRACTION**: Scan both sources for signatures. Identify authors who signed PHYSICALLY or ELECTRONICALLY. If no signature found, OMIT - do not guess.
8. **CLAIM NUMBER EXTRACTION**: Scan both sources for claim number. Extract exact value if present. If not found, OMIT - do not fabricate.

🔍 SPECIAL INSTRUCTIONS FOR PATIENT DETAILS VALIDATION:

**CRITICAL - PATIENT DETAILS CROSS-VALIDATION**:
If the raw_text contains a "--- PATIENT DETAILS ---" section:
1. **FIRST**: Extract the patient details from that section (Patient Name, DOB, DOI, Claim Number)
2. **SECOND**: Cross-validate each detail against the FULL TEXT EXTRACTION (text parameter) as sometimes the full text is not properly formatted, so the fields and values are not aligned properly, but the full text must have the correct details, and if we are getting the pateint details from the patient details section, we need to make sure they are accurate by cross-checking with the full text extraction
3. **VALIDATION RULES**:
   ✅ If the detail MATCHES what's in the full text extraction → USE IT (it's accurate)
   ✅ If the detail is CLOSE but has minor formatting differences → USE the formatted version from patient details section
   ❌ If the detail CONTRADICTS the full text extraction → IGNORE the patient details section value and extract directly from full text
   ❌ If the detail is MISSING or shows "N/A" → Extract directly from full text extraction
4. **FINAL CHECK**: Ensure all patient details (Name, DOB, DOI, Claim Number) are accurate and consistent with the document content

**Example Validation Process**:
- Patient Details section shows: "Patient Name: John Smith"
- Full text contains: "Patient: John Smith" → ✅ VALID - Use "John Smith"
- Patient Details section shows: "DOB: N/A"
- Full text contains: "Date of Birth: 05/15/1975" → ❌ INVALID - Use "05/15/1975" from full text
- Patient Details section shows: "Claim Number: 12345-ABC"
- Full text contains: "Claim #: 12345-ABC" → ✅ VALID - Use "12345-ABC"

UNIVERSAL EXTRACTION FOCUS:

For MEDICAL DOCUMENTS:
I. PATIENT & CLINICAL CONTEXT
II. CRITICAL FINDINGS & DIAGNOSES
III. LAB/IMAGING RESULTS
IV. TREATMENT & OBSERVATIONS
V. STATUS & RECOMMENDATIONS

For ADMINISTRATIVE DOCUMENTS:
I. DOCUMENT OVERVIEW
II. KEY PARTIES & INFORMATION
III. ACTION ITEMS & DEADLINES
IV. CONTACT & FOLLOW-UP

Now analyze this COMPLETE document using the DUAL-CONTEXT approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY:
""")

        # User prompt with DUAL-CONTEXT input
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PRIMARY SOURCE - ACCURATE CONTEXT (USE THIS FIRST):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{primary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 SUPPLEMENTARY SOURCE - FULL TEXT (USE ONLY FOR MISSING DETAILS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{supplementary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ REPORT DATE INSTRUCTION:
- Extract the ACTUAL report/examination/document date from the sources above
- DO NOT use current/today's date - only use dates explicitly mentioned in the document
- IMPORTANT: US date format is MM/DD/YYYY. Example: 11/25/2025 means November 25, 2025 (NOT day 11 of month 25)
- If no date found, use "00/00/0000" as placeholder

**INSTRUCTIONS**: Generate a comprehensive structured summary following the DUAL-CONTEXT hierarchy. Prioritize PRIMARY SOURCE for clinical/administrative context. Use SUPPLEMENTARY SOURCE only for specific details (dosages, claim numbers, exact dates, additional names) if not found in PRIMARY SOURCE.

Generate the long summary in this EXACT STRUCTURED FORMAT (adapt to medical or administrative content):

For MEDICAL CONTENT:
📋 MEDICAL DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: [extracted]
Report Date: [extract ACTUAL date from document; if not found use "00/00/0000"]
Claim Number: [extracted if present; otherwise omit]
Patient Name: [extracted]
Provider: [extracted]
                                                         
👤 PATIENT & CLINICAL INFORMATION
--------------------------------------------------
Name: [extracted]
DOB: [extracted]
Chief Complaint: [extracted]
Clinical History: [extracted]
                                                               
All Doctors Involved:
• [list all extracted doctors with names and titles]
━━━ ALL DOCTORS EXTRACTION ━━━
- Extract ALL physician/doctor names mentioned ANYWHERE in both sources.
- Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
- Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
- Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
- Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
- If no doctors found, leave list empty [].

━━━ CLAIM NUMBER EXTRACTION PATTERNS ━━━
CRITICAL: Scan BOTH SOURCES (header, footer, cc: lines, letterhead) for claim numbers.

Common claim number patterns (case-insensitive):
- if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
- "[Claim #XXXXXXXXX]" or "[Claim #XXXXX-XXX]"
- "Claim Number: XXXXXXXXX" or "Claim #: XXXXXXXXX"
- "Claim: XXXXXXXXX" or "Claim #XXXXXXXXX"
- "WC Claim: XXXXXXXXX" or "Workers Comp Claim: XXXXXXXXX"
- "Policy/Claim: XXXXXXXXX"
- In "cc:" lines: "Broadspire [Claim #XXXXXXXXX]"
- In subject lines or reference fields: "Claim #XXXXXXX"
                                                               
🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 5 critical items from PRIMARY SOURCE]

🏥 DIAGNOSES & ASSESSMENTS
--------------------------------------------------
Primary Diagnosis: [from PRIMARY SOURCE]
Secondary Diagnoses:
• [list up to 3 from PRIMARY SOURCE]
Lab Results:
• [list up to 5 key results from PRIMARY SOURCE]
• [list key results with values/ranges]
Imaging Findings:
• [list key observations]

💊 TREATMENT & OBSERVATIONS
--------------------------------------------------
Current Medications:
• [list with doses if stated]
Clinical Observations:
• [list vital signs, exam findings]
Procedures/Treatments:
• [list recent or ongoing]

💼 STATUS & RECOMMENDATIONS
--------------------------------------------------
Work Status: [extracted]
MMI: [extracted]
Recommendations:
• [list up to 5 next steps]

For ADMINISTRATIVE CONTENT:
📋 ADMINISTRATIVE DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Document Date: [extract ACTUAL date from document; if not found use "00/00/0000"]
Claim Number: [extracted if present; otherwise omit]
Purpose: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]

👥 KEY PARTIES
--------------------------------------------------
Patient: [extracted]
Provider: [extracted]
Referring Party: [extracted]

📄 KEY INFORMATION
--------------------------------------------------
Important Dates: [extracted]
Reference Numbers: [extracted]
Administrative Details: [extracted]

✅ ACTION ITEMS
--------------------------------------------------
Required Actions:
• [list up to 5]
Deadlines: [extracted]

📞 CONTACT & FOLLOW-UP
--------------------------------------------------
Contact Information: [extracted]
Next Steps: [extracted]

⚠️ MANDATORY EXTRACTION RULES (donot include in output, for LLM use only):
1. Adapt structure to content: Medical if clinical data present, Administrative if not
2. Extract ONLY explicit information - omit sections with no data
3. Use exact wording for medical terms, dates, names, signatures, and claim numbers
4. Bullet points for lists, clear headings
5. No assumptions or additions
6. For signatures: Look for end-of-document sign blocks, /s/ notations, scanned signatures, or explicit "signed by" statements. Distinguish physical (e.g., "Handwritten by Dr. X") vs. electronic (e.g., "Electronically signed by Dr. Y").
7. For claim number: Search for patterns like "Claim #", "Claim Number", "WC Claim", etc., and extract the alphanumeric value exactly.

🚨 FINAL VERIFICATION (CRITICAL):
Before outputting, verify EVERY piece of information:
- Can I point to the exact text in PRIMARY or SUPPLEMENTARY source? → YES = Include | NO = OMIT
- Am I assuming or inferring this? → If YES = REMOVE IT
- Is this a "typical" or "standard" value I'm adding? → If YES = REMOVE IT
- Did I fabricate any dates, names, numbers, or findings? → If YES = REMOVE THEM
**Output ONLY what is explicitly stated in the sources. Leave fields blank rather than guess.**

📋 OUTPUT FORMAT INSTRUCTIONS:
You MUST output your response as a valid JSON object following this exact schema:
{format_instructions}

⚠️ IMPORTANT JSON RULES:
- Use empty string "" for text fields with no data (NOT null, NOT "N/A", NOT "unknown")
- Use empty array [] for list fields with no data
- Use null ONLY for optional fields like claim_number when not present
- For content_type: use "medical" if clinical data present, "administrative" if not
- For all_doctors_involved: each doctor must have "name" (required), "title" (optional), "role" (optional)
- For signature_type: use "physical" or "electronic" or null if not found
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for structured long summary generation with Pydantic validation...")
            
            # Single LLM call with DUAL-CONTEXT (primary + supplementary sources)
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "primary_source": raw_text,  # PRIMARY: Document AI summarizer output
                "supplementary_source": text,  # SUPPLEMENTARY: Full OCR text
                "doc_type": doc_type,
                "fallback_date": fallback_date,
                "format_instructions": format_instructions
            })
            
            raw_response = result.content.strip()
            logger.info(f"📝 Raw LLM response length: {len(raw_response)} chars")
            
            # Parse and validate with Pydantic
            try:
                parsed_summary = self.long_summary_parser.parse(raw_response)
                logger.info(f"✅ Pydantic validation successful - content_type: {parsed_summary.content_type}")
                
                # Format the validated Pydantic model back to text format
                long_summary = format_universal_long_summary(parsed_summary)
                
                logger.info(f"⚡ Structured long summary generation completed with Pydantic validation")
                logger.info(f"✅ Generated long summary using:")
                logger.info(f"   - PRIMARY SOURCE: {len(raw_text):,} chars")
                logger.info(f"   - SUPPLEMENTARY SOURCE: {len(text):,} chars")
                
                return long_summary
                
            except Exception as parse_error:
                logger.warning(f"⚠️ Pydantic parsing failed: {parse_error}")
                logger.warning("⚠️ Falling back to raw LLM output (may contain inconsistencies)")
                
                # Try to extract JSON from the response and create a fallback
                try:
                    # Attempt to find JSON in the response
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        json_str = json_match.group()
                        parsed_dict = json.loads(json_str)
                        # Create Pydantic model with defaults for missing fields
                        parsed_summary = UniversalLongSummary(**parsed_dict)
                        long_summary = format_universal_long_summary(parsed_summary)
                        logger.info("✅ Successfully recovered with JSON extraction fallback")
                        return long_summary
                except Exception as json_error:
                    logger.warning(f"⚠️ JSON extraction fallback failed: {json_error}")
                
                # Final fallback: use raw response if it looks like formatted text
                if "📋" in raw_response or "DOCUMENT OVERVIEW" in raw_response:
                    logger.info("✅ Using raw formatted text response")
                    return raw_response
                
                # Create minimal fallback summary
                fallback = create_fallback_long_summary(doc_type, fallback_date)
                return format_universal_long_summary(fallback)
            
        except Exception as e:
            logger.error(f"❌ Direct universal long summary generation failed: {e}", exc_info=True)
            
            # Fallback: Generate a minimal summary using Pydantic model
            fallback = create_fallback_long_summary(doc_type, fallback_date)
            return format_universal_long_summary(fallback)

    def _generate_short_summary_from_long_summary(self, raw_text: str, doc_type: str) -> dict:
        """
        Generate a structured, UI-ready summary from raw_text (Document AI summarizer output).
        Delegates to the reusable helper function.
        
        Args:
            raw_text: The Document AI summarizer output (primary context)
            doc_type: Document type
            
        Returns:
            dict: Structured summary with header, findings, recommendations, status
        """
        return generate_structured_short_summary(self.llm, raw_text, doc_type)

    def _remove_empty_segments(self, text: str) -> str:
        """
        Remove any segments that contain empty values, placeholders, or unwanted markers.
        """
        # Split by pipe
        segments = text.split('|')
        cleaned_segments = []
        
        # Patterns to detect empty/placeholder values
        empty_patterns = [
            '[empty]', '[unknown]', '[not provided]', '[n/a]', '[na]',
            'unknown', 'not specified', 'not provided', 'not available',
            'not mentioned', 'not stated', 'none', 'n/a', 'na'
        ]
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Check if segment contains a colon (Key: Value format)
            if ':' in segment:
                key, value = segment.split(':', 1)
                value = value.strip()
                
                # Skip if value is empty or matches empty patterns
                if not value:
                    continue
                
                # Check against empty patterns (case-insensitive)
                value_lower = value.lower()
                if any(pattern in value_lower for pattern in empty_patterns):
                    continue
                
                # Keep this segment
                cleaned_segments.append(segment)
            else:
                # If no colon, keep as-is (shouldn't happen but safety)
                cleaned_segments.append(segment)
        
        return ' | '.join(cleaned_segments)
    
    def _clean_empty_fields(self, data: Dict, fallback_date: str) -> Dict:
        """Remove all empty fields and ensure clean structure."""
        cleaned = {}
        
        # Only include patient_details if it has any non-empty values
        if "patient_details" in data:
            patient_clean = {}
            for key, value in data["patient_details"].items():
                if value and str(value).strip():
                    patient_clean[key] = value
            if patient_clean:
                cleaned["patient_details"] = patient_clean
        
        # Only include critical_findings if non-empty
        critical_findings = data.get("critical_findings", [])
        if critical_findings and any(finding and str(finding).strip() for finding in critical_findings):
            cleaned["critical_findings"] = [f for f in critical_findings if f and str(f).strip()]
        
        # Only include diagnoses if non-empty
        diagnoses = data.get("diagnoses", [])
        if diagnoses and any(dx and str(dx).strip() for dx in diagnoses):
            cleaned["diagnoses"] = [dx for dx in diagnoses if dx and str(dx).strip()]
        
        # Only include lab_results if non-empty
        lab_results = data.get("lab_results", [])
        if lab_results and any(lab and str(lab).strip() for lab in lab_results):
            cleaned["lab_results"] = [lab for lab in lab_results if lab and str(lab).strip()]
        
        # Only include imaging_findings if non-empty
        imaging_findings = data.get("imaging_findings", [])
        if imaging_findings and any(img and str(img).strip() for img in imaging_findings):
            cleaned["imaging_findings"] = [img for img in imaging_findings if img and str(img).strip()]
        
        # Only include clinical_observations if non-empty
        clinical_obs = data.get("clinical_observations", [])
        if clinical_obs and any(obs and str(obs).strip() for obs in clinical_obs):
            cleaned["clinical_observations"] = [obs for obs in clinical_obs if obs and str(obs).strip()]
        
        # Only include current_treatment if it has any non-empty values
        if "current_treatment" in data:
            treatment_clean = {}
            for key, items in data["current_treatment"].items():
                if isinstance(items, list):
                    if items and any(item and str(item).strip() for item in items):
                        treatment_clean[key] = [item for item in items if item and str(item).strip()]
                elif items and str(items).strip():
                    treatment_clean[key] = items
            if treatment_clean:
                cleaned["current_treatment"] = treatment_clean
        
        # Only include work_status if it has any non-empty values
        if "work_status" in data:
            work_clean = {}
            for key, value in data["work_status"].items():
                if value and str(value).strip():
                    work_clean[key] = value
            if work_clean:
                cleaned["work_status"] = work_clean
        
        # Always include important_dates with at least report_date
        cleaned["important_dates"] = {
            "report_date": data.get("important_dates", {}).get("report_date") or fallback_date
        }
        # Add other dates only if they exist
        for key in ["examination_date", "follow_up_dates", "service_date", "collection_date"]:
            value = data.get("important_dates", {}).get(key)
            if value and str(value).strip():
                cleaned["important_dates"][key] = value
        
        # Only include document_context if it has any non-empty values
        if "document_context" in data:
            context_clean = {}
            for key, value in data["document_context"].items():
                if value and str(value).strip():
                    context_clean[key] = value
            if context_clean:
                cleaned["document_context"] = context_clean
        
        return cleaned
    
    def _clean_summary_text(self, text: str) -> str:
        """Remove any unwanted phrases from summary text."""
        unwanted_phrases = [
            "unknown", "not specified", "not provided", "none", 
            "no information", "missing", "unavailable", "unspecified",
            "not mentioned", "not available", "not stated"
        ]
        
        cleaned = text
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Clean up extra spaces but preserve formatting
        lines = cleaned.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = ' '.join(line.split())
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_short_summary(self, text: str) -> str:
        """Clean short summary text."""
        # Remove unwanted phrases
        text = self._clean_summary_text(text)
        
        # Clean formatting
        text = ' '.join(text.split())
        text = text.strip('"').strip("'")
        
        return text 
    def _create_basic_medical_summary(self, extracted_data: Dict, doc_type: str) -> str:
        """Create a basic medical summary from extracted data when LLM fails."""
        dates = extracted_data.get("important_dates", {})
        report_date = dates.get("report_date", "")
        
        sections = []
        
        # Header
        header = f"MEDICAL SUMMARY - {doc_type.upper()}"
        if report_date:
            header += f"\nReport Date: {report_date}"
        sections.append(header)
        
        # Patient info
        patient = extracted_data.get("patient_details", {})
        if patient:
            patient_info = ["PATIENT INFORMATION"]
            if patient.get("name"):
                patient_info.append(f"• Name: {patient['name']}")
            if patient.get("dob"):
                patient_info.append(f"• DOB: {patient['dob']}")
            if patient.get("mrn"):
                patient_info.append(f"• MRN: {patient['mrn']}")
            if len(patient_info) > 1:
                sections.append("\n".join(patient_info))
        
        # Lab results
        lab_results = extracted_data.get("lab_results", [])
        if lab_results:
            lab_section = ["LABORATORY RESULTS"]
            for result in lab_results[:10]:
                lab_section.append(f"• {result}")
            sections.append("\n".join(lab_section))
        
        # Imaging findings
        imaging = extracted_data.get("imaging_findings", [])
        if imaging:
            img_section = ["IMAGING FINDINGS"]
            for finding in imaging[:10]:
                img_section.append(f"• {finding}")
            sections.append("\n".join(img_section))
        
        # Critical findings
        critical = extracted_data.get("critical_findings", [])
        if critical:
            crit_section = ["CRITICAL FINDINGS"]
            for finding in critical[:5]:
                crit_section.append(f"• {finding}")
            sections.append("\n".join(crit_section))
        
        if len(sections) > 1:
            return "\n\n".join(sections)
        
        return f"{header}\n\nBasic medical information extracted. Review original document for details."
    
    def _create_basic_administrative_summary(self, extracted_data: Dict, doc_type: str) -> str:
        """Create a basic administrative summary when LLM fails."""
        dates = extracted_data.get("important_dates", {})
        report_date = dates.get("report_date", "")
        
        header = f"ADMINISTRATIVE SUMMARY - {doc_type.upper()}"
        if report_date:
            header += f"\nDocument Date: {report_date}"
        
        return f"{header}\n\nAdministrative document processed. No clinical content identified."
    
    def _clean_pipes_from_summary(self, short_summary: str) -> str:
        """
        Clean empty pipes from short summary to avoid consecutive pipes or trailing pipes.
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
    
    def _create_fallback_extraction(self, fallback_date: str, doc_type: str) -> Dict:
        """Create fallback extraction data with clean empty fields."""
        return {
            "important_dates": {
                "report_date": fallback_date
            }
        }
    
    def _create_error_response(self, doc_type: str, error_msg: str, fallback_date: str) -> Dict:
        """Create error response that still provides a summary."""
        return {
            "long_summary": f"DOCUMENT SUMMARY - {doc_type.upper()}\n\nDocument processed. Basic information extracted.",
            "short_summary": f"Report Title: {doc_type} | Date: {fallback_date}",
            "content_type": "unknown"
        }