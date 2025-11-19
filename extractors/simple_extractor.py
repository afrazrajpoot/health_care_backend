"""
Simplified Four-LLM Chain Extractor with Conditional Routing
Version: 4.0 - Smart Document Routing
"""

import logging
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch

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
        logger.info("✅ SimpleExtractor v4.0 initialized with Four-LLM Conditional Chain")
    
    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Four-step LLM chain extraction with conditional routing.
        """
        logger.info(f"🚀 Four-LLM Conditional Extraction: {doc_type}")
        
        try:
            # STEP 1: First LLM - Extract key data
            extracted_data = self._extract_key_data(text, doc_type, fallback_date)
            
            # STEP 2: Conditional routing - Check if document has medical content
            has_medical_content = self._check_medical_content(extracted_data, text)
            
            # STEP 3: Generate appropriate long summary based on content type
            if has_medical_content:
                logger.info("🩺 Document contains medical content - generating medical summary")
                long_summary = self._generate_medical_long_summary(extracted_data, doc_type, text)
            else:
                logger.info("📄 Document is administrative - generating administrative summary")
                long_summary = self._generate_administrative_summary(text, doc_type, extracted_data)
            
            # STEP 4: Generate short summary
            short_summary = self._generate_short_summary(long_summary, doc_type)
            
            logger.info(f"✅ Conditional chain extraction completed")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "extracted_data": extracted_data,
                "content_type": "medical" if has_medical_content else "administrative"
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _check_medical_content(self, extracted_data: Dict, text: str) -> bool:
        """
        Fourth LLM: Determine if document contains medical content.
        """
        logger.info("🔍 Fourth LLM - Checking medical content...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a document classification expert. Determine if the document contains meaningful medical/clinical content.

MEDICAL CONTENT INDICATORS:
- Diagnoses, symptoms, or medical conditions
- Medications, treatments, or procedures
- Physical exam findings or test results
- Patient clinical status or progress
- Medical recommendations or plans

ADMINISTRATIVE CONTENT INDICATORS:
- Forms, cover letters, or cover sheets
- Appointment scheduling or billing
- Document processing notifications
- Request for information or records
- Minimal clinical data (just names/dates)

Return ONLY "true" if medical content exists, "false" if only administrative content.
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXTRACTED DATA:
{extracted_data}

DOCUMENT TEXT PREVIEW (first 500 chars):
{text_preview}

Does this document contain meaningful medical/clinical content?
Answer with ONLY "true" or "false":
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "extracted_data": str(extracted_data),
                "text_preview": text[:500]
            })
            
            result = response.content.strip().lower()
            has_medical = result == "true"
            
            logger.info(f"✅ Medical content check: {has_medical}")
            return has_medical
            
        except Exception as e:
            logger.error(f"❌ Medical content check failed: {e}")
            # Fallback: Check if we have any medical data in extraction
            return self._fallback_medical_check(extracted_data)
    
    def _fallback_medical_check(self, extracted_data: Dict) -> bool:
        """Fallback method to check for medical content."""
        # Check if we have any medical data
        medical_indicators = [
            extracted_data.get("diagnoses"),
            extracted_data.get("critical_findings"),
            extracted_data.get("current_treatment", {}).get("medications"),
            extracted_data.get("current_treatment", {}).get("procedures"),
        ]
        
        has_medical = any(indicator for indicator in medical_indicators if indicator)
        logger.info(f"🔄 Fallback medical check: {has_medical}")
        return has_medical
    
    def _extract_key_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        First LLM: Extract key findings, critical data, and patient details.
        """
        logger.info("🔍 First LLM - Extracting key data...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical document analysis expert. Extract ALL available information from the document.

EXTRACT THESE KEY CATEGORIES:

1. PATIENT DETAILS:
   - Name, DOB, Age, Gender
   - Chief complaint
   - Injury date and mechanism

2. CRITICAL FINDINGS (URGENT):
   - Life-threatening conditions
   - Surgical emergencies  
   - Medication alerts
   - Abnormal vital signs
   - Neurological deficits

3. KEY DIAGNOSES:
   - Primary diagnosis with ICD-10 if available
   - Secondary diagnoses
   - Affected body parts

4. CURRENT TREATMENT:
   - Current medications (name, dose, frequency)
   - Recent procedures
   - Ongoing therapies

5. WORK STATUS:
   - Current work capacity
   - Restrictions and limitations
   - MMI status if mentioned

6. IMPORTANT DATES:
   - Report date
   - Examination date
   - Follow-up dates

7. DOCUMENT CONTEXT:
   - Document purpose
   - Any administrative information
   - Next steps or deadlines

CRITICAL RULES:
- Extract ANY information available, even if minimal
- If no data found for a field, DO NOT include it
- Empty arrays for no items, empty strings for no text
- Include administrative details if medical data is sparse

Return JSON format - ONLY include fields with actual data:
{{{{
  "patient_details": {{}},
  "critical_findings": [],
  "diagnoses": [],
  "current_treatment": {{}},
  "work_status": {{}},
  "important_dates": {{}},
  "document_context": {{}}
}}}}
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

DOCUMENT TEXT:
{text}

Extract ALL available information. Include administrative details if medical data is limited:
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm | self.parser
            extracted = chain.invoke({
                "text": text[:20000],
                "doc_type": doc_type
            })
            
            # Clean empty fields
            extracted = self._clean_empty_fields(extracted, fallback_date)
            
            logger.info("✅ Key data extraction complete")
            return extracted
            
        except Exception as e:
            logger.error(f"❌ Key data extraction failed: {e}")
            return self._create_fallback_extraction(fallback_date, doc_type)
    
    def _generate_medical_long_summary(self, extracted_data: Dict, doc_type: str, original_text: str = "") -> str:
        """
        Second LLM: Generate comprehensive medical long summary from extracted data.
        """
        logger.info("📝 Second LLM - Generating medical long summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical summarization expert. Create a professionally formatted medical summary.

CRITICAL RULES:
- Use proper medical formatting with clear sections and bullet points
- ONLY include information that is explicitly available
- OMIT entire sections if no data exists
- NEVER mention missing information
- Use professional medical terminology
- Structure the summary for quick physician review

PROFESSIONAL FORMATTING GUIDE:

Start with a clear header:
MEDICAL SUMMARY - [Document Type]
[Report Date]

Then include ONLY these sections if data exists:

1. PATIENT INFORMATION
   • Name: [Name]
   • DOB: [Date of Birth] 
   • Chief Complaint: [Complaint]

2. DIAGNOSES
   • Primary: [Diagnosis] ([ICD-10 if available])
   • Secondary: [Diagnosis]

3. PROCEDURES & TREATMENT
   • Procedure: [Procedure Name] ([Date])
   • Surgeon: [Surgeon Name]
   • Findings: [Key surgical findings]
   • Medications: [List with doses]

4. CURRENT STATUS
   • Work Status: [Status]
   • Restrictions: [Specific restrictions]
   • MMI Status: [If applicable]

5. FOLLOW-UP PLAN
   • Next Appointment: [Date with provider]
   • Ongoing Care: [Therapies, treatments]
   • Deadlines: [Important dates]

Use bullet points (•) for lists and clean section headers. Maintain professional medical tone.
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

EXTRACTED MEDICAL INFORMATION:
{extracted_data}

Create a professionally formatted medical summary:
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "extracted_data": str(extracted_data),
                "doc_type": doc_type
            })
            
            long_summary = response.content.strip()
            long_summary = self._clean_summary_text(long_summary)
            
            logger.info(f"✅ Medical long summary generated ({len(long_summary)} chars)")
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Medical long summary generation failed: {e}")
            return self._create_basic_medical_summary(extracted_data, doc_type)
    
    def _generate_administrative_summary(self, text: str, doc_type: str, extracted_data: Dict) -> str:
        """
        Fourth LLM: Generate administrative summary for non-medical documents.
        """
        logger.info("📋 Fourth LLM - Generating administrative summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an administrative document summarization expert. Create a clear, concise summary of administrative documents.

DOCUMENT TYPES:
- Cover letters, forms, or cover sheets
- Appointment notifications
- Billing or insurance documents
- Record requests
- Processing notifications

SUMMARY STRUCTURE:

ADMINISTRATIVE SUMMARY - [Document Type]
[Document Date]

KEY INFORMATION:
• Document Purpose: [Why this document was created]
• Key Parties: [Names mentioned - patients, providers, organizations]
• Important Dates: [Deadlines, appointment dates, due dates]
• Action Required: [What needs to be done next]
• Contact Information: [Who to contact if mentioned]

RULES:
- Focus on practical, actionable information
- Extract names, dates, and deadlines
- Highlight next steps or required actions
- Keep it concise and professional
- Omit sections if no information available
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

DOCUMENT TEXT (first 2000 characters):
{text}

EXTRACTED DATA CONTEXT:
{extracted_data}

Create a clear administrative summary:
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "text": text[:2000],
                "doc_type": doc_type,
                "extracted_data": str(extracted_data)
            })
            
            admin_summary = response.content.strip()
            admin_summary = self._clean_summary_text(admin_summary)
            
            logger.info(f"✅ Administrative summary generated ({len(admin_summary)} chars)")
            return admin_summary
            
        except Exception as e:
            logger.error(f"❌ Administrative summary generation failed: {e}")
            return self._create_basic_administrative_summary(extracted_data, doc_type)
    
    def _generate_short_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Third LLM: Generate concise short summary from long summary.
        """
        logger.info("🎯 Third LLM - Generating short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited summaries for documents.

 STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:
    [Report Title] | [Author/Physician or The person who signed the report] | [Date] | [Body parts] | [Diagnosis] | [Medication] | [MMI Status] | [Key Action Items] | [Work Status] | [Recommendation] | [Critical Finding] | Urgent Next Steps

    3. DO NOT fabricate or infer missing data — simply SKIP fields that do not exist.
    4. Use ONLY information explicitly found in the long summary.
    5. Output must be a SINGLE LINE (no line breaks).
    6. Content priority:
    - report title
    - author name
    - date
    - affected body parts
    - primary diagnosis
    - medications (if present)
    - MMI status (if present)
    - work status (if present)
    - key recommendation(s) (if present)
    - one critical finding (if present)
    - urgent next steps (if present)
    - follow-up plan (if present)

    7. ABSOLUTE NO:
    - assumptions
    - clinical interpretation
    - invented medications
    - invented dates
    - narrative sentences

    8. If a field is missing, SKIP IT—do NOT write "None" or "Not provided" and simply leave the field empty also donot use | for this field as if 2 fileds are missing then it shows ||

    Your final output must be 30–60 words and MUST follow the exact pipe-delimited format above.
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG SUMMARY:
{long_summary}

Create a clean pipe-delimited short summary with ONLY available information:
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary[:3000]})
            short_summary = response.content.strip()
            
            short_summary = self._clean_short_summary(short_summary)
            word_count = len(short_summary.split())
            logger.info(f"✅ Short summary: {word_count} words")
            
            # Fallback if too long
            if word_count > 80:
                words = short_summary.split()
                short_summary = ' '.join(words[:60]) + "..."
            
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return f"{doc_type} | | |"
    
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
        
        # Only include current_treatment if it has any non-empty values
        if "current_treatment" in data:
            treatment_clean = {}
            for key, items in data["current_treatment"].items():
                if items and any(item and str(item).strip() for item in items):
                    treatment_clean[key] = [item for item in items if item and str(item).strip()]
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
        for key in ["examination_date", "follow_up_dates"]:
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
        
        # Ensure proper pipe format
        parts = text.split('|')
        if len(parts) == 4:  # All sections present
            cleaned_parts = [part.strip() for part in parts]
            return ' | '.join(cleaned_parts)
        
        return text
    
    def _create_basic_medical_summary(self, extracted_data: Dict, doc_type: str) -> str:
        """Create a basic medical summary from extracted data when LLM fails."""
        dates = extracted_data.get("important_dates", {})
        report_date = dates.get("report_date", "")
        
        header = f"MEDICAL SUMMARY - {doc_type.upper()}"
        if report_date:
            header += f"\nReport Date: {report_date}"
        
        return f"{header}\n\nBasic medical information extracted. Review original document for details."
    
    def _create_basic_administrative_summary(self, extracted_data: Dict, doc_type: str) -> str:
        """Create a basic administrative summary when LLM fails."""
        dates = extracted_data.get("important_dates", {})
        report_date = dates.get("report_date", "")
        
        header = f"ADMINISTRATIVE SUMMARY - {doc_type.upper()}"
        if report_date:
            header += f"\nDocument Date: {report_date}"
        
        return f"{header}\n\nAdministrative document processed. No clinical content identified."
    
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
            "short_summary": f"{doc_type} | | |",
            "extracted_data": self._create_fallback_extraction(fallback_date, doc_type),
            "content_type": "unknown"
        }