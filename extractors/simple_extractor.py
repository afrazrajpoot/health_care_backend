"""
Simplified Four-LLM Chain Extractor with Conditional Routing
Version: 4.1 - Enhanced Universal Document Handling
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
        logger.info("✅ SimpleExtractor v4.1 initialized with Enhanced Universal Document Handling")
    
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
- Laboratory results, imaging findings
- Patient clinical status or progress
- Medical recommendations or plans
- Vital signs, biometric data
- Clinical observations or assessments

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
            extracted_data.get("lab_results"),
            extracted_data.get("imaging_findings"),
            extracted_data.get("clinical_observations")
        ]
        
        has_medical = any(indicator for indicator in medical_indicators if indicator)
        logger.info(f"📄 Fallback medical check: {has_medical}")
        return has_medical
    
    def _extract_key_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        First LLM: Extract key findings, critical data, and patient details.
        Enhanced to handle ANY document type including lab reports, imaging, etc.
        """
        logger.info("🔍 First LLM - Extracting key data...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical document analysis expert. Extract ALL available information from ANY type of medical document.

EXTRACT THESE KEY CATEGORIES (include ALL that are present):

1. PATIENT DETAILS:
   - Name, DOB, Age, Gender
   - MRN, Account Number
   - Chief complaint
   - Injury date and mechanism

2. CRITICAL FINDINGS (URGENT):
   - Life-threatening conditions
   - Critical lab values (HIGH/LOW alerts)
   - Abnormal imaging findings
   - Surgical emergencies  
   - Medication alerts
   - Abnormal vital signs
   - Neurological deficits

3. KEY DIAGNOSES:
   - Primary diagnosis with ICD-10 if available
   - Secondary diagnoses
   - Affected body parts
   - Clinical impressions

4. LAB RESULTS (if lab report):
   - Test names with results and reference ranges
   - Abnormal values (HIGH/LOW flags)
   - Critical values
   - Collection date/time
   - Ordering physician

5. IMAGING FINDINGS (if imaging report):
   - Study type (X-ray, MRI, CT, Ultrasound, etc.)
   - Body part examined
   - Key findings
   - Impressions/conclusions
   - Comparison to prior studies

6. CLINICAL OBSERVATIONS:
   - Physical exam findings
   - Vital signs
   - Symptoms reported
   - Functional status
   - Clinical assessments

7. CURRENT TREATMENT:
   - Current medications (name, dose, frequency)
   - Recent procedures
   - Ongoing therapies
   - Treatment plans

8. WORK STATUS:
   - Current work capacity
   - Restrictions and limitations
   - MMI status if mentioned

9. IMPORTANT DATES:
   - Report date
   - Service date / Collection date
   - Examination date
   - Follow-up dates

10. DOCUMENT CONTEXT:
    - Document type/purpose
    - Ordering/referring physician
    - Facility/department
    - Report status (preliminary/final)
    - Any administrative information
    - Next steps or recommendations

CRITICAL RULES:
- Extract ANY and ALL information available, even if minimal
- For lab reports: focus on test results, abnormal values, critical findings
- For imaging: focus on findings, impressions, recommendations
- If no data found for a field, DO NOT include it in output
- Empty arrays for no items, omit empty objects
- Include administrative details if medical data is sparse
- Look for ANY clinical data: tests, results, observations, assessments

Return JSON format - ONLY include fields with actual data:
{{{{
  "patient_details": {{}},
  "critical_findings": [],
  "diagnoses": [],
  "lab_results": [],
  "imaging_findings": [],
  "clinical_observations": [],
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

Extract ALL available information from this document. Look for any clinical data, test results, findings, or observations:
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
        Enhanced to handle lab reports, imaging, and other document types.
        """
        logger.info("🔍 Second LLM - Generating medical long summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical summarization expert. Create a professionally formatted medical summary for ANY type of medical document.

CRITICAL RULES:
- Use proper medical formatting with clear sections and bullet points
- ONLY include information that is explicitly available
- OMIT entire sections if no data exists
- NEVER mention missing information
- Use professional medical terminology
- Structure the summary for quick physician review
- Adapt sections based on document type (lab report, imaging, clinical note, etc.)

PROFESSIONAL FORMATTING GUIDE:

Start with a clear header:
MEDICAL SUMMARY - [Document Type]
[Report Date]

Then include ONLY these sections if data exists:

1. PATIENT INFORMATION
   • Name: [Name]
   • DOB: [Date of Birth]
   • MRN: [Medical Record Number]

2. CRITICAL FINDINGS (if any urgent findings)
   • [List any critical or urgent findings]
   • [Abnormal lab values with HIGH/LOW flags]
   • [Critical imaging findings]

3. LABORATORY RESULTS (for lab reports)
   • Test Name: [Result] ([Reference Range]) [FLAG if abnormal]
   • [List all test results with values]
   • Notable Abnormalities: [Highlight critical values]

4. IMAGING FINDINGS (for imaging reports)
   • Study Type: [MRI/CT/X-ray/Ultrasound]
   • Body Part: [Anatomical region]
   • Key Findings: [Major observations]
   • Impression: [Radiologist's conclusion]
   • Recommendations: [Follow-up imaging or studies]

5. DIAGNOSES (if available)
   • Primary: [Diagnosis] ([ICD-10 if available])
   • Secondary: [Diagnosis]

6. CLINICAL OBSERVATIONS (if available)
   • Physical Exam: [Findings]
   • Vital Signs: [Values]
   • Symptoms: [Patient reported symptoms]

7. PROCEDURES & TREATMENT (if available)
   • Procedure: [Procedure Name] ([Date])
   • Surgeon/Provider: [Name]
   • Findings: [Key findings]
   • Medications: [List with doses]

8. CURRENT STATUS (if available)
   • Work Status: [Status]
   • Restrictions: [Specific restrictions]
   • MMI Status: [If applicable]

9. RECOMMENDATIONS & FOLLOW-UP
   • Next Steps: [Recommended actions]
   • Follow-up: [Appointments or tests needed]
   • Referrals: [Specialist consultations]
   • Deadlines: [Important dates]

Use bullet points (•) for lists and clear section headers. Maintain professional medical tone.
Adapt the structure based on what information is available in the extracted data.
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

EXTRACTED MEDICAL INFORMATION:
{extracted_data}

Create a professionally formatted medical summary that includes ALL available information:
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
        Enhanced to handle any document type with available information.
        """
        logger.info("🎯 Third LLM - Generating short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited summaries for ANY type of medical document.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be pipe-delimited with ONLY fields that have actual data.
3. Possible fields (include ONLY if data exists):
   - Report Title
   - Author/Physician
   - Date
   - Body Parts
   - Diagnosis
   - Lab Results (key abnormal findings)
   - Imaging Findings (key observations)
   - Medication
   - MMI Status
   - Key Action Items
   - Work Status
   - Recommendation
   - Critical Finding
   - Urgent Next Steps

***IMPORTANT FORMAT RULES***
- Each segment must be **Key: Value**
- If a field has NO VALUE, SKIP THE ENTIRE SEGMENT
- NEVER output empty fields or keys without values
- NEVER produce double pipes (||)
- ONLY include segments with real data
- Keep keys descriptive and relevant

EXAMPLES:

Lab Report:
"Report Title: Lab Results | Date: 10/22/2025 | Critical Finding: Elevated WBC 15.2 (H), Glucose 245 mg/dL (H) | Lab Results: Hemoglobin 12.1, Creatinine 1.2 | Recommendation: Repeat CBC in 1 week, endocrinology consult for diabetes management"

Imaging Report:
"Report Title: MRI Lumbar Spine | Date: 09/15/2025 | Body Parts: L4-L5, L5-S1 | Imaging Findings: Moderate central stenosis L4-L5, broad-based disc herniation L5-S1 with nerve root impingement | Recommendation: Consider epidural steroid injection, neurosurgery consultation if conservative management fails"

Clinical Note:
"Report Title: Follow-up Visit | Physician: Dr. Smith | Date: 08/20/2025 | Body Parts: Right knee | Diagnosis: Post-operative status ACL reconstruction | Work Status: Modified duty, no squatting/kneeling | Recommendation: Continue PT 2x/week, f/u 6 weeks"

3. DO NOT fabricate or infer missing data – simply SKIP segments that don't exist
4. Use ONLY information explicitly found in the long summary
5. Output must be a SINGLE LINE (no line breaks)
6. Priority information (include if present):
   - Report title/type
   - Date
   - Critical findings or abnormal results
   - Key test results (labs/imaging)
   - Diagnoses
   - Recommendations or next steps
   - Work status if mentioned
   - Medications if mentioned

7. ABSOLUTE NO:
   - Assumptions or inferences
   - Empty fields or placeholders
   - Invented data
   - Narrative sentences
   - Extra pipes for missing fields

Your final output must be **30–60 words** with ONLY available information in pipe-delimited format.
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
            short_summary = self._clean_pipes_from_summary(short_summary)
            
            word_count = len(short_summary.split())
            logger.info(f"✅ Short summary: {word_count} words")
            
            # Fallback if too long
            if word_count > 80:
                words = short_summary.split()
                short_summary = ' '.join(words[:60]) + "..."
            
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return self._clean_pipes_from_summary(f"Report Title: {doc_type} | Date: Unknown")
    
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
            "extracted_data": self._create_fallback_extraction(fallback_date, doc_type),
            "content_type": "unknown"
        }