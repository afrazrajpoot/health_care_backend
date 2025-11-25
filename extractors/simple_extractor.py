"""
Simplified Four-LLM Chain Extractor with Conditional Routing
Version: 4.3 - Enhanced Claim Number Extraction
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
        logger.info("✅ SimpleExtractor v4.3 initialized with Enhanced Claim Number Extraction")
    
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
        print(text,'oooppp')
        try:
            # STEP 1: Directly generate long summary with full context (no intermediate extraction)
            long_summary = self._generate_long_summary_direct(text, doc_type, fallback_date)
            
            # STEP 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)
            
            logger.info(f"✅ Conditional chain extraction completed")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "content_type": "universal"
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _generate_long_summary_direct(self, text: str, doc_type: str, fallback_date: str) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Adapted to handle both medical and administrative content universally.
        Enhanced to extract physical and electronic signers/authors and claim number.
        """
        logger.info("🔍 Processing ENTIRE document in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Universal prompt that handles medical or administrative based on content
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a universal medical and administrative document summarization expert analyzing a COMPLETE document.

PRIMARY PURPOSE: Generate a comprehensive, structured long summary that adapts to the document type (medical or administrative).

DETERMINE DOCUMENT TYPE AUTOMATICALLY:
- MEDICAL: If contains diagnoses, treatments, labs, imaging, clinical observations
- ADMINISTRATIVE: If primarily forms, letters, notifications, billing, scheduling

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing comprehensive extraction without loss.

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):
1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** - Empty if not mentioned
2. **NO ASSUMPTIONS** - Do not infer or add typical values
3. **ADAPT STRUCTURE** - Use medical sections for clinical content, administrative for non-clinical
4. **EMPTY FIELDS BETTER THAN GUESSES** - Omit sections if no data
5. **SIGNATURE EXTRACTION**: Scan the entire document for signatures. Identify authors who signed PHYSICALLY (e.g., handwritten, wet signature) or ELECTRONICALLY (e.g., e-signature, typed name with /s/, digital stamp). Distinguish and list separately if both present. Use exact names/titles from the document.
6. **CLAIM NUMBER EXTRACTION**: Scan the entire document for claim number (e.g., "Claim #12345", "Claim Number: ABC-123"). Extract the exact value if present; omit if not found.

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

Now analyze this COMPLETE document and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points; adapt sections based on content type):
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

COMPLETE DOCUMENT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no date found; adapt to medical or administrative content):

For MEDICAL CONTENT:
📋 MEDICAL DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: [extracted or {fallback_date}]
Claim Number: [extracted if present; otherwise omit]
Patient Name: [extracted]
Provider: [extracted]
Signer/Author:
• Physical Signature: [extracted name/title if physical signature present; otherwise omit]
• Electronic Signature: [extracted name/title if electronic signature present; otherwise omit]

👤 PATIENT & CLINICAL INFORMATION
--------------------------------------------------
Name: [extracted]
DOB: [extracted]
Chief Complaint: [extracted]
Clinical History: [extracted]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 5 critical items, e.g., abnormal labs, urgent diagnoses]

🏥 DIAGNOSES & ASSESSMENTS
--------------------------------------------------
Primary Diagnosis: [extracted]
Secondary Diagnoses:
• [list up to 3]
Lab Results:
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
Document Date: [extracted or {fallback_date}]
Claim Number: [extracted if present; otherwise omit]
Purpose: [extracted]
Signer/Author:
• Physical Signature: [extracted name/title if physical signature present; otherwise omit]
• Electronic Signature: [extracted name/title if electronic signature present; otherwise omit]

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

⚠️ MANDATORY EXTRACTION RULES:
1. Adapt structure to content: Medical if clinical data present, Administrative if not
2. Extract ONLY explicit information - omit sections with no data
3. Use exact wording for medical terms, dates, names, signatures, and claim numbers
4. Bullet points for lists, clear headings
5. No assumptions or additions
6. For signatures: Look for end-of-document sign blocks, /s/ notations, scanned signatures, or explicit "signed by" statements. Distinguish physical (e.g., "Handwritten by Dr. X") vs. electronic (e.g., "Electronically signed by Dr. Y").
7. For claim number: Search for patterns like "Claim #", "Claim Number", "WC Claim", etc., and extract the alphanumeric value exactly.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for direct full-context universal long summary generation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            logger.info(f"⚡ Direct universal long summary generation completed")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char document")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct universal long summary generation failed: {e}", exc_info=True)
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate concise short summary from long summary.
        Enhanced to handle any document type with available information.
        Includes author/signer and claim number but excludes patient details.
        """
        logger.info("🎯 Third LLM - Generating short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited summaries for ANY type of medical document.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be pipe-delimited with ONLY fields that have actual data.
3. Possible fields (include ONLY if data exists; NO patient details like name, DOB, MRN):
   - Report Title
   - Claim Number
   - Author/Physician/Signer
   - Signature Type (Physical/Electronic)
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
- For Author/Signer: Combine name and type, e.g., "Author/Physician/Signer: Dr. Smith (Electronic)"
- For Signature Type: Only if distinct from author field

EXAMPLES:

Lab Report:
"Report Title: Lab Results | Claim Number: ABC-123 | Date: 10/22/2025 | Author/Physician/Signer: Dr. Jones (Electronic) | Critical Finding: Elevated WBC 15.2 (H), Glucose 245 mg/dL (H) | Lab Results: Hemoglobin 12.1, Creatinine 1.2 | Recommendation: Repeat CBC in 1 week, endocrinology consult for diabetes management"

Imaging Report:
"Report Title: MRI Lumbar Spine | Claim Number: 45678 | Date: 09/15/2025 | Author/Physician/Signer: Dr. Lee (Physical) | Body Parts: L4-L5, L5-S1 | Imaging Findings: Moderate central stenosis L4-L5, broad-based disc herniation L5-S1 with nerve root impingement | Recommendation: Consider epidural steroid injection, neurosurgery consultation if conservative management fails"

Clinical Note:
"Report Title: Follow-up Visit | Author/Physician/Signer: Dr. Smith (Electronic) | Date: 08/20/2025 | Body Parts: Right knee | Diagnosis: Post-operative status ACL reconstruction | Work Status: Modified duty, no squatting/kneeling | Recommendation: Continue PT 2x/week, f/u 6 weeks"

3. DO NOT fabricate or infer missing data – simply SKIP segments that don't exist
4. Use ONLY information explicitly found in the long summary
5. Output must be a SINGLE LINE (no line breaks)
6. Priority information (include if present):
   - Report title/type
   - Claim Number
   - Author/Signer (with type if available)
   - Date
   - Critical findings or abnormal results
   - Key test results (labs/imaging)
   - Diagnoses
   - Recommendations or next steps
   - Work status if mentioned
   - Medications if mentioned
7. ABSOLUTE NO:
   - Patient details (name, DOB, MRN, etc.)
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

Create a clean pipe-delimited short summary with ONLY available information (exclude patient details):
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
            "content_type": "unknown"
        }