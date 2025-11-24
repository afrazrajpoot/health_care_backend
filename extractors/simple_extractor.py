"""
Simplified Four-LLM Chain Extractor with Conditional Routing
Version: 4.2 - Enhanced for General Medicine and Workers' Compensation
"""
import logging
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from datetime import datetime  # Added for dynamic current date handling


logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Smart four-LLM chain extractor with conditional routing, enhanced for General Medicine and Workers' Compensation.
    Key Updates in v4.2:
    - Integrated mode-specific prompting for 'general medicine' and 'worker comp'.
    - Dynamic fallback_date using current date if not provided.
    - Streamlined code by removing unused fallback extraction methods.
    - Enhanced prompts with worker comp focus (e.g., causal relationship, impairment, MMI).
    - Improved short summary fields for work-related details.
    """
    
    def __init__(self, llm: AzureChatOpenAI, mode: str = "general medicine"):
        self.llm = llm
        self.mode = mode.lower()  # Normalize mode
        self.parser = JsonOutputParser()
        logger.info(f"✅ SimpleExtractor v4.2 initialized with mode: {self.mode}")
    
    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: Optional[str] = None
    ) -> Dict:
        """
        Four-step LLM chain extraction with conditional routing.
        Updates: Auto-generate fallback_date if None.
        """
        if fallback_date is None:
            fallback_date = datetime.now().strftime("%B %d, %Y")  # e.g., "November 24, 2025"
        
        logger.info(f"🚀 Four-LLM Conditional Extraction: {doc_type} (Mode: {self.mode})")
        
        try:
            # STEP 1: Directly generate long summary with full context (no intermediate extraction)
            long_summary = self._generate_long_summary_direct(text, doc_type, fallback_date)
            
            # STEP 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)
            
            logger.info(f"✅ Conditional chain extraction completed")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "content_type": "universal",
                "mode": self.mode
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _generate_long_summary_direct(self, text: str, doc_type: str, fallback_date: str) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Adapted to handle both medical and administrative content universally, with mode-specific emphasis.
        """
        logger.info("🔍 Processing ENTIRE document in single context window for direct long summary...")
        
        # Enhanced System Prompt for Direct Long Summary Generation
        # Universal prompt that handles medical or administrative based on content, tailored by mode
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a universal medical and administrative document summarization expert analyzing a COMPLETE document.
Mode: {mode} - Tailor emphasis: For 'worker comp', prioritize work status, causal relationship, impairment %, MMI, and return-to-work recommendations.

PRIMARY PURPOSE: Generate a comprehensive, structured long summary that adapts to the document type (medical or administrative).

DETERMINE DOCUMENT TYPE AUTOMATICALLY:
- MEDICAL: If contains diagnoses, treatments, labs, imaging, clinical observations (common in general medicine or worker comp)
- ADMINISTRATIVE: If primarily forms, letters, notifications, billing, scheduling

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing comprehensive extraction without loss.

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):
1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** - Empty if not mentioned
2. **NO ASSUMPTIONS** - Do not infer or add typical values
3. **ADAPT STRUCTURE** - Use medical sections for clinical content, administrative for non-clinical
4. **EMPTY FIELDS BETTER THAN GUESSES** - Omit sections if no data
5. **WORKER COMP FOCUS**: If mode is 'worker comp', extract causal link to injury, impairment rating, MMI status, work restrictions.

UNIVERSAL EXTRACTION FOCUS:

For MEDICAL DOCUMENTS (General Medicine or Worker Comp):
I. PATIENT & CLINICAL CONTEXT
II. CRITICAL FINDINGS & DIAGNOSES
III. LAB/IMAGING RESULTS
IV. TREATMENT & OBSERVATIONS
V. STATUS & RECOMMENDATIONS (emphasize work-related for worker comp)

For ADMINISTRATIVE DOCUMENTS:
I. DOCUMENT OVERVIEW
II. KEY PARTIES & INFORMATION
III. ACTION ITEMS & DEADLINES
IV. CONTACT & FOLLOW-UP

Now analyze this COMPLETE document and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points; adapt sections based on content type):
""")

        # Enhanced User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}
MODE: {mode}

COMPLETE DOCUMENT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no date found; adapt to medical or administrative content):

For MEDICAL CONTENT:
📋 MEDICAL DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: [extracted or {fallback_date}]
Patient Name: [extracted]
Provider: [extracted]

👤 PATIENT & CLINICAL INFORMATION
--------------------------------------------------
Name: [extracted]
DOB: [extracted]
Chief Complaint: [extracted]
Clinical History: [extracted]
Date of Injury/Onset: [extracted, emphasize for worker comp]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 5 critical items, e.g., abnormal labs, urgent diagnoses, causal relationship to work injury]

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
Work Status: [extracted, e.g., missed work dates, limitations, return to duty]
MMI: [extracted, Maximum Medical Improvement status]
Impairment Rating: [extracted, e.g., 0-100% temporary/permanent]
Causal Relationship: [extracted, link to work incident]
Recommendations:
• [list up to 5 next steps, prioritize worker comp follow-ups]

For ADMINISTRATIVE CONTENT:
📋 ADMINISTRATIVE DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Document Date: [extracted or {fallback_date}]
Purpose: [extracted]

👥 KEY PARTIES
--------------------------------------------------
Patient: [extracted]
Provider: [extracted]
Referring Party: [extracted]
Employer/Carrier: [extracted, for worker comp]

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
3. Use exact wording for medical terms, dates, names
4. Bullet points for lists, clear headings
5. No assumptions or additions
6. For worker comp mode: Always check for work-related fields even in medical sections
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for direct full-context universal long summary generation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "doc_type": doc_type,
                "fallback_date": fallback_date,
                "mode": self.mode
            })
            
            long_summary = result.content.strip()
            
            logger.info(f"⚡ Direct universal long summary generation completed")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char document")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct universal long summary generation failed: {e}", exc_info=True)
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date} (Mode: {self.mode}): Document processing failed due to {str(e)}"

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate concise short summary from long summary.
        Enhanced to handle any document type with available information, with worker comp fields.
        """
        logger.info("🎯 Third LLM - Generating short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited summaries for ANY type of medical document.
Mode: {mode} - Include worker comp fields if present (e.g., Impairment Rating, Causal Relationship, MMI).

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
   - Impairment Rating
   - Causal Relationship
   - Key Action Items
   - Work Status
   - Recommendation
   - Critical Finding
   - Urgent Next Steps
   - Employer/Carrier (for worker comp)

***IMPORTANT FORMAT RULES***
- Each segment must be **Key: Value**
- If a field has NO VALUE, SKIP THE ENTIRE SEGMENT
- NEVER output empty fields or keys without values
- NEVER produce double pipes (||)
- ONLY include segments with real data
- Keep keys descriptive and relevant

EXAMPLES:

Lab Report (General Medicine):
"Report Title: Lab Results | Date: 10/22/2025 | Critical Finding: Elevated WBC 15.2 (H), Glucose 245 mg/dL (H) | Lab Results: Hemoglobin 12.1, Creatinine 1.2 | Recommendation: Repeat CBC in 1 week, endocrinology consult for diabetes management"

Worker Comp Report:
"Report Title: Initial Injury Evaluation | Date: 09/15/2025 | Body Parts: Low Back | Diagnosis: Lumbar Strain | Work Status: Off duty since 09/10/2025 | Causal Relationship: Direct result of lifting incident | Impairment Rating: 15% temporary | MMI: Not reached | Recommendation: PT 3x/week, f/u in 4 weeks"

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
   - Work status, impairment, MMI if mentioned (worker comp priority)
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
            response = chain.invoke({
                "long_summary": long_summary[:3000],
                "mode": self.mode
            })
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
            return self._clean_pipes_from_summary(f"Report Title: {doc_type} | Date: Unknown | Mode: {self.mode}")
    
    def _clean_short_summary(self, text: str) -> str:
        """Clean short summary text."""
        # Remove unwanted phrases
        unwanted_phrases = [
            "unknown", "not specified", "not provided", "none", 
            "no information", "missing", "unavailable", "unspecified",
            "not mentioned", "not available", "not stated"
        ]
        
        cleaned = text
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Clean formatting
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.strip('"').strip("'")
        
        return cleaned
    
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
    
    def _create_error_response(self, doc_type: str, error_msg: str, fallback_date: str) -> Dict:
        """Create error response that still provides a summary."""
        return {
            "long_summary": f"DOCUMENT SUMMARY - {doc_type.upper()}\n\nDocument processed. Basic information extracted. Error: {error_msg}",
            "short_summary": f"Report Title: {doc_type} | Date: {fallback_date}",
            "content_type": "unknown",
            "mode": self.mode
        }