"""
FormalMedicalReportExtractor - Enhanced Extractor for Comprehensive Medical Reports
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


class FormalMedicalReportExtractor:
    """
    Enhanced Formal Medical Report extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, Endoscopy, Genetics, Discharge Summaries
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.report_type_patterns = {
            'surgery': re.compile(r'\b(surgery|surgical|pre[- ]?op|post[- ]?op|operative|procedure)\b', re.IGNORECASE),
            'anesthesia': re.compile(r'\b(anesthesia|anesthetic|sedation|airway|intubation)\b', re.IGNORECASE),
            'emg': re.compile(r'\b(EMG|NCS|electromyography|nerve conduction|needle exam)\b', re.IGNORECASE),
            'pathology': re.compile(r'\b(pathology|biopsy|specimen|histology|microscopic)\b', re.IGNORECASE),
            'cardiology': re.compile(r'\b(cardiology|EKG|ECG|echocardiogram|stress test|holter)\b', re.IGNORECASE),
            'sleep': re.compile(r'\b(sleep study|polysomnography|PSG|apnea|hypopnea)\b', re.IGNORECASE),
            'endoscopy': re.compile(r'\b(endoscopy|colonoscopy|EGD|gastroscopy|bronchoscopy)\b', re.IGNORECASE),
            'genetics': re.compile(r'\b(genetic|mutation|variant|DNA|RNA|chromosome)\b', re.IGNORECASE),
            'discharge': re.compile(r'\b(discharge|admission|hospital course|disposition)\b', re.IGNORECASE)
        }
        
        # Medical procedure patterns
        self.procedure_patterns = {
            'cpt_codes': re.compile(r'\bCPT[:\s]*(\d{4,5})', re.IGNORECASE),
            'icd_codes': re.compile(r'\b(ICD[-]?10[:\s]*([A-Z]\d{2,})|([A-Z]\d{2,}))', re.IGNORECASE),
            'medications': re.compile(r'\b(\d+\s*(mg|mcg|g|ml)\s*[\w\s]+\s*(PO|IV|IM|SC|QD|BID|TID|QID|PRN))', re.IGNORECASE)
        }
        
        logger.info("✅ FormalMedicalReportExtractor initialized (Full Context)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
    
    ) -> Dict:
        """
        Extract Formal Medical Report data with FULL CONTEXT.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Surgery, Anesthesia, EMG, Pathology, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING FORMAL MEDICAL REPORT EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        # Use text directly for LLM extraction
        text_to_use = text
        logger.info("📄 Using text for LLM extraction")
        
        # Auto-detect specific report type if not specified
        detected_type = self._detect_report_type(text_to_use, doc_type)
        logger.info(f"📋 Report Type: {detected_type} (original: {doc_type})")
        
        # Check document size
        text_length = len(text_to_use)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Directly generate long summary with FULL CONTEXT (no intermediate extraction)
        long_summary = self._generate_long_summary_direct(
            text=text_to_use,
            doc_type=detected_type,
            fallback_date=fallback_date
        )

        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)

        logger.info("=" * 80)
        logger.info("✅ FORMAL MEDICAL REPORT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_report_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific medical report type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for report_type, pattern in self.report_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[report_type] = len(matches)
        
        # Boost scores for procedure-specific terminology
        if self.procedure_patterns['cpt_codes'].search(text):
            for report_type in ['surgery', 'procedure']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 2
        
        if self.procedure_patterns['icd_codes'].search(text):
            for report_type in ['pathology', 'discharge']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].upper().replace('_', ' ')
                logger.info(f"🔍 Auto-detected report type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect report type, using: {original_type}")
        return original_type or "MEDICAL_REPORT"

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
        logger.info("🔍 Processing ENTIRE medical report in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Reuses core anti-hallucination rules and medical report focus from original extraction prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical documentation specialist analyzing a COMPLETE {doc_type} report.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE medical report at once, allowing you to:
- Understand the complete clinical picture from history to conclusions
- Connect pre-procedure assessments with intraoperative findings and post-procedure outcomes
- Identify relationships between clinical indications, procedures performed, and results
- Provide comprehensive extraction without information loss

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate medical information
   - DO NOT fill in "typical" or "common" medical values
   - DO NOT use medical knowledge to "complete" incomplete information
   
2. **PROCEDURES & FINDINGS - EXACT WORDING ONLY**
   - Extract procedures using EXACT wording from report
   - Extract findings verbatim - do not interpret or rephrase
   - For pathology: extract microscopic descriptions EXACTLY as stated
   - For lab values: extract numbers and units EXACTLY as written
   
3. **MEDICATIONS & DOSES - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY medications explicitly listed in medication sections
   - Include dosages, routes, frequencies ONLY if explicitly stated
   - DO NOT extract medications mentioned as examples or comparisons
   
4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
5. **MEDICAL CODING - EXACT REFERENCES ONLY**
   - Extract CPT/ICD codes ONLY if explicitly listed in the report
   - DO NOT assign codes based on procedure descriptions
   - Include code descriptors ONLY if provided

EXTRACTION FOCUS - 8 CRITICAL MEDICAL REPORT CATEGORIES:

I. REPORT IDENTITY & CONTEXT
- Report type, dates, identification numbers
- Facility and department information
- All healthcare providers involved

II. PATIENT CLINICAL CONTEXT
- Patient demographics and identifiers
- Clinical history and presenting symptoms
- Pre-existing conditions and risk factors
- Indications for procedure/study

III. PROCEDURE/TEST DETAILS (CORE CONTENT)
- Procedure/test name and type
- Anatomical locations and specific sites
- Technique/methodology used
- Duration and technical details
- Specimen information (for pathology)

IV. INTRAOPERATIVE/INTRA-PROCEDURAL FINDINGS
- Detailed findings during procedure
- Anatomical observations
- Complications or unexpected findings
- Blood loss, fluids, vital signs

V. SPECIMEN/PATHOLOGY DETAILS (if applicable)
- Specimen descriptions and labeling
- Gross examination findings
- Microscopic examination details
- Special stains and results

VI. RESULTS & INTERPRETATIONS
- Test results with values and units
- Physician interpretations and conclusions
- Diagnostic impressions
- Correlation with clinical information

VII. MEDICATIONS & ANESTHESIA (if applicable)
- Anesthetic agents and techniques
- Medications administered during procedure
- Dosages, routes, and timing
- Anesthesia complications

VIII. FOLLOW-UP & RECOMMENDATIONS
- Post-procedure instructions
- Medication prescriptions
- Follow-up scheduling
- Additional testing recommendations

⚠️ FINAL REMINDER:
- If information is NOT in the report, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate medical information
- PROCEDURE DETAILS: Use exact wording from report
- It is BETTER to have empty fields than incorrect medical information

Now analyze this COMPLETE {doc_type} medical report and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} MEDICAL REPORT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no report date found):

📋 MEDICAL REPORT OVERVIEW
--------------------------------------------------
Report Type: {doc_type}
Report Date: [extracted or {fallback_date}]
Procedure Date: [extracted]
Accession Number: [extracted]
Facility: [extracted]
Department: [extracted]

👤 PATIENT INFORMATION
--------------------------------------------------
Name: [extracted]
DOB: [extracted]
Age: [extracted]
Gender: [extracted]
Allergies: [extracted]
MRN: [extracted]

👨‍⚕️ HEALTHCARE PROVIDERS
--------------------------------------------------
Performing Physician: [name]
  Specialty: [extracted]
Ordering Physician: [name]
  Specialty: [extracted]
Anesthesiologist: [name]

🏥 CLINICAL CONTEXT
--------------------------------------------------
Indications: [extracted]
Preoperative Diagnosis: [extracted]
Postoperative Diagnosis: [extracted]
Clinical History: [extracted]

🔧 PROCEDURE DETAILS
--------------------------------------------------
Procedure: [extracted]
Type: [extracted]
Anatomical Sites: [list up to 3]
Laterality: [extracted]
Anesthesia: [extracted]
Duration: [extracted]
CPT Codes: [list up to 3]

🔍 FINDINGS & RESULTS
--------------------------------------------------
Intraoperative Findings: [extracted]
Pathological Diagnosis: [extracted]
Microscopic: [extracted, truncate if >200 chars]
Results Summary: [extracted]
Interpretation: [extracted]

💊 MEDICATIONS & ANESTHESIA
--------------------------------------------------
Anesthetic Agents:
• [list up to 5 with doses if stated]
Intraoperative Medications:
• [list up to 5 with doses if stated]

🎯 CONCLUSIONS & RECOMMENDATIONS
--------------------------------------------------
Final Diagnosis: [extracted]
Clinical Impressions: [extracted]
Recommendations:
• [list up to 5]
Follow-up Plan: [extracted]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 8 most significant items]

⚠️ CRITICAL MEDICAL REMINDERS:
1. For "procedure_details": Extract EXACT procedure names from report
   - Include anatomical specifics ONLY if explicitly stated
   - Include CPT/ICD codes ONLY if explicitly listed

2. For "pathology_findings": Extract microscopic descriptions VERBATIM
   - Do not interpret or summarize pathological findings
   - Include tumor characteristics ONLY if explicitly measured/stated

3. For "test_results": Extract values and units EXACTLY as written
   - Include reference ranges ONLY if provided
   - Do not interpret abnormal vs normal - extract values only

4. For "medications_anesthesia": Extract ONLY medications explicitly administered
   - Include dosages and routes ONLY if explicitly stated
   - Do not include medications mentioned in history or recommendations

5. For "critical_findings": Include only clinically significant findings
   - Malignancies or positive cancer diagnoses
   - Critical abnormal lab values
   - Significant complications
   - Life-threatening conditions
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context medical report long summary generation...")
            
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
            
            logger.info(f"⚡ Direct medical report long summary generation completed in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char medical report")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct medical report long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Medical report exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large reports")
            
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
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word structured medical summary in key-value format.
        Zero hallucinations, pipe-delimited, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word medical structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-report summarization specialist.

TASK:
Create a concise, factual summary of a medical report using ONLY information explicitly present in the long summary.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Medication:[value] | MMI Status:[value] | Work Status:[value] | Restrictions:[value] | Treatment Progress:[value] | Critical Finding:[value] | Follow-up:[value]

FORMAT & RULES:
- MUST be **30–60 words**.
- MUST be **ONE LINE**, pipe-delimited, no line breaks.
- NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
- NEVER fabricate: no invented dates, meds, restrictions, exam findings, or recommendations.
- NO narrative sentences. Use short factual fragments ONLY.
- First three fields (Report Title, Author, Date) appear without keys
- All other fields use key-value format: Key:[value]

CONTENT PRIORITY (only if provided in the long summary):
1. Report Title  
2. Author  
3. Visit Date  
4. Body parts  
5. Diagnosis  
6. Medications  
7. MMI status  
8. Work status & restrictions  
9. Treatment progress  
10. Critical finding  
11. Follow-up plan

ABSOLUTELY FORBIDDEN:
- assumptions, interpretations, invented medications, or inferred diagnoses
- narrative writing
- placeholder text or "Not provided"
- duplicate pipes or empty pipe fields (e.g., "||")

Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
MEDICAL REPORT LONG SUMMARY:

{long_summary}

Produce a 30–60 word structured medical summary following ALL rules.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Remove pipe cleaning to keep pipes as-is
            # summary = self._clean_pipes_from_summary(summary)

            # Validate 30–60 word range
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Medical summary out of range ({wc} words). Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous output contained {wc} words. Rewrite it to be **between 30 and 60 words**, keeping all factual content, maintaining the key-value pipe-delimited format, and adding NO invented details. Remember: [Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after regeneration

            logger.info(f"✅ Medical summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Medical summary generation failed: {e}")
            return "Summary unavailable due to processing error."
  
    def _create_medical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback medical summary directly from long summary"""
        
        # Extract key medical information using regex patterns
        patterns = {
            'procedure': r'Procedure:\s*([^\n]+)',
            'physician': r'Performing Physician:\s*([^\n]+)',
            'diagnosis': r'Final Diagnosis:\s*([^\n]+)',
            'findings': r'Intraoperative Findings:\s*([^\n]+)',
            'pathology': r'Pathological Diagnosis:\s*([^\n]+)',
            'recommendations': r'Recommendations:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type
        parts.append(f"{doc_type} report")
        
        if 'procedure' in extracted:
            parts.append(f"Procedure: {extracted['procedure']}")
        
        # Add physician context
        if 'physician' in extracted:
            parts.append(f"by {extracted['physician']}")
        
        # Add findings
        if 'findings' in extracted:
            first_finding = extracted['findings'][:80] + "..." if len(extracted['findings']) > 80 else extracted['findings']
            parts.append(f"Findings: {first_finding}")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"Diagnosis: {extracted['diagnosis']}")
        elif 'pathology' in extracted:
            parts.append(f"Pathology: {extracted['pathology']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with standard medical follow-up", "following established protocols", "with routine clinical monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used medical fallback summary: {len(summary.split())} words")
        return summary