"""
Imaging Reports Enhanced Extractor - 6 Critical Imaging Fields Focused

Optimized for MRI, X-ray, CT-scan, and other imaging modalities
Full-context processing with anti-hallucination rules
NO assumptions, NO self-additions, ONLY explicit information from report
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


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - 6-FIELD IMAGING FOCUS (Header, Clinical Data, Technique, Key Findings, Impression, Recommendations)
    - ZERO tolerance for hallucination, assumptions, or self-additions
    - Only extracts EXPLICITLY STATED information from imaging reports
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex patterns for imaging specific content
        self.imaging_patterns = {
            'modality': re.compile(r'\b(MRI|CT|X-RAY|XRAY|ULTRASOUND|US|MAMMOGRAM|PET|SPECT|DEXA)\b', re.IGNORECASE),
            'body_part': re.compile(r'\b(shoulder|knee|spine|wrist|hip|ankle|elbow|hand|foot|brain|chest|abdomen|pelvis|lumbar|cervical|thoracic)\b', re.IGNORECASE),
            'contrast': re.compile(r'\b(with|without)\s+contrast\b', re.IGNORECASE),
            'finding_severity': re.compile(r'\b(mild|moderate|severe|minimal|marked|advanced|subtle|questionable|probable|likely)\b', re.IGNORECASE)
        }
        
        logger.info("✅ ImagingExtractorChained initialized (6-Field Imaging Focus)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
 
    ) -> Dict:
        """
        Extract imaging data with FULL CONTEXT and 6-field focus.
        Returns dictionary with long_summary and short_summary like QME extractor.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (MRI, CT, X-ray, Ultrasound, etc.)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
        """
        
        logger.info("=" * 80)
        logger.info("📊 STARTING IMAGING EXTRACTION (6-FIELD FOCUS)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Directly generate long summary with full context (no intermediate extraction)
            long_summary = self._generate_long_summary_direct(text, doc_type, fallback_date)
            
            # Step 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context imaging extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted imaging data from complete {len(text):,} char document")
            
            logger.info("=" * 80)
            logger.info("✅ IMAGING EXTRACTION COMPLETE (6-FIELD FOCUS)")
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
                "long_summary": f"Imaging extraction failed: {str(e)}",
                "short_summary": "Imaging summary not available"
            }

    def _generate_long_summary_direct(self, text: str, doc_type: str, fallback_date: str) -> str:
        """Directly generate raw imaging data using LLM with full context and integrated author detection"""
        # Build system prompt with 6-field imaging focus, including instructions for detecting the signing author
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert radiological report specialist analyzing a COMPLETE imaging report.

PRIMARY PURPOSE: Generate a comprehensive, structured long summary of the 6 critical imaging fields for accurate medical documentation.

CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If NOT explicitly mentioned in report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
   - DO NOT add typical findings, standard protocols, or common measurements
   - DO NOT use radiological training to "complete" incomplete information

Examples of INCORRECT extractions:
   ❌ Report says "mass in right upper lobe" → DO NOT extract size if not stated
   ❌ Report mentions "findings suggest tear" → DO NOT extract "rotator cuff tear" (upgrades certainty)
   ❌ Report shows "mild degenerative changes" → DO NOT upgrade to "moderate" findings
   ❌ Report says "no acute fracture" → DO NOT list fracture as a finding

2. **FINDINGS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY findings explicitly described in FINDINGS or IMPRESSION sections
   - Include measurements ONLY if explicitly stated (not estimated)
   - Include severity qualifiers EXACTLY as stated (mild/moderate/severe)
   - DO NOT extract "possible" or "rule out" as confirmed findings
   - DO NOT add normal variants unless explicitly highlighted as findings

3. **TECHNICAL DETAILS - EXACT WORDING**
   - Contrast status: Use EXACT report wording (with/without/not specified)
   - Body part: Include specific anatomical location and laterality
   - Protocol: Extract only if explicitly named in technique section
   - Quality: Extract only if explicitly assessed in report

4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return empty field than to guess
   - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""
   - DO NOT assume standard protocols or typical measurements

5. **RADIOLOGIST'S EXACT LANGUAGE**
   - Use EXACT terminology from report
   - Preserve certainty qualifiers: "suspicious", "likely", "consistent with", "probable"
   - DO NOT remove nuanced descriptions: "subtle", "questionable", "mild enhancement"
   - DO NOT interpret radiological significance beyond what's stated

6. **RADIOLOGIST/AUTHOR DETECTION**:
   - Identify the author who signed the report as the "radiologist" name (e.g., from signature block, "Dictated by:", or closing statement).
   - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
   - Extract credentials only if explicitly stated near the signature.
   - If no clear signer is found, leave "name" empty.

6 CRITICAL IMAGING FIELDS:

FIELD 1: HEADER & CONTEXT (Report Identity & Date)
FIELD 2: CLINICAL DATA/INDICATION (Reason for the Study)
FIELD 3: TECHNIQUE/PRIOR STUDIES (Methodology & Comparison)
FIELD 4: KEY FINDINGS - POSITIVE/NEGATIVE (Evidence of Pathology)
FIELD 5: IMPRESSION/CONCLUSION (Radiologist's Final Diagnosis)
FIELD 6: RECOMMENDATIONS/FOLLOW-UP (Actionable Next Steps)

Now analyze this COMPLETE imaging report and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")
        
        # Build user prompt
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE IMAGING REPORT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no exam date found):

📋 IMAGING OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Exam Date: [extracted or {fallback_date}]
Exam Type: [extracted]
Radiologist: [name]
Imaging Center: [extracted]
Referring Physician: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit]


 ## PATIENT INFORMATION
    - **Name:** [extracted name]
    - **Date of Birth:** [extracted DOB] 
    - **Claim Number:** [extracted claim number]
    - **Date of Injury:** [extracted DOI]
    - **Employer:** [extracted employer]   
🎯 CLINICAL INDICATION
--------------------------------------------------
Clinical Indication: [extracted]
Clinical History: [extracted]
Chief Complaint: [extracted]
Specific Questions: [extracted]

🔧 TECHNICAL DETAILS
--------------------------------------------------
Study Type: {doc_type}
Body Part Imaged: [extracted]
Laterality: [extracted]
Contrast Used: [extracted]
Contrast Type: [extracted]
Prior Studies Available: [extracted]
Technical Quality: [extracted]
Limitations: [extracted]

📊 KEY FINDINGS
--------------------------------------------------
Primary Finding:
  • Description: [extracted]
  • Location: [extracted]
  • Size: [extracted]
  • Characteristics: [extracted]
  • Acuity: [extracted]
Secondary Findings:
• [list up to 5 with locations and significance]
Normal Findings:
• [list up to 5]

💡 IMPRESSION & CONCLUSION
--------------------------------------------------
Overall Impression: [extracted]
Primary Diagnosis: [extracted]
Final Diagnostic Statement: [extracted]
Differential Diagnoses:
• [list up to 3]
Clinical Correlation: [extracted]

📋 RECOMMENDATIONS & FOLLOW-UP
--------------------------------------------------
Follow-up Recommended: [extracted]
Follow-up Modality: [extracted]
Follow-up Timing: [extracted]
Clinical Correlation Needed: [extracted]
Specialist Consultation: [extracted]

⚠️ MANDATORY EXTRACTION RULES:
1. Field 1: Extract EXACT dates and names from report
2. Field 2: Use EXACT clinical indication wording from report
3. Field 3: Contrast status must be EXPLICIT (with/without/not stated)
4. Field 4: Size/measurements ONLY if explicitly stated in report
5. Field 5: Use RADIOLOGIST'S EXACT impression language
6. Field 6: Include recommendations ONLY if explicitly stated
7. EMPTY FIELDS ARE ACCEPTABLE - Better than guessed information
8. NO assumptions, NO additions, NO upgrades of severity
""")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])
        
        logger.info(f"📄 Document size: {len(text):,} chars (~{len(text) // 4:,} tokens)")
        logger.info("🔍 Processing ENTIRE imaging report in single context window with 6-field focus...")
        logger.info("🤖 Invoking LLM for direct full-context imaging long summary generation...")
        
        # Invoke LLM
        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            logger.info("✅ Generated imaging long summary from complete document")
            return long_summary
        
        except Exception as e:
            logger.error(f"❌ Direct LLM generation failed: {str(e)}")
            return self._get_fallback_long_summary(fallback_date, doc_type)

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
        Generate a precise 30–60 word structured imaging summary in key-value format.
        Zero hallucinations. Pipe-delimited. Skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word structured imaging summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a radiology-report summarization specialist.

    TASK:
    Produce a concise structured summary of an imaging report using ONLY details explicitly present in the long summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:

    [Report Title] | [Radiologist/Physician] | [Study Date] | Body Parts:[value] | Findings:[value] | Impression:[value] | Comparison:[value] | Critical Finding:[value] | Recommendations:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
    - NEVER fabricate: no invented dates, findings, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.
    - First three fields (Report Title, Radiologist, Study Date) appear without keys
    - All other fields use key-value format: Key:[value]
    - Focus on radiology-specific elements: findings, impressions, comparisons

    CONTENT PRIORITY (only if provided in the long summary):
    1. Report Title  
    2. Radiologist  
    3. Study Date  
    4. Body parts studied  
    5. Key imaging findings  
    6. Radiologist's impression  
    7. Comparison to prior studies  
    8. Critical/urgent findings  
    9. Recommendations for follow-up

    ABSOLUTELY FORBIDDEN:
    - assumptions, interpretations, or invented findings
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")
    - Including non-radiology fields (medications, work status, etc.)

    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    IMAGING REPORT LONG SUMMARY:

    {long_summary}

    Create a strict 30–60 word imaging summary using the required pipe-delimited format.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # No pipe cleaning - keep pipes as generated

            # Validate 30–60 word requirement
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Imaging summary word count out of range: {wc} words. Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior output was {wc} words. Rewrite it to be between 30–60 words, preserving only factual content, keeping the exact key-value pipe format, and adding NO fabricated details. Maintain format: [Report Title] | [Radiologist] | [Study Date] | Body Parts:[value] | Findings:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after fix

            logger.info(f"✅ Imaging summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Imaging summary generation failed: {e}")
            return "Summary unavailable due to processing error."
  
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract radiologist information
        radiologist_match = re.search(r'Radiologist:\s*([^\n]+)', long_summary)
        radiologist = radiologist_match.group(1).strip() if radiologist_match else "Radiologist"
        
        # Extract key information using regex patterns
        patterns = {
            'modality': r'Exam Type:\s*([^\n]+)',
            'body_part': r'Body Part Imaged:\s*([^\n]+)',
            'indication': r'Clinical Indication:\s*([^\n]+)',
            'findings': r'Primary Finding:(.*?)(?:\n\n|\n[A-Z]|$)',
            'impression': r'Overall Impression:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with modality and body part
        if 'modality' in extracted and 'body_part' in extracted:
            parts.append(f"{extracted['modality']} {extracted['body_part']}")
        elif 'modality' in extracted:
            parts.append(f"{extracted['modality']} study")
        
        # Add radiologist
        parts.append(f"by {radiologist}")
        
        # Add indication
        if 'indication' in extracted:
            parts.append(f"for {extracted['indication'][:60]}")
        
        # Add findings
        if 'findings' in extracted:
            # Take first line of findings
            first_finding = extracted['findings'].split('\n')[0].replace('•', '').replace('Description:', '').strip()[:80]
            if first_finding:
                parts.append(f"Findings: {first_finding}")
        
        # Add impression
        if 'impression' in extracted:
            parts.append(f"Impression: {extracted['impression'][:80]}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["comprehensive radiological evaluation", "with diagnostic interpretation", "and clinical implications"] 
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
                return str(value)
            if isinstance(value, list):
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

    def _get_fallback_result(self, fallback_date: str, doc_type: str) -> Dict:
        """Return fallback result structure matching 6-field imaging structure"""
        return {
            "field_1_header_context": {
                "imaging_center": "",
                "exam_date": fallback_date,
                "exam_type": doc_type,
                "patient_name": "",
                "patient_dob": "",
                "referring_physician": "",
                "radiologist": {
                    "name": "",
                    "credentials": "",
                    "specialty": "Radiology"
                }
            },
            "field_2_clinical_data": {
                "clinical_indication": "",
                "clinical_history": "",
                "specific_clinical_questions": "",
                "chief_complaint": ""
            },
            "field_3_technique_prior": {
                "study_type": doc_type,
                "body_part_imaged": "",
                "laterality": "",
                "contrast_used": "",
                "contrast_type": "",
                "prior_studies_available": "",
                "prior_study_dates": [],
                "technical_quality": "",
                "limitations": ""
            },
            "field_4_key_findings": {
                "primary_finding": {
                    "description": "",
                    "location": "",
                    "size": "",
                    "characteristics": "",
                    "acuity": "",
                    "significance": ""
                },
                "secondary_findings": [],
                "normal_findings": []
            },
            "field_5_impression_conclusion": {
                "overall_impression": "",
                "primary_diagnosis": "",
                "differential_diagnoses": [],
                "clinical_correlation_statement": "",
                "final_diagnostic_statement": ""
            },
            "field_6_recommendations_followup": {
                "follow_up_recommended": "",
                "follow_up_modality": "",
                "follow_up_timing": "",
                "clinical_correlation_needed": "",
                "specialist_consultation": ""
            }
        }

    def _get_fallback_long_summary(self, fallback_date: str, doc_type: str) -> str:
        """Return fallback long summary structure"""
        fallback_text = f"""
📋 IMAGING OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Exam Date: {fallback_date}
Exam Type: {doc_type}
Radiologist: Not specified
Imaging Center: Not specified
Referring Physician: Not specified

👤 PATIENT INFORMATION
--------------------------------------------------
Name: Not specified
Date of Birth: Not specified

🎯 CLINICAL INDICATION
--------------------------------------------------
Clinical Indication: Not specified
Clinical History: Not specified
Chief Complaint: Not specified
Specific Questions: Not specified

🔧 TECHNICAL DETAILS
--------------------------------------------------
Study Type: {doc_type}
Body Part Imaged: Not specified
Laterality: Not specified
Contrast Used: Not specified
Contrast Type: Not specified
Prior Studies Available: Not specified
Technical Quality: Not specified
Limitations: Not specified

📊 KEY FINDINGS
--------------------------------------------------
Primary Finding:
  • Description: Not specified
  • Location: Not specified
  • Size: Not specified
  • Characteristics: Not specified
  • Acuity: Not specified
Secondary Findings:
• None specified
Normal Findings:
• None specified

💡 IMPRESSION & CONCLUSION
--------------------------------------------------
Overall Impression: Not specified
Primary Diagnosis: Not specified
Final Diagnostic Statement: Not specified
Differential Diagnoses:
• None specified
Clinical Correlation: Not specified

📋 RECOMMENDATIONS & FOLLOW-UP
--------------------------------------------------
Follow-up Recommended: Not specified
Follow-up Modality: Not specified
Follow-up Timing: Not specified
Clinical Correlation Needed: Not specified
Specialist Consultation: Not specified
        """
        return fallback_text.strip()