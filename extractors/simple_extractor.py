"""
Simplified Four-LLM Chain Extractor with Enhanced Detail Extraction
Version: 4.3 - Added dedicated patient detail extraction layer
"""
import logging
import re
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from datetime import datetime

logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Smart four-LLM chain extractor with enhanced patient detail extraction.
    Key Updates in v4.3:
    - Added dedicated patient detail extraction BEFORE long summary
    - Enhanced regex patterns for common medical identifiers
    - Improved fallback mechanisms for critical fields
    - Better handling of complex document structures
    """
    
    def __init__(self, llm: AzureChatOpenAI, mode: str = "general medicine"):
        self.llm = llm
        self.mode = mode.lower()
        self.parser = JsonOutputParser()
        logger.info(f"✅ SimpleExtractor v4.3 initialized with mode: {self.mode}")
    
    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: Optional[str] = None
    ) -> Dict:
        """
        Enhanced extraction with dedicated patient detail capture.
        """
        print(doc_type,'doc type')
        if fallback_date is None:
            fallback_date = datetime.now().strftime("%B %d, %Y")
        
        logger.info(f"🚀 Enhanced Extraction with Detail Capture: {doc_type} (Mode: {self.mode})")
        
        try:
            # STEP 0: FIRST extract critical patient details from full text
            patient_details = self._extract_patient_details(text, doc_type, fallback_date)
            
            # STEP 1: Generate long summary with full context
            long_summary = self._generate_long_summary_direct(text, doc_type, fallback_date, patient_details)
            
            # STEP 2: Generate short summary from long summary
            short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)
            
            logger.info(f"✅ Enhanced extraction completed with patient details captured")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "patient_details": patient_details,  # Include extracted details
                "content_type": "universal",
                "mode": self.mode
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _extract_patient_details(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        DEDICATED extraction of critical patient details from full text.
        This runs FIRST to ensure key details are captured.
        """
        logger.info("🔍 DEDICATED PATIENT DETAIL EXTRACTION from full text...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical document detail extraction specialist. Your SOLE task is to extract CRITICAL PATIENT IDENTIFICATION details.

MANDATORY FIELDS TO EXTRACT:
1. Patient Name (full name)
2. Date of Birth (DOB)
3. Patient ID/Medical Record Number
4. Claim Number (for insurance/worker comp)
5. Date of Service/Report Date
6. Provider/Physician Name
7. Insurance/Carrier Information

CRITICAL RULES:
- EXTRACT ONLY EXPLICITLY STATED INFORMATION
- NO INFERENCES or ASSUMPTIONS
- If field not found, use "Not Found"
- Prefer accuracy over completeness
- Look for common patterns:
  * Names: Title + First + Last (Dr., Mr., Ms., Patient:)
  * DOB: MM/DD/YYYY, DD-MM-YYYY, etc.
  * IDs: Numbers, alphanumeric codes
  * Claim #: "Claim", "Case", "Reference"

OUTPUT FORMAT: JSON only
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}
FALLBACK DATE: {fallback_date}

FULL DOCUMENT TEXT:
{full_text}

Extract ONLY the explicitly stated patient details in JSON format:
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for dedicated patient detail extraction...")
            
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "full_text": text[:8000],  # Use substantial chunk but not necessarily full doc
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            # Parse the JSON response
            detail_text = result.content.strip()
            patient_details = self._parse_detail_response(detail_text)
            
            # Enhanced regex fallback for critical fields
            patient_details = self._enhance_with_regex_fallback(text, patient_details)
            
            logger.info(f"✅ Patient details extracted: {len(patient_details)} fields")
            return patient_details
            
        except Exception as e:
            logger.error(f"❌ Patient detail extraction failed: {e}")
            return self._create_default_patient_details()
    
    def _parse_detail_response(self, detail_text: str) -> Dict:
        """Parse the detail extraction response into structured data."""
        try:
            # Try to extract JSON if wrapped in code blocks
            json_match = re.search(r'\{.*\}', detail_text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: manual parsing
        details = {
            "patient_name": "Not Found",
            "date_of_birth": "Not Found", 
            "patient_id": "Not Found",
            "claim_number": "Not Found",
            "service_date": "Not Found",
            "provider_name": "Not Found",
            "insurance_carrier": "Not Found"
        }
        
        # Simple keyword extraction as fallback
        lines = detail_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'name:' in line_lower and 'patient' in line_lower:
                details["patient_name"] = line.split(':', 1)[1].strip()
            elif 'dob:' in line_lower or 'date of birth:' in line_lower:
                details["date_of_birth"] = line.split(':', 1)[1].strip()
            elif 'claim' in line_lower and 'number' in line_lower:
                details["claim_number"] = line.split(':', 1)[1].strip()
        
        return details
    
    def _enhance_with_regex_fallback(self, text: str, patient_details: Dict) -> Dict:
        """Enhanced regex patterns to find critical patient details."""
        enhanced_details = patient_details.copy()
        
        # Regex patterns for common medical document patterns
        patterns = {
            "patient_name": [
                r'Patient:\s*([A-Za-z\s,]+)',
                r'Patient Name:\s*([A-Za-z\s,]+)',
                r'Name:\s*([A-Za-z\s,]+)',
                r'PATIENT:\s*([A-Za-z\s,]+)'
            ],
            "date_of_birth": [
                r'DOB:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Date of Birth:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Birth Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'DOB\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            "claim_number": [
                r'Claim[\s#]*[:\-]*\s*([A-Z0-9\-]+)',
                r'Claim Number:\s*([A-Z0-9\-]+)',
                r'Case Number:\s*([A-Z0-9\-]+)',
                r'Reference Number:\s*([A-Z0-9\-]+)'
            ],
            "patient_id": [
                r'Patient ID:\s*([A-Z0-9\-]+)',
                r'Medical Record Number:\s*([A-Z0-9\-]+)',
                r'MRN:\s*([A-Z0-9\-]+)',
                r'ID:\s*([A-Z0-9\-]+)'
            ]
        }
        
        # Apply regex patterns
        for field, regex_list in patterns.items():
            if enhanced_details.get(field) in ["Not Found", "", None]:
                for pattern in regex_list:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        enhanced_details[field] = match.group(1).strip()
                        logger.info(f"🔍 Regex found {field}: {enhanced_details[field]}")
                        break
        
        return enhanced_details
    
    def _generate_long_summary_direct(self, text: str, doc_type: str, fallback_date: str, patient_details: Dict) -> str:
        """
        Enhanced long summary generation with pre-extracted patient details.
        """
        logger.info("🔍 Processing ENTIRE document with pre-extracted patient details...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a universal medical document summarization expert analyzing a COMPLETE document.
Mode: {mode}

PRIMARY PURPOSE: Generate a comprehensive, structured long summary.

PRE-EXTRACTED PATIENT DETAILS (use these when available):
{patient_details_json}

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING + PRE-EXTRACTED DETAILS:
You have both the complete document AND pre-verified patient details.

⚠️ CRITICAL RULES:
1. **USE PRE-EXTRACTED DETAILS** when available - they are verified
2. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** for other fields
3. **NO ASSUMPTIONS** - Do not infer or add typical values
4. **EMPTY FIELDS BETTER THAN GUESSES**

Now analyze this COMPLETE document and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}
MODE: {mode}
PRE-EXTRACTED PATIENT DETAILS: 
{patient_details_json}

COMPLETE DOCUMENT TEXT:
{full_document_text}

Generate the long summary in STRUCTURED FORMAT using this template:

📋 DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: [use pre-extracted or extracted or {fallback_date}]
Patient Name: [{patient_name}]
Provider: [{provider_name}]
Patient ID: [{patient_id}]
Claim Number: [{claim_number}]

👤 PATIENT DEMOGRAPHICS (FROM PRE-EXTRACTED)
--------------------------------------------------
Name: {patient_name}
DOB: {date_of_birth}
Patient ID: {patient_id}
Insurance/Carrier: {insurance_carrier}

🚨 CLINICAL FINDINGS & ASSESSMENTS
--------------------------------------------------
[Extract clinical content, diagnoses, findings]

💊 TREATMENT & OBSERVATIONS  
--------------------------------------------------
[Extract medications, procedures, observations]

💼 STATUS & RECOMMENDATIONS
--------------------------------------------------
[Extract work status, recommendations, follow-ups]

⚠️ EXTRACTION PRIORITY:
1. Use PRE-EXTRACTED details when available
2. Extract additional clinical information from full text
3. Omit sections with no data
4. No assumptions or inventions
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for enhanced long summary with patient details...")
            
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "full_document_text": text,
                "doc_type": doc_type,
                "fallback_date": fallback_date,
                "mode": self.mode,
                "patient_details_json": str(patient_details),
                "patient_name": patient_details.get("patient_name", "Not Found"),
                "provider_name": patient_details.get("provider_name", "Not Found"),
                "patient_id": patient_details.get("patient_id", "Not Found"),
                "claim_number": patient_details.get("claim_number", "Not Found"),
                "date_of_birth": patient_details.get("date_of_birth", "Not Found"),
                "insurance_carrier": patient_details.get("insurance_carrier", "Not Found")
            })
            
            long_summary = result.content.strip()
            logger.info(f"✅ Enhanced long summary generated with patient details")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Enhanced long summary generation failed: {e}")
            # Fallback that includes patient details
            return self._create_fallback_long_summary(doc_type, fallback_date, patient_details)
    
    def _create_fallback_long_summary(self, doc_type: str, fallback_date: str, patient_details: Dict) -> str:
        """Create fallback long summary with available patient details."""
        return f"""
📋 DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: {fallback_date}
Patient Name: {patient_details.get('patient_name', 'Not Found')}
Provider: {patient_details.get('provider_name', 'Not Found')}
Patient ID: {patient_details.get('patient_id', 'Not Found')}
Claim Number: {patient_details.get('claim_number', 'Not Found')}

👤 PATIENT DEMOGRAPHICS
--------------------------------------------------
Name: {patient_details.get('patient_name', 'Not Found')}
DOB: {patient_details.get('date_of_birth', 'Not Found')}
Patient ID: {patient_details.get('patient_id', 'Not Found')}
Insurance/Carrier: {patient_details.get('insurance_carrier', 'Not Found')}

Note: Detailed clinical extraction failed, but patient identification details are captured.
"""
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """Generate concise short summary from long summary."""
        # [Keep your existing short summary method unchanged]
        logger.info("🎯 Generating short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited summaries for ANY type of medical document.
Mode: {mode}

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be pipe-delimited with ONLY fields that have actual data.
3. Include patient details if available in the long summary.
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
            
            if word_count > 80:
                words = short_summary.split()
                short_summary = ' '.join(words[:60]) + "..."
            
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return f"Report Title: {doc_type} | Patient Details: Extracted | Mode: {self.mode}"
    
    def _create_default_patient_details(self) -> Dict:
        """Create default patient details structure."""
        return {
            "patient_name": "Not Found",
            "date_of_birth": "Not Found",
            "patient_id": "Not Found", 
            "claim_number": "Not Found",
            "service_date": "Not Found",
            "provider_name": "Not Found",
            "insurance_carrier": "Not Found"
        }
    
    def _clean_short_summary(self, text: str) -> str:
        """Clean short summary text."""
        unwanted_phrases = [
            "unknown", "not specified", "not provided", "none", 
            "no information", "missing", "unavailable", "unspecified",
            "not mentioned", "not available", "not stated"
        ]
        
        cleaned = text
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.strip('"').strip("'")
        
        return cleaned
    
    def _clean_pipes_from_summary(self, short_summary: str) -> str:
        """Clean empty pipes from short summary."""
        if not short_summary or '|' not in short_summary:
            return short_summary
        
        parts = short_summary.split('|')
        cleaned_parts = []
        
        for part in parts:
            stripped_part = part.strip()
            if stripped_part:
                cleaned_parts.append(stripped_part)
        
        cleaned_summary = ' | '.join(cleaned_parts)
        return cleaned_summary
    
    def _create_error_response(self, doc_type: str, error_msg: str, fallback_date: str) -> Dict:
        """Create error response."""
        return {
            "long_summary": f"DOCUMENT SUMMARY - {doc_type.upper()}\n\nDocument processed. Basic information extracted. Error: {error_msg}",
            "short_summary": f"Report Title: {doc_type} | Date: {fallback_date}",
            "patient_details": self._create_default_patient_details(),
            "content_type": "unknown", 
            "mode": self.mode
        }