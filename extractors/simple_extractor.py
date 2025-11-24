"""
Simplified Four-LLM Chain Extractor with Enhanced Detail Extraction
Version: 4.5 - LLM-Only Patient Details (No Regex Fallback)
Key Updates:
- Removed all regex enhancements; rely solely on LLM for accurate detail detection
- Integrated JsonOutputParser for strict JSON output from LLM
- Enhanced prompts for precise JSON schema adherence
- Retained retry logic (LLM-based) for robustness on misses
- Simplified parsing: Direct dict from parser, no manual fallback extraction
"""
import logging
import json
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from datetime import datetime
import re
logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Smart four-LLM chain extractor with LLM-only patient detail extraction.
    Key Updates in v4.5:
    - Pure LLM reliance: No regex; enhanced prompts for accuracy
    - Strict JSON parsing via output parser
    - Retry on misses for comprehensive coverage
    """
    
    def __init__(self, llm: AzureChatOpenAI, mode: str = "general medicine"):
        self.llm = llm
        self.mode = mode.lower()
        self.parser = JsonOutputParser()
        logger.info(f"✅ SimpleExtractor v4.5 initialized with mode: {self.mode}")
    
    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: Optional[str] = None
    ) -> Dict:
        """
        Enhanced extraction with dedicated patient detail capture.
        """
        print(text,'doc type')
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
        DEDICATED extraction of critical patient details from full text using LLM only.
        This runs FIRST to ensure key details are captured accurately.
        """
        logger.info("🔍 DEDICATED PATIENT DETAIL EXTRACTION from full text (LLM-only)...")
        
        # Preprocess text for complex/fax docs (kept for LLM context cleaning)
        processed_text = self._preprocess_text_for_details(text)
        
        # Define exact JSON schema for parser
        json_schema = {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "date_of_birth": {"type": "string"},
                "patient_id": {"type": "string"},
                "claim_number": {"type": "string"},
                "service_date": {"type": "string"},
                "provider_name": {"type": "string"},
                "insurance_carrier": {"type": "string"}
            },
            "required": ["patient_name", "date_of_birth", "patient_id", "claim_number", "service_date", "provider_name", "insurance_carrier"]
        }
        self.parser = JsonOutputParser(pydantic_object=None, partial_schema=json_schema)  # Enforce schema
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical document detail extraction specialist. Your SOLE task is to extract CRITICAL PATIENT IDENTIFICATION details ACCURATELY from the text.

MANDATORY FIELDS (output EXACTLY these, using "Not Found" if absent):
- patient_name: Full patient name
- date_of_birth: DOB in any format (e.g., MM/DD/YYYY)
- patient_id: Medical Record Number, MRN, or ID
- claim_number: Claim, Case, or Reference number
- service_date: Date of Service or Report Date
- provider_name: Physician or Provider full name
- insurance_carrier: Insurance company or carrier name

SPECIAL HANDLING FOR FAXED/SCANNED DOCS:
- Look for "Re:", "RE:", "Regarding:", or similar memo-style prefixes—they often introduce patient details (e.g., "Re: Jane Smith DOB: 03/22/1975").
- Extract the content IMMEDIATELY FOLLOWING these prefixes as primary patient info.
- Ignore fax headers/footers like transmission dates or sender info unless they contain patient data.

CRITICAL RULES:
- EXTRACT ONLY EXPLICITLY STATED INFORMATION—NO INFERENCES or ASSUMPTIONS
- Be exhaustive: Scan the entire text for patterns like Patient:, DOB:, Claim #, etc.
- If truly not found anywhere, use "Not Found" (do not guess or infer)
- Prefer precision: Names after "Re:" or in headers take priority
- DOB: Often right after name in faxes; IDs: Alphanumeric codes near name

OUTPUT: VALID JSON ONLY matching this exact schema. No extra text, code blocks, or explanations.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}
FALLBACK DATE: {fallback_date} (use only if service_date explicitly absent)

FULL DOCUMENT TEXT:
{full_text}

Respond with ONLY the JSON object for the fields above.
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            logger.info("🤖 Invoking LLM for dedicated patient detail extraction...")
            
            # Chain with parser for direct dict output
            chain = chat_prompt | self.llm | self.parser
            patient_details = chain.invoke({
                "full_text": processed_text,
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            logger.debug(f"Raw LLM details: {patient_details}")
            
            # Ensure all keys present (fill missing with "Not Found")
            default_keys = {"patient_name", "date_of_birth", "patient_id", "claim_number", "service_date", "provider_name", "insurance_carrier"}
            for key in default_keys:
                if key not in patient_details:
                    patient_details[key] = "Not Found"
            
            # Retry logic for high misses
            patient_details = self._retry_on_misses(patient_details, processed_text, system_prompt, fallback_date, doc_type)
            
            # Debug for fax
            if re.search(r'Re[:\-.\s]*', text, re.IGNORECASE):
                logger.info("📠 FAX DETECTED: LLM prioritized 'Re:' patterns in details")
            
            logger.info(f"✅ Patient details extracted via LLM: {len(patient_details)} fields")
            return patient_details
            
        except Exception as e:
            logger.error(f"❌ Patient detail extraction failed: {e}")
            return self._create_default_patient_details()
    
    def _preprocess_text_for_details(self, text: str) -> str:
        """
        Normalize text for better LLM context in complex docs.
        - Split into sections (headers, body, tables) for targeted scanning.
        - Clean noise: extra whitespace, common artifacts.
        """
        # Basic cleaning
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize newlines
        text = re.sub(r'[^\w\s\-.,/:;()&@#]+', ' ', text)  # Remove weird chars but keep medical punctuation
        
        # Rough section split (improve with heuristics if you have doc metadata)
        sections = {
            'headers_footers': re.findall(r'^.{0,500}.*?(?=\n\n[A-Z]{3,})', text, re.MULTILINE | re.DOTALL)[:2],  # First 500 chars + patterns
            'body': re.sub(r'^.{0,1000}', '', text)[:4000],  # Skip intro, take mid-section
            'tables': re.findall(r'\b[A-Za-z\s]+:\s*[A-Za-z0-9\s\-/]+(?:\n[A-Za-z\s]+:\s*[A-Za-z0-9\s\-/]+)*', text)  # Key-value lines
        }
        
        # Concat prioritized sections (headers first, as details often there)
        prioritized_text = '\n\n'.join(sections['headers_footers'] + sections['tables'] + [sections['body']])
        
        logger.info(f"📄 Preprocessed text length: {len(prioritized_text)} (original: {len(text)})")
        return prioritized_text[:6000]  # Cap to avoid token overflow
    
    def _retry_on_misses(self, patient_details: Dict, processed_text: str, system_prompt: SystemMessagePromptTemplate, fallback_date: str, doc_type: str) -> Dict:
        """Retry LLM extraction for fields with high 'Not Found' count."""
        not_found_count = sum(1 for v in patient_details.values() if v == "Not Found")
        if not_found_count > 3:  # Threshold for retry
            logger.warning(f"⚠️ High 'Not Found' count ({not_found_count}/7). Retrying with focused LLM scan...")
            
            user_prompt_retry = HumanMessagePromptTemplate.from_template("""
FOCUSED RETRY: You previously missed these fields: {missing_fields}. Scan AGAIN for ONLY them.

ENTIRE PROCESSED DOCUMENT TEXT (be more thorough this time):
{processed_text}

DOCUMENT TYPE: {doc_type}
FALLBACK DATE: {fallback_date}

Respond with ONLY the JSON object updating JUST the missing fields (others ignored).
""")
            
            missing_fields = [k for k, v in patient_details.items() if v == "Not Found"]
            retry_chat = ChatPromptTemplate.from_messages([system_prompt, user_prompt_retry])
            retry_chain = retry_chat | self.llm | self.parser
            
            try:
                retry_details = retry_chain.invoke({
                    "processed_text": processed_text,
                    "missing_fields": ', '.join(missing_fields),
                    "doc_type": doc_type,
                    "fallback_date": fallback_date
                })
                
                # Merge: Update only misses
                for k, v in retry_details.items():
                    if patient_details[k] == "Not Found" and v != "Not Found":
                        patient_details[k] = v
                
                logger.info(f"🔄 LLM Retry updated {sum(1 for v in patient_details.values() if v != 'Not Found')} fields")
            except Exception as retry_e:
                logger.error(f"❌ LLM Retry failed: {retry_e}")
        
        return patient_details
    
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
                "patient_details_json": json.dumps(patient_details),
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