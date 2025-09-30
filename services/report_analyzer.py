from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re
import logging
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

# Pydantic models for structured extraction
class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema"""
    patient_name: str = Field(..., description="Full name of the patient")
    claim_number: str = Field(..., description="Claim number or case ID. Use 'Not specified' if not found")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    doi: str = Field(..., description="Date of injury in YYYY-MM-DD format")
    status: str = Field(..., description="Current status: normal, urgent, critical, etc.")
    
    # Summary Snapshot fields - POINT FORM (2-3 words)
    diagnosis: str = Field(..., description="Primary diagnosis in 2-3 words")
    key_concern: str = Field(..., description="Main clinical concern in 2-3 words")
    next_step: str = Field(..., description="Recommended next steps in 2-3 words")
    
    # ADL fields - POINT FORM (2-3 words)
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
    
    # Document Summary
    document_type: str = Field(..., description="Type of document")
    summary_points: List[str] = Field(..., description="3-5 key points in 2-3 words each")

class ReportAnalyzer:
    """Service for extracting structured data from medical documents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.1,
            timeout=120
        )
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)

    def create_extraction_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Extract structured information from the following medical document.
        
        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        EXTRACT THE FOLLOWING INFORMATION IN POINT FORM (2-3 WORDS MAX PER FIELD):
        
        - Patient name (full name)
        - Claim number or case ID (look for patterns like WC-, CL-, Case No., Claim #, etc. If not found, use "Not specified")
        - Date of birth (DOB) in YYYY-MM-DD format
        - Date of injury (DOI) in YYYY-MM-DD format  
        - Current status (normal, urgent, critical)
        - Primary diagnosis (2-3 words only)
        - Key clinical concerns (2-3 words only)
        - Recommended next steps (2-3 words only)
        - Activities of daily living affected (2-3 words only)
        - Work restrictions (2-3 words only)
        - Document type
        - 3-5 key summary points (each point 2-3 words only)
        
        CRITICAL INSTRUCTIONS:
        - For diagnosis, key_concern, next_step, adls_affected, work_restrictions - USE ONLY 2-3 WORDS MAX.
        - For summary_points, provide 3-5 bullet points, each 2-3 words.
        - If claim number is not explicitly found in the document, use "Not specified"
        - Do NOT invent or generate claim numbers. Only use what's actually in the document.
        - Look for claim number patterns: WC-2024-001, CL-12345, Case No. 123, Claim # ABC123
        
        If any information is not found in the document, use "Not specified".
        
        {format_instructions}
        
        Return ONLY valid JSON. No additional text.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def extract_claim_number_from_text(self, document_text: str) -> Optional[str]:
        """
        Helper function to extract claim number using regex patterns
        This provides a fallback if the LLM misses it
        """
        try:
            # Common claim number patterns
            patterns = [
                r'WC[-\s]*(\d+[-]\d+)',  # WC-2024-001
                r'CL[-\s]*(\d+[-]?\d*)',  # CL-12345, CL-2024-001
                r'Claim[#\s]*([A-Z0-9-]+)',  # Claim #ABC123, Claim WC-001
                r'Case[#\s\w]*([A-Z0-9-]+)',  # Case No. 123, Case Number WC-001
                r'Claim\s*Number[:\s]*([A-Z0-9-]+)',  # Claim Number: WC-2024-001
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, document_text, re.IGNORECASE)
                if matches:
                    claim_number = matches[0].strip()
                    logger.info(f"🔍 Found claim number via regex: {claim_number}")
                    return claim_number
            
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting claim number via regex: {str(e)}")
            return None

    def extract_document_data(self, document_text: str) -> DocumentAnalysis:
        """Extract structured data from document text"""
        try:
            prompt = self.create_extraction_prompt()
            chain = prompt | self.llm | self.parser
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            result = chain.invoke({
                "document_text": document_text[:15000],
                "current_date": current_date
            })
            
            # If LLM returned "Not specified" for claim number, try regex extraction
            if result.get('claim_number') in ['Not specified', 'not specified', 'Not Specified', None, '']:
                regex_claim = self.extract_claim_number_from_text(document_text)
                if regex_claim:
                    result['claim_number'] = regex_claim
                    logger.info(f"🔄 Updated claim number via regex: {regex_claim}")
            
            logger.info(f"✅ Extracted data: Patient={result['patient_name']}, Claim={result['claim_number']}")
            return DocumentAnalysis(**result)
            
        except Exception as e:
            logger.error(f"❌ Document analysis failed: {str(e)}")
            # Return fallback analysis
            return self.create_fallback_analysis()

    def create_fallback_analysis(self) -> DocumentAnalysis:
        """Create a fallback analysis when extraction fails"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=datetime.now().strftime("%Y-%m-%d"),
            doi=datetime.now().strftime("%Y-%m-%d"),
            status="normal",
            diagnosis="Not specified",
            key_concern="Not specified",
            next_step="Not specified",
            adls_affected="Not specified",
            work_restrictions="Not specified",
            document_type="Medical Document",
            summary_points=["Processing completed", "Analysis unavailable"]
        )

    def compare_with_previous_document(
        self, 
        current_analysis: DocumentAnalysis, 
        previous_document: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Compare current document with previous one to determine what's new
        Returns structured data for the WhatsNew table
        """
        try:
            whats_new = {
                "diagnostic": "No changes",
                "qme": "No changes", 
                "urDecision": "No changes",
                "legal": "No changes"
            }
            
            if not previous_document:
                whats_new = {
                    "diagnostic": "Initial assessment",
                    "qme": "First evaluation",
                    "urDecision": "Case opened",
                    "legal": "New case"
                }
                logger.info("🆕 First document for this patient/claim")
                return whats_new
            
            # If current claim number is "Not specified", we can't reliably compare
            if current_analysis.claim_number == "Not specified":
                logger.warning("⚠️ Cannot compare - claim number not specified")
                return {
                    "diagnostic": "Claim unspecified",
                    "qme": "Claim unspecified",
                    "urDecision": "Claim unspecified",
                    "legal": "Claim unspecified"
                }
            
            # Get previous summary snapshot
            prev_summary = previous_document.get('summarySnapshot')
            prev_diagnosis = prev_summary.get('dx', '') if prev_summary else ''
            prev_concern = prev_summary.get('keyConcern', '') if prev_summary else ''
            prev_next_step = prev_summary.get('nextStep', '') if prev_summary else ''
            
            # Compare diagnostic information (2-3 words)
            if current_analysis.diagnosis != prev_diagnosis and current_analysis.diagnosis != "Not specified":
                whats_new["diagnostic"] = f"DX: {current_analysis.diagnosis}"[:30]
                logger.info(f"🔄 Diagnosis changed: {prev_diagnosis} -> {current_analysis.diagnosis}")
            
            # Compare QME/medical evaluation (2-3 words)
            if current_analysis.key_concern != prev_concern and current_analysis.key_concern != "Not specified":
                whats_new["qme"] = f"Concern: {current_analysis.key_concern}"[:30]
                logger.info(f"🔄 Key concern changed: {prev_concern} -> {current_analysis.key_concern}")
            
            # Compare ADL/work restrictions (2-3 words)
            prev_adl = previous_document.get('adl')
            prev_restrictions = prev_adl.get('workRestrictions', '') if prev_adl else ''
            
            if current_analysis.work_restrictions != prev_restrictions and current_analysis.work_restrictions != "Not specified":
                whats_new["urDecision"] = f"Work: {current_analysis.work_restrictions}"[:30]
                logger.info(f"🔄 Work restrictions changed: {prev_restrictions} -> {current_analysis.work_restrictions}")
            
            # Legal changes (2-3 words)
            if current_analysis.status != previous_document.get('status'):
                whats_new["legal"] = f"Status: {current_analysis.status}"[:20]
                logger.info(f"🔄 Status changed: {previous_document.get('status')} -> {current_analysis.status}")
            
            logger.info("🔄 Document comparison completed")
            return whats_new
            
        except Exception as e:
            logger.error(f"❌ Document comparison failed: {str(e)}")
            return {
                "diagnostic": "Compare error",
                "qme": "Compare error",
                "urDecision": "Compare error", 
                "legal": "Compare error"
            }

    def detect_document_type_preview(self, document_text: str) -> str:
        """Quick document type detection"""
        text_lower = document_text.lower()
        if any(term in text_lower for term in ['mri', 'ct scan', 'x-ray', 'ultrasound', 'mammography', 'radiolog']):
            return "Medical Imaging Report"
        if any(term in text_lower for term in ['lab result', 'pathology', 'blood test', 'urinalysis', 'biopsy']):
            return "Laboratory/Pathology Report"
        if any(term in text_lower for term in ['progress report', 'pr-2', 'follow-up', 'treatment progress']):
            return "Progress Report"
        if any(term in text_lower for term in ['independent medical examination', 'ime', 'medical evaluation']):
            return "Independent Medical Examination"
        if any(term in text_lower for term in ['request for authorization', 'rfa', 'pre-authorization', 'treatment request']):
            return "Request for Authorization (RFA)"
        if any(term in text_lower for term in ['denied', 'denial', 'not authorized', 'coverage denied']):
            return "Denial/Coverage Decision"
        if any(term in text_lower for term in ['ttd', 'temporary total disability', 'work restriction', 'return to work']):
            return "Work Status Document"
        if any(term in text_lower for term in ['legal opinion', 'attorney', 'litigation', 'deposition']):
            return "Legal Document"
        if any(term in text_lower for term in ['patient', 'diagnosis', 'treatment', 'medical', 'physician']):
            return "Medical Report"
        return "Unknown Document Type"