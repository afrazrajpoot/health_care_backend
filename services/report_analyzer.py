"""
Main report analyzer orchestrator with LLM chaining - Optimized version.
Coordinates all extractors and maintains the extraction pipeline.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict
from langchain_openai import AzureChatOpenAI

from config.settings import CONFIG
from models.data_models import DocumentType, ExtractionResult
from utils.document_detector import detect_document_type
from utils.extraction_verifier import ExtractionVerifier
from extractors.qme_extractor import QMEExtractorChained
from extractors.imaging_extractor import ImagingExtractorChained
from extractors.pr2_extractor import PR2ExtractorChained
from extractors.consult_extractor import ConsultExtractorChained
from extractors.simple_extractor import SimpleExtractor
from extractors.ur_extractor import DecisionDocumentExtractor
from extractors.formal_medical_extractor import FormalMedicalReportExtractor
from extractors.clinical_extractor import ClinicalNoteExtractor
from extractors.administritive_extractor import AdministrativeExtractor

logger = logging.getLogger("document_ai")


class ReportAnalyzer:
    """
    Optimized orchestrator with LLM chaining.
    Removed page zones and document context analyzer for better performance.
    Uses raw text directly for all extractions.
    """
    
    def __init__(self, mode):
        """Initialize LLM and all extraction components"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120,
        )
        self.mode = mode
        
        # Create a second LLM instance for analysis
        self.analysis_llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.1,
            timeout=120
        )
        
        self.verifier = ExtractionVerifier(self.llm)
        
        # Initialize specialized extractors with dual LLMs
        self.qme_extractor = QMEExtractorChained(self.llm, mode=mode)
        self.imaging_extractor = ImagingExtractorChained(self.llm)
        self.pr2_extractor = PR2ExtractorChained(self.llm)
        self.consult_extractor = ConsultExtractorChained(self.llm)
        self.simple_extractor = SimpleExtractor(self.llm)
        self.decision_extractor = DecisionDocumentExtractor(self.llm)
        self.formal_medical_extractor = FormalMedicalReportExtractor(self.llm)
        self.clinical_note_extractor = ClinicalNoteExtractor(self.llm)
        self.administrative_extractor = AdministrativeExtractor(self.llm)
        
        logger.info("✅ ReportAnalyzer initialized (Optimized - no page zones or context analyzer)")
    
    def compare_with_previous_documents(
        self, 
        current_raw_text: str,
        doc_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Main extraction pipeline - returns dictionary with both summaries.
        
        Args:
            current_raw_text: Raw document text to extract
            doc_type: Optional document type if already detected
            
        Returns:
            Dict with 'long_summary' and 'short_summary'
        """
        try:
            # extract_document returns dict with both summaries
            result_dict = self.extract_document(current_raw_text, doc_type)
            
            # Return the dictionary directly (no bullet points)
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Extraction pipeline failed: {e}")
            fallback_date = datetime.now().strftime("%m/%d/%y")
            error_msg = f"{fallback_date}: Extraction failed - manual review required"
            return {
                "long_summary": error_msg,
                "short_summary": error_msg
            }
    
    def extract_document(self, text: str, doc_type: Optional[str] = None) -> Dict[str, str]:
        """
        Optimized pipeline - returns dictionary with both summaries.
        
        Args:
            text: Raw document text to extract
            doc_type: Optional document type if already detected
            
        Returns:
            Dict with 'long_summary' and 'short_summary'
        """
        fallback_date = datetime.now().strftime("%m/%d/%y")
        
        try:
            # Stage 1: Detect document type (simplified)
            if doc_type:
                doc_type_str = doc_type
                logger.info(f"📄 Document type provided: {doc_type_str}")
            else:
                detection_result = detect_document_type(text)
                doc_type_str = detection_result.get("doc_type", "Unknown")
            
            logger.info(f"📄 Document type: {doc_type_str}")
            
            # Convert to DocumentType enum
            doc_type = self._parse_document_type(doc_type_str)
            
            # Stage 2: Direct extraction - get dictionary with both summaries
            result_dict = self._route_to_extractor(
                text=text,
                doctype=doc_type,
                fallback_date=fallback_date
            )
            
            logger.info(f"✅ Extraction complete")
            logger.info(f"   Long summary: {len(result_dict.get('long_summary', ''))} chars")
            logger.info(f"   Short summary: {result_dict.get('short_summary', '')}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}", exc_info=True)
            error_msg = f"{fallback_date}: Extraction failed - manual review required"
            return {
                "long_summary": error_msg,
                "short_summary": error_msg
            }

    def _route_to_extractor(
        self,
        text: str,
        doctype: DocumentType,
        fallback_date: str
    ) -> Dict[str, str]:
        """
        Route document to appropriate extractor - simplified version.
        Returns dictionary with both summaries.
        Uses raw text directly.
        """
        
        # QME/AME/IME - use context-aware extraction (already returns dict)
        if doctype in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            logger.info(f"🎯 Routing to QME extractor")
            qme_result = self.qme_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            if isinstance(qme_result, dict):
                return qme_result
            else:
                return {
                    "long_summary": str(qme_result) if qme_result else f"{fallback_date}: QME extraction completed",
                    "short_summary": f"{fallback_date}: QME report processed"
                }
        
        # Decision Documents - UR/IMR, Appeals, Authorizations, RFA, DFR
        elif doctype in [DocumentType.UR, DocumentType.IMR, DocumentType.APPEAL, 
                        DocumentType.AUTHORIZATION, DocumentType.RFA, DocumentType.DFR]:
            logger.info(f"🎯 Routing to Decision Document extractor for {doctype.value}")
            decision_result = self.decision_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            if isinstance(decision_result, dict):
                return decision_result
            else:
                return {
                    "long_summary": str(decision_result) if decision_result else f"{fallback_date}: {doctype.value} extraction completed",
                    "short_summary": f"{fallback_date}: {doctype.value} decision processed"
                }

        # Formal Medical Reports
        elif doctype in [DocumentType.SURGERY_REPORT, DocumentType.ANESTHESIA_REPORT, 
                        DocumentType.PATHOLOGY, DocumentType.BIOPSY, DocumentType.GENETIC_TESTING,
                        DocumentType.CARDIOLOGY, DocumentType.SLEEP_STUDY, DocumentType.DISCHARGE,
                        DocumentType.ADMISSION_NOTE, DocumentType.HOSPITAL_COURSE, DocumentType.ER_REPORT,
                        DocumentType.EMERGENCY_ROOM, DocumentType.OPERATIVE_NOTE, DocumentType.PRE_OP,
                        DocumentType.POST_OP, DocumentType.NEUROLOGY, DocumentType.ORTHOPEDICS,
                        DocumentType.RHEUMATOLOGY, DocumentType.ENDOCRINOLOGY, DocumentType.GASTROENTEROLOGY,
                        DocumentType.PULMONOLOGY, DocumentType.EKG, DocumentType.ECG, DocumentType.ECHO,
                        DocumentType.HOLTER_MONITOR, DocumentType.STRESS_TEST, DocumentType.NERVE_CONDUCTION]:
            logger.info(f"🎯 Routing to Formal Medical Report extractor for {doctype.value}")
            formal_medical_result = self.formal_medical_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            if isinstance(formal_medical_result, dict):
                return formal_medical_result
            else:
                return {
                    "long_summary": str(formal_medical_result) if formal_medical_result else f"{fallback_date}: {doctype.value} extraction completed",
                    "short_summary": f"{fallback_date}: {doctype.value} report processed"
                }

        # Clinical Notes
        elif doctype in [DocumentType.PROGRESS_NOTE, DocumentType.OFFICE_VISIT, DocumentType.CLINIC_NOTE,
                        DocumentType.PHYSICAL_THERAPY, DocumentType.OCCUPATIONAL_THERAPY, DocumentType.CHIROPRACTIC,
                        DocumentType.ACUPUNCTURE, DocumentType.PAIN_MANAGEMENT, DocumentType.PSYCHIATRY,
                        DocumentType.PSYCHOLOGY, DocumentType.PSYCHOTHERAPY, DocumentType.BEHAVIORAL_HEALTH,
                        DocumentType.MED_REFILL, DocumentType.PRESCRIPTION, DocumentType.TELEMEDICINE,
                        DocumentType.MASSAGE_THERAPY, DocumentType.CLINICAL_NOTE, DocumentType.NURSING,
                        DocumentType.NURSING_NOTE, DocumentType.VITAL_SIGNS, DocumentType.MEDICATION_ADMINISTRATION]:
            logger.info(f"🎯 Routing to Clinical Note extractor for {doctype.value}")
            clinical_note_result = self.clinical_note_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            if isinstance(clinical_note_result, dict):
                return clinical_note_result
            else:
                return {
                    "long_summary": str(clinical_note_result) if clinical_note_result else f"{fallback_date}: {doctype.value} extraction completed",
                    "short_summary": f"{fallback_date}: {doctype.value} note processed"
                }

        # Administrative Documents
        elif doctype in [DocumentType.ADJUSTER, DocumentType.ATTORNEY, DocumentType.NCM,
                        DocumentType.SIGNATURE_REQUEST, DocumentType.REFERRAL, DocumentType.CORRESPONDENCE,
                        DocumentType.DENIAL_LETTER, DocumentType.APPROVAL_LETTER, DocumentType.WORK_STATUS,
                        DocumentType.WORK_RESTRICTIONS, DocumentType.RETURN_TO_WORK, DocumentType.DISABILITY,
                        DocumentType.CLAIM_FORM, DocumentType.EMPLOYER_REPORT, DocumentType.VOCATIONAL_REHAB,
                        DocumentType.JOB_ANALYSIS, DocumentType.WORK_CAPACITY, DocumentType.PHARMACY,
                        DocumentType.MEDICATION_LIST, DocumentType.PRIOR_AUTH, DocumentType.DEPOSITION,
                        DocumentType.INTERROGATORY, DocumentType.SUBPOENA, DocumentType.AFFIDAVIT,
                        DocumentType.ADMINISTRATIVE]:
            logger.info(f"🎯 Routing to Administrative extractor for {doctype.value}")
            administrative_result = self.administrative_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            if isinstance(administrative_result, dict):
                return administrative_result
            else:
                return {
                    "long_summary": str(administrative_result) if administrative_result else f"{fallback_date}: {doctype.value} extraction completed",
                    "short_summary": f"{fallback_date}: {doctype.value} document processed"
                }
        
        # Imaging reports - convert ExtractionResult to dict
        elif doctype in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY, 
                        DocumentType.ULTRASOUND, DocumentType.EMG, DocumentType.MAMMOGRAM,
                        DocumentType.PET_SCAN, DocumentType.BONE_SCAN, DocumentType.DEXA_SCAN,
                        DocumentType.FLUOROSCOPY, DocumentType.ANGIOGRAM]:
            logger.info(f"🎯 Routing to Imaging extractor")
            imaging_result = self.imaging_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            return self._convert_extraction_result_to_dict(imaging_result, fallback_date)
        
        # Progress reports (PR-2) - convert ExtractionResult to dict
        elif doctype == DocumentType.PR2:
            logger.info(f"🎯 Routing to PR-2 extractor")
            pr2_result = self.pr2_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            return self._convert_extraction_result_to_dict(pr2_result, fallback_date)
        
        # Specialist consults - convert ExtractionResult to dict
        elif doctype == DocumentType.CONSULT:
            logger.info(f"🎯 Routing to Consult extractor")
            consult_result = self.consult_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                raw_text=text  # Explicitly pass raw text
            )
            return self._convert_extraction_result_to_dict(consult_result, fallback_date)
        
        # All other simple document types - convert ExtractionResult to dict
        else:
            logger.info(f"🎯 Routing to Simple extractor for {doctype.value}")
            simple_result = self.simple_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                # raw_text=text  # Explicitly pass raw text
            )
            return self._convert_extraction_result_to_dict(simple_result, fallback_date)

    def _convert_extraction_result_to_dict(self, result, fallback_date: str) -> Dict[str, str]:
        """
        Convert ExtractionResult OR dict to dictionary with both summaries.
        Handles both old ExtractionResult objects and new dict returns.
        """
        try:
            # If it's already a dict (from new extractors), return it directly
            if isinstance(result, dict):
                long_summary = result.get("long_summary", "")
                short_summary = result.get("short_summary", "")
                
                # If short_summary is missing but long_summary exists, generate one
                if not short_summary and long_summary:
                    short_summary = self._generate_short_summary_from_long(long_summary)
                
                return {
                    "long_summary": long_summary or f"{fallback_date}: Document processed",
                    "short_summary": short_summary or f"{fallback_date}: Report processed"
                }
            
            # If it's an ExtractionResult object (old style)
            elif hasattr(result, 'short_summary') and hasattr(result, 'summary_line'):
                short_summary = result.short_summary
                if not short_summary and result.summary_line:
                    short_summary = self._generate_short_summary_from_long(result.summary_line)
                
                return {
                    "long_summary": result.summary_line or f"{fallback_date}: Document processed",
                    "short_summary": short_summary or f"{fallback_date}: {getattr(result, 'document_type', 'Unknown')} report"
                }
            
            # Fallback for any other type
            else:
                logger.warning(f"⚠️ Unknown result type: {type(result)}")
                return {
                    "long_summary": f"{fallback_date}: Document processed",
                    "short_summary": f"{fallback_date}: Report processed"
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting result to dict: {e}")
            return {
                "long_summary": f"{fallback_date}: Document processed",
                "short_summary": f"{fallback_date}: Report processed"
            }

    def _generate_short_summary_from_long(self, long_summary: str) -> str:
        """
        Generate short summary from long summary using LLM.
        """
        try:
            system_prompt = """You are a medical documentation specialist creating concise summaries.
            Create a 1-2 sentence summary focusing on the most critical findings.
            Be brief and focus on key medical-legal points."""
            
            user_prompt = f"LONG SUMMARY:\n{long_summary}\n\nCreate a 1-2 sentence concise summary:"
            
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            
            system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
            human_msg = HumanMessagePromptTemplate.from_template(user_prompt)
            chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            # Fallback: take first 100 chars or meaningful truncation
            if len(long_summary) > 100:
                return long_summary[:97] + "..."
            return long_summary

    def _parse_document_type(self, doc_type_str: str) -> DocumentType:
        """
        Parse document type string to DocumentType enum.
        """
        try:
            # Normalize the string to match enum (e.g., "X-ray" -> "XRAY", "PR-2" -> "PR2")
            normalized = doc_type_str.upper().replace("-", "").replace(" ", "_")
            
            # Try direct match first
            if normalized in DocumentType.__members__:
                return DocumentType[normalized]
            
            # Handle special cases
            if doc_type_str == "X-ray":
                return DocumentType.XRAY
            elif doc_type_str in ["PR-2", "PR2"]:
                return DocumentType.PR2
            elif doc_type_str in ["PR-4", "PR4"]:
                return DocumentType.PR4
            elif doc_type_str in ["UR", "Utilization Review"]:
                return DocumentType.UR
            elif doc_type_str in ["IMR", "Independent Medical Review"]:
                return DocumentType.IMR
            elif doc_type_str == "Appeal":
                return DocumentType.APPEAL
            elif doc_type_str in ["Authorization", "Treatment Authorization"]:
                return DocumentType.AUTHORIZATION
            elif doc_type_str in ["RFA", "Request for Authorization"]:
                return DocumentType.RFA
            elif doc_type_str in ["DFR", "Doctor First Report"]:
                return DocumentType.DFR
            else:
                logger.warning(f"⚠️ Unknown document type '{doc_type_str}', using UNKNOWN")
                return DocumentType.UNKNOWN
        except Exception as e:
            logger.warning(f"⚠️ Error parsing document type '{doc_type_str}': {e}")
            return DocumentType.UNKNOWN

    def format_whats_new_as_highlights(self, bullet_points: List[str]) -> List[str]:
        """
        Format bullet points as highlights (for backward compatibility).
        """
        return bullet_points if bullet_points else [
            "• No significant new findings identified in current document"
        ]
    
    def get_structured_extraction(self, text: str) -> Dict[str, Any]:
        """
        Get full structured extraction as dictionary (for API/storage).
        """
        try:
            # Get both summaries
            summaries_dict = self.extract_document(text)
            
            # Get additional metadata
            detection_result = detect_document_type(text)
            doc_type_str = detection_result.get("doc_type", "Unknown")
            
            return {
                "document_type": doc_type_str,
                "long_summary": summaries_dict.get("long_summary", ""),
                "short_summary": summaries_dict.get("short_summary", ""),
                "extracted_at": datetime.now().isoformat(),
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"❌ Structured extraction failed: {e}")
            fallback_date = datetime.now().strftime("%m/%d/%y")
            return {
                "document_type": "Unknown",
                "long_summary": f"{fallback_date}: Extraction failed",
                "short_summary": f"{fallback_date}: Extraction failed",
                "error": str(e)
            }
    
    def get_extraction_metadata(self, text: str) -> Dict[str, Any]:
        """
        Get extraction metadata without full processing (useful for validation).
        """
        try:
            detection_result = detect_document_type(text)
            doc_type_str = detection_result.get("doc_type", "Unknown")
            return {
                "document_type": doc_type_str,
                "detected_at": datetime.now().isoformat(),
                "text_length": len(text),
                "preview": text[:200] + "..." if len(text) > 200 else text
            }
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            return {
                "document_type": "Unknown",
                "error": str(e)
            }