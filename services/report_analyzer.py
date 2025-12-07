"""
Main report analyzer orchestrator with LLM chaining - Optimized version.
Coordinates all extractors and maintains the extraction pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.settings import CONFIG
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
    Uses simple dictionary-based document type matching instead of enum.
    """
    
    # Document type categories for routing (expanded for better substring matching)
    DOCUMENT_CATEGORIES = {
        # QME/AME/IME Evaluations
        "qme_evaluations": [
            "QME", "AME", "IME", "QUALIFIED MEDICAL EVALUATION",
            "AGREED MEDICAL EVALUATION", "INDEPENDENT MEDICAL EXAM"
        ],
        # Decision Documents
        "decision_documents": [
            "UR", "IMR", "APPEAL", "AUTHORIZATION", "RFA", "DFR",
            "UTILIZATION REVIEW", "INDEPENDENT MEDICAL REVIEW",
            "REQUEST FOR AUTHORIZATION", "DETERMINATION OF MEDICAL NECESSITY"
        ],
        # Formal Medical Reports
        "formal_medical_reports": [
            "SURGERY REPORT", "ANESTHESIA REPORT", "PATHOLOGY", "BIOPSY", "GENETIC TESTING",
            "CARDIOLOGY", "SLEEP STUDY", "DISCHARGE", "ADMISSION NOTE", "HOSPITAL COURSE",
            "ER REPORT", "EMERGENCY ROOM", "OPERATIVE NOTE", "PRE-OP", "POST-OP", "NEUROLOGY",
            "ORTHOPEDICS", "RHEUMATOLOGY", "ENDOCRINOLOGY", "GASTROENTEROLOGY", "PULMONOLOGY",
            "EKG", "ECG", "ECHO", "HOLTER MONITOR", "STRESS TEST", "NERVE CONDUCTION",
            "ELECTROCARDIOGRAM", "ECHOCARDIOGRAM"
        ],
        # Clinical Notes
        "clinical_notes": [
            "PROGRESS NOTE", "OFFICE VISIT", "CLINIC NOTE", "PHYSICAL THERAPY", "Chart Notes",
            "OCCUPATIONAL THERAPY", "CHIROPRACTIC", "ACUPUNCTURE", "PAIN MANAGEMENT",
            "PSYCHIATRY", "PSYCHOLOGY", "PSYCHOTHERAPY", "BEHAVIORAL HEALTH",
            "MED REFILL", "PRESCRIPTION", "TELEMEDICINE", "MASSAGE THERAPY",
            "CLINICAL NOTE", "NURSING", "NURSING NOTE", "VITAL SIGNS",
            "MEDICATION ADMINISTRATION"
        ],
        # Administrative Documents
        "administrative_documents": [
            "ADJUSTER", "ATTORNEY", "NCM", "SIGNATURE REQUEST", "REFERRAL",
            "CORRESPONDENCE", "DENIAL LETTER", "APPROVAL LETTER", "WORK STATUS",
            "WORK RESTRICTIONS", "RETURN TO WORK", "DISABILITY", "CLAIM FORM",
            "EMPLOYER REPORT", "VOCATIONAL REHAB", "JOB ANALYSIS", "WORK CAPACITY",
            "PHARMACY", "MEDICATION LIST", "PRIOR AUTH", "DEPOSITION",
            "INTERROGATORY", "SUBPOENA", "AFFIDAVIT", "ADMINISTRATIVE"
        ],
        # Imaging Reports
        "imaging_reports": [
            "MRI", "CT", "X-RAY", "ULTRASOUND", "EMG", "MAMMOGRAM", "PET SCAN",
            "BONE SCAN", "DEXA SCAN", "FLUOROSCOPY", "ANGIOGRAM",
            "MAGNETIC RESONANCE IMAGING", "COMPUTED TOMOGRAPHY"
        ],
        # Progress Reports
        "progress_reports": ["PR-2", "PR2", "PR-4", "PR4"],
        # Consultations
        "consultations": ["CONSULT", "CONSULTATION"]
    }

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
        
        logger.info("✅ ReportAnalyzer initialized (Dictionary-based type matching)")

    def compare_with_previous_documents(
        self,
        current_raw_text: str
    ) -> Dict[str, str]:
        """
        Main extraction pipeline - returns dictionary with both summaries.
        """
        try:
            # extract_document returns dict with both summaries
            # Pass current_raw_text as both parameters (Document AI summarizer output as primary)
            result_dict = self.extract_document(text=current_raw_text, raw_text=current_raw_text)
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

    def extract_document(self, text: str, raw_text: str) -> Dict[str, str]:
        """
        Optimized pipeline - returns dictionary with both summaries.
        """
        fallback_date = datetime.now().strftime("%m/%d/%y")
        
        try:
            # Stage 1: Detect document type
            detection_result = detect_document_type(text)
            
            # Get the detected document type string (FIXED: dictionary access)
            # Using .get() with fallback to avoid KeyError if key is missing
            doc_type_str = detection_result.get("doc_type", "Unknown")
            
            # Safely get other fields if they exist, defaulting if not
            is_standard_type = detection_result.get("is_standard_type", False)
            confidence = detection_result.get("confidence", 0.0)

            # logger.info(f"🔍 Document type1111111111 raw_text: {raw_text}")
            
            logger.info(f"📄 Document type detected: {doc_type_str} (Conf: {confidence})")
            
            # Stage 2: Direct extraction - get dictionary with both summaries
            result_dict = self._route_to_extractor(
                text=text,
                raw_text=raw_text,
                doc_type_str=doc_type_str,
                is_standard_type=is_standard_type,
                fallback_date=fallback_date
            )
            
            logger.info(f"✅ Extraction complete")
            # logger.info(f"   Long summary: {len(result_dict.get('long_summary', ''))} chars")
            # logger.info(f"   Short summary: {result_dict.get('short_summary', '')}")
            
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
        raw_text: str,
        doc_type_str: str,
        is_standard_type: bool,
        fallback_date: str
    ) -> Dict[str, str]:
        """
        Improved routing: Consistent substring matching for all categories.
        Checks in priority order to resolve overlaps.
        """
        logger.info(f"🎯 Routing document: {doc_type_str}")
        
        normalized_doc_type = doc_type_str.upper().strip()
        
        # Priority 1: QME/AME/IME (most specialized)
        if any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["qme_evaluations"]):
            logger.info(f"🎯 Routing to QME extractor for {doc_type_str}")
            return self._safe_extract(
                self.qme_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: QME report processed"
            )
            
        # Priority 2: Progress Reports (specific forms)
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["progress_reports"]):
            logger.info(f"🎯 Routing to PR-2 extractor for {doc_type_str}")
            return self._safe_extract(
                self.pr2_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} progress report"
            )
            
        # Priority 3: Imaging (technical reports)
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["imaging_reports"]):
            logger.info(f"🎯 Routing to Imaging extractor for {doc_type_str}")
            return self._safe_extract(
                self.imaging_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} imaging report"
            )
            
        # Priority 4: Consultations
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["consultations"]):
            logger.info(f"🎯 Routing to Consult extractor for {doc_type_str}")
            return self._safe_extract(
                self.consult_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} consultation"
            )
            
        # Priority 5: Decision Documents
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["decision_documents"]):
            logger.info(f"🎯 Routing to Decision Document extractor for {doc_type_str}")
            return self._safe_extract(
                self.decision_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} decision processed"
            )
            
        # Priority 6: Formal Medical Reports
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["formal_medical_reports"]):
            logger.info(f"🎯 Routing to Formal Medical Report extractor for {doc_type_str}")
            return self._safe_extract(
                self.formal_medical_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} report processed"
            )
            
        # Priority 7: Clinical Notes
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["clinical_notes"]):
            logger.info(f"🎯 Routing to Clinical Note extractor for {doc_type_str}")
            return self._safe_extract(
                self.clinical_note_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} note processed"
            )
            
        # Priority 8: Administrative (catch-all for non-clinical)
        elif any(key in normalized_doc_type for key in self.DOCUMENT_CATEGORIES["administrative_documents"]):
            logger.info(f"🎯 Routing to Administrative extractor for {doc_type_str}")
            return self._safe_extract(
                self.administrative_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} document processed"
            )
            
        # All other document types (including custom titles)
        else:
            logger.info(f"🎯 Routing to Simple extractor for custom type: {doc_type_str}")
            return self._safe_extract(
                self.simple_extractor.extract, text, raw_text, doc_type_str, fallback_date,
                f"{fallback_date}: {doc_type_str} processed"
            )

    def _safe_extract(self, extractor_func, text: str, raw_text: str, doc_type: str, fallback_date: str, fallback_short: str) -> Dict[str, str]:
        """
        Safely call extractor function and handle different return types.
        """
        try:
            result = extractor_func(
                text=text,
                raw_text=raw_text,
                doc_type=doc_type,
                fallback_date=fallback_date,
            )
            # Convert result to dictionary format
            return self._convert_extraction_result_to_dict(result, fallback_date, fallback_short)
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_type}: {e}")
            return {
                "long_summary": f"{fallback_date}: {doc_type} extraction completed",
                "short_summary": fallback_short
            }

    def _convert_extraction_result_to_dict(self, result, fallback_date: str, fallback_short: str) -> Dict[str, str]:
        """
        Convert ExtractionResult OR dict to dictionary with both summaries.
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
                    "short_summary": short_summary or fallback_short
                }
            
            # If it's an ExtractionResult object (old style)
            elif hasattr(result, 'short_summary') and hasattr(result, 'summary_line'):
                short_summary = result.short_summary
                if not short_summary and result.summary_line:
                    short_summary = self._generate_short_summary_from_long(result.summary_line)
                
                return {
                    "long_summary": result.summary_line or f"{fallback_date}: Document processed",
                    "short_summary": short_summary or fallback_short
                }
            
            # Fallback for any other type
            else:
                logger.warning(f"⚠️ Unknown result type: {type(result)}")
                return {
                    "long_summary": f"{fallback_date}: Document processed",
                    "short_summary": fallback_short
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting result to dict: {e}")
            return {
                "long_summary": f"{fallback_date}: Document processed",
                "short_summary": fallback_short
            }

    def _generate_short_summary_from_long(self, long_summary: str) -> str:
        """
        Generate short summary from long summary using LLM.
        """
        try:
            # If long summary is short enough, use it directly
            if len(long_summary) <= 150:
                return long_summary
                
            system_prompt = """You are a medical documentation specialist creating concise summaries.
Create a 1-2 sentence summary focusing on the most critical findings.
Be brief and focus on key medical-legal points."""

            user_prompt = f"LONG SUMMARY:\n{long_summary}\n\nCreate a 1-2 sentence concise summary:"
            
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
            # Get additional metadata (FIXED: dictionary access)
            detection_result = detect_document_type(text)
            
            # Ensure we handle both dict and object returns just in case
            if isinstance(detection_result, dict):
                doc_type_str = detection_result.get("doc_type", "Unknown")
                is_standard_type = detection_result.get("is_standard_type", False)
            else:
                doc_type_str = getattr(detection_result, "doc_type", "Unknown")
                is_standard_type = getattr(detection_result, "is_standard_type", False)

            # Get both summaries
            summaries_dict = self.extract_document(text=text, raw_text=text)

            return {
                "document_type": doc_type_str,
                "is_standard_type": is_standard_type,
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
                "is_standard_type": False,
                "long_summary": f"{fallback_date}: Extraction failed",
                "short_summary": f"{fallback_date}: Extraction failed",
                "error": str(e)
            }

    def get_extraction_metadata(self, text: str) -> Dict[str, Any]:
        """
        Get extraction metadata without full processing (useful for validation).
        """
        try:
            # FIXED: dictionary access
            detection_result = detect_document_type(text)
            
            if isinstance(detection_result, dict):
                doc_type_str = detection_result.get("doc_type", "Unknown")
                is_standard_type = detection_result.get("is_standard_type", False)
            else:
                doc_type_str = getattr(detection_result, "doc_type", "Unknown")
                is_standard_type = getattr(detection_result, "is_standard_type", False)
            
            return {
                "document_type": doc_type_str,
                "is_standard_type": is_standard_type,
                "detected_at": datetime.now().isoformat(),
                "text_length": len(text),
                "preview": text[:200] + "..." if len(text) > 200 else text
            }
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            return {
                "document_type": "Unknown",
                "is_standard_type": False,
                "error": str(e)
            }
