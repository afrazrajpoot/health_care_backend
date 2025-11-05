"""
Main report analyzer orchestrator with LLM chaining and verification.
Coordinates all extractors and maintains the extraction pipeline.
"""
import logging
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict
from langchain_openai import AzureChatOpenAI

from config.settings import CONFIG
from models.data_models import DocumentType, ExtractionResult
from utils.document_detector import DocumentTypeDetector
from utils.extraction_verifier import ExtractionVerifier
from extractors.qme_extractor import QMEExtractorChained
from extractors.imaging_extractor import ImagingExtractor
from extractors.pr2_extractor import PR2Extractor
from extractors.consult_extractor import ConsultExtractor
from extractors.simple_extractor import SimpleExtractor

logger = logging.getLogger("document_ai")


class ReportAnalyzer:
    """
    Enhanced orchestrator with LLM chaining and verification.
    Maintains backward compatibility while adding robustness.
    """
    
    def __init__(self):
        """Initialize LLM and all extraction components"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )
        
        # Initialize components
        self.detector = DocumentTypeDetector(self.llm)
        self.verifier = ExtractionVerifier(self.llm)
        
        # Initialize specialized extractors
        self.qme_extractor = QMEExtractorChained(self.llm)
        self.imaging_extractor = ImagingExtractor(self.llm)
        self.pr2_extractor = PR2Extractor(self.llm)
        self.consult_extractor = ConsultExtractor(self.llm)
        self.simple_extractor = SimpleExtractor(self.llm)
        
        logger.info("✅ ReportAnalyzer initialized with all extractors")
    
    def compare_with_previous_documents(self, current_raw_text: str) -> List[str]:
        """
        Main extraction pipeline - returns bullet-formatted summary.
        Compatible with existing interface.
        
        Args:
            current_raw_text: Raw document text to extract
            
        Returns:
            List containing single bullet-formatted summary
        """
        try:
            result = self.extract_document(current_raw_text)
            return [f"• {result.summary_line}"]
        except Exception as e:
            logger.error(f"❌ Extraction pipeline failed: {e}")
            fallback_date = datetime.now().strftime("%m/%d/%y")
            return [f"• {fallback_date}: Extraction failed - manual review required"]
    
    def extract_document(self, text: str) -> ExtractionResult:
        """
        Full extraction pipeline with chaining and verification.
        
        Pipeline stages:
        1. Document type detection (pattern + LLM)
        2. Specialized extraction (type-specific extractors)
        3. Verification and correction (format validation)
        
        Args:
            text: Raw document text
            
        Returns:
            ExtractionResult with structured data and summary
        """
        fallback_date = datetime.now().strftime("%m/%d/%y")
        
        try:
            # Stage 1: Detect document type
            doc_type = self.detector.detect(text)
            logger.info(f"📄 Detected document type: {doc_type.value}")
            
            # Stage 2: Route to specialized extractor
            result = self._route_to_extractor(text, doc_type, fallback_date)
            
            # Stage 3: Verify and correct (if needed)
            final_result = self._verify_result(result, doc_type)
            
            logger.info(f"✅ Extraction complete: {final_result.summary_line}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return ExtractionResult(
                document_type="Unknown",
                document_date=fallback_date,
                summary_line=f"{fallback_date}: Extraction failed - manual review required",
                raw_data={}
            )
    
    def _route_to_extractor(self, text: str, doc_type: DocumentType, fallback_date: str) -> ExtractionResult:
        """
        Route document to appropriate specialized extractor.
        
        Args:
            text: Document text
            doc_type: Detected document type
            fallback_date: Fallback date if extraction fails
            
        Returns:
            Initial ExtractionResult from specialized extractor
        """
        # Med-Legal reports (QME/AME/IME) - uses chained extractor
        if doc_type in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            return self.qme_extractor.extract(text, doc_type.value, fallback_date)
        
        # Imaging reports
        elif doc_type in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY,
                         DocumentType.ULTRASOUND, DocumentType.EMG]:
            return self.imaging_extractor.extract(text, doc_type.value, fallback_date)
        
        # Progress reports
        elif doc_type == DocumentType.PR2:
            return self.pr2_extractor.extract(text, doc_type.value, fallback_date)
        
        # Specialist consults
        elif doc_type == DocumentType.CONSULT:
            return self.consult_extractor.extract(text, doc_type.value, fallback_date)
        
        # All other simple document types
        else:
            return self.simple_extractor.extract(text, doc_type.value, fallback_date)
    
    def _verify_result(self, result: ExtractionResult, doc_type: DocumentType) -> ExtractionResult:
        """
        Verify and correct extraction result if needed.
        
        Note: QME extractor already includes verification in its chain,
        so we skip re-verification for those.
        
        Args:
            result: Initial extraction result
            doc_type: Document type
            
        Returns:
            Verified/corrected ExtractionResult
        """
        # QME/AME/IME already verified in their extractor chain
        if doc_type in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            return result
        
        # Verify all other document types
        return self.verifier.verify_and_fix(result)
    
    def format_whats_new_as_highlights(self, bullet_points: List[str]) -> List[str]:
        """
        Format bullet points as highlights (for backward compatibility).
        
        Args:
            bullet_points: List of bullet-formatted summaries
            
        Returns:
            Same list or default message if empty
        """
        return bullet_points if bullet_points else [
            "• No significant new findings identified in current document"
        ]
    
    def get_structured_extraction(self, text: str) -> Dict[str, Any]:
        """
        Get full structured extraction as dictionary (for API/storage).
        
        Args:
            text: Document text to extract
            
        Returns:
            Dictionary with all extraction fields
        """
        result = self.extract_document(text)
        return asdict(result)
    
    def get_extraction_metadata(self, text: str) -> Dict[str, Any]:
        """
        Get extraction metadata without full processing (useful for validation).
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document type and basic metadata
        """
        try:
            doc_type = self.detector.detect(text)
            return {
                "document_type": doc_type.value,
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
