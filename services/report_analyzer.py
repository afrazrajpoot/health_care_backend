"""
Main report analyzer orchestrator with LLM chaining and verification.
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
from utils.document_context_analyzer import DocumentContextAnalyzer
from extractors.qme_extractor import QMEExtractorChained
from extractors.imaging_extractor import ImagingExtractorChained
from extractors.pr2_extractor import PR2ExtractorChained
from extractors.consult_extractor import ConsultExtractorChained
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
        
        # Create a second LLM instance for analysis (can use same or different deployment)
        self.analysis_llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),  # Can use same or different deployment
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.1,  # Slightly higher temperature for analysis
            timeout=120
        )
        
    # No longer need DocumentTypeDetector instance; use detect_document_type function
        self.verifier = ExtractionVerifier(self.llm)
        self.document_analyzer = DocumentContextAnalyzer(self.analysis_llm)
        
        # Initialize specialized extractors with dual LLMs
        self.qme_extractor = QMEExtractorChained(self.llm)
        self.imaging_extractor = ImagingExtractorChained(self.llm)
        self.pr2_extractor = PR2ExtractorChained(self.llm)
        self.consult_extractor = ConsultExtractorChained(self.llm)
        self.simple_extractor = SimpleExtractor(self.llm)
        
        logger.info("✅ ReportAnalyzer initialized with all extractors and dual-LLM support")
    
    def compare_with_previous_documents(
        self, 
        current_raw_text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> List[str]:
        """
        Main extraction pipeline - returns bullet-formatted summary.
        Compatible with existing interface.
        
        Args:
            current_raw_text: Raw document text to extract
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            
        Returns:
            List containing single bullet-formatted summary
        """
        try:
            result = self.extract_document(current_raw_text, page_zones)
            return [f"• {result.summary_line}"]
        except Exception as e:
            logger.error(f"❌ Extraction pipeline failed: {e}")
            fallback_date = datetime.now().strftime("%m/%d/%y")
            return [f"• {fallback_date}: Extraction failed - manual review required"]
    
    def extract_document(self, text: str, page_zones: Optional[Dict] = None) -> ExtractionResult:
        """
        Enhanced pipeline with context analysis
        """
        fallback_date = datetime.now().strftime("%m/%d/%y")
        
        try:
            # Stage 0: Analyze document context
            logger.info("🧠 Stage 0: Analyzing document context...")
            context_analysis = self.document_analyzer.analyze_document_structure(
                text=text,
                doc_type_hint=None
            )
            
            # Stage 1: Detect document type
            detection_result = detect_document_type(text)
            doc_type_str = detection_result.get("doc_type", "Unknown")
            
            # Update context analysis with detected type
            context_analysis["detected_doc_type"] = doc_type_str
            
            logger.info(f"📄 Document type: {doc_type_str}")
            
            # Log context analysis details
            if context_analysis:
                primary_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
                focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
                
                logger.info(f"🎯 Critical sections identified: {focus_sections}")
                logger.info(f"👨‍⚕️ Primary physician: {primary_physician.get('name', 'Unknown')}")
                logger.info(f"   Role: {primary_physician.get('role', 'Unknown')}, Confidence: {primary_physician.get('confidence', 'low')}")
            
            # Convert to DocumentType enum
            doc_type = self._parse_document_type(doc_type_str)
            
            # Stage 2: Context-aware extraction
            result = self._route_to_extractor_with_context(  # ✅ Use the context-aware routing!
                text=text,
                doctype=doc_type,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis  # ✅ Pass context!
            )
            
            # Stage 3: Verify
            final_result = self._verify_result(result, doc_type)
            
            logger.info(f"✅ Extraction complete: {final_result.summary_line}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                document_type="Unknown",
                document_date=fallback_date,
                summary_line=f"{fallback_date}: Extraction failed - manual review required",
                raw_data={}
            )


    def _route_to_extractor_with_context(
        self,
        text: str,
        doctype: DocumentType,
        fallback_date: str,
        page_zones: Optional[Dict],
        context_analysis: Dict  # NEW
    ) -> ExtractionResult:
        """
        Route document to appropriate extractor WITH context analysis.
        
        Args:
            text: Document text
            doctype: Detected document type
            fallback_date: Fallback date if extraction fails
            page_zones: Per-page zone extraction
            context_analysis: Context from DocumentContextAnalyzer (CRITICAL)
        
        Returns:
            ExtractionResult from specialized extractor
        """
        
        # QME/AME/IME - use context-aware extraction
        if doctype in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            logger.info(f"🎯 Routing to QME extractor WITH context analysis")
            return self.qme_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis,  # ✅ FIXED: Pass context!
                raw_text=None
            )
        
        # Imaging reports
        elif doctype in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY, 
                        DocumentType.ULTRASOUND, DocumentType.EMG]:
            logger.info(f"🎯 Routing to Imaging extractor")
            return self.imaging_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                page_zones=page_zones
            )
        
        # Progress reports (PR-2)
        elif doctype == DocumentType.PR2:
            logger.info(f"🎯 Routing to PR-2 extractor")
            return self.pr2_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,  # ✅ Pass context!
                page_zones=page_zones
            )
        
        # Specialist consults
        elif doctype == DocumentType.CONSULT:
            logger.info(f"🎯 Routing to Consult extractor")
            return self.consult_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,  # ✅ Pass context!
                page_zones=page_zones
            )
        
        # All other simple document types
        else:
            logger.info(f"🎯 Routing to Simple extractor for {doctype.value}")
            return self.simple_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,  # ✅ Pass context!
                page_zones=page_zones
            )
    
             
    def _route_to_extractor(self, text: str, doc_type: DocumentType, fallback_date: str, page_zones: Optional[Dict[str, Dict[str, str]]] = None) -> ExtractionResult:
        """
        Route document to appropriate specialized extractor.
        
        Args:
            text: Document text
            doc_type: Detected document type
            fallback_date: Fallback date if extraction fails
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            
        Returns:
            Initial ExtractionResult from specialized extractor
        """
        # Med-Legal reports (QME/AME/IME) - uses chained extractor
        if doc_type in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            return self.qme_extractor.extract(text, doc_type.value, fallback_date, page_zones=page_zones)
        
        # Imaging reports
        elif doc_type in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY,
                         DocumentType.ULTRASOUND, DocumentType.EMG]:
            return self.imaging_extractor.extract(text, doc_type.value, fallback_date, page_zones=page_zones)
        
        # Progress reports
        elif doc_type == DocumentType.PR2:
            return self.pr2_extractor.extract(text, doc_type.value, fallback_date, page_zones=page_zones)
        
        # Specialist consults
        elif doc_type == DocumentType.CONSULT:
            return self.consult_extractor.extract(text, doc_type.value, fallback_date, page_zones=page_zones)
        
        # All other simple document types
        else:
            return self.simple_extractor.extract(text, doc_type.value, fallback_date, page_zones=page_zones)
    
    def _parse_document_type(self, doc_type_str: str) -> DocumentType:
        """
        Parse document type string to DocumentType enum.
        
        Args:
            doc_type_str: Document type string from detection
            
        Returns:
            DocumentType enum value
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
            else:
                logger.warning(f"⚠️ Unknown document type '{doc_type_str}', using UNKNOWN")
                return DocumentType.UNKNOWN
        except Exception as e:
            logger.warning(f"⚠️ Error parsing document type '{doc_type_str}': {e}")
            return DocumentType.UNKNOWN
    
    def _verify_result(self, result: ExtractionResult, doc_type: DocumentType) -> ExtractionResult:
        """
        Verify and correct extraction result if needed.
        
        Note: Skip verification for imaging reports to preserve physician names.
        
        Args:
            result: Initial extraction result
            doc_type: Document type
            
        Returns:
            Verified/corrected ExtractionResult
        """
        # Skip verification for imaging reports to preserve physician names
        if doc_type in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY,
                       DocumentType.ULTRASOUND, DocumentType.EMG]:
            logger.info(f"🛡️ Skipping verification for {doc_type.value} to preserve physician name")
            return result
        
        # QME/AME/IME already verified in their extractor chain
        if doc_type in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            return result
        
        # Verify all other document types
        logger.info(f"🔍 Verifying extraction for {doc_type.value}")
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
            detection_result = detect_document_type(text)
            doc_type_str = detection_result.get("final_doc_type", "Unknown")
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