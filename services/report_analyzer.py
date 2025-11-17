"""
Main report analyzer orchestrator with LLM chaining and verification.
Coordinates all extractors and maintains the extraction pipeline.
"""
import logging
from typing import List, Dict, Any, Optional, Union
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
    
    UPDATED: Returns ONLY dual summaries {"summary": {"long": "...", "short": "..."}} for all extractions.
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
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        return_raw_data: bool = True  # DEPRECATED: Now always returns dual summary
    ) -> Dict[str, Dict[str, str]]:
        """
        Main extraction pipeline - returns ONLY dual summary.
        
        Args:
            current_raw_text: Raw document text to extract
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            return_raw_data: Ignored (deprecated) - always returns dual summary
            
        Returns:
            {"summary": {"long": "...", "short": "..."}} - dual summary structure
        """
        try:
            result = self.extract_document(current_raw_text, page_zones)
            return result
        except Exception as e:
            logger.error(f"❌ Extraction pipeline failed: {e}")
            fallback_date = datetime.now().strftime("%m/%d/%y")
            
            # Return structured error summary
            return {
                "summary": {
                    "long": f"EXTRACTION ERROR - {fallback_date}:\nError: {str(e)}\nManual review required.",
                    "short": f"{fallback_date}: Extraction failed - manual review required."
                }
            }
    
    def extract_document(self, text: str, page_zones: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
        """
        Enhanced pipeline with context analysis - returns ONLY dual summary.
        
        Args:
            text: Document text to process
            page_zones: Optional per-page zone extraction data
            
        Returns:
            {"summary": {"long": "...", "short": "..."}} - dual summary structure
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
            
            # Stage 2: Context-aware extraction (now returns dual summary)
            dual_summary = self._route_to_extractor_with_context(
                text=text,
                doctype=doc_type,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis
            )
            
            # Stage 3: Verify (if applicable) - but since we're only returning summaries, skip or adapt
            # Note: Verification would need to be adapted for summaries only; skipping for now
            logger.info(f"✅ Extraction complete: {dual_summary['summary']['short']}")
            return dual_summary
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}", exc_info=True)
            return {
                "summary": {
                    "long": f"EXTRACTION ERROR - {fallback_date}:\nError: {str(e)}\nManual review required.",
                    "short": f"{fallback_date}: Extraction failed - manual review required."
                }
            }

    def _route_to_extractor_with_context(
        self,
        text: str,
        doctype: DocumentType,
        fallback_date: str,
        page_zones: Optional[Dict],
        context_analysis: Dict
    ) -> Dict[str, Dict[str, str]]:
        """
        Route document to appropriate extractor WITH context analysis.
        
        UPDATED: Returns ONLY dual summary from extractor.
        
        Args:
            text: Document text
            doctype: Detected document type
            fallback_date: Fallback date if extraction fails
            page_zones: Per-page zone extraction
            context_analysis: Context from DocumentContextAnalyzer
        
        Returns:
            {"summary": {"long": "...", "short": "..."}} from specialized extractor
        """
        
        # QME/AME/IME - use context-aware extraction (already returns dual summary)
        if doctype in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            logger.info(f"🎯 Routing to QME extractor WITH context analysis")
            return self.qme_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis,
                raw_text=None
            )
        
        # Imaging reports - assume extractor updated to return dual summary
        elif doctype in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY, 
                        DocumentType.ULTRASOUND, DocumentType.EMG]:
            logger.info(f"🎯 Routing to Imaging extractor")
            # Note: ImagingExtractorChained needs similar update to return dual summary
            # For now, fallback to simple summary generation
            return self._generate_fallback_dual_summary(text, doctype.value, fallback_date)
        
        # Progress reports (PR-2) - assume extractor updated
        elif doctype == DocumentType.PR2:
            logger.info(f"🎯 Routing to PR-2 extractor")
            # Note: PR2ExtractorChained needs similar update
            return self._generate_fallback_dual_summary(text, doctype.value, fallback_date)
        
        # Specialist consults - assume extractor updated
        elif doctype == DocumentType.CONSULT:
            logger.info(f"🎯 Routing to Consult extractor")
            # Note: ConsultExtractorChained needs similar update
            return self._generate_fallback_dual_summary(text, doctype.value, fallback_date)
        
        # All other simple document types - assume extractor updated
        else:
            logger.info(f"🎯 Routing to Simple extractor for {doctype.value}")
            # Note: SimpleExtractor needs similar update
            return self._generate_fallback_dual_summary(text, doctype.value, fallback_date)
    
    def _generate_fallback_dual_summary(self, text: str, doc_type: str, fallback_date: str) -> Dict[str, Dict[str, str]]:
        """
        Fallback dual summary generation for extractors not yet updated.
        
        Returns:
            Basic dual summary structure
        """
        logger.warning(f"⚠️ Using fallback summary for {doc_type} - update extractor for full support")
        
        # Simple extraction from text (can be enhanced with LLM)
        preview = text[:500] + "..." if len(text) > 500 else text
        
        long_summary = f"DOCUMENT SUMMARY - {fallback_date}:\nType: {doc_type}\nPreview: {preview}\nFull extraction pending extractor update."
        
        short_summary = f"{fallback_date}: {doc_type} document processed. Update extractor for detailed summaries."
        
        return {
            "summary": {
                "long": long_summary,
                "short": short_summary
            }
        }
    
    def _route_to_extractor(self, text: str, doc_type: DocumentType, fallback_date: str, page_zones: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Dict[str, str]]:
        """
        DEPRECATED: Route document to appropriate specialized extractor.
        
        UPDATED: Returns dual summary (for backward compatibility).
        """
        logger.warning("⚠️ _route_to_extractor deprecated - use _route_to_extractor_with_context")
        return self._route_to_extractor_with_context(text, doc_type, fallback_date, page_zones, {})
    
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
    
    def _verify_result(self, result: Dict[str, Dict[str, str]], doc_type: DocumentType) -> Dict[str, Dict[str, str]]:
        """
        DEPRECATED: Verify and correct extraction result if needed.
        
        UPDATED: Skipped for summaries only.
        """
        logger.info(f"🛡️ Skipping verification for summaries only ({doc_type.value})")
        return result
    
    def format_whats_new_as_highlights(self, bullet_points: List[str]) -> List[str]:
        """
        DEPRECATED: Format bullet points as highlights (for backward compatibility).
        
        Args:
            bullet_points: List of bullet-formatted summaries
            
        Returns:
            Same list or default message if empty
        """
        logger.warning("⚠️ format_whats_new_as_highlights deprecated - use dual summary")
        return bullet_points if bullet_points else [
            "• No significant new findings identified in current document"
        ]
    
    def get_structured_extraction(self, text: str) -> Dict[str, Any]:
        """
        DEPRECATED: Get full structured extraction as dictionary (for API/storage).
        
        UPDATED: Returns dual summary as structured dict.
        """
        logger.warning("⚠️ get_structured_extraction deprecated - use extract_document for dual summary")
        result = self.extract_document(text)
        return result
    
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