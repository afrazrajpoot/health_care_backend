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
    ) -> Dict[str, str]:
        """
        Main extraction pipeline - returns dictionary with both summaries.
        
        Args:
            current_raw_text: Raw document text to extract
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            
        Returns:
            Dict with 'long_summary' and 'short_summary'
        """
        try:
            # extract_document returns dict with both summaries
            result_dict = self.extract_document(current_raw_text, page_zones)
            
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
    
    def extract_document(self, text: str, page_zones: Optional[Dict] = None) -> Dict[str, str]:
        """
        Enhanced pipeline with context analysis - returns dictionary with both summaries.
        
        Args:
            text: Document text to extract
            page_zones: Per-page zone extraction
            
        Returns:
            Dict with 'long_summary' and 'short_summary'
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
            
            # Stage 2: Context-aware extraction - get dictionary with both summaries
            result_dict = self._route_to_extractor_with_context(
                text=text,
                doctype=doc_type,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis
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

    def _route_to_extractor_with_context(
        self,
        text: str,
        doctype: DocumentType,
        fallback_date: str,
        page_zones: Optional[Dict],
        context_analysis: Dict
    ) -> Dict[str, str]:
        """
        Route document to appropriate extractor WITH context analysis.
        Returns dictionary with both summaries.
        
        Args:
            text: Document text
            doctype: Detected document type
            fallback_date: Fallback date if extraction fails
            page_zones: Per-page zone extraction
            context_analysis: Context from DocumentContextAnalyzer
        
        Returns:
            Dict with 'long_summary' and 'short_summary'
        """
        
        # QME/AME/IME - use context-aware extraction (already returns dict)
        if doctype in [DocumentType.QME, DocumentType.AME, DocumentType.IME]:
            logger.info(f"🎯 Routing to QME extractor WITH context analysis")
            qme_result = self.qme_extractor.extract(
                text=text,
                doc_type=doctype.value,
                fallback_date=fallback_date,
                page_zones=page_zones,
                context_analysis=context_analysis,
                raw_text=None
            )
            # QME extractor already returns dict with both summaries
            if isinstance(qme_result, dict):
                return qme_result
            else:
                # Fallback if not dict (backward compat)
                return {
                    "long_summary": str(qme_result) if qme_result else f"{fallback_date}: QME extraction completed",
                    "short_summary": f"{fallback_date}: QME report processed"
                }
        
        # Imaging reports - convert ExtractionResult to dict
        elif doctype in [DocumentType.MRI, DocumentType.CT, DocumentType.XRAY, 
                        DocumentType.ULTRASOUND, DocumentType.EMG]:
            logger.info(f"🎯 Routing to Imaging extractor")
            imaging_result = self.imaging_extractor.extract(
                text, 
                doctype.value, 
                fallback_date,
                context_analysis=context_analysis,
                page_zones=page_zones
            )
            return self._convert_extraction_result_to_dict(imaging_result, fallback_date)
        
        # Progress reports (PR-2) - convert ExtractionResult to dict
        elif doctype == DocumentType.PR2:
            logger.info(f"🎯 Routing to PR-2 extractor")
            pr2_result = self.pr2_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,
                page_zones=page_zones
            )
            return self._convert_extraction_result_to_dict(pr2_result, fallback_date)
        
        # Specialist consults - convert ExtractionResult to dict
        elif doctype == DocumentType.CONSULT:
            logger.info(f"🎯 Routing to Consult extractor")
            consult_result = self.consult_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,
                page_zones=page_zones
            )
            return self._convert_extraction_result_to_dict(consult_result, fallback_date)
        
        # All other simple document types - convert ExtractionResult to dict
        else:
            logger.info(f"🎯 Routing to Simple extractor for {doctype.value}")
            simple_result = self.simple_extractor.extract(
                text, 
                doctype.value, 
                fallback_date, 
                context_analysis=context_analysis,
                page_zones=page_zones
            )
            return self._convert_extraction_result_to_dict(simple_result, fallback_date)

    def _convert_extraction_result_to_dict(self, result: ExtractionResult, fallback_date: str) -> Dict[str, str]:
        """
        Convert ExtractionResult to dictionary with both summaries.
        
        Args:
            result: ExtractionResult object
            fallback_date: Fallback date for error cases
            
        Returns:
            Dict with 'long_summary' and 'short_summary'
        """
        try:
            # Use short_summary if available, otherwise generate from long_summary
            short_summary = result.short_summary
            if not short_summary and result.summary_line:
                short_summary = self._generate_short_summary_from_long(result.summary_line)
            
            return {
                "long_summary": result.summary_line or f"{fallback_date}: Document processed",
                "short_summary": short_summary or f"{fallback_date}: {result.document_type} report"
            }
        except Exception as e:
            logger.error(f"❌ Error converting ExtractionResult to dict: {e}")
            return {
                "long_summary": f"{fallback_date}: Document processed",
                "short_summary": f"{fallback_date}: {result.document_type if hasattr(result, 'document_type') else 'Unknown'} report"
            }

    def _generate_short_summary_from_long(self, long_summary: str) -> str:
        """
        Generate short summary from long summary using LLM.
        
        Args:
            long_summary: Long detailed summary
            
        Returns:
            Short concise summary
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
            Dictionary with all extraction fields including both summaries
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
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document type and basic metadata
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