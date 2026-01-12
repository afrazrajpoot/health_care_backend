# document_ai_service.py

"""
Enhanced Document AI Service using Document Summarizer processor.
REPLACED doc-ocr with document-summarizer for better accuracy.
UPDATED: Added batch processing API for large documents (>15 pages) for better context understanding.

This is the main orchestration service that uses:
- utils/pdf_splitter.py - PDF splitting utilities
- utils/gcs_batch_handler.py - GCS batch processing
- utils/layout_parser.py - Document structure extraction
- utils/document_helpers.py - Helper functions
"""

import os
import json
import time
import logging
from typing import List, Optional
from PyPDF2 import PdfReader
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient
from google.api_core import exceptions as google_exceptions

from models.schemas import ExtractionResult
from config.settings import CONFIG
from utils.patient_details_extractor import get_patient_extractor
from helpers.helpers import build_llm_friendly_json, extract_text_from_summarizer
from utils.multi_report_detector import detect_multiple_reports
from utils.pdf_splitter import PDFSplitter, LayoutPreservingTextExtractor
from utils.gcs_batch_handler import GCSBatchHandler
from utils.layout_parser import LayoutParser
from utils.document_helpers import (
    get_mime_type,
    format_patient_details_text,
    log_patient_details,
    merge_extraction_results,
    extract_summary_from_result
)
from utils.signature_extractor import extract_author_signature

logger = logging.getLogger("document_ai")


class DocumentAIProcessor:
    """Service for Document AI Summarizer processing with batch API support for large documents"""
    
    # Batch processing configuration
    BATCH_GCS_BUCKET = "hiregenix"
    BATCH_INPUT_PREFIX = "docai-batch-input/"
    BATCH_OUTPUT_PREFIX = "docai-batch-output/"
    BATCH_TIMEOUT_SECONDS = 600  # 10 minutes max wait
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.summarizer_processor_path: Optional[str] = None
        
        # Initialize helper components
        self.pdf_splitter = PDFSplitter(max_pages_per_chunk=10)
        self.layout_extractor = LayoutPreservingTextExtractor()
        self.gcs_handler: Optional[GCSBatchHandler] = None
        self.layout_parser: Optional[LayoutParser] = None
        
        self._initialize_client()
        self._initialize_helpers()
    
    def _initialize_client(self):
        """Initialize Document AI client with SUMMARIZER processor"""
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CONFIG["credentials_path"]
            logger.info(f"🔑 Using credentials: {CONFIG['credentials_path']}")
            logger.info(f"🆔 Project ID: {CONFIG['project_id']}")
            
            api_endpoint = f"{CONFIG['location']}-documentai.googleapis.com"
            logger.info(f"🌐 API Endpoint: {api_endpoint}")
            
            self.client = DocumentProcessorServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            
            # Use summarizer_processor_id
            summarizer_id = CONFIG.get("summarizer_processor_id") or CONFIG.get("processor_id")
            
            self.summarizer_processor_path = self.client.processor_path(
                CONFIG["project_id"], CONFIG["location"], summarizer_id
            )
            
            logger.info(f"✅ Document AI Summarizer initialized: {summarizer_id}")
            logger.info("📝 Using Document AI Summarizer (pretrained-foundation-model-v1.0-2023-08-22)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document AI Client: {str(e)}")
            raise
    
    def _initialize_helpers(self):
        """Initialize helper components (GCS handler, Layout parser)"""
        # Initialize GCS handler for batch processing
        try:
            self.gcs_handler = GCSBatchHandler(
                bucket_name=self.BATCH_GCS_BUCKET,
                input_prefix=self.BATCH_INPUT_PREFIX,
                output_prefix=self.BATCH_OUTPUT_PREFIX,
                timeout_seconds=self.BATCH_TIMEOUT_SECONDS
            )
        except Exception as e:
            logger.warning(f"⚠️ GCS handler initialization failed: {e}")
            self.gcs_handler = None
        
        # Initialize Layout Parser
        layout_parser_id = CONFIG.get("layout_parser_processor_id")
        if layout_parser_id:
            layout_parser_path = self.client.processor_path(
                CONFIG["project_id"], CONFIG["location"], layout_parser_id
            )
            self.layout_parser = LayoutParser(self.client, layout_parser_path)
            logger.info(f"✅ Document AI Layout Parser initialized: {layout_parser_id}")
        else:
            self.layout_parser = LayoutParser(self.client, None)
            logger.warning("⚠️ Layout Parser processor ID not configured, layout parsing will be skipped")
    
    def _process_document_direct(
        self,
        filepath: str,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True
    ) -> ExtractionResult:
        """
        Process document using Document AI SUMMARIZER (not OCR).
        
        Args:
            filepath: Path to the document
            is_first_chunk: Whether this is the first chunk (extract patient details)
            is_last_chunk: Whether this is the last chunk
        """
        try:
            mime_type = get_mime_type(filepath)
            logger.info(f"📄 Processing document with SUMMARIZER: {filepath}")
            logger.info(f"📋 MIME type: {mime_type}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"📦 File size: {file_size} bytes")
            
            # Read file content
            with open(filepath, "rb") as file:
                file_content = file.read()
            
            # Create raw document for summarizer
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=mime_type
            )
            
            # Create process request
            request = documentai.ProcessRequest(
                name=self.summarizer_processor_path,
                raw_document=raw_document
            )
            
            logger.info("📤 Sending request to Document AI Summarizer...")
            logger.info("   Using processor default settings (Length: MODERATE, Format: PARAGRAPH)")
            
            # Process document with retry logic
            result = self._process_with_retry(request)
            
            # Extract summary text
            summary_text = extract_summary_from_result(result)
            
            logger.info(f"📝 Summary text length: {len(summary_text)} characters")
            logger.info(f"📄 Pages analyzed: {len(result.pages) if result.pages else 0}")
            
            # Log the summary output
            logger.info("=" * 80)
            logger.info("🤖 DOCUMENT AI SUMMARIZER OUTPUT:")
            logger.info("=" * 80)
            logger.info(summary_text)
            logger.info("=" * 80)
            
            # Run multi-report detection
            # multi_report_result = detect_multiple_reports(summary_text)
            
            # Log full text comparison for debugging
            if result.text and len(result.text) != len(summary_text):
                logger.info("=" * 80)
                logger.info("📄 FULL OCR TEXT (for comparison):")
                logger.info("=" * 80)
                logger.info(f"Full text length: {len(result.text)} chars")
                logger.info(f"First 500 chars: {result.text[:500]}")
                logger.info("=" * 80)
            
            # Extract author signature from document text
            # Use raw OCR text (result.text) for better signature detection as it contains full document content
            raw_document_text = result.text or summary_text
            signature_info = extract_author_signature(raw_document_text)
            if signature_info:
                logger.info(f"✍️ Signature extracted - Author: {signature_info.get('author')}, Confidence: {signature_info.get('confidence')}")
            else:
                logger.info("⚠️ No author signature found in document")
            
            # Extract text using simplified extractor (pass signature info for author detection)
            layout_data = extract_text_from_summarizer(result, signature_info=signature_info)
            layout_data["layout_preserved"] = summary_text
            layout_data["raw_text"] = summary_text
            layout_data["structured_document"]["metadata"]["summary_chars"] = len(summary_text)
            layout_data["structured_document"]["metadata"]["full_text_chars"] = len(result.text) if result.text else 0
            
            logger.info(f"🔍 Text extraction complete:")
            logger.info(f"   - Summary text: {len(summary_text)} chars")
            logger.info(f"   - Pages analyzed: {layout_data['structured_document']['document_structure']['total_pages']}")
            
            # Extract patient details from first 3 pages (only for first chunk)
            patient_details = {}
            total_pages = len(result.pages) if result.pages else 0
            
            if total_pages > 0 and is_first_chunk and self.layout_parser.is_configured:
                patient_details = self._extract_patient_details(
                    filepath, summary_text, layout_data, total_pages
                )
            else:
                logger.info(f"⏭️ Skipping layout extraction (is_first_chunk={is_first_chunk})")
            
            # Build LLM-friendly JSON
            llm_json = build_llm_friendly_json(layout_data['structured_document'])
            llm_json["content"]["summary"] = summary_text
            llm_text = json.dumps(llm_json, indent=2)

            logger.info(f'summary text----------------- : {summary_text}')
            
            # Prepend patient details to raw_text if available
            final_raw_text = layout_data["raw_text"]
            if patient_details:
                log_patient_details(patient_details, "PATIENT DETAILS (from layout extraction)")
                patient_details_text = format_patient_details_text(patient_details)
                final_raw_text = patient_details_text + layout_data["raw_text"]
                
                logger.info("🔍 DEBUG: Patient details prepended to raw_text")
                logger.info(f"🔍 First 500 chars of final_raw_text:\n{final_raw_text[:500]}")
            
            # Create extraction result
            processed_result = ExtractionResult(
                text=result.text,
                raw_text=final_raw_text,
                llm_text=llm_text,
                page_zones=layout_data["page_zones"],
                pages=len(result.pages) if result.pages else 0,
                entities=[],
                tables=[],
                formFields=[],
                symbols=[],
                confidence=1.0,
                success=True,
                metadata={"patient_details": patient_details} if patient_details else {},
                # is_multiple_reports=multi_report_result.get("is_multiple", False),
                # multi_report_info=multi_report_result
            )
            
            # ✅ Log metadata for debugging
            if patient_details:
                logger.info(f"📋 Patient details saved to metadata: {patient_details}")
            else:
                logger.info("⚠️ No patient details extracted - metadata will be empty")
            
            logger.info("📊 Extraction summary:")
            logger.info(f"   - Summary characters: {len(processed_result.text)}")
            logger.info(f"   - Raw text (with layout): {len(processed_result.raw_text)} chars")
            logger.info(f"   - Pages: {processed_result.pages}")
            logger.info(f"   - Confidence: 100% (foundation model)")
            logger.info(f"   - Metadata keys: {list(processed_result.metadata.keys()) if processed_result.metadata else 'None'}")
            
            return processed_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error processing document with summarizer: {error_msg}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ExtractionResult(success=False, error=error_msg)
    
    def _process_with_retry(self, request: documentai.ProcessRequest, max_retries: int = 3):
        """Process document with retry logic for transient errors"""
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.process_document(request=request)
                logger.info("✅ Document AI Summarizer processed successfully!")
                return response.document
                
            except google_exceptions.Unknown as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Transient error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    raise
                    
            except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Service error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    raise
    
    def _extract_patient_details(
        self,
        filepath: str,
        summary_text: str,
        layout_data: dict,
        total_pages: int
    ) -> dict:
        """
        Extract patient details using smart AI-first approach.
        
        Strategy:
        1. First: Try AI extraction from summarizer output (fast, accurate)
        2. If all fields found: Skip layout parser entirely
        3. If some fields missing: Use layout parser to fill gaps
        4. Use signature extractor as fallback for author if still missing
        5. Merge: Combine results for completeness
        """
        patient_details = {}
        
        # Get patient extractor instance
        patient_extractor = get_patient_extractor()
        
        # Get signature info from layout_data if available (extracted earlier)
        signature_info = layout_data.get("signature_info")
        signature_author = signature_info.get("author") if signature_info else None
        if signature_author:
            logger.info(f"✍️ Signature author available for fallback: {signature_author} ({signature_info.get('confidence')} confidence)")
        
        # Step 1: Try AI extraction from summarizer first (pass signature_info for author detection)
        logger.info("🤖 Step 1: Attempting AI extraction from summarizer output...")
        ai_result = patient_extractor.extract_from_summarizer_ai(summary_text, signature_info=signature_info)
        
        # Check which fields are found by AI
        ai_fields_found = [k for k in ['patient_name', 'dob', 'doi', 'claim_number', 'author'] if ai_result.get(k)]
        ai_missing_fields = [k for k in ['patient_name', 'dob', 'doi', 'claim_number', 'author'] if not ai_result.get(k)]
        
        logger.info(f"📊 AI extraction found {len(ai_fields_found)}/5 fields: {ai_fields_found}")
        
        # Step 2: If all fields found by AI, skip layout parser
        if not ai_missing_fields:
            logger.info("✅ All patient details found via AI - skipping layout parser extraction")
            patient_details = ai_result
            
            # Still update layout_data metadata
            if layout_data.get("structured_document", {}).get("metadata") is not None:
                layout_data["structured_document"]["metadata"]["patient_details"] = patient_details
            
            log_patient_details(patient_details, "PATIENT DETAILS (AI extraction - complete)")
            return patient_details
        
        # Step 3: Some fields missing - use layout parser to fill gaps
        logger.info(f"⚠️ AI missing fields: {ai_missing_fields} - using layout parser fallback")
        
        # Determine pages to extract (first 3 pages)
        pages_to_extract = list(range(1, min(4, total_pages + 1)))
        
        if not pages_to_extract:
            logger.warning("⚠️ No pages to extract, returning AI results only")
            return ai_result
        
        logger.info(f"📄 Extracting layout from first {len(pages_to_extract)} pages for missing fields")
        
        mime_type = get_mime_type(filepath)
        layout_result = self.layout_parser.process_pages(filepath, pages_to_extract, mime_type)
        
        document_dict = layout_result.get("document_dict", {})
        formatted_text = layout_result.get("formatted_text", "")
        
        logger.info(f"🔍 Layout result keys: {list(layout_result.keys())}")
        logger.info(f"📄 Document dict keys: {list(document_dict.keys()) if document_dict else 'None'}")
        
        if document_dict:
            try:
                # Extract from layout parser
                layout_patient_details = patient_extractor.extract_from_layout_json(document_dict)
                
                # Step 4: Merge results - AI takes priority, layout fills gaps, signature as final fallback for author
                patient_details = {
                    "patient_name": ai_result.get("patient_name") or layout_patient_details.get("patient_name"),
                    "dob": ai_result.get("dob") or layout_patient_details.get("dob"),
                    "doi": ai_result.get("doi") or layout_patient_details.get("doi"),
                    "claim_number": ai_result.get("claim_number") or layout_patient_details.get("claim_number"),
                    "author": ai_result.get("author") or layout_patient_details.get("author") or signature_author,
                    "date_of_report": ai_result.get("date_of_report")  # Only from AI
                }
                
                # Log merge results
                logger.info("=" * 60)
                logger.info("🔗 MERGED PATIENT DETAILS (AI + Layout Parser + Signature):")
                logger.info("=" * 60)
                for field in ['patient_name', 'dob', 'doi', 'claim_number', 'author']:
                    ai_val = ai_result.get(field)
                    layout_val = layout_patient_details.get(field)
                    merged_val = patient_details.get(field)
                    if field == 'author':
                        source = "AI" if ai_val else ("Layout" if layout_val else ("Signature" if signature_author and merged_val == signature_author else "Not found"))
                    else:
                        source = "AI" if ai_val else ("Layout" if layout_val else "Not found")
                    logger.info(f"  {field}: {merged_val} (source: {source})")
                logger.info("=" * 60)
                
                # Add patient details to metadata
                if layout_data.get("structured_document", {}).get("metadata") is not None:
                    layout_data["structured_document"]["metadata"]["patient_details"] = patient_details
                
            except Exception as e:
                logger.error(f"❌ Layout parser patient extraction failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fall back to AI-only results
                patient_details = ai_result
            
            # Append structured layout to raw_text
            json_output = json.dumps(document_dict, indent=2, ensure_ascii=False)
            logger.info(f"📝 JSON output layout: {json_output}")
            layout_data["raw_text"] = (
                summary_text + 
                "\n\n--- STRUCTURED LAYOUT (Formatted) ---\n\n" + formatted_text +
                "\n\n--- STRUCTURED LAYOUT (Full JSON) ---\n\n" + json_output
            )
        else:
            logger.warning("⚠️ document_dict is empty, using AI results only")
            patient_details = ai_result
        
        return patient_details
    
    def _process_document_batch(self, filepath: str) -> ExtractionResult:
        """
        Process document using Document AI batch API.
        Used for large documents (>15 pages) to maintain full context.
        """
        gcs_input_uri = None
        gcs_output_uri = None
        
        try:
            logger.info("=" * 80)
            logger.info("🔄 BATCH PROCESSING: Starting holistic document summarization")
            logger.info("=" * 80)
            
            # Get page count
            with open(filepath, "rb") as f:
                pdf_reader = PdfReader(f)
                total_pages = len(pdf_reader.pages)
            logger.info(f"📄 Document has {total_pages} pages - using batch API for full context")
            
            # Upload to GCS
            gcs_input_uri = self.gcs_handler.upload_for_batch(filepath)
            gcs_output_uri = self.gcs_handler.get_output_uri(gcs_input_uri)
            
            # Run batch processing
            mime_type = get_mime_type(filepath)
            success = self.gcs_handler.run_batch_process(
                self.client,
                self.summarizer_processor_path,
                gcs_input_uri,
                gcs_output_uri,
                mime_type
            )
            
            if not success:
                raise Exception("Batch processing operation failed")
            
            # Download results
            summary_text = self.gcs_handler.download_results(gcs_output_uri)
            
            if not summary_text:
                raise Exception("No summary text extracted from batch results")
            
            # Extract patient details using smart AI-first approach
            logger.info("📋 Extracting patient details using smart AI-first approach...")
            patient_details = {}
            patient_extractor = get_patient_extractor()
            
            # Extract author signature from summary text for batch processing
            signature_info = extract_author_signature(summary_text)
            if signature_info:
                logger.info(f"✍️ Signature extracted - Author: {signature_info.get('author')}, Confidence: {signature_info.get('confidence')}")
            
            # Step 1: Try AI extraction from summarizer first (pass signature_info for author detection)
            ai_result = patient_extractor.extract_from_summarizer_ai(summary_text, signature_info=signature_info)
            ai_fields_found = [k for k in ['patient_name', 'dob', 'doi', 'claim_number', 'author'] if ai_result.get(k)]
            ai_missing_fields = [k for k in ['patient_name', 'dob', 'doi', 'claim_number', 'author'] if not ai_result.get(k)]
            
            logger.info(f"📊 AI extraction found {len(ai_fields_found)}/4 fields: {ai_fields_found}")
            
            # Step 2: If all fields found, use AI results
            if not ai_missing_fields:
                logger.info("✅ All patient details found via AI - skipping layout parser")
                patient_details = ai_result
            else:
                # Step 3: Use layout parser to fill gaps
                logger.info(f"⚠️ AI missing fields: {ai_missing_fields} - using layout parser fallback")
                
                if self.layout_parser.is_configured:
                    try:
                        layout_result = self.layout_parser.process_pages(filepath, [1, 2, 3], mime_type)
                        if layout_result and layout_result.get("document_dict"):
                            document_dict = layout_result.get("document_dict", {})
                            layout_patient_details = patient_extractor.extract_from_layout_json(document_dict)
                            
                            # Merge results - AI takes priority
                            patient_details = {
                                "patient_name": ai_result.get("patient_name") or layout_patient_details.get("patient_name"),
                                "dob": ai_result.get("dob") or layout_patient_details.get("dob"),
                                "doi": ai_result.get("doi") or layout_patient_details.get("doi"),
                                "claim_number": ai_result.get("claim_number") or layout_patient_details.get("claim_number"),
                                "date_of_report": ai_result.get("date_of_report")
                            }
                            
                            logger.info("🔗 Merged AI + Layout parser results")
                        else:
                            patient_details = ai_result
                    except Exception as layout_err:
                        logger.warning(f"⚠️ Layout parser extraction failed: {layout_err}")
                        patient_details = ai_result
                else:
                    patient_details = ai_result
            
            # Build final result
            final_raw_text = summary_text
            if patient_details:
                patient_details_text = format_patient_details_text(patient_details)
                final_raw_text = patient_details_text + summary_text
            
            logger.info("=" * 80)
            logger.info("✅ BATCH PROCESSING COMPLETE")
            logger.info(f"   Summary length: {len(summary_text)} chars")
            logger.info(f"   Total pages: {total_pages}")
            logger.info(f"   Patient details: {'Found' if patient_details else 'Not found'}")
            log_patient_details(patient_details, "BATCH PROCESSING PATIENT DETAILS")
            logger.info("=" * 80)
            
            # Log complete summary
            logger.info("=" * 80)
            logger.info("🤖 COMPLETE BATCH SUMMARY OUTPUT:")
            logger.info("=" * 80)
            logger.info(summary_text)
            logger.info("=" * 80)
            
            # Run multi-report detection
            # multi_report_result = detect_multiple_reports(summary_text)
            
            return ExtractionResult(
                text=summary_text,
                raw_text=final_raw_text,
                llm_text=final_raw_text,
                page_zones={},
                pages=total_pages,
                entities=[],
                tables=[],
                formFields=[],
                symbols=[],
                confidence=1.0,
                success=True,
                metadata={"patient_details": patient_details} if patient_details else {},
                # is_multiple_reports=multi_report_result.get("is_multiple", False),
                # multi_report_info=multi_report_result
            )
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
            
        finally:
            if self.gcs_handler and (gcs_input_uri or gcs_output_uri):
                self.gcs_handler.cleanup(gcs_input_uri, gcs_output_uri)
    
    def _process_large_document_chunked(self, filepath: str) -> ExtractionResult:
        """
        Fallback method: Process large documents by splitting into chunks.
        Used when batch API is not available or fails.
        """
        try:
            logger.info("=" * 80)
            logger.info("🔄 CHUNKED PROCESSING: Splitting document into smaller parts")
            logger.info("=" * 80)
            
            chunk_files = self.pdf_splitter.split_pdf(filepath)
            
            if len(chunk_files) == 1:
                return self._process_document_direct(chunk_files[0], is_first_chunk=True, is_last_chunk=True)
            
            logger.info(f"📦 Processing {len(chunk_files)} chunks with summarizer")
            logger.info(f"📄 Patient details will be extracted from first 3 pages of chunk 1 only")
            
            all_results = []
            for i, chunk_file in enumerate(chunk_files):
                is_first = (i == 0)
                is_last = (i == len(chunk_files) - 1)
                logger.info(f"🔄 Processing chunk {i + 1}/{len(chunk_files)} with summarizer (first={is_first}, last={is_last})")
                try:
                    chunk_result = self._process_document_direct(chunk_file, is_first_chunk=is_first, is_last_chunk=is_last)
                    if chunk_result.success:
                        all_results.append(chunk_result)
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i + 1}: {str(e)}")
            
            self.pdf_splitter.cleanup_chunks(chunk_files)
            
            if not all_results:
                return ExtractionResult(success=False, error="All chunks failed")
            
            return merge_extraction_results(all_results, filepath)
            
        except Exception as e:
            logger.error(f"❌ Error in chunked processing: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")
    
    def process_large_document(self, filepath: str) -> ExtractionResult:
        """
        Process large documents using batch API for full context understanding.
        Falls back to chunking method if batch API fails.
        """
        try:
            if self.gcs_handler and self.gcs_handler.is_available:
                logger.info("🔄 Attempting batch processing for holistic summarization...")
                try:
                    return self._process_document_batch(filepath)
                except Exception as batch_error:
                    logger.warning(f"⚠️ Batch processing failed, falling back to chunking: {batch_error}")
            else:
                logger.warning("⚠️ GCS client not available, using chunking method")
            
            return self._process_large_document_chunked(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error processing large document: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")
    
    def process_document(self, filepath: str) -> ExtractionResult:
        """
        Main document processing method using SUMMARIZER.
        
        Strategy based on page count:
        - ≤15 pages: Use synchronous API (fast, seconds)
        - >15 pages: Use batch API (slower but handles large docs with full context)
        
        Patient details are extracted from first 3 pages using layout parser.
        """
        try:
            mime_type = get_mime_type(filepath)
            
            if mime_type == "application/pdf":
                try:
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        page_count = len(pdf_reader.pages)
                    
                    logger.info(f"📄 Document has {page_count} pages")
                    
                    # Synchronous API limit is 15 pages for Document AI Summarizer
                    if page_count <= 15:
                        logger.info("⚡ Using synchronous API (fast processing for ≤15 pages)")
                        return self._process_document_direct(filepath)
                    else:
                        logger.info(f"📦 Document has {page_count} pages (>15) - using batch API for full context")
                        logger.info("⏳ Note: Batch API takes 2-6 minutes but preserves full document context")
                        
                        if self.gcs_handler and self.gcs_handler.is_available:
                            try:
                                return self._process_document_batch(filepath)
                            except Exception as batch_error:
                                logger.warning(f"⚠️ Batch processing failed, falling back to chunked: {batch_error}")
                                return self._process_large_document_chunked(filepath)
                        else:
                            logger.warning("⚠️ GCS client not available, using chunked processing")
                            return self._process_large_document_chunked(filepath)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not check page count: {str(e)}")
                    return self._process_document_direct(filepath)
            else:
                # Non-PDF files use direct processing
                return self._process_document_direct(filepath)
                
        except Exception as e:
            logger.error(f"❌ Error in main processing: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")


# Global processor instance
processor_instance = None


def get_document_ai_processor() -> DocumentAIProcessor:
    """Get singleton DocumentAIProcessor instance"""
    global processor_instance
    if processor_instance is None:
        try:
            logger.info("🚀 Initializing Document AI Summarizer processor...")
            processor_instance = DocumentAIProcessor()
            logger.info("✅ Document AI Summarizer processor ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            raise
    return processor_instance


def process_document_smart(filepath: str) -> ExtractionResult:
    """Smart document processing using Document AI Summarizer"""
    processor = get_document_ai_processor()
    return processor.process_document(filepath)
