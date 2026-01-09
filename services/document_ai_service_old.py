# document_ai_service.py

"""
Enhanced Document AI Service using Document Summarizer processor.
REPLACED doc-ocr with document-summarizer for better accuracy.
UPDATED: Added batch processing API for large documents (>15 pages) for better context understanding.
"""

import os
import base64
import tempfile
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient, types
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
from google.api_core.operation import Operation
import logging
from PyPDF2 import PdfReader, PdfWriter
from models.schemas import ExtractionResult, FileInfo
from config.settings import CONFIG
from utils.patient_details_extractor import get_patient_extractor
from helpers.helpers import build_llm_friendly_json , extract_text_from_summarizer
from utils.multi_report_detector import detect_multiple_reports
from utils.pdf_splitter import PDFSplitter, LayoutPreservingTextExtractor

logger = logging.getLogger("document_ai")


class DocumentAIProcessor:
    """Service for Document AI Summarizer processing with batch API support for large documents"""
    
    # Batch processing configuration
    BATCH_GCS_BUCKET = "hiregenix"
    BATCH_INPUT_PREFIX = "docai-batch-input/"
    BATCH_OUTPUT_PREFIX = "docai-batch-output/"
    BATCH_TIMEOUT_SECONDS = 600  # 10 minutes max wait
    BATCH_POLL_INTERVAL = 5  # seconds between status checks
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.summarizer_processor_path: Optional[str] = None
        self.layout_parser_processor_path: Optional[str] = None  # NEW: Layout parser
        self.pdf_splitter = PDFSplitter(max_pages_per_chunk=10)
        self.layout_extractor = LayoutPreservingTextExtractor()
        self.storage_client: Optional[storage.Client] = None
        self.gcs_bucket = None
        self._initialize_client()
        self._initialize_gcs_client()
    
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
            
            # REPLACED: Use summarizer_processor_id instead of processor_id
            summarizer_id = CONFIG.get("summarizer_processor_id") or CONFIG.get("processor_id")
            
            self.summarizer_processor_path = self.client.processor_path(
                CONFIG["project_id"], CONFIG["location"], summarizer_id
            )
            
            logger.info(f"✅ Document AI Summarizer initialized: {summarizer_id}")
            logger.info("📝 Using Document AI Summarizer (pretrained-foundation-model-v1.0-2023-08-22)")
            
            # NEW: Initialize Layout Parser processor
            layout_parser_id = CONFIG.get("layout_parser_processor_id")
            if layout_parser_id:
                self.layout_parser_processor_path = self.client.processor_path(
                    CONFIG["project_id"], CONFIG["location"], layout_parser_id
                )
                logger.info(f"✅ Document AI Layout Parser initialized: {layout_parser_id}")
            else:
                logger.warning("⚠️ Layout Parser processor ID not configured, layout parsing will be skipped")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document AI Client: {str(e)}")
            raise
    
    def _initialize_gcs_client(self):
        """Initialize GCS client for batch processing"""
        try:
            self.storage_client = storage.Client()
            self.gcs_bucket = self.storage_client.bucket(self.BATCH_GCS_BUCKET)
            logger.info(f"✅ GCS client initialized for batch processing (bucket: {self.BATCH_GCS_BUCKET})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GCS client for batch processing: {str(e)}")
            logger.warning("⚠️ Batch processing will fall back to chunking method")
            self.storage_client = None
            self.gcs_bucket = None
    
    def get_mime_type(self, filepath: str) -> str:
        """Get MIME type based on file extension"""
        mime_mapping = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        file_ext = Path(filepath).suffix.lower()
        return mime_mapping.get(file_ext, "application/octet-stream")
    
    def _process_with_layout_parser(self, filepath: str, page_numbers: List[int]) -> Dict[str, Any]:
        """
        Process specific pages with Layout Parser to preserve structure.
        Used for first 3 and last 3 pages to extract patient details and signatures.
        
        Args:
            filepath: Path to the document
            page_numbers: List of page numbers to process (1-indexed)
        
        Returns:
            Dictionary with structured layout data including blocks with nearest neighbor info
        """
        if not self.layout_parser_processor_path:
            logger.warning("⚠️ Layout Parser not configured, skipping layout extraction")
            return {"blocks": [], "text": ""}
        
        try:
            # Create PDF with only requested pages
            temp_pdf_path = None
            if len(page_numbers) > 0:
                with open(filepath, "rb") as file:
                    pdf_reader = PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    # Filter valid page numbers
                    valid_pages = [p for p in page_numbers if 0 < p <= total_pages]
                    
                    if not valid_pages:
                        return {"blocks": [], "text": ""}
                    
                    # Create temporary PDF with selected pages
                    pdf_writer = PdfWriter()
                    for page_num in valid_pages:
                        pdf_writer.add_page(pdf_reader.pages[page_num - 1])
                    
                    temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
                    with open(temp_pdf_path, 'wb') as output_file:
                        pdf_writer.write(output_file)
                
                file_to_process = temp_pdf_path
            else:
                file_to_process = filepath
            
            # Read file content
            with open(file_to_process, "rb") as file:
                file_content = file.read()
            
            mime_type = self.get_mime_type(file_to_process)
            
            # Create raw document
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=mime_type
            )
            
            # Create process request for layout parser
            request = documentai.ProcessRequest(
                name=self.layout_parser_processor_path,
                raw_document=raw_document
            )
            
            logger.info(f"📤 Sending {len(page_numbers)} pages to Layout Parser...")
            
            # Process with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.client.process_document(request=request)
                    break
                except google_exceptions.Unknown as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Layout parser error (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
            
            processed_document = response.document
            
            # Convert the entire Document object to dictionary using Google's built-in method
            document_dict = types.Document.to_dict(processed_document)
            
            # Build formatted text with field-value detection from document_layout blocks
            formatted_text = ""
            blocks = []
            
            if document_dict.get('document_layout') and document_dict['document_layout'].get('blocks'):
                blocks = document_dict['document_layout']['blocks']
                formatted_text = self._extract_fields_from_blocks(blocks)
                logger.info(f"✅ Layout Parser extracted {len(blocks)} blocks from document_layout")
            else:
                logger.warning(f"⚠️ No document_layout blocks found in response")
                # Check if blocks exist elsewhere
                if document_dict.get('pages'):
                    total_blocks = sum(len(page.get('blocks', [])) for page in document_dict['pages'])
                    logger.info(f"📄 Found {total_blocks} blocks in pages structure")
            
            # Clean up temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            # Log document structure for debugging
            logger.info(f"📊 Document structure keys: {list(document_dict.keys())}")
            if 'pages' in document_dict:
                logger.info(f"📄 Total pages in layout result: {len(document_dict['pages'])}")
            
            return {
                "document_dict": document_dict,  # Full structured output from Google
                "blocks": blocks,
                "formatted_text": formatted_text,
                "full_text": document_dict.get('text', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Layout Parser error: {e}")
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            return {"document_dict": {}, "blocks": [], "formatted_text": ""}
    
    def _extract_fields_from_blocks(self, blocks: List[Dict]) -> str:
        """
        Extract field-value pairs from blocks using nearest neighbor logic.
        Handles both 'field: value' and 'value field' patterns.
        """
        # Common field keywords for medical documents
        field_keywords = [
            "claim", "claim number", "claim no", "claim#", "cl", "claim #", ""
            "dob", "date of birth", "birth date", "dob"
            "doi", "date of injury", "injury date", "doi"
            "patient", "applicant", "name",
            "ssn", "social security",
            "panel", "panel no",
            "employer",
            "signed", "signature", "author"
        ]
        
        formatted_lines = []
        i = 0
        
        while i < len(blocks):
            current_block = blocks[i]
            current_text = current_block.get("text", "").strip()
            
            if not current_text:
                i += 1
                continue
            
            # Check if current block is a field keyword
            is_field = any(keyword in current_text.lower() for keyword in field_keywords)
            
            if is_field:
                # Look for value in next block (field: value pattern)
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    next_text = next_block.get("text", "").strip()
                    
                    # Check if they're on same page and close by block ID
                    current_page = current_block.get("pageSpan", {}).get("pageStart", 0)
                    next_page = next_block.get("pageSpan", {}).get("pageStart", 0)
                    
                    if current_page == next_page and next_text:
                        formatted_lines.append(f"{current_text}: {next_text}")
                        i += 2  # Skip next block as it's been processed
                        continue
                
                formatted_lines.append(current_text)
            else:
                # Check if next block is a field keyword (value field pattern)
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    next_text = next_block.get("text", "").strip()
                    next_is_field = any(keyword in next_text.lower() for keyword in field_keywords)
                    
                    if next_is_field:
                        current_page = current_block.get("pageSpan", {}).get("pageStart", 0)
                        next_page = next_block.get("pageSpan", {}).get("pageStart", 0)
                        
                        if current_page == next_page:
                            formatted_lines.append(f"{next_text}: {current_text}")
                            i += 2
                            continue
                
                formatted_lines.append(current_text)
            
            i += 1
        
        return "\n".join(formatted_lines)
    
    def _process_document_direct(self, filepath: str, is_first_chunk: bool = True, is_last_chunk: bool = True) -> ExtractionResult:
        """
        Process document using Document AI SUMMARIZER (not OCR).
        FIXED: Extract summary field instead of full text.
        
        Args:
            filepath: Path to the document
            is_first_chunk: Whether this is the first chunk (extract first 3 pages for patient details)
            is_last_chunk: Whether this is the last chunk (extract last 3 pages for patient details)
        """
        try:
            mime_type = self.get_mime_type(filepath)
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
            
            # Create simple process request (uses processor default settings)
            request = documentai.ProcessRequest(
                name=self.summarizer_processor_path,
                raw_document=raw_document
            )
            
            logger.info("📤 Sending request to Document AI Summarizer...")
            logger.info("   Using processor default settings (Length: MODERATE, Format: PARAGRAPH)")
            
            # Process document with retry logic for transient errors
            max_retries = 3
            retry_delay = 1  # Start with 1 second
            
            for attempt in range(max_retries):
                try:
                    response = self.client.process_document(request=request)
                    result = response.document
                    logger.info("✅ Document AI Summarizer processed successfully!")
                    break  # Success, exit retry loop
                    
                except google_exceptions.Unknown as e:
                    # Transient network/SSL errors
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Transient error (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"❌ Failed after {max_retries} attempts")
                        raise
                        
                except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable) as e:
                    # Timeout or service unavailable
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Service error (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"❌ Failed after {max_retries} attempts")
                        raise
            
            # FIXED: Extract the SUMMARY field, not the full text
            summary_text = ""
            
            # Check if document has summary field
            if hasattr(result, 'chunked_document') and result.chunked_document:
                # For chunked documents, get summary from chunks
                logger.info(f"📦 Found chunked document with {len(result.chunked_document.chunks)} chunks")
                summary_parts = []
                for chunk in result.chunked_document.chunks:
                    if hasattr(chunk, 'content') and chunk.content:
                        summary_parts.append(chunk.content)
                summary_text = "\n\n".join(summary_parts)
            
            # Try to get entity-based summary (for newer API versions)
            if not summary_text and hasattr(result, 'entities'):
                for entity in result.entities:
                    if entity.type_ == 'summary' or 'summary' in entity.type_.lower():
                        summary_text = entity.mention_text
                        break
            
            # Fallback: Check for summary in document text (last resort)
            # Some processors return summary as the primary text field
            if not summary_text:
                # Check if text is significantly shorter than expected (indicates summarization occurred)
                full_text = result.text or ""
                page_count = len(result.pages) if result.pages else 0
                
                # If text is less than ~500 chars per page, it's likely a summary, not full OCR
                avg_chars_per_page = len(full_text) / page_count if page_count > 0 else 0
                
                if avg_chars_per_page < 500 or page_count == 0:
                    # This is likely the summary
                    summary_text = full_text
                    logger.info(f"✅ Detected summary in text field (avg {avg_chars_per_page:.0f} chars/page)")
                else:
                    # This is full OCR text - summarizer didn't work properly
                    logger.warning(f"⚠️ Got full OCR text instead of summary (avg {avg_chars_per_page:.0f} chars/page)")
                    logger.warning("⚠️ Processor may not be configured as a summarizer")
                    summary_text = full_text  # Use it anyway for compatibility
            
            logger.info(f"📝 Summary text length: {len(summary_text)} characters")
            logger.info(f"📄 Pages analyzed: {len(result.pages) if result.pages else 0}")
            
            # Log the actual SUMMARY (not full OCR text)
            logger.info("=" * 80)
            logger.info("🤖 DOCUMENT AI SUMMARIZER OUTPUT:")
            logger.info("=" * 80)
            logger.info(summary_text)
            logger.info("=" * 80)
            
            # NEW: Run multi-report detection on summarizer output (using o3-pro AI model)
            # multi_report_result = detect_multiple_reports(summary_text)
            
            # Also log full text comparison for debugging
            if result.text and len(result.text) != len(summary_text):
                logger.info("=" * 80)
                logger.info("📄 FULL OCR TEXT (for comparison):")
                logger.info("=" * 80)
                logger.info(f"Full text length: {len(result.text)} chars")
                logger.info(f"First 500 chars: {result.text[:500]}")
                logger.info("=" * 80)
            
            # Extract text using simplified extractor
            layout_data = extract_text_from_summarizer(result)
            # Override with actual summary
            layout_data["layout_preserved"] = summary_text
            layout_data["raw_text"] = summary_text
            layout_data["structured_document"]["metadata"]["summary_chars"] = len(summary_text)
            layout_data["structured_document"]["metadata"]["full_text_chars"] = len(result.text) if result.text else 0
            
            logger.info(f"🔍 Text extraction complete:")
            logger.info(f"   - Summary text: {len(summary_text)} chars")
            logger.info(f"   - Pages analyzed: {layout_data['structured_document']['document_structure']['total_pages']}")
            
            # NEW: Extract patient details only from first 3 pages of first chunk
            layout_extracted_text = ""
            total_pages = len(result.pages) if result.pages else 0
            
            # Only extract patient details from first chunk (first 3 pages only)
            if total_pages > 0 and is_first_chunk:
                pages_to_extract = []
                
                # Determine which pages to extract - only first 3 pages
                if total_pages <= 3:
                    # Process all pages if 3 or fewer
                    pages_to_extract = list(range(1, total_pages + 1))
                else:
                    # Extract only first 3 pages for patient details
                    pages_to_extract = list(range(1, 4))
                
                if pages_to_extract:
                    logger.info(f"📄 Extracting layout from first {len(pages_to_extract)} pages for patient details")
                    layout_result = self._process_with_layout_parser(filepath, pages_to_extract)
                    
                    # Get full structured JSON from layout parser
                    document_dict = layout_result.get("document_dict", {})
                    formatted_text = layout_result.get("formatted_text", "")
                    
                    logger.info(f"🔍 Layout result keys: {list(layout_result.keys())}")
                    logger.info(f"📄 Document dict keys: {list(document_dict.keys()) if document_dict else 'None'}")
                    
                    if document_dict:                        
                        # Extract patient details from layout JSON (only from first chunk)
                        try:
                            patient_extractor = get_patient_extractor()
                            patient_details = patient_extractor.extract_from_layout_json(document_dict, summary_text)
                            
                            # Log extracted patient details
                            logger.info("=" * 80)
                            logger.info("👤 EXTRACTED PATIENT DETAILS:")
                            logger.info("=" * 80)
                            logger.info(f"  Patient Name: {patient_details.get('patient_name', 'Not found')}")
                            logger.info(f"  DOB: {patient_details.get('dob', 'Not found')}")
                            logger.info(f"  DOI: {patient_details.get('doi', 'Not found')}")
                            logger.info(f"  Claim Number: {patient_details.get('claim_number', 'Not found')}")
                            logger.info("=" * 80)
                            
                            # Add patient details to metadata
                            layout_data["structured_document"]["metadata"]["patient_details"] = patient_details
                            
                        except Exception as e:
                            logger.error(f"❌ Patient details extraction failed: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        # Append both formatted text and full JSON to raw_text
                        json_output = json.dumps(document_dict, indent=2, ensure_ascii=False)
                        logger.info(f"📝 JSON output length: {len(json_output)} characters")
                        layout_data["raw_text"] = (
                            summary_text + 
                            "\n\n--- STRUCTURED LAYOUT (Formatted) ---\n\n" + formatted_text +
                            "\n\n--- STRUCTURED LAYOUT (Full JSON) ---\n\n" + json_output
                        )
                    else:
                        logger.warning("⚠️ document_dict is empty, skipping patient extraction")
            else:
                logger.info(f"⏭️ Skipping layout extraction (is_first_chunk={is_first_chunk}, is_last_chunk={is_last_chunk})")
            
            # Build LLM-friendly JSON
            llm_json = build_llm_friendly_json(layout_data['structured_document'])
            llm_json["content"]["summary"] = summary_text
            llm_text = json.dumps(llm_json, indent=2)

            logger.info(f'summary text----------------- : {summary_text}')
            
            # Extract patient details from metadata
            patient_details = layout_data.get("structured_document", {}).get("metadata", {}).get("patient_details", {})
            
            # Prepend patient details to raw_text if available (for single chunk documents)
            final_raw_text = layout_data["raw_text"]
            if patient_details:
                logger.info("=" * 80)
                logger.info("👤 PATIENT DETAILS (from layout extraction):")
                logger.info("=" * 80)
                logger.info(f"  Patient Name: {patient_details.get('patient_name', 'Not found')}")
                logger.info(f"  DOB: {patient_details.get('dob', 'Not found')}")
                logger.info(f"  DOI: {patient_details.get('doi', 'Not found')}")
                logger.info(f"  Claim Number: {patient_details.get('claim_number', 'Not found')}")
                logger.info("=" * 80)
                
                # Add patient details section to the beginning of raw_text
                patient_details_text = "--- PATIENT DETAILS ---\n"
                patient_details_text += f"Patient Name: {patient_details.get('patient_name', 'N/A')}\n"
                patient_details_text += f"Date of Birth: {patient_details.get('dob', 'N/A')}\n"
                patient_details_text += f"Date of Injury: {patient_details.get('doi', 'N/A')}\n"
                patient_details_text += f"Claim Number: {patient_details.get('claim_number', 'N/A')}\n"
                patient_details_text += "--- END PATIENT DETAILS ---\n\n"
                
                final_raw_text = patient_details_text + layout_data["raw_text"]
                
                logger.info("🔍 DEBUG: Patient details prepended to raw_text")
                logger.info(f"🔍 First 500 chars of final_raw_text:\n{final_raw_text[:500]}")
            
            # Create extraction result with merged raw_text and patient_details in metadata
            processed_result = ExtractionResult(
                text=result.text,  #full OCR text
                raw_text=final_raw_text,  # Summary + Patient Details + Layout Parser structured text
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
            
            logger.info("📊 Extraction summary:")
            logger.info(f"   - Summary characters: {len(processed_result.text)}")
            logger.info(f"   - Raw text (with layout): {len(processed_result.raw_text)} chars")
            logger.info(f"   - Pages: {processed_result.pages}")
            logger.info(f"   - Confidence: 100% (foundation model)")
            
            return processed_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error processing document with summarizer: {error_msg}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ExtractionResult(success=False, error=error_msg)

    # ==================== BATCH PROCESSING METHODS ====================
    
    def _upload_to_gcs_for_batch(self, filepath: str) -> str:
        """
        Upload a local file to GCS for batch processing.
        Returns the GCS URI (gs://bucket/path/file.pdf)
        
        NOTE: This is NOT a duplicate upload. The Document AI batch API requires files
        to be in GCS (cannot process local files). The standard flow is:
        1. User uploads file → saved as LOCAL temp file
        2. Document AI processes the file (batch API needs it in GCS temporarily)
        3. AFTER processing, file is uploaded to permanent "uploads/" folder
        
        This temporary upload to "docai-batch-input/" is cleaned up after processing.
        """
        try:
            # Generate unique filename to avoid collisions
            unique_id = str(uuid.uuid4())[:8]
            original_name = Path(filepath).stem
            extension = Path(filepath).suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_filename = f"{original_name}_{timestamp}_{unique_id}{extension}"
            
            blob_path = f"{self.BATCH_INPUT_PREFIX}{gcs_filename}"
            blob = self.gcs_bucket.blob(blob_path)
            
            # Upload file
            blob.upload_from_filename(filepath)
            gcs_uri = f"gs://{self.BATCH_GCS_BUCKET}/{blob_path}"
            
            logger.info(f"✅ Uploaded to GCS for batch processing: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to GCS: {str(e)}")
            raise
    
    def _get_batch_output_uri(self, input_uri: str) -> str:
        """Generate output GCS URI for batch processing results"""
        # Extract filename from input URI
        input_filename = Path(input_uri).stem
        unique_id = str(uuid.uuid4())[:8]
        output_folder = f"{self.BATCH_OUTPUT_PREFIX}{input_filename}_{unique_id}/"
        return f"gs://{self.BATCH_GCS_BUCKET}/{output_folder}"
    
    def _batch_process_document(self, gcs_input_uri: str, gcs_output_uri: str, mime_type: str = "application/pdf") -> bool:
        """
        Call Document AI batch_process_documents API.
        Returns True if operation completed successfully.
        """
        try:
            logger.info(f"🚀 Starting batch processing...")
            logger.info(f"   Input: {gcs_input_uri}")
            logger.info(f"   Output: {gcs_output_uri}")
            
            # Create batch process request
            gcs_document = documentai.GcsDocument(
                gcs_uri=gcs_input_uri,
                mime_type=mime_type
            )
            
            gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
            
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=gcs_documents
            )
            
            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=gcs_output_uri
                )
            )
            
            request = documentai.BatchProcessRequest(
                name=self.summarizer_processor_path,
                input_documents=input_config,
                document_output_config=output_config
            )
            
            # Start the batch operation (returns a long-running operation)
            operation = self.client.batch_process_documents(request=request)
            
            logger.info(f"⏳ Batch operation started: {operation.operation.name}")
            logger.info(f"⏳ Waiting for completion (timeout: {self.BATCH_TIMEOUT_SECONDS}s)...")
            
            # Wait for the operation to complete (synchronous blocking)
            result = operation.result(timeout=self.BATCH_TIMEOUT_SECONDS)
            
            logger.info(f"✅ Batch processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _download_batch_results(self, gcs_output_uri: str) -> str:
        """
        Download and parse results from batch processing output location.
        Returns the extracted summary text.
        """
        try:
            # Extract bucket and prefix from GCS URI
            # gs://bucket/path/ -> bucket, path/
            gcs_output_uri = gcs_output_uri.rstrip('/')
            parts = gcs_output_uri.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            logger.info(f"📥 Downloading batch results from: {gcs_output_uri}")
            logger.info(f"   Bucket: {bucket_name}, Prefix: {prefix}")
            
            # List all blobs in the output folder
            blobs = list(self.storage_client.list_blobs(bucket_name, prefix=prefix))
            
            if not blobs:
                logger.warning(f"⚠️ No output files found at {gcs_output_uri}")
                return ""
            
            logger.info(f"📄 Found {len(blobs)} output file(s)")
            
            # Batch processing outputs JSON files (shards)
            all_text = []
            all_summaries = []
            
            for blob in blobs:
                if blob.name.endswith(".json"):
                    logger.info(f"   Processing: {blob.name}")
                    content = blob.download_as_text()
                    
                    try:
                        # Parse the Document AI output JSON
                        doc_json = json.loads(content)
                        
                        # Extract text from the document
                        if "text" in doc_json:
                            all_text.append(doc_json["text"])
                        
                        # Extract summary from chunkedDocument or entities
                        if "chunkedDocument" in doc_json:
                            chunked_doc = doc_json["chunkedDocument"]
                            if "chunks" in chunked_doc:
                                for chunk in chunked_doc["chunks"]:
                                    if "content" in chunk:
                                        all_summaries.append(chunk["content"])
                        
                        # Also check entities for summary
                        if "entities" in doc_json:
                            for entity in doc_json["entities"]:
                                if entity.get("type") == "summary" or "summary" in entity.get("type", "").lower():
                                    mention_text = entity.get("mentionText", "")
                                    if mention_text:
                                        all_summaries.append(mention_text)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Failed to parse JSON from {blob.name}: {e}")
                        continue
            
            # Combine results - prefer summaries if available, otherwise use text
            if all_summaries:
                result_text = "\n\n".join(all_summaries)
                logger.info(f"✅ Extracted {len(all_summaries)} summary chunk(s), total {len(result_text)} chars")
            elif all_text:
                result_text = "\n\n".join(all_text)
                logger.info(f"✅ Extracted full text, total {len(result_text)} chars")
            else:
                result_text = ""
                logger.warning("⚠️ No text or summary found in batch results")
            
            return result_text
            
        except Exception as e:
            logger.error(f"❌ Failed to download batch results: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ""
    
    def _cleanup_batch_gcs_files(self, gcs_input_uri: str, gcs_output_uri: str):
        """Clean up temporary GCS files after batch processing"""
        try:
            # Delete input file
            if gcs_input_uri:
                input_path = gcs_input_uri.replace(f"gs://{self.BATCH_GCS_BUCKET}/", "")
                input_blob = self.gcs_bucket.blob(input_path)
                if input_blob.exists():
                    input_blob.delete()
                    logger.debug(f"🧹 Deleted input file: {gcs_input_uri}")
            
            # Delete output folder and contents
            if gcs_output_uri:
                output_prefix = gcs_output_uri.replace(f"gs://{self.BATCH_GCS_BUCKET}/", "").rstrip("/")
                blobs = list(self.storage_client.list_blobs(self.BATCH_GCS_BUCKET, prefix=output_prefix))
                for blob in blobs:
                    blob.delete()
                logger.debug(f"🧹 Deleted {len(blobs)} output file(s) from: {gcs_output_uri}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up GCS files: {str(e)}")
    
    def _process_document_batch(self, filepath: str) -> ExtractionResult:
        """
        Process document using Document AI batch API.
        This is the main batch processing method that:
        1. Uploads document to GCS
        2. Calls batch_process_documents API
        3. Waits for completion
        4. Downloads and parses results
        5. Still extracts patient details from first 3 pages using layout parser
        """
        gcs_input_uri = None
        gcs_output_uri = None
        
        try:
            logger.info("=" * 80)
            logger.info("🔄 BATCH PROCESSING: Starting holistic document summarization")
            logger.info("=" * 80)
            
            # Get page count for logging
            with open(filepath, "rb") as f:
                pdf_reader = PdfReader(f)
                total_pages = len(pdf_reader.pages)
            logger.info(f"📄 Document has {total_pages} pages - using batch API for full context")
            
            # Step 1: Upload to GCS
            gcs_input_uri = self._upload_to_gcs_for_batch(filepath)
            
            # Step 2: Generate output URI
            gcs_output_uri = self._get_batch_output_uri(gcs_input_uri)
            
            # Step 3: Run batch processing
            mime_type = self.get_mime_type(filepath)
            success = self._batch_process_document(gcs_input_uri, gcs_output_uri, mime_type)
            
            if not success:
                raise Exception("Batch processing operation failed")
            
            # Step 4: Download results
            summary_text = self._download_batch_results(gcs_output_uri)
            
            if not summary_text:
                raise Exception("No summary text extracted from batch results")
            
            # Step 5: Extract patient details from first 3 pages using layout parser
            logger.info("📋 Extracting patient details from first 3 pages using layout parser...")
            patient_details = {}
            layout_data = {"raw_text": "", "structured_document": {}, "page_zones": {}}
            
            try:
                layout_data = self._process_with_layout_parser(filepath, [1, 2, 3])
                if layout_data and layout_data.get("blocks"):
                    logger.info(f"📊 Layout extraction found {len(layout_data.get('blocks', []))} blocks")
                    
                    # Extract patient details using the patient extractor
                    patient_extractor = get_patient_extractor()
                    document_dict = layout_data.get("structured_document", {})
                    
                    if document_dict:
                        extracted_patients = patient_extractor.extract_patient_details_from_document(document_dict)
                        if extracted_patients:
                            patient_details = extracted_patients[0] if isinstance(extracted_patients, list) else extracted_patients
                            logger.info(f"👤 Patient details extracted: {patient_details.get('patient_name', 'N/A')}")
            except Exception as layout_err:
                logger.warning(f"⚠️ Layout parser extraction failed: {layout_err}")
            
            # Step 6: Build final result
            final_raw_text = summary_text
            
            # Prepend patient details if found
            if patient_details:
                patient_details_text = "--- PATIENT DETAILS ---\n"
                patient_details_text += f"Patient Name: {patient_details.get('patient_name', 'N/A')}\n"
                patient_details_text += f"Date of Birth: {patient_details.get('dob', 'N/A')}\n"
                patient_details_text += f"Date of Injury: {patient_details.get('doi', 'N/A')}\n"
                patient_details_text += f"Claim Number: {patient_details.get('claim_number', 'N/A')}\n"
                patient_details_text += "--- END PATIENT DETAILS ---\n\n"
                final_raw_text = patient_details_text + summary_text
            
            logger.info("=" * 80)
            logger.info("✅ BATCH PROCESSING COMPLETE")
            logger.info(f"   Summary length: {len(summary_text)} chars")
            logger.info(f"   Total pages: {total_pages}")
            logger.info(f"   Patient details: {'Found' if patient_details else 'Not found'}")
            logger.info("=" * 80)
            
            # Log the complete summary for debugging/verification
            logger.info("=" * 80)
            logger.info("🤖 COMPLETE BATCH SUMMARY OUTPUT:")
            logger.info("=" * 80)
            logger.info(summary_text)
            logger.info("=" * 80)
            
            # Run multi-report detection on batch summary (using o3-pro AI model)
            # multi_report_result = detect_multiple_reports(summary_text)
            
            return ExtractionResult(
                text=summary_text,
                raw_text=final_raw_text,
                llm_text=final_raw_text,
                page_zones=layout_data.get("page_zones", {}),
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
            raise  # Re-raise to trigger fallback to chunking
            
        finally:
            # Clean up GCS files
            if gcs_input_uri or gcs_output_uri:
                self._cleanup_batch_gcs_files(gcs_input_uri, gcs_output_uri)
    
    # ==================== END BATCH PROCESSING METHODS ====================

    def process_large_document(self, filepath: str) -> ExtractionResult:
        """
        Process large documents using batch API for full context understanding.
        Falls back to chunking method if batch API fails.
        """
        try:
            # First, try batch processing for better context understanding
            if self.storage_client and self.gcs_bucket:
                logger.info("🔄 Attempting batch processing for holistic summarization...")
                try:
                    return self._process_document_batch(filepath)
                except Exception as batch_error:
                    logger.warning(f"⚠️ Batch processing failed, falling back to chunking: {batch_error}")
            else:
                logger.warning("⚠️ GCS client not available, using chunking method")
            
            # Fallback to chunking method
            return self._process_large_document_chunked(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error processing large document: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")
    
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
            
            return self._merge_results(all_results, filepath)
            
        except Exception as e:
            logger.error(f"❌ Error in chunked processing: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")
    
    def _merge_results(self, results: List[ExtractionResult], original_file: str) -> ExtractionResult:
        """Merge results from multiple chunks"""
        if not results:
            return ExtractionResult(success=False, error="No successful chunks")
        
        merged_text = ""
        merged_raw_text = ""
        
        logger.info(f"🔗 Starting merge of {len(results)} summarizer chunks...")
        
        for i, result in enumerate(results):
            chunk_num = i + 1
            logger.info(f"📦 Processing chunk {chunk_num}:")
            logger.info(f"   - Has text: {bool(result.text)}")
            logger.info(f"   - Pages in chunk: {result.pages}")
            
            if result.text:
                if i > 0:
                    merged_text += f"\n\n{'='*80}\nCHUNK {i + 1}\n{'='*80}\n\n"
                merged_text += result.text
            
            if hasattr(result, 'raw_text') and result.raw_text:
                merged_raw_text += result.raw_text + "\n\n"
        
        total_pages = sum(r.pages for r in results)
        
        # Extract patient_details from the first chunk (only first chunk has patient details)
        patient_details = {}
        if results and hasattr(results[0], 'metadata') and results[0].metadata:
            patient_details = results[0].metadata.get('patient_details', {})
            if patient_details:
                logger.info("=" * 80)
                logger.info("👤 MERGED PATIENT DETAILS (from first chunk):")
                logger.info("=" * 80)
                logger.info(f"  Patient Name: {patient_details.get('patient_name', 'Not found')}")
                logger.info(f"  DOB: {patient_details.get('dob', 'Not found')}")
                logger.info(f"  DOI: {patient_details.get('doi', 'Not found')}")
                logger.info(f"  Claim Number: {patient_details.get('claim_number', 'Not found')}")
                logger.info("=" * 80)
                
                # Add patient details to merged raw text (at the very beginning, no leading newlines)
                patient_details_text = "--- PATIENT DETAILS ---\n"
                patient_details_text += f"Patient Name: {patient_details.get('patient_name', 'N/A')}\n"
                patient_details_text += f"Date of Birth: {patient_details.get('dob', 'N/A')}\n"
                patient_details_text += f"Date of Injury: {patient_details.get('doi', 'N/A')}\n"
                patient_details_text += f"Claim Number: {patient_details.get('claim_number', 'N/A')}\n"
                patient_details_text += "--- END PATIENT DETAILS ---\n\n"
                
                # Prepend to merged_raw_text
                merged_raw_text = patient_details_text + merged_raw_text
                
                logger.info("🔍 DEBUG: Patient details prepended to merged_raw_text")
                logger.info(f"🔍 First 500 chars of merged_raw_text:\n{merged_raw_text[:500]}")
        
        logger.info(f"🔗 Merge complete:")
        logger.info(f"   - Total pages: {total_pages}")
        logger.info(f"   - Total text-merged---------------: {(merged_raw_text)}")

        
        # Log the complete merged summarizer output
        logger.info("=" * 80)
        logger.info("🤖 COMPLETE MERGED SUMMARIZER OUTPUT (ALL CHUNKS COMBINED):")
        logger.info("=" * 80)
        logger.info(f"Total chunks processed: {len(results)}")
        logger.info(f"Total pages: {total_pages}")
        logger.info(f"Total characters: {len(merged_text)}")
        logger.info("=" * 80)
        logger.info(merged_text)
        logger.info("=" * 80)
        
        # Run multi-report detection on merged text (using o3-pro AI model)
        # multi_report_result = detect_multiple_reports(merged_text)
        
        merged_result = ExtractionResult(
            text=merged_raw_text,
            raw_text=merged_raw_text,
            llm_text=merged_raw_text,
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
        
        return merged_result
    
    def process_document(self, filepath: str) -> ExtractionResult:
        """
        Main document processing method using SUMMARIZER.
        
        Strategy based on page count:
        - ≤15 pages: Use synchronous API (fast, seconds)
        - >15 pages: Use batch API (slower but handles large docs with full context)
        
        Patient details are extracted from first 3 pages using layout parser.
        """
        try:
            mime_type = self.get_mime_type(filepath)
            
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
                        # Large documents need batch API for full context
                        logger.info(f"📦 Document has {page_count} pages (>15) - using batch API for full context")
                        logger.info("⏳ Note: Batch API takes 2-6 minutes but preserves full document context")
                        
                        if self.storage_client and self.gcs_bucket:
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
