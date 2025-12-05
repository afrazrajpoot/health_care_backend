# document_ai_service.py

"""
Enhanced Document AI Service using Document Summarizer processor.
REPLACED doc-ocr with document-summarizer for better accuracy.
"""

import os
import base64
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient, types
from google.api_core import exceptions as google_exceptions
import logging
from PyPDF2 import PdfReader, PdfWriter
from models.schemas import ExtractionResult, FileInfo
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

class LayoutPreservingTextExtractor:
    """
    Extract text from Document AI Summarizer results.
    Simplified since summarizer returns clean text without complex layout.
    """
    
    @staticmethod
    def extract_text_from_summarizer(document) -> Dict[str, Any]:
        """
        Extract text from summarizer response.
        Summarizer returns clean, organized text with natural paragraph structure.
        """
        summarized_text = document.text or ""
        
        # Extract page count if available
        page_count = len(document.pages) if document.pages else 0
        
        # Build simple structure for compatibility with existing code
        return {
            "layout_preserved": summarized_text,
            "raw_text": summarized_text,
            "page_zones": {},  # Summarizer doesn't provide detailed zones
            "structured_document": {
                "document_structure": {
                    "total_pages": page_count,
                    "summarized": True
                },
                "pages": [],
                "metadata": {
                    "has_summary": True,
                    "total_chars": len(summarized_text)
                }
            }
        }

def build_llm_friendly_json(structured_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build LLM-optimized JSON structure from summarizer output.
    """
    return {
        "document_type_hints": {
            "header_text": "",
            "first_page_context": "",
            "has_form_structure": False,
            "is_summarized": True
        },
        "content": {
            "summary": structured_document.get("summarized_text", ""),
            "page_count": structured_document.get("document_structure", {}).get("total_pages", 0)
        },
        "metadata": structured_document.get("metadata", {})
    }

class PDFSplitter:
    """Utility to split large PDFs into smaller chunks"""
    
    def __init__(self, max_pages_per_chunk: int = 10):
        self.max_pages_per_chunk = max_pages_per_chunk
    
    def split_pdf(self, filepath: str) -> List[str]:
        """Split PDF into multiple chunks"""
        try:
            logger.info(f"🔍 Splitting PDF: {filepath}")
            with open(filepath, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
            
            logger.info(f"📄 Total pages: {total_pages}")
            logger.info(f"📦 Max pages per chunk: {self.max_pages_per_chunk}")
            
            if total_pages <= self.max_pages_per_chunk:
                logger.info("✅ No splitting needed")
                return [filepath]
            
            num_chunks = (total_pages + self.max_pages_per_chunk - 1) // self.max_pages_per_chunk
            logger.info(f"✂️ Splitting into {num_chunks} chunks")
            
            chunk_files = []
            for chunk_num in range(num_chunks):
                start_page = chunk_num * self.max_pages_per_chunk
                end_page = min((chunk_num + 1) * self.max_pages_per_chunk, total_pages)
                chunk_file = self._create_chunk(filepath, start_page, end_page, chunk_num)
                chunk_files.append(chunk_file)
                logger.info(f"✅ Created chunk {chunk_num + 1}: pages {start_page + 1}-{end_page}")
            
            return chunk_files
            
        except Exception as e:
            logger.error(f"❌ Error splitting PDF: {str(e)}")
            raise
    
    def _create_chunk(self, original_path: str, start: int, end: int, chunk_num: int) -> str:
        """Create a single PDF chunk"""
        with open(original_path, "rb") as file:
            pdf_reader = PdfReader(file)
            pdf_writer = PdfWriter()
            
            for page_num in range(start, end):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            original_stem = Path(original_path).stem
            timestamp = datetime.now().strftime("%H%M%S%f")
            output_filename = f"{original_stem}_chunk{chunk_num + 1}_{timestamp}.pdf"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            with open(output_path, "wb") as output_file:
                pdf_writer.write(output_file)
            
            return output_path
    
    def cleanup_chunks(self, chunk_files: List[str]):
        """Clean up temporary chunk files"""
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file) and "chunk" in chunk_file:
                    os.remove(chunk_file)
                    logger.debug(f"🧹 Cleaned up {chunk_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up {chunk_file}: {str(e)}")

class DocumentAIProcessor:
    """Service for Document AI Summarizer processing"""
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.summarizer_processor_path: Optional[str] = None
        self.layout_parser_processor_path: Optional[str] = None  # NEW: Layout parser
        self.pdf_splitter = PDFSplitter(max_pages_per_chunk=10)
        self.layout_extractor = LayoutPreservingTextExtractor()
        self._initialize_client()
    
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
            
            # Clean up temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            logger.info(f"✅ Layout Parser extracted {len(blocks)} blocks")
            
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
    
    def _process_document_direct(self, filepath: str) -> ExtractionResult:
        """
        Process document using Document AI SUMMARIZER (not OCR).
        FIXED: Extract summary field instead of full text.
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
            
            # Also log full text comparison for debugging
            if result.text and len(result.text) != len(summary_text):
                logger.info("=" * 80)
                logger.info("📄 FULL OCR TEXT (for comparison):")
                logger.info("=" * 80)
                logger.info(f"Full text length: {len(result.text)} chars")
                logger.info(f"First 500 chars: {result.text}")
                logger.info("=" * 80)
            
            # Extract text using simplified extractor
            layout_data = self.layout_extractor.extract_text_from_summarizer(result)
            # Override with actual summary
            layout_data["layout_preserved"] = summary_text
            layout_data["raw_text"] = summary_text
            layout_data["structured_document"]["metadata"]["summary_chars"] = len(summary_text)
            layout_data["structured_document"]["metadata"]["full_text_chars"] = len(result.text) if result.text else 0
            
            logger.info(f"🔍 Text extraction complete:")
            logger.info(f"   - Summary text: {len(summary_text)} chars")
            logger.info(f"   - Pages analyzed: {layout_data['structured_document']['document_structure']['total_pages']}")
            
            # NEW: Extract first 3 and last 3 pages with Layout Parser
            layout_extracted_text = ""
            total_pages = len(result.pages) if result.pages else 0
            
            if total_pages > 0:
                # Determine which pages to extract
                if total_pages <= 2:
                    # Process all pages if only 2 or fewer
                    pages_to_extract = list(range(1, total_pages + 1))
                else:
                    # First 3 and last 3 pages
                    first_pages = list(range(1, min(4, total_pages + 1)))
                    last_pages = list(range(max(total_pages - 2, 4), total_pages + 1))
                    pages_to_extract = sorted(set(first_pages + last_pages))
                
                logger.info(f"📄 Extracting layout from pages: {pages_to_extract}")
                layout_result = self._process_with_layout_parser(filepath, pages_to_extract)
                
                # Get full structured JSON from layout parser
                document_dict = layout_result.get("document_dict", {})
                formatted_text = layout_result.get("formatted_text", "")
                
                if document_dict:
                    logger.info(f"✅ Layout Parser extracted {len(layout_result.get('blocks', []))} blocks")
                    # Append both formatted text and full JSON to raw_text
                    json_output = json.dumps(document_dict, indent=2, ensure_ascii=False)
                    layout_data["raw_text"] = (
                        summary_text + 
                        "\n\n--- STRUCTURED LAYOUT (Formatted) ---\n\n" + formatted_text +
                        "\n\n--- STRUCTURED LAYOUT (Full JSON) ---\n\n" + json_output
                    )
            
            # Build LLM-friendly JSON
            llm_json = build_llm_friendly_json(layout_data['structured_document'])
            llm_json["content"]["summary"] = summary_text
            llm_text = json.dumps(llm_json, indent=2)

            logger.info(f'summary text----------------- : {summary_text}')
            
            # Create extraction result with merged raw_text
            processed_result = ExtractionResult(
                text=result.text,  #full OCR text
                raw_text=layout_data["raw_text"],  # Summary + Layout Parser structured text
                llm_text=llm_text,
                page_zones=layout_data["page_zones"],
                pages=len(result.pages) if result.pages else 0,
                entities=[],
                tables=[],
                formFields=[],
                symbols=[],
                confidence=1.0,
                success=True,
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

    def process_large_document(self, filepath: str) -> ExtractionResult:
        """Process large documents by splitting them first"""
        try:
            chunk_files = self.pdf_splitter.split_pdf(filepath)
            
            if len(chunk_files) == 1:
                return self._process_document_direct(chunk_files[0])
            
            logger.info(f"📦 Processing {len(chunk_files)} chunks with summarizer")
            
            all_results = []
            for i, chunk_file in enumerate(chunk_files):
                logger.info(f"🔄 Processing chunk {i + 1}/{len(chunk_files)} with summarizer")
                try:
                    chunk_result = self._process_document_direct(chunk_file)
                    if chunk_result.success:
                        all_results.append(chunk_result)
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i + 1}: {str(e)}")
            
            self.pdf_splitter.cleanup_chunks(chunk_files)
            
            if not all_results:
                return ExtractionResult(success=False, error="All chunks failed")
            
            return self._merge_results(all_results, filepath)
            
        except Exception as e:
            logger.error(f"❌ Error processing large document: {str(e)}")
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
        )
        
        return merged_result
    
    def process_document(self, filepath: str) -> ExtractionResult:
        """Main document processing method using SUMMARIZER"""
        try:
            mime_type = self.get_mime_type(filepath)
            
            if mime_type == "application/pdf":
                try:
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        page_count = len(pdf_reader.pages)
                    
                    logger.info(f"📄 Document has {page_count} pages")
                    
                    if page_count > 10:
                        logger.info("📦 Using chunked processing with summarizer")
                        return self.process_large_document(filepath)
                    else:
                        return self._process_document_direct(filepath)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not check page count: {str(e)}")
                    return self._process_document_direct(filepath)
            else:
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
