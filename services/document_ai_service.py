import os
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from google.cloud.documentai_v1 import DocumentProcessorServiceClient
import logging
from PyPDF2 import PdfReader, PdfWriter

from models.schemas import ExtractionResult, FileInfo
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

class PDFSplitter:
    """Utility to split large PDFs into smaller chunks"""
    
    def __init__(self, max_pages_per_chunk: int = 10):
        self.max_pages_per_chunk = max_pages_per_chunk
    
    def split_pdf(self, file_path: str) -> List[str]:
        """Split PDF into multiple chunks"""
        try:
            logger.info(f"📂 Splitting PDF: {file_path}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"📄 Total pages: {total_pages}")
                logger.info(f"📦 Max pages per chunk: {self.max_pages_per_chunk}")
                
                if total_pages <= self.max_pages_per_chunk:
                    logger.info("✅ No splitting needed")
                    return [file_path]
                
                # Calculate number of chunks needed
                num_chunks = (total_pages + self.max_pages_per_chunk - 1) // self.max_pages_per_chunk
                logger.info(f"🔢 Splitting into {num_chunks} chunks")
                
                chunk_files = []
                
                for chunk_num in range(num_chunks):
                    start_page = chunk_num * self.max_pages_per_chunk
                    end_page = min((chunk_num + 1) * self.max_pages_per_chunk, total_pages)
                    
                    chunk_file = self._create_chunk(pdf_reader, file_path, start_page, end_page, chunk_num)
                    chunk_files.append(chunk_file)
                    
                    logger.info(f"✅ Created chunk {chunk_num + 1}: pages {start_page + 1}-{end_page}")
                
                return chunk_files
                
        except Exception as e:
            logger.error(f"❌ Error splitting PDF: {str(e)}")
            raise
    
    def _create_chunk(self, pdf_reader, original_path: str, start: int, end: int, chunk_num: int) -> str:
        """Create a single PDF chunk"""
        pdf_writer = PdfWriter()
        
        # Add pages to chunk
        for page_num in range(start, end):
            pdf_writer.add_page(pdf_reader.pages[page_num])
        
        # Create output filename with unique identifier to avoid conflicts
        original_stem = Path(original_path).stem
        # Add timestamp to make filename unique
        timestamp = datetime.now().strftime("%H%M%S%f")
        output_filename = f"{original_stem}_chunk_{chunk_num + 1}_{timestamp}.pdf"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Write chunk to file
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        return output_path

    def cleanup_chunks(self, chunk_files: List[str]):
        """Clean up temporary chunk files"""
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file) and "_chunk_" in chunk_file:
                    os.remove(chunk_file)
                    logger.debug(f"🧹 Cleaned up: {chunk_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up {chunk_file}: {str(e)}")

class DocumentAIProcessor:
    """Service for Document AI processing with PDF splitting"""
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.processor_path: Optional[str] = None
        self.pdf_splitter = PDFSplitter(max_pages_per_chunk=10)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Document AI client with error handling"""
        try:
            # Set credentials via environment variable (this is correct)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CONFIG["credentials_path"]
            
            logger.info(f"🔑 Using credentials: {CONFIG['credentials_path']}")
            logger.info(f"📍 Project ID: {CONFIG['project_id']}")
            logger.info(f"🌍 Location: {CONFIG['location']}")
            logger.info(f"⚙️ Processor ID: {CONFIG['processor_id']}")
            
            # API endpoint
            api_endpoint = f"{CONFIG['location']}-documentai.googleapis.com"
            logger.info(f"🔗 API Endpoint: {api_endpoint}")
            
            # Initialize client WITHOUT credentials_path parameter
            # The client will automatically use GOOGLE_APPLICATION_CREDENTIALS env var
            self.client = DocumentProcessorServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            
            # Build processor path
            self.processor_path = self.client.processor_path(
                CONFIG["project_id"],
                CONFIG["location"],
                CONFIG["processor_id"]
            )
            
            logger.info(f"✅ Document AI Client initialized successfully")
            logger.info(f"📄 Processor path: {self.processor_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document AI Client: {str(e)}")
            raise
    
    def get_mime_type(self, file_path: str) -> str:
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
        
        file_ext = Path(file_path).suffix.lower()
        mime_type = mime_mapping.get(file_ext, "application/octet-stream")
        
        if file_ext in ['.docx', '.pptx', '.xlsx']:
            logger.warning(f"⚠️ Office document format reached Document AI service: {file_ext}")
            logger.warning("This file should have been converted to PDF first!")
        
        return mime_type
    
    def extract_text_from_layout(self, text_anchor, full_text: str) -> str:
        """Extract text from layout text anchor"""
        if not text_anchor or not text_anchor.text_segments:
            return ""
        
        start_index = text_anchor.text_segments[0].start_index or 0
        end_index = text_anchor.text_segments[0].end_index
        
        return full_text[start_index:end_index] if end_index else ""
    
    def extract_entities(self, document) -> List[Dict[str, Any]]:
        """Extract entities from document"""
        entities = []
        if document.entities:
            for entity in document.entities:
                entities.append({
                    "type": entity.type_,
                    "mentionText": entity.mention_text,
                    "confidence": float(entity.confidence),
                    "id": entity.id
                })
            logger.info(f"🏷️ Found {len(entities)} entities")
        return entities
    
    def extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        tables = []
        
        if not document.pages:
            return tables
        
        for page_index, page in enumerate(document.pages):
            if page.tables:
                for table_index, table in enumerate(page.tables):
                    table_data = {
                        "pageNumber": page_index + 1,
                        "tableIndex": table_index + 1,
                        "headerRows": [],
                        "bodyRows": []
                    }
                    
                    # Extract header rows
                    if table.header_rows:
                        for row in table.header_rows:
                            row_data = []
                            for cell in row.cells:
                                cell_text = self.extract_text_from_layout(
                                    cell.layout.text_anchor, document.text
                                ) if cell.layout and cell.layout.text_anchor else ""
                                row_data.append({
                                    "text": cell_text.strip(),
                                    "confidence": float(cell.layout.confidence) if cell.layout else 0.0
                                })
                            table_data["headerRows"].append(row_data)
                    
                    # Extract body rows
                    if table.body_rows:
                        for row in table.body_rows:
                            row_data = []
                            for cell in row.cells:
                                cell_text = self.extract_text_from_layout(
                                    cell.layout.text_anchor, document.text
                                ) if cell.layout and cell.layout.text_anchor else ""
                                row_data.append({
                                    "text": cell_text.strip(),
                                    "confidence": float(cell.layout.confidence) if cell.layout else 0.0
                                })
                            table_data["bodyRows"].append(row_data)
                    
                    tables.append(table_data)
        
        if tables:
            logger.info(f"📊 Found {len(tables)} tables")
        return tables
    
    def extract_form_fields(self, document) -> List[Dict[str, Any]]:
        """Extract form fields from document"""
        form_fields = []
        
        if not document.pages:
            return form_fields
        
        for page in document.pages:
            if page.form_fields:
                for field in page.form_fields:
                    field_name = ""
                    field_value = ""
                    
                    if field.field_name and field.field_name.text_anchor:
                        field_name = self.extract_text_from_layout(
                            field.field_name.text_anchor, document.text
                        )
                    
                    if field.field_value and field.field_value.text_anchor:
                        field_value = self.extract_text_from_layout(
                            field.field_value.text_anchor, document.text
                        )
                    
                    form_fields.append({
                        "name": field_name.strip(),
                        "value": field_value.strip(),
                        "nameConfidence": float(field.field_name.confidence) if field.field_name else 0.0,
                        "valueConfidence": float(field.field_value.confidence) if field.field_value else 0.0
                    })
        
        if form_fields:
            logger.info(f"📋 Found {len(form_fields)} form fields")
        return form_fields
    
    def calculate_overall_confidence(self, document) -> float:
        """Calculate overall confidence score"""
        if not document.pages:
            return 0.0
        
        total_confidence = 0
        count = 0
        
        for page in document.pages:
            if page.layout and page.layout.confidence:
                total_confidence += float(page.layout.confidence)
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def process_large_document(self, file_path: str) -> ExtractionResult:
        """Process large documents by splitting them first"""
        try:
            # Split PDF if needed
            chunk_files = self.pdf_splitter.split_pdf(file_path)
            
            if len(chunk_files) == 1:
                logger.info("📄 Processing as single document")
                return self._process_document_direct(chunk_files[0])
            
            logger.info(f"🔄 Processing {len(chunk_files)} chunks")
            
            all_results = []
            processed_chunks = set()  # Local tracking for this document only
            
            for i, chunk_file in enumerate(chunk_files):
                logger.info(f"🔍 Processing chunk {i + 1}/{len(chunk_files)}: {chunk_file}")
                
                # Skip if we've already processed this specific chunk in this session
                if chunk_file in processed_chunks:
                    logger.warning(f"⚠️ Skipping duplicate chunk: {chunk_file}")
                    continue
                    
                processed_chunks.add(chunk_file)
                
                try:
                    chunk_result = self._process_document_direct(chunk_file)
                    if chunk_result.success:
                        all_results.append(chunk_result)
                        logger.info(f"✅ Chunk {i + 1} processed successfully")
                    else:
                        logger.error(f"❌ Chunk {i + 1} failed: {chunk_result.error}")
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i + 1}: {str(e)}")
            
            # Cleanup temporary files
            self.pdf_splitter.cleanup_chunks(chunk_files)
            
            # Merge results
            if not all_results:
                logger.error("❌ All chunks failed to process")
                return ExtractionResult(
                    success=False,
                    error="All chunks failed to process"
                )
            
            merged_result = self._merge_results(all_results, file_path)
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ Error processing large document: {str(e)}")
            return ExtractionResult(
                success=False,
                error=f"Failed to process large document: {str(e)}"
            )
    
    def _process_document_direct(self, file_path: str) -> ExtractionResult:
        """Direct document processing without recursion"""
        try:
            mime_type = self.get_mime_type(file_path)
            
            logger.info(f"📄 Processing document: {file_path}")
            logger.info(f"📋 MIME type: {mime_type}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"📊 File size: {file_size} bytes")
            
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            logger.info(f"🔢 Content encoded, length: {len(encoded_content)} characters")
            
            request = {
                "name": self.processor_path,
                "raw_document": {
                    "content": encoded_content,
                    "mime_type": mime_type,
                },
            }
            
            logger.info(f"🚀 Sending request to Document AI...")
            
            response = self.client.process_document(request=request)
            result = response.document
            
            logger.info("✅ Document processed successfully!")
            logger.info(f"📝 Extracted text length: {len(result.text) if result.text else 0} characters")
            logger.info(f"📄 Pages found: {len(result.pages) if result.pages else 0}")
            
            if result.text:
                preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                logger.info(f"📖 Text preview: \"{preview}\"")
            
            processed_result = ExtractionResult(
                text=result.text or "",
                pages=len(result.pages) if result.pages else 0,
                entities=self.extract_entities(result),
                tables=self.extract_tables(result),
                formFields=self.extract_form_fields(result),
                confidence=self.calculate_overall_confidence(result),
                success=True
            )
            
            logger.info("📊 Extraction summary:")
            logger.info(f"   - Text characters: {len(processed_result.text)}")
            logger.info(f"   - Pages: {processed_result.pages}")
            logger.info(f"   - Entities: {len(processed_result.entities)}")
            logger.info(f"   - Tables: {len(processed_result.tables)}")
            logger.info(f"   - Form fields: {len(processed_result.formFields)}")
            logger.info(f"   - Overall confidence: {(processed_result.confidence * 100):.2f}%")
            
            return processed_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error processing document: {error_msg}")
            
            # Check if it's a page limit error
            if any(keyword in error_msg for keyword in ["PAGE_LIMIT_EXCEEDED", "pages exceed the limit", "page_limit", "non-imageless mode"]):
                logger.error(f"🚨 Page limit exceeded: {error_msg}")
                
                # For documents that exceed the limit, try splitting with smaller chunks
                if mime_type == "application/pdf":
                    try:
                        with open(file_path, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            page_count = len(pdf_reader.pages)
                            
                            logger.info(f"📄 Document has {page_count} pages, trying smaller chunks...")
                            # Use smaller chunks for this specific document
                            self.pdf_splitter.max_pages_per_chunk = 5
                            return self.process_large_document(file_path)
                    except Exception as pdf_error:
                        logger.error(f"❌ Could not check PDF page count: {str(pdf_error)}")
                
                return ExtractionResult(
                    success=False,
                    error=f"Page limit exceeded: {error_msg}"
                )
            
            # Other error handling
            if "permission" in error_msg.lower():
                logger.error("🚫 Permission error - check service account roles")
            elif "credentials" in error_msg.lower():
                logger.error("🔑 Credentials error - check JSON file path")
            elif "processor" in error_msg.lower():
                logger.error("⚙️ Processor error - check processor ID and location")
            
            return ExtractionResult(
                success=False,
                error=error_msg
            )
    
    def _merge_results(self, results: List[ExtractionResult], original_file: str) -> ExtractionResult:
        """Merge results from multiple chunks"""
        if not results:
            return ExtractionResult(success=False, error="No successful chunks to merge")
        
        # Merge text (add page breaks between chunks)
        merged_text = ""
        for i, result in enumerate(results):
            if result.text:
                if i > 0:
                    merged_text += f"\n\n--- Chunk {i + 1} ---\n\n"
                merged_text += result.text
        
        # Merge other fields
        merged_entities = []
        merged_tables = []
        merged_form_fields = []
        total_pages = 0
        total_confidence = 0.0
        
        for result in results:
            merged_entities.extend(result.entities)
            merged_tables.extend(result.tables)
            merged_form_fields.extend(result.formFields)
            total_pages += result.pages
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        logger.info(f"✅ Merged {len(results)} chunks:")
        logger.info(f"   - Total pages: {total_pages}")
        logger.info(f"   - Total text characters: {len(merged_text)}")
        logger.info(f"   - Total entities: {len(merged_entities)}")
        logger.info(f"   - Total tables: {len(merged_tables)}")
        logger.info(f"   - Total form fields: {len(merged_form_fields)}")
        logger.info(f"   - Average confidence: {(avg_confidence * 100):.2f}%")
        
        return ExtractionResult(
            text=merged_text,
            pages=total_pages,
            entities=merged_entities,
            tables=merged_tables,
            formFields=merged_form_fields,
            confidence=avg_confidence,
            success=True
        )

    def process_document(self, file_path: str) -> ExtractionResult:
        """Main document processing method - split documents with more than 10 pages"""
        try:
            # Check if it's a PDF and page count
            mime_type = self.get_mime_type(file_path)
            
            if mime_type == "application/pdf":
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        page_count = len(pdf_reader.pages)
                        
                        logger.info(f"📄 Document has {page_count} pages")
                        
                        # Split documents with more than 10 pages
                        if page_count > 10:
                            logger.info(f"🔄 Document has {page_count} pages (>10), using chunked processing")
                            return self.process_large_document(file_path)
                        else:
                            logger.info(f"✅ Document has {page_count} pages (≤10), processing directly")
                            return self._process_document_direct(file_path)
                except Exception as e:
                    logger.warning(f"⚠️ Could not check PDF page count: {str(e)}")
                    # Fall back to direct processing
                    return self._process_document_direct(file_path)
            else:
                # Non-PDF files processed directly
                logger.info("📄 Non-PDF document, processing directly")
                return self._process_document_direct(file_path)
                
        except Exception as e:
            logger.error(f"❌ Error in main document processing: {str(e)}")
            return ExtractionResult(
                success=False,
                error=f"Main processing failed: {str(e)}"
            )

# Global processor instance
_processor_instance = None

def get_document_ai_processor() -> DocumentAIProcessor:
    """Get singleton DocumentAIProcessor instance"""
    global _processor_instance
    if _processor_instance is None:
        try:
            logger.info("🚀 Initializing Document AI processor...")
            _processor_instance = DocumentAIProcessor()
            logger.info("✅ Document AI processor ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document AI processor: {str(e)}")
            raise
    return _processor_instance

def process_document_smart(file_path: str) -> ExtractionResult:
    """Smart document processing with chunking for documents >10 pages"""
    processor = get_document_ai_processor()
    return processor.process_document(file_path)