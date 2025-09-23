import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from google.cloud.documentai_v1 import DocumentProcessorServiceClient
import logging

from models.schemas import ExtractionResult, FileInfo
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

class DocumentAIProcessor:
    """Service for Document AI processing"""
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.processor_path: Optional[str] = None
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
                # Remove credentials_path - it doesn't exist as a parameter!
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
            # Note: DOCX files should be converted to PDF before reaching this point
        }
        
        file_ext = Path(file_path).suffix.lower()
        mime_type = mime_mapping.get(file_ext, "application/octet-stream")
        
        # Log warning if DOCX reaches this point (should have been converted)
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
    
    def process_document(self, file_path: str) -> ExtractionResult:
        """Process document with Document AI"""
        try:
            mime_type = self.get_mime_type(file_path)
            logger.info(f"📄 Processing document: {file_path}")
            logger.info(f"📋 MIME type: {mime_type}")
            
            # Check file exists and get size
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"📊 File size: {file_size} bytes")
            
            # Read and encode file
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            logger.info(f"🔢 Content encoded, length: {len(encoded_content)} characters")
            
            # Create request
            request = {
                "name": self.processor_path,
                "raw_document": {
                    "content": encoded_content,
                    "mime_type": mime_type,
                },
            }
            
            logger.info(f"🚀 Sending request to Document AI...")
            logger.info(f"📍 Processor path: {self.processor_path}")
            
            # Process document
            response = self.client.process_document(request=request)
            result = response.document
            
            logger.info("✅ Document processed successfully!")
            logger.info(f"📝 Extracted text length: {len(result.text) if result.text else 0} characters")
            logger.info(f"📄 Pages found: {len(result.pages) if result.pages else 0}")
            
            # Log text preview
            if result.text:
                preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                logger.info(f"📖 Text preview: \"{preview}\"")
            
            # Build result
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
            logger.error(f"❌ Error processing document: {str(e)}")
            
            # Specific error handling
            if "permission" in str(e).lower():
                logger.error("🚫 Permission error - check service account roles")
            elif "credentials" in str(e).lower():
                logger.error("🔑 Credentials error - check JSON file path")
            elif "processor" in str(e).lower():
                logger.error("⚙️ Processor error - check processor ID and location")
            
            return ExtractionResult(
                success=False,
                error=str(e)
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