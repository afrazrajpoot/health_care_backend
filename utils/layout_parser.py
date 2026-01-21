# layout_parser.py

"""
Layout Parser utilities for Document AI.
Handles document structure extraction and field-value pair detection.
"""

import os
import tempfile
import time
import logging
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader, PdfWriter
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient, types
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger("document_ai")


# Common field keywords for medical documents
MEDICAL_FIELD_KEYWORDS = [
    "claim", "claim number", "claim no", "claim#", "cl", "claim #",
    "dob", "date of birth", "birth date",
    "doi", "date of injury", "injury date",
    "patient", "applicant", "name",
    "ssn", "social security",
    "panel", "panel no",
    "employer",
    "signed", "signature", "author"
]


def extract_fields_from_blocks(blocks: List[Dict], field_keywords: List[str] = None) -> str:
    """
    Extract field-value pairs from blocks using nearest neighbor logic.
    Handles both 'field: value' and 'value field' patterns.
    
    Args:
        blocks: List of document blocks from layout parser
        field_keywords: List of keywords to identify as fields (defaults to medical keywords)
    
    Returns:
        Formatted string with field-value pairs
    """
    if field_keywords is None:
        field_keywords = MEDICAL_FIELD_KEYWORDS
    
    # Validate that blocks is actually a list
    if not isinstance(blocks, list):
        logger.warning(f"⚠️ Blocks is not a list (type: {type(blocks).__name__})")
        return ""
    
    formatted_lines = []
    i = 0
    
    while i < len(blocks):
        current_block = blocks[i]
        
        # Ensure current_block is a dictionary
        if not isinstance(current_block, dict):
            logger.warning(f"⚠️ Block at index {i} is not a dict (type: {type(current_block).__name__})")
            i += 1
            continue
        
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
                
                # Ensure next_block is also a dictionary
                if not isinstance(next_block, dict):
                    formatted_lines.append(current_text)
                    i += 1
                    continue
                
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
                
                # Ensure next_block is a dictionary
                if not isinstance(next_block, dict):
                    formatted_lines.append(current_text)
                    i += 1
                    continue
                
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


class LayoutParser:
    """Handles document layout parsing using Google Document AI Layout Parser"""
    
    def __init__(
        self,
        client: DocumentProcessorServiceClient,
        processor_path: Optional[str] = None
    ):
        self.client = client
        self.processor_path = processor_path
    
    @property
    def is_configured(self) -> bool:
        """Check if layout parser is configured"""
        return self.processor_path is not None
    
    def process_pages(
        self,
        filepath: str,
        page_numbers: List[int],
        mime_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        Process specific pages with Layout Parser to preserve structure.
        
        Args:
            filepath: Path to the document
            page_numbers: List of page numbers to process (1-indexed)
            mime_type: MIME type of the document
        
        Returns:
            Dictionary with structured layout data including blocks
        """
        if not self.is_configured:
            logger.warning("⚠️ Layout Parser not configured, skipping layout extraction")
            return {"blocks": [], "text": "", "document_dict": {}, "formatted_text": ""}
        
        temp_pdf_path = None
        
        try:
            # Create PDF with only requested pages
            if len(page_numbers) > 0:
                with open(filepath, "rb") as file:
                    pdf_reader = PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    # Filter valid page numbers
                    valid_pages = [p for p in page_numbers if 0 < p <= total_pages]
                    
                    if not valid_pages:
                        return {"blocks": [], "text": "", "document_dict": {}, "formatted_text": ""}
                    
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
            
            # Create raw document
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=mime_type
            )
            
            # Create process request for layout parser
            request = documentai.ProcessRequest(
                name=self.processor_path,
                raw_document=raw_document
            )
            
            logger.info(f"📤 Sending {len(page_numbers)} pages to Layout Parser...")
            
            # Process with retry logic
            response = self._process_with_retry(request)
            processed_document = response.document
            
            # Convert the entire Document object to dictionary
            document_dict = types.Document.to_dict(processed_document)
            
            # Build formatted text with field-value detection
            formatted_text = ""
            blocks = []
            
            if document_dict.get('document_layout') and document_dict['document_layout'].get('blocks'):
                blocks = document_dict['document_layout']['blocks']
                logger.info(f"✅ Layout Parser extracted {len(blocks)} blocks from document_layout")
                
                # Validate block structure
                if blocks and len(blocks) > 0:
                    first_block_type = type(blocks[0]).__name__
                    if first_block_type != 'dict':
                        logger.warning(f"⚠️ Blocks contain {first_block_type} instead of dict - blocks may not be properly structured")
                        # Try to extract text directly if blocks are strings
                        if isinstance(blocks[0], str):
                            formatted_text = "\n".join(str(b) for b in blocks if b)
                        else:
                            formatted_text = ""
                    else:
                        formatted_text = extract_fields_from_blocks(blocks)
                else:
                    formatted_text = ""
            else:
                logger.warning(f"⚠️ No document_layout blocks found in response")
            
            # Log document structure for debugging
            logger.info(f"📊 Document structure keys: {list(document_dict.keys())}")
            if 'pages' in document_dict:
                pages = document_dict['pages']
                # Handle case where pages might be a string or other non-list type
                if isinstance(pages, list):
                    logger.info(f"📄 Total pages in layout result: {len(pages)}")
                    # Count total blocks in pages if available
                    if not blocks:  # Only if we didn't get blocks from document_layout
                        total_blocks = sum(len(page.get('blocks', [])) for page in pages if isinstance(page, dict))
                        if total_blocks > 0:
                            logger.info(f"📄 Found {total_blocks} blocks in pages structure")
                else:
                    logger.warning(f"⚠️ 'pages' is not a list (type: {type(pages).__name__})")
            
            return {
                "document_dict": document_dict,
                "blocks": blocks,
                "formatted_text": formatted_text,
                "full_text": document_dict.get('text', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Layout Parser error: {e}")
            return {"document_dict": {}, "blocks": [], "formatted_text": "", "full_text": ""}
            
        finally:
            # Clean up temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    def _process_with_retry(
        self,
        request: documentai.ProcessRequest,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        """Process document with retry logic for transient errors"""
        retry_delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.process_document(request=request)
                return response
            except google_exceptions.Unknown as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Layout parser error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        
        raise Exception("Max retries exceeded")