"""
Page Extractor - Extracts text from specific pages of a document
Supports PDF, DOCX, and other document formats
"""
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
from services.file_service import FileService
from services.document_ai_service import get_document_ai_processor
from services.document_converter import DocumentConverter
from utils.logger import logger

logger = logging.getLogger("page_extractor")


class PageExtractor:
    """
    Extracts text from specific pages of a document.
    Can work with PDF files from GCS or local files.
    """
    
    def __init__(self):
        self.file_service = FileService()
        self.document_processor = get_document_ai_processor()
    
    def extract_pages_from_pdf(self, pdf_content: bytes, page_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text from specific page ranges of a PDF.
        
        Args:
            pdf_content: PDF file content as bytes
            page_ranges: List of dicts with 'start_page', 'end_page', and 'report_title'
                Example: [
                    {"start_page": 1, "end_page": 5, "report_title": "QME"},
                    {"start_page": 6, "end_page": 10, "report_title": "PR2"}
                ]
        
        Returns:
            List of extracted page groups with text content
        """
        temp_file_path = None
        try:
            # Validate PDF content
            if not pdf_content or len(pdf_content) < 100:
                raise ValueError("PDF content is empty or too small")
            
            # Check if it's a valid PDF (starts with %PDF)
            if not pdf_content.startswith(b'%PDF'):
                raise ValueError("Invalid PDF format: file does not start with %PDF marker")
            
            # Save PDF to temp file with proper flushing
            temp_file_path = tempfile.mktemp(suffix='.pdf')
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(pdf_content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
            
            # Read PDF to get total pages (use strict=False to handle some corrupted PDFs)
            try:
                with open(temp_file_path, "rb") as f:
                    pdf_reader = PdfReader(f, strict=False)
                    total_pages = len(pdf_reader.pages)
            except Exception as pdf_error:
                logger.error(f"❌ Error reading PDF: {str(pdf_error)}")
                raise ValueError(f"Failed to read PDF file: {str(pdf_error)}")
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"📄 PDF has {total_pages} total pages")
            
            extracted_groups = []
            
            for idx, page_range in enumerate(page_ranges):
                chunk_temp_path = None
                try:
                    start_page = page_range.get("start_page", 1)
                    end_page = page_range.get("end_page", start_page)
                    report_title = page_range.get("report_title", f"Report_{idx + 1}")
                    
                    # Validate page numbers (1-indexed)
                    start_page = max(1, min(start_page, total_pages))
                    end_page = max(start_page, min(end_page, total_pages))
                    
                    logger.info(f"📄 Extracting pages {start_page}-{end_page} for {report_title}")
                    
                    # Re-read PDF for each chunk to avoid issues with shared reader
                    with open(temp_file_path, "rb") as f:
                        pdf_reader = PdfReader(f, strict=False)
                        
                        # Create a temporary PDF with only the specified pages
                        from PyPDF2 import PdfWriter
                        pdf_writer = PdfWriter()
                        
                        # Pages are 0-indexed in PyPDF2
                        for page_num in range(start_page - 1, end_page):
                            if page_num < len(pdf_reader.pages):
                                try:
                                    pdf_writer.add_page(pdf_reader.pages[page_num])
                                except Exception as page_error:
                                    logger.warning(f"⚠️ Error adding page {page_num + 1}: {str(page_error)}")
                                    continue
                        
                        # Save chunk to temp file with proper flushing
                        chunk_temp_path = tempfile.mktemp(suffix='.pdf')
                        with open(chunk_temp_path, "wb") as chunk_file:
                            pdf_writer.write(chunk_file)
                            chunk_file.flush()
                            os.fsync(chunk_file.fileno())  # Force write to disk
                    
                    # Process the chunk with Document AI to extract text
                    try:
                        extraction_result = self.document_processor.process_document(chunk_temp_path)
                        
                        if extraction_result.success and extraction_result.text:
                            extracted_text = extraction_result.text
                            raw_text = extraction_result.raw_text if hasattr(extraction_result, 'raw_text') else extracted_text
                            
                            extracted_groups.append({
                                "report_title": report_title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "text": extracted_text,
                                "raw_text": raw_text,
                                "page_count": end_page - start_page + 1
                            })
                            
                            logger.info(f"✅ Extracted {len(extracted_text)} characters from pages {start_page}-{end_page}")
                        else:
                            logger.warning(f"⚠️ Failed to extract text from pages {start_page}-{end_page}")
                            extracted_groups.append({
                                "report_title": report_title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "text": "",
                                "raw_text": "",
                                "page_count": end_page - start_page + 1,
                                "error": "Failed to extract text"
                            })
                    
                    except Exception as e:
                        logger.error(f"❌ Error processing pages {start_page}-{end_page}: {str(e)}", exc_info=True)
                        extracted_groups.append({
                            "report_title": report_title,
                            "start_page": start_page,
                            "end_page": end_page,
                            "text": "",
                            "raw_text": "",
                            "page_count": end_page - start_page + 1,
                            "error": str(e)
                        })
                    
                finally:
                    # Clean up chunk temp file
                    if chunk_temp_path:
                        try:
                            if os.path.exists(chunk_temp_path):
                                os.remove(chunk_temp_path)
                        except Exception as cleanup_error:
                            logger.warning(f"⚠️ Failed to cleanup chunk temp file: {str(cleanup_error)}")
            
            return extracted_groups
            
        except Exception as e:
            logger.error(f"❌ Error extracting pages: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up main temp file
            if temp_file_path:
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to cleanup main temp file: {str(cleanup_error)}")
    
    def _detect_file_type(self, content: bytes, blob_path: str) -> str:
        """
        Detect file type from content and blob path.
        
        Returns:
            File extension (e.g., '.pdf', '.docx')
        """
        # Check blob path extension first
        blob_ext = Path(blob_path).suffix.lower()
        if blob_ext in ['.pdf', '.docx', '.doc', '.pptx', '.xlsx']:
            return blob_ext
        
        # Check content headers
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif content.startswith(b'PK\x03\x04'):  # ZIP-based formats (DOCX, PPTX, XLSX)
            # Check for DOCX signature
            if b'word/' in content[:2000] or b'[Content_Types].xml' in content[:2000]:
                return '.docx'
            elif b'ppt/' in content[:2000]:
                return '.pptx'
            elif b'xl/' in content[:2000]:
                return '.xlsx'
            return '.docx'  # Default to DOCX for ZIP files
        
        # Default to PDF if unknown
        logger.warning(f"⚠️ Could not detect file type for {blob_path}, defaulting to PDF")
        return '.pdf'
    
    def extract_pages_from_docx(self, docx_content: bytes, page_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text from DOCX file by converting to PDF first, then extracting pages.
        This ensures accurate page-based splitting.
        
        Args:
            docx_content: DOCX file content as bytes
            page_ranges: List of page ranges to extract
        
        Returns:
            List of extracted page groups with text content
        """
        docx_temp_path = None
        pdf_temp_path = None
        try:
            # Save DOCX to temp file
            docx_temp_path = tempfile.mktemp(suffix='.docx')
            with open(docx_temp_path, "wb") as temp_file:
                temp_file.write(docx_content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            
            logger.info(f"🔄 Converting DOCX to PDF for accurate page extraction...")
            
            # Convert DOCX to PDF
            try:
                pdf_temp_path = DocumentConverter.convert_docx_to_pdf(docx_temp_path)
                logger.info(f"✅ Successfully converted DOCX to PDF: {pdf_temp_path}")
            except Exception as convert_error:
                logger.error(f"❌ Failed to convert DOCX to PDF: {str(convert_error)}")
                # Fallback: extract text directly from DOCX
                logger.info(f"🔄 Falling back to direct text extraction from DOCX...")
                full_text = DocumentConverter.extract_text_from_docx(docx_temp_path)
                
                if not full_text or len(full_text.strip()) < 50:
                    raise ValueError("DOCX file appears to be empty or has insufficient text")
                
                # For fallback, return all text as a single "page"
                extracted_groups = []
                for idx, page_range in enumerate(page_ranges):
                    report_title = page_range.get("report_title", f"Report_{idx + 1}")
                    extracted_groups.append({
                        "report_title": report_title,
                        "start_page": page_range.get("start_page", 1),
                        "end_page": page_range.get("end_page", 1),
                        "text": full_text,
                        "raw_text": full_text,
                        "page_count": 1,
                        "warning": "DOCX conversion to PDF failed, using full document text"
                    })
                return extracted_groups
            
            # Read the converted PDF
            with open(pdf_temp_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
            
            # Now extract pages from the PDF
            logger.info(f"📄 Extracting pages from converted PDF...")
            return self.extract_pages_from_pdf(pdf_content, page_ranges)
            
        except Exception as e:
            logger.error(f"❌ Error extracting pages from DOCX: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temp files
            for path in [docx_temp_path, pdf_temp_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Failed to cleanup temp file {path}: {str(cleanup_error)}")
    
    def extract_pages_from_gcs(self, blob_path: str, page_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text from specific pages of a document stored in GCS.
        Supports PDF, DOCX, and other formats.
        
        Args:
            blob_path: GCS blob path to the document file
            page_ranges: List of page ranges to extract
        
        Returns:
            List of extracted page groups with text content
        """
        try:
            logger.info(f"📥 Downloading document from GCS: {blob_path}")
            file_content = self.file_service.download_from_gcs(blob_path)
            
            if not file_content:
                raise ValueError(f"Downloaded file from GCS is empty for blob: {blob_path}")
            
            if len(file_content) < 100:
                raise ValueError(f"Downloaded file from GCS is too small ({len(file_content)} bytes) for blob: {blob_path}")
            
            # Detect file type
            file_type = self._detect_file_type(file_content, blob_path)
            logger.info(f"📄 Detected file type: {file_type}")
            
            if file_type == '.pdf':
                # Validate PDF header
                if not file_content.startswith(b'%PDF'):
                    raise ValueError(f"Downloaded file from GCS does not appear to be a valid PDF (blob: {blob_path})")
                
                logger.info(f"✅ Downloaded {len(file_content)} bytes from GCS (valid PDF)")
                return self.extract_pages_from_pdf(file_content, page_ranges)
            
            elif file_type == '.docx':
                logger.info(f"✅ Downloaded {len(file_content)} bytes from GCS (DOCX file)")
                return self.extract_pages_from_docx(file_content, page_ranges)
            
            else:
                # Try to convert to PDF first, then process
                logger.info(f"🔄 Attempting to convert {file_type} to PDF for processing...")
                temp_file_path = tempfile.mktemp(suffix=file_type)
                try:
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(file_content)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                    
                    # Convert to PDF
                    pdf_path, was_converted = DocumentConverter.convert_document(temp_file_path, "pdf")
                    
                    if was_converted and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_content = pdf_file.read()
                        return self.extract_pages_from_pdf(pdf_content, page_ranges)
                    else:
                        raise ValueError(f"Failed to convert {file_type} to PDF")
                finally:
                    for path in [temp_file_path, pdf_path if 'pdf_path' in locals() else None]:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except:
                                pass
            
        except Exception as e:
            logger.error(f"❌ Error extracting pages from GCS: {str(e)}", exc_info=True)
            raise


# Singleton instance
_extractor_instance = None

def get_page_extractor() -> PageExtractor:
    """Get singleton PageExtractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        logger.info("🚀 Initializing Page Extractor...")
        _extractor_instance = PageExtractor()
    return _extractor_instance

