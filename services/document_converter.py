import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import logging
from docx import Document
import shutil

logger = logging.getLogger("document_ai")

class DocumentConverter:
    """Service for converting documents to formats supported by Document AI"""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.bmp', '.webp'}
    SUPPORTED_DOCUMENT_FORMATS = {'.pdf'}
    CONVERTIBLE_FORMATS = {'.docx', '.pptx', '.xlsx'}
    
    @classmethod
    def get_supported_formats(cls) -> set:
        """Get all supported and convertible file formats"""
        return cls.SUPPORTED_IMAGE_FORMATS | cls.SUPPORTED_DOCUMENT_FORMATS | cls.CONVERTIBLE_FORMATS
    
    @classmethod
    def is_supported_format(cls, file_path: str) -> bool:
        """Check if file format is supported or convertible"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in cls.get_supported_formats()
    
    @classmethod
    def needs_conversion(cls, file_path: str) -> bool:
        """Check if file needs conversion to be processed by Document AI"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in cls.CONVERTIBLE_FORMATS
    
    @classmethod
    def convert_docx_to_pdf(cls, docx_path: str, output_dir: Optional[str] = None) -> str:
        """
        Convert DOCX file to PDF format using LibreOffice
        
        Args:
            docx_path: Path to the DOCX file
            output_dir: Directory to save the PDF (optional, uses temp dir if None)
            
        Returns:
            Path to the converted PDF file
        """
        try:
            logger.info(f"🔄 Converting DOCX to PDF using LibreOffice: {docx_path}")
            
            # Validate input file
            if not os.path.exists(docx_path):
                raise FileNotFoundError(f"DOCX file not found: {docx_path}")
            
            # Determine output directory
            if output_dir is None:
                output_dir = tempfile.gettempdir()
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Use LibreOffice to convert DOCX to PDF
            cmd = [
                "libreoffice", 
                "--headless",           # Run without GUI
                "--convert-to", "pdf",  # Convert to PDF format
                "--outdir", output_dir, # Output directory
                docx_path              # Input file
            ]
            
            logger.info(f"🚀 Running LibreOffice command: {' '.join(cmd)}")
            
            # Run the conversion
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                error_msg = f"LibreOffice conversion failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Determine expected PDF path
            docx_name = Path(docx_path).stem
            pdf_path = os.path.join(output_dir, f"{docx_name}.pdf")
            
            # Verify PDF was created
            if not os.path.exists(pdf_path):
                raise Exception(f"PDF conversion failed - expected output file not found: {pdf_path}")
            
            pdf_size = os.path.getsize(pdf_path)
            if pdf_size == 0:
                raise Exception("PDF conversion failed - output file is empty")
            
            logger.info(f"✅ Successfully converted to PDF: {pdf_path}")
            logger.info(f"📏 PDF file size: {pdf_size} bytes")
            
            # Log LibreOffice output if available
            if result.stdout:
                logger.info(f"📋 LibreOffice output: {result.stdout.strip()}")
            
            return pdf_path
            
        except subprocess.TimeoutExpired:
            logger.error("❌ LibreOffice conversion timed out after 60 seconds")
            raise Exception("DOCX to PDF conversion timed out")
        except Exception as e:
            logger.error(f"❌ Error converting DOCX to PDF: {str(e)}")
            raise Exception(f"DOCX to PDF conversion failed: {str(e)}")
    
    @classmethod
    def extract_text_from_docx(cls, docx_path: str) -> str:
        """
        Extract plain text from DOCX file as fallback option
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"📄 Extracting text from DOCX: {docx_path}")
            
            document = Document(docx_path)
            
            # Extract text from paragraphs
            text_content = []
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            # Extract text from tables
            for table in document.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))
            
            extracted_text = "\n".join(text_content)
            logger.info(f"✅ Extracted {len(extracted_text)} characters from DOCX")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from DOCX: {str(e)}")
            raise Exception(f"DOCX text extraction failed: {str(e)}")
    
    @classmethod
    def convert_document(cls, input_path: str, target_format: str = "pdf") -> Tuple[str, bool]:
        """
        Convert document to target format if needed
        
        Args:
            input_path: Path to input document
            target_format: Target format ('pdf' or 'text')
            
        Returns:
            Tuple of (converted_file_path, was_converted)
        """
        file_ext = Path(input_path).suffix.lower()
        
        # If already in supported format, return as-is
        if file_ext in (cls.SUPPORTED_IMAGE_FORMATS | cls.SUPPORTED_DOCUMENT_FORMATS):
            logger.info(f"✅ File already in supported format: {file_ext}")
            return input_path, False
        
        # Convert based on file type and target format
        if file_ext == '.docx':
            if target_format == "pdf":
                converted_path = cls.convert_docx_to_pdf(input_path)
                return converted_path, True
            elif target_format == "text":
                # For text extraction, we don't create a new file
                # This is handled differently in the calling code
                return input_path, False
        
        # Add more conversion logic for other formats as needed
        # elif file_ext == '.pptx':
        #     # Future implementation for PowerPoint
        #     pass
        # elif file_ext == '.xlsx':
        #     # Future implementation for Excel
        #     pass
        
        raise ValueError(f"Unsupported file format for conversion: {file_ext}")
    
    @classmethod
    def cleanup_converted_file(cls, file_path: str, was_converted: bool):
        """Clean up converted file if it was created during conversion"""
        if was_converted and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up converted file: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up converted file {file_path}: {str(e)}")