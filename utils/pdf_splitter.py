# pdf_splitter.py

"""
PDF splitting and text extraction utilities for Document AI processing.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger("document_ai")


class LayoutPreservingTextExtractor:
    """
    Extract text from Document AI Summarizer results.
    Simplified since summarizer returns clean text without complex layout.
    """


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
