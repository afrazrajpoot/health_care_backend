"""
Text Chunker for Citation Workflow
Splits OCR text into semantic chunks with positional metadata for embedding and citation matching.
"""
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("document_ai")


@dataclass
class TextChunk:
    """Represents a chunk of text with its source metadata."""
    chunk_id: int
    text: str
    page_number: Optional[int] = None
    paragraph_index: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class TextChunker:
    """
    Splits OCR text into semantic chunks for embedding generation.
    
    Strategies:
    - Paragraph-based: Split by double newlines (default)
    - Sentence-based: Split by sentence boundaries
    - Fixed-size: Split by token/character count with overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 500,  # Target characters per chunk
        chunk_overlap: int = 50,  # Overlap between chunks
        min_chunk_size: int = 50,  # Minimum chunk size to keep
        strategy: str = "paragraph"  # "paragraph", "sentence", "fixed"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy
    
    def chunk_text(
        self, 
        text: str, 
        page_boundaries: Optional[List[int]] = None,
        total_pages: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The full OCR text
            page_boundaries: Optional list of character positions where pages start
            total_pages: Optional total number of pages for estimation
            
        Returns:
            List of TextChunk objects with metadata
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided to chunker")
            return []
        
        # Try to detect page boundaries if not provided
        if not page_boundaries:
            page_boundaries = self.extract_page_boundaries_from_text(text)
        
        if self.strategy == "paragraph":
            chunks = self._chunk_by_paragraph(text, page_boundaries, total_pages)
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentence(text, page_boundaries, total_pages)
        else:
            chunks = self._chunk_by_fixed_size(text, page_boundaries, total_pages)
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if len(c.text.strip()) >= self.min_chunk_size]
        
        # Ensure all chunks have paragraph indices assigned sequentially
        for idx, chunk in enumerate(chunks):
            if chunk.paragraph_index is None:
                chunk.paragraph_index = idx + 1
        
        logger.info(f"📝 Created {len(chunks)} text chunks (strategy: {self.strategy})")
        return chunks
    
    def _chunk_by_paragraph(
        self, 
        text: str, 
        page_boundaries: Optional[List[int]] = None,
        total_pages: Optional[int] = None
    ) -> List[TextChunk]:
        """Split text by paragraphs (double newlines or bullet points)."""
        chunks = []
        
        # Split by double newlines, multiple newlines, or bullet point patterns
        # This handles summary text that uses bullet points (•)
        paragraphs = re.split(r'\n\s*\n+|(?=•\s)', text)
        
        current_pos = 0
        current_line = 1
        total_text_len = len(text)
        
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            # Find the actual position in original text
            char_start = text.find(para, current_pos)
            if char_start == -1:
                char_start = current_pos
            char_end = char_start + len(para)
            
            # Calculate line numbers
            line_start = current_line
            line_end = line_start + para.count('\n')
            
            # Determine page number from boundaries or estimate
            page_num = self._get_page_number(char_start, page_boundaries, total_text_len, total_pages)
            
            # If paragraph is too long, split it further
            if len(para) > self.chunk_size * 2:
                sub_chunks = self._split_long_paragraph(
                    para, char_start, line_start, page_num, len(chunks), total_text_len, total_pages
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(TextChunk(
                    chunk_id=len(chunks),
                    text=para,
                    page_number=page_num,
                    paragraph_index=len(chunks) + 1,  # 1-indexed paragraph
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end
                ))
            
            current_pos = char_end
            current_line = line_end + 1
        
        return chunks
    
    def _chunk_by_sentence(
        self, 
        text: str, 
        page_boundaries: Optional[List[int]] = None,
        total_pages: Optional[int] = None
    ) -> List[TextChunk]:
        """Split text by sentences, grouping small sentences together."""
        chunks = []
        total_text_len = len(text)
        
        # Sentence splitting pattern (handles common abbreviations)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        current_chunk_text = ""
        chunk_start_pos = 0
        current_search_pos = 0
        paragraph_idx = 1

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk_text) + len(sentence) > self.chunk_size and current_chunk_text:
                # Save current chunk
                page_num = self._get_page_number(chunk_start_pos, page_boundaries, total_text_len, total_pages)
                chunks.append(TextChunk(
                    chunk_id=len(chunks),
                    text=current_chunk_text.strip(),
                    page_number=page_num,
                    paragraph_index=paragraph_idx,
                    char_start=chunk_start_pos,
                    char_end=chunk_start_pos + len(current_chunk_text)
                ))
                paragraph_idx += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk_text[-self.chunk_overlap:] if len(current_chunk_text) > self.chunk_overlap else ""
                current_chunk_text = overlap_text + " " + sentence
                sentence_pos = text.find(sentence, current_search_pos)
                chunk_start_pos = sentence_pos if sentence_pos >= 0 else current_search_pos
            else:
                if not current_chunk_text:
                    sentence_pos = text.find(sentence, current_search_pos)
                    chunk_start_pos = sentence_pos if sentence_pos >= 0 else current_search_pos
                current_chunk_text += " " + sentence if current_chunk_text else sentence
            
            sentence_pos = text.find(sentence, current_search_pos)
            if sentence_pos >= 0:
                current_search_pos = sentence_pos + len(sentence)
        
        # Don't forget the last chunk
        if current_chunk_text.strip():
            page_num = self._get_page_number(chunk_start_pos, page_boundaries, total_text_len, total_pages)
            chunks.append(TextChunk(
                chunk_id=len(chunks),
                text=current_chunk_text.strip(),
                page_number=page_num,
                paragraph_index=paragraph_idx,
                char_start=chunk_start_pos,
                char_end=chunk_start_pos + len(current_chunk_text)
            ))
        
        return chunks
    
    def _chunk_by_fixed_size(
        self, 
        text: str, 
        page_boundaries: Optional[List[int]] = None,
        total_pages: Optional[int] = None
    ) -> List[TextChunk]:
        """Split text by fixed character size with overlap."""
        chunks = []
        total_text_len = len(text)
        paragraph_idx = 1
        
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a word boundary
            if end < len(text):
                # Look for last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.chunk_size // 2:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                page_num = self._get_page_number(start, page_boundaries, total_text_len, total_pages)
                chunks.append(TextChunk(
                    chunk_id=len(chunks),
                    text=chunk_text,
                    page_number=page_num,
                    paragraph_index=paragraph_idx,
                    char_start=start,
                    char_end=end
                ))
                paragraph_idx += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def _split_long_paragraph(
        self, 
        para: str, 
        char_start: int, 
        line_start: int, 
        page_num: Optional[int],
        chunk_id_start: int,
        total_text_len: int = 0,
        total_pages: Optional[int] = None
    ) -> List[TextChunk]:
        """Split a long paragraph into smaller chunks at sentence boundaries."""
        chunks = []
        
        # Split by sentences or bullet points
        sentences = re.split(r'(?<=[.!?])\s+|(?=•\s)', para)
        
        current_text = ""
        current_start = char_start
        para_idx_offset = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_text) + len(sentence) > self.chunk_size and current_text:
                # Estimate page if not provided
                effective_page = page_num
                if effective_page is None and total_pages and total_text_len > 0:
                    effective_page = self._estimate_page_number(current_start, total_text_len, total_pages)
                
                chunks.append(TextChunk(
                    chunk_id=chunk_id_start + len(chunks),
                    text=current_text.strip(),
                    page_number=effective_page,
                    paragraph_index=chunk_id_start + para_idx_offset + 1,
                    char_start=current_start,
                    char_end=current_start + len(current_text)
                ))
                para_idx_offset += 1
                current_start = current_start + len(current_text)
                current_text = sentence
            else:
                current_text += " " + sentence if current_text else sentence
        
        if current_text.strip():
            effective_page = page_num
            if effective_page is None and total_pages and total_text_len > 0:
                effective_page = self._estimate_page_number(current_start, total_text_len, total_pages)
            
            chunks.append(TextChunk(
                chunk_id=chunk_id_start + len(chunks),
                text=current_text.strip(),
                page_number=effective_page,
                paragraph_index=chunk_id_start + para_idx_offset + 1,
                char_start=current_start,
                char_end=current_start + len(current_text)
            ))
        
        return chunks
    
    def _get_page_number(
        self, 
        char_pos: int, 
        page_boundaries: Optional[List[int]],
        total_text_len: int = 0,
        total_pages: Optional[int] = None
    ) -> Optional[int]:
        """
        Determine page number from character position.
        
        Uses page_boundaries if available, otherwise estimates based on position
        in text relative to total pages.
        """
        # If explicit boundaries exist, use them
        if page_boundaries:
            for i, boundary in enumerate(page_boundaries):
                if char_pos < boundary:
                    return i + 1
            return len(page_boundaries)
        
        # Estimate page based on position in text
        if total_pages and total_pages > 0 and total_text_len > 0:
            return self._estimate_page_number(char_pos, total_text_len, total_pages)
        
        return None
    
    def _estimate_page_number(
        self, 
        char_pos: int, 
        total_text_len: int, 
        total_pages: int
    ) -> int:
        """
        Estimate page number based on character position.
        Assumes roughly even distribution of text across pages.
        
        Returns:
            Estimated page number (1-indexed)
        """
        if total_text_len <= 0 or total_pages <= 0:
            return 1
        
        # Calculate position ratio and map to page
        position_ratio = char_pos / total_text_len
        estimated_page = int(position_ratio * total_pages) + 1
        
        # Clamp to valid range
        return max(1, min(estimated_page, total_pages))
    
    @staticmethod
    def extract_page_boundaries_from_text(text: str) -> List[int]:
        """
        Try to extract page boundaries from text markers.
        Looks for patterns like "Page X", "--- Page X ---", etc.
        """
        boundaries = [0]
        
        # Common page marker patterns
        patterns = [
            r'(?:^|\n)[-=]{3,}\s*Page\s*\d+\s*[-=]{3,}',
            r'(?:^|\n)Page\s*\d+\s*(?:\n|$)',
            r'\[Page\s*\d+\]',
            r'(?:^|\n)\d+\s*(?:\n|$)',  # Just page numbers at start of line
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                if match.start() not in boundaries:
                    boundaries.append(match.start())
        
        boundaries.sort()
        return boundaries if len(boundaries) > 1 else []


def get_text_chunker(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "paragraph"
) -> TextChunker:
    """Factory function to get a configured TextChunker instance."""
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
    )
