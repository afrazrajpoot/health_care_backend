# """
# Citation Service with FAISS Vector Store
# Matches summary statements to source OCR text chunks using semantic similarity.
# """
# import logging
# import re
# from typing import List, Dict, Optional, Any, Tuple
# from dataclasses import dataclass, asdict
# import numpy as np

# try:
#     import faiss
#     FAISS_AVAILABLE = True
# except ImportError:
#     FAISS_AVAILABLE = False
#     logging.warning("⚠️ FAISS not installed. Run: pip install faiss-cpu")

# from utils.text_chunker import TextChunk, get_text_chunker
# from services.embedding_service import get_embedding_service

# logger = logging.getLogger("document_ai")


# @dataclass
# class Citation:
#     """Represents a citation linking a summary statement to source text."""
#     statement: str
#     source_text: str
#     page_number: Optional[int]
#     paragraph_index: Optional[int]
#     text_snippet: str  # Truncated preview of source
#     confidence: float
#     confidence_level: str  # "high", "medium", "low"
#     chunk_id: int
    
#     def to_dict(self) -> dict:
#         return asdict(self)


# class CitationService:
#     """
#     Service for attaching verifiable citations to summary statements.
#     Uses FAISS for efficient similarity search.
#     """
    
#     # Confidence thresholds
#     HIGH_CONFIDENCE_THRESHOLD = 0.90
#     MEDIUM_CONFIDENCE_THRESHOLD = 0.80  # Minimum to include
#     SNIPPET_MAX_LENGTH = 150  # Max chars for text snippet
    
#     def __init__(
#         self,
#         min_confidence: float = 0.80,
#         chunk_size: int = 500,
#         chunk_overlap: int = 50
#     ):
#         if not FAISS_AVAILABLE:
#             raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
#         self.min_confidence = min_confidence
#         self.embedding_service = get_embedding_service()
#         self.text_chunker = get_text_chunker(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy="paragraph"
#         )
        
#         # FAISS index (created per document)
#         self.index: Optional[faiss.IndexFlatIP] = None
#         self.chunks: List[TextChunk] = []
#         self.chunk_embeddings: List[np.ndarray] = []
        
#         logger.info(f"✅ CitationService initialized (min_confidence: {min_confidence})")
    
#     def build_index_from_text(
#         self, 
#         ocr_text: str, 
#         page_boundaries: Optional[List[int]] = None,
#         total_pages: Optional[int] = None
#     ) -> bool:
#         """
#         Build FAISS index from OCR text.
        
#         Args:
#             ocr_text: Full OCR text from document
#             page_boundaries: Optional list of char positions where pages start
#             total_pages: Optional total number of pages for estimation
            
#         Returns:
#             True if index was built successfully
#         """
#         try:
#             # Store total pages for later use
#             self.total_pages = total_pages
            
#             # Step 1: Chunk the text
#             self.chunks = self.text_chunker.chunk_text(ocr_text, page_boundaries, total_pages)
            
#             if not self.chunks:
#                 logger.warning("⚠️ No chunks created from OCR text")
#                 return False
            
#             logger.info(f"📝 Created {len(self.chunks)} chunks from OCR text (total_pages={total_pages})")
            
#             # Step 2: Generate embeddings for chunks
#             chunk_texts = [chunk.text for chunk in self.chunks]
#             embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
#             # Filter out failed embeddings
#             valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
#             if not valid_indices:
#                 logger.error("❌ No valid embeddings generated")
#                 return False
            
#             # Keep only chunks with valid embeddings
#             self.chunks = [self.chunks[i] for i in valid_indices]
#             self.chunk_embeddings = [np.array(embeddings[i], dtype=np.float32) for i in valid_indices]
            
#             # Step 3: Build FAISS index
#             dimension = len(self.chunk_embeddings[0])
#             self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
            
#             # Normalize vectors for cosine similarity
#             embeddings_matrix = np.vstack(self.chunk_embeddings).astype(np.float32)
#             faiss.normalize_L2(embeddings_matrix)
            
#             self.index.add(embeddings_matrix)
            
#             logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
#             return True
            
#         except Exception as e:
#             logger.error(f"❌ Failed to build citation index: {e}")
#             return False
    
#     def find_citation(self, statement: str, top_k: int = 3) -> Optional[Citation]:
#         """
#         Find the best matching source for a summary statement.
        
#         Args:
#             statement: Summary statement to find source for
#             top_k: Number of top matches to consider
            
#         Returns:
#             Citation object if match found above threshold, None otherwise
#         """
#         if not self.index or self.index.ntotal == 0:
#             logger.warning("⚠️ Citation index not built")
#             return None
        
#         if not statement or not statement.strip():
#             return None
        
#         try:
#             # Generate embedding for statement
#             statement_embedding = self.embedding_service.generate_embedding(statement)
#             if statement_embedding is None:
#                 return None
            
#             # Convert to numpy and normalize
#             query_vec = np.array([statement_embedding], dtype=np.float32)
#             faiss.normalize_L2(query_vec)
            
#             # Search
#             scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
            
#             if len(indices[0]) == 0:
#                 return None
            
#             # Get best match
#             best_idx = indices[0][0]
#             best_score = float(scores[0][0])
            
#             # Check confidence threshold
#             if best_score < self.min_confidence:
#                 logger.debug(f"⚠️ Low confidence ({best_score:.2f}) for: {statement[:50]}...")
#                 return None
            
#             # Get source chunk
#             source_chunk = self.chunks[best_idx]
            
#             # Create citation
#             citation = Citation(
#                 statement=statement,
#                 source_text=source_chunk.text,
#                 page_number=source_chunk.page_number,
#                 paragraph_index=source_chunk.paragraph_index,
#                 text_snippet=self._create_snippet(source_chunk.text, statement),
#                 confidence=round(best_score, 3),
#                 confidence_level=self._get_confidence_level(best_score),
#                 chunk_id=source_chunk.chunk_id
#             )
            
#             return citation
            
#         except Exception as e:
#             logger.error(f"❌ Citation search failed: {e}")
#             return None
    
#     def attach_citations_to_summary(
#         self, 
#         short_summary: dict, 
#         ocr_text: str,
#         total_pages: Optional[int] = None
#     ) -> dict:
#         """
#         Attach citations to an already-generated short summary.
        
#         Args:
#             short_summary: The structured short summary dict
#             ocr_text: Full OCR text for source matching
#             total_pages: Optional total number of pages for page estimation
            
#         Returns:
#             Enhanced short_summary with citations attached to each item
#         """
#         if not short_summary or not ocr_text:
#             logger.warning("⚠️ Missing summary or OCR text for citation attachment")
#             return short_summary
        
#         try:
#             # Build index from OCR text
#             if not self.build_index_from_text(ocr_text, total_pages=total_pages):
#                 logger.warning("⚠️ Failed to build citation index, returning summary without citations")
#                 return short_summary
            
#             # Get summary items
#             items = short_summary.get("summary", {}).get("items", [])
#             if not items:
#                 logger.info("📝 No summary items to attach citations to")
#                 return short_summary
            
#             # Process each item
#             total_citations = 0
#             for item in items:
#                 citations = []
                
#                 # Extract statements from expanded text (bullet points)
#                 expanded = item.get("expanded", "")
#                 statements = self._extract_statements_from_expanded(expanded)
                
#                 # Also try collapsed as a single statement
#                 collapsed = item.get("collapsed", "")
#                 if collapsed and collapsed not in statements:
#                     statements.insert(0, collapsed)
                
#                 # Find citations for each statement
#                 for statement in statements:
#                     citation = self.find_citation(statement)
#                     if citation:
#                         citations.append(citation.to_dict())
#                         total_citations += 1
                
#                 # Attach citations to item
#                 if citations:
#                     item["citations"] = citations
            
#             logger.info(f"✅ Attached {total_citations} citations to summary")
            
#             # Update summary with citation metadata
#             short_summary["_citation_metadata"] = {
#                 "total_citations": total_citations,
#                 "chunks_indexed": len(self.chunks),
#                 "min_confidence_threshold": self.min_confidence
#             }
            
#             return short_summary
            
#         except Exception as e:
#             logger.error(f"❌ Citation attachment failed: {e}")
#             return short_summary
    
#     def _extract_statements_from_expanded(self, expanded_text: str) -> List[str]:
#         """Extract individual statements from bullet-point expanded text."""
#         if not expanded_text:
#             return []
        
#         statements = []
        
#         # Split by bullet points
#         lines = expanded_text.split('\n')
#         for line in lines:
#             # Remove bullet point prefix
#             line = re.sub(r'^[•\-\*]\s*', '', line.strip())
#             if line and len(line) > 10:  # Skip very short lines
#                 statements.append(line)
        
#         # If no bullets found, try splitting by periods
#         if not statements:
#             sentences = re.split(r'(?<=[.!?])\s+', expanded_text)
#             statements = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
#         return statements
    
#     def _create_snippet(self, source_text: str, statement: str, max_length: int = None) -> str:
#         """Create a relevant snippet from source text."""
#         max_length = max_length or self.SNIPPET_MAX_LENGTH
        
#         if len(source_text) <= max_length:
#             return source_text
        
#         # Try to find the most relevant part by looking for keywords from statement
#         keywords = set(re.findall(r'\b\w{4,}\b', statement.lower()))
        
#         # Score each position in source text
#         best_start = 0
#         best_score = 0
        
#         words = source_text.split()
#         for i in range(len(words)):
#             window_text = ' '.join(words[i:i+20]).lower()
#             score = sum(1 for kw in keywords if kw in window_text)
#             if score > best_score:
#                 best_score = score
#                 best_start = len(' '.join(words[:i]))
        
#         # Extract snippet around best position
#         start = max(0, best_start - 20)
#         end = min(len(source_text), start + max_length)
        
#         snippet = source_text[start:end]
        
#         # Clean up edges
#         if start > 0:
#             snippet = "..." + snippet.lstrip()
#         if end < len(source_text):
#             snippet = snippet.rstrip() + "..."
        
#         return snippet
    
#     def _get_confidence_level(self, score: float) -> str:
#         """Determine confidence level from score."""
#         if score >= self.HIGH_CONFIDENCE_THRESHOLD:
#             return "high"
#         elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
#             return "medium"
#         else:
#             return "low"
    
#     def clear_index(self):
#         """Clear the current FAISS index."""
#         self.index = None
#         self.chunks = []
#         self.chunk_embeddings = []
#         self.total_pages = None


# def attach_citations_to_short_summary(
#     short_summary: dict, 
#     ocr_text: str,
#     min_confidence: float = 0.80,
#     total_pages: Optional[int] = None
# ) -> dict:
#     """
#     Convenience function to attach citations to a short summary.
    
#     Args:
#         short_summary: The structured short summary dict
#         ocr_text: Full OCR text from the document
#         min_confidence: Minimum similarity score for citations (default 0.80)
#         total_pages: Optional total number of pages for page estimation
        
#     Returns:
#         Enhanced short_summary with citations
#     """
#     try:
#         service = CitationService(min_confidence=min_confidence)
#         return service.attach_citations_to_summary(short_summary, ocr_text, total_pages=total_pages)
#     except ImportError as e:
#         logger.error(f"❌ Citation service unavailable: {e}")
#         return short_summary
#     except Exception as e:
#         logger.error(f"❌ Citation attachment failed: {e}")
#         return short_summary
