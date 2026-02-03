"""
Context Expansion Service
Enriches summary bullet points with supporting context from the original document
using semantic similarity (embeddings + cosine similarity).

This service provides accurate, verifiable context for each summary statement
by finding the most semantically relevant passages from the source document.
"""
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

from services.embedding_service import get_embedding_service, EmbeddingService
from utils.text_chunker import TextChunker, TextChunk, get_text_chunker

logger = logging.getLogger("document_ai")


@dataclass
class ExpandedContext:
    """Represents expanded context for a bullet point."""
    bullet_point: str
    supporting_context: str  # The relevant source text
    similarity_score: float  # How confident we are in the match
    extracted_details: Dict[str, Any] = field(default_factory=dict)  # Structured details if extracted
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BulletExpansion:
    """Represents a bullet point with its expanded context."""
    original_bullet: str
    contexts: List[ExpandedContext]  # Multiple supporting contexts ranked by relevance
    combined_context: str  # Merged context for display
    
    def to_dict(self) -> dict:
        return {
            "original_bullet": self.original_bullet,
            "contexts": [c.to_dict() for c in self.contexts],
            "combined_context": self.combined_context
        }


class ContextExpansionService:
    """
    Service for enriching summary bullet points with supporting context
    from the original document using semantic similarity.
    
    Uses embeddings to understand the meaning of each bullet point and
    retrieve the most relevant supporting text from the document.
    """
    
    # Similarity thresholds
    HIGH_RELEVANCE_THRESHOLD = 0.85
    MEDIUM_RELEVANCE_THRESHOLD = 0.75
    MIN_RELEVANCE_THRESHOLD = 0.65  # Below this, context is considered unreliable
    
    # Context extraction settings
    MAX_CONTEXT_LENGTH = 800  # Max chars for combined context
    TOP_K_CHUNKS = 3  # Number of top chunks to consider per bullet
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        min_relevance: float = 0.65
    ):
        """
        Initialize the context expansion service.
        
        Args:
            chunk_size: Target size for document chunks
            chunk_overlap: Overlap between chunks for continuity
            min_relevance: Minimum similarity score to include context
        """
        self.embedding_service: EmbeddingService = get_embedding_service()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=30,
            strategy="paragraph"
        )
        self.min_relevance = min_relevance
        
        # Cache for document chunks and embeddings
        self._document_chunks: List[TextChunk] = []
        self._chunk_embeddings: List[np.ndarray] = []
        self._document_hash: Optional[str] = None
        
        logger.info(f"✅ ContextExpansionService initialized (min_relevance: {min_relevance})")
    
    def _compute_document_hash(self, text: str) -> str:
        """Compute a simple hash for document caching."""
        import hashlib
        return hashlib.md5(text[:5000].encode()).hexdigest()
    
    def _index_document(self, document_text: str) -> bool:
        """
        Index the document by chunking and generating embeddings.
        
        Args:
            document_text: The full original document text
            
        Returns:
            True if indexing succeeded
        """
        try:
            # Check cache
            doc_hash = self._compute_document_hash(document_text)
            if doc_hash == self._document_hash and self._chunk_embeddings:
                logger.debug("📦 Using cached document index")
                return True
            
            # Chunk the document
            self._document_chunks = self.text_chunker.chunk_text(document_text)
            
            if not self._document_chunks:
                logger.warning("⚠️ No chunks created from document")
                return False
            
            logger.info(f"📝 Created {len(self._document_chunks)} chunks from document")
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in self._document_chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            # Filter to valid embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for i, emb in enumerate(embeddings):
                if emb is not None:
                    valid_chunks.append(self._document_chunks[i])
                    valid_embeddings.append(np.array(emb, dtype=np.float32))
            
            if not valid_embeddings:
                logger.error("❌ No valid embeddings generated for document")
                return False
            
            self._document_chunks = valid_chunks
            self._chunk_embeddings = valid_embeddings
            self._document_hash = doc_hash
            
            logger.info(f"✅ Indexed {len(self._chunk_embeddings)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}")
            return False
    
    def _find_relevant_chunks(
        self, 
        query_text: str, 
        top_k: int = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Find the most relevant document chunks for a query using semantic similarity.
        
        Args:
            query_text: The bullet point or query to find context for
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples sorted by relevance
        """
        top_k = top_k or self.TOP_K_CHUNKS
        
        if not self._chunk_embeddings:
            logger.warning("⚠️ Document not indexed")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query_text)
            if query_embedding is None:
                return []
            
            query_vec = np.array(query_embedding, dtype=np.float32)
            
            # Compute similarities with all chunks
            similarities = []
            for i, chunk_emb in enumerate(self._chunk_embeddings):
                score = self.embedding_service.compute_cosine_similarity(query_vec, chunk_emb)
                if score >= self.min_relevance:
                    similarities.append((self._document_chunks[i], score))
            
            # Sort by similarity (descending) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Chunk retrieval failed: {e}")
            return []
    
    def expand_bullet_point(
        self, 
        bullet_point: str, 
        document_text: str
    ) -> BulletExpansion:
        """
        Expand a single bullet point with supporting context from the document.
        
        Args:
            bullet_point: The summary bullet point to expand
            document_text: The full original document text
            
        Returns:
            BulletExpansion with supporting context
        """
        # Ensure document is indexed
        if not self._index_document(document_text):
            return BulletExpansion(
                original_bullet=bullet_point,
                contexts=[],
                combined_context=""
            )
        
        # Clean bullet point for searching
        clean_bullet = self._clean_bullet_for_search(bullet_point)
        
        # Find relevant chunks
        relevant_chunks = self._find_relevant_chunks(clean_bullet)
        
        if not relevant_chunks:
            logger.debug(f"⚠️ No relevant context found for: {bullet_point[:50]}...")
            return BulletExpansion(
                original_bullet=bullet_point,
                contexts=[],
                combined_context=""
            )
        
        # Build expanded contexts
        contexts = []
        for chunk, score in relevant_chunks:
            # Extract relevant details from the chunk
            extracted = self._extract_relevant_details(bullet_point, chunk.text)
            
            context = ExpandedContext(
                bullet_point=bullet_point,
                supporting_context=chunk.text,
                similarity_score=round(score, 3),
                extracted_details=extracted
            )
            contexts.append(context)
        
        # Combine contexts intelligently
        combined = self._combine_contexts(contexts, bullet_point)
        
        return BulletExpansion(
            original_bullet=bullet_point,
            contexts=contexts,
            combined_context=combined
        )
    
    def expand_summary_items(
        self, 
        summary_items: List[Dict], 
        document_text: str
    ) -> List[Dict]:
        """
        Expand all summary items with supporting context.
        
        Args:
            summary_items: List of summary item dicts with 'collapsed' and 'expanded' fields
            document_text: The full original document text
            
        Returns:
            Enhanced summary items with 'context_expansion' field added
        """
        if not summary_items or not document_text:
            logger.warning("⚠️ Missing summary items or document text")
            return summary_items
        
        # Index document once
        if not self._index_document(document_text):
            logger.warning("⚠️ Failed to index document, returning items without context")
            return summary_items
        
        enhanced_items = []
        total_expansions = 0
        
        for item in summary_items:
            enhanced_item = item.copy()
            
            # Extract bullet points from expanded text
            expanded_text = item.get("expanded", "")
            bullets = self._extract_bullets(expanded_text)
            
            # Also include collapsed as a statement to expand
            collapsed = item.get("collapsed", "")
            
            # Expand each bullet
            bullet_expansions = []
            for bullet in bullets:
                expansion = self.expand_bullet_point(bullet, document_text)
                if expansion.combined_context:
                    bullet_expansions.append({
                        "bullet": bullet,
                        "context": expansion.combined_context,
                        "confidence": max([c.similarity_score for c in expansion.contexts]) if expansion.contexts else 0
                    })
                    total_expansions += 1
            
            # If no bullet expansions but we have collapsed, try expanding that
            if not bullet_expansions and collapsed:
                expansion = self.expand_bullet_point(collapsed, document_text)
                if expansion.combined_context:
                    bullet_expansions.append({
                        "bullet": collapsed,
                        "context": expansion.combined_context,
                        "confidence": max([c.similarity_score for c in expansion.contexts]) if expansion.contexts else 0
                    })
                    total_expansions += 1
            
            # Add context expansion to item
            if bullet_expansions:
                enhanced_item["context_expansion"] = bullet_expansions
            
            enhanced_items.append(enhanced_item)
        
        logger.info(f"✅ Added context expansion to {total_expansions} bullet points")
        return enhanced_items
    
    def _clean_bullet_for_search(self, bullet: str) -> str:
        """Clean bullet point text for better semantic matching."""
        # Remove bullet markers
        bullet = re.sub(r'^[•\-\*]\s*', '', bullet.strip())
        
        # Remove common attribution phrases that might dilute the semantic meaning
        attributions = [
            r'^(?:was|were)\s+(?:documented|noted|reported|described|stated)\s+(?:that|as|:)?\s*',
            r'^(?:the\s+)?(?:following|report|document)\s+(?:documented|noted|indicates?|shows?)\s+(?:that)?\s*',
            r'^(?:it\s+)?was\s+(?:documented|noted|reported)\s+(?:that)?\s*',
        ]
        
        for pattern in attributions:
            bullet = re.sub(pattern, '', bullet, flags=re.IGNORECASE)
        
        return bullet.strip()
    
    def _extract_bullets(self, expanded_text: str) -> List[str]:
        """Extract individual bullet points from expanded text."""
        if not expanded_text:
            return []
        
        bullets = []
        lines = expanded_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for bullet markers
            if re.match(r'^[•\-\*]\s*', line):
                bullet_text = re.sub(r'^[•\-\*]\s*', '', line).strip()
                if bullet_text and len(bullet_text) > 10:
                    bullets.append(bullet_text)
            elif len(line) > 10:
                # Non-bullet line that might still be a statement
                bullets.append(line)
        
        return bullets
    
    def _extract_relevant_details(self, bullet: str, context: str) -> Dict[str, Any]:
        """
        Extract structured details from context relevant to the bullet point.
        
        This extracts specific medical details like medications, dosages,
        therapy types, dates, etc.
        """
        details = {}
        context_lower = context.lower()
        bullet_lower = bullet.lower()
        
        # Extract medications if bullet mentions therapy/medication
        if any(term in bullet_lower for term in ['medication', 'prescribed', 'drug', 'medicine']):
            meds = self._extract_medications(context)
            if meds:
                details['medications'] = meds
        
        # Extract therapy details
        if any(term in bullet_lower for term in ['therapy', 'treatment', 'physical therapy', 'pt']):
            therapies = self._extract_therapy_details(context)
            if therapies:
                details['therapy_details'] = therapies
        
        # Extract dosages
        dosages = self._extract_dosages(context)
        if dosages:
            details['dosages'] = dosages
        
        # Extract dates/timeframes
        dates = self._extract_dates(context)
        if dates:
            details['timeframes'] = dates
        
        # Extract numeric values (impairment ratings, visit counts, etc.)
        if any(term in bullet_lower for term in ['%', 'impairment', 'rating', 'visits', 'sessions']):
            numbers = self._extract_numeric_values(context)
            if numbers:
                details['numeric_values'] = numbers
        
        return details
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication names from text."""
        # Common medication patterns
        med_patterns = [
            r'\b([A-Z][a-z]+(?:in|ol|am|il|an|ex|ax|ix|one|ide|ate|ine)\b)',  # Common drug suffixes
            r'\b([A-Z][a-z]+)\s+\d+\s*(?:mg|mcg|ml|g)\b',  # Drug with dosage
        ]
        
        meds = set()
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 3:  # Filter out very short matches
                    meds.add(match)
        
        return list(meds)[:5]  # Limit to 5 medications
    
    def _extract_therapy_details(self, text: str) -> List[str]:
        """Extract therapy-related details from text."""
        therapy_types = []
        
        patterns = [
            r'(physical therapy|occupational therapy|speech therapy|chiropractic|acupuncture)',
            r'(PT|OT)\s+(?:treatment|session|visit)',
            r'(\d+)\s*(?:visits?|sessions?|treatments?)',
            r'(active|passive)\s+(?:treatment|therapy|modalities)',
            r'(massage|ultrasound|electrical stimulation|TENS|heat|ice|stretching)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    therapy_types.extend(match)
                else:
                    therapy_types.append(match)
        
        return list(set(t for t in therapy_types if t))[:5]
    
    def _extract_dosages(self, text: str) -> List[str]:
        """Extract dosage information from text."""
        dosage_patterns = [
            r'\b\d+\s*(?:mg|mcg|ml|g|units?)\b',
            r'\b(?:once|twice|three times|four times)\s+(?:daily|a day|per day)\b',
            r'\b(?:QD|BID|TID|QID|PRN|HS)\b',
            r'\b\d+\s*(?:tablets?|capsules?|pills?)\b',
        ]
        
        dosages = []
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dosages.extend(matches)
        
        return list(set(dosages))[:5]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates and timeframes from text."""
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b',
            r'\b\d+\s*(?:days?|weeks?|months?|years?)\b',
            r'\b(?:for|over|past)\s+\d+\s*(?:days?|weeks?|months?)\b',
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))[:5]
    
    def _extract_numeric_values(self, text: str) -> List[str]:
        """Extract numeric values with context."""
        patterns = [
            r'\b(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'\b(\d+)\s*(?:visits?|sessions?|treatments?)',  # Counts
            r'(?:rating|score|scale)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)',  # Ratings
        ]
        
        values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                values.append(match)
        
        return list(set(values))[:5]
    
    def _combine_contexts(
        self, 
        contexts: List[ExpandedContext], 
        bullet: str
    ) -> str:
        """
        Intelligently combine multiple relevant contexts into a coherent summary.
        
        Prioritizes high-relevance matches and removes redundancy.
        """
        if not contexts:
            return ""
        
        # If only one context with high relevance, use it directly
        if len(contexts) == 1 and contexts[0].similarity_score >= self.HIGH_RELEVANCE_THRESHOLD:
            return self._trim_context(contexts[0].supporting_context)
        
        # Combine contexts, prioritizing higher scores
        combined_parts = []
        total_length = 0
        seen_content = set()
        
        for ctx in contexts:
            # Skip low-relevance contexts if we have enough
            if ctx.similarity_score < self.MEDIUM_RELEVANCE_THRESHOLD and combined_parts:
                continue
            
            # Deduplicate similar content
            content_key = ctx.supporting_context[:100].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            # Trim and add context
            trimmed = self._trim_context(ctx.supporting_context)
            if trimmed:
                new_length = total_length + len(trimmed)
                if new_length <= self.MAX_CONTEXT_LENGTH:
                    combined_parts.append(trimmed)
                    total_length = new_length
                elif not combined_parts:
                    # At least include one context even if long
                    combined_parts.append(trimmed[:self.MAX_CONTEXT_LENGTH])
                    break
        
        # Join with clear separation
        if len(combined_parts) > 1:
            return " [...] ".join(combined_parts)
        elif combined_parts:
            return combined_parts[0]
        
        return ""
    
    def _trim_context(self, context: str, max_length: int = None) -> str:
        """Trim context to a reasonable length while preserving meaning."""
        max_length = max_length or self.MAX_CONTEXT_LENGTH
        
        if len(context) <= max_length:
            return context.strip()
        
        # Try to cut at a sentence boundary
        truncated = context[:max_length]
        
        # Look for last sentence end
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        
        cut_point = max(last_period, last_question, last_exclaim)
        
        if cut_point > max_length * 0.6:  # At least 60% of content
            return truncated[:cut_point + 1].strip()
        
        # Otherwise cut at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            return truncated[:last_space].strip() + "..."
        
        return truncated.strip() + "..."
    
    def clear_cache(self):
        """Clear the document cache."""
        self._document_chunks = []
        self._chunk_embeddings = []
        self._document_hash = None


# Singleton instance
_context_expansion_service: Optional[ContextExpansionService] = None


def get_context_expansion_service(
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    min_relevance: float = 0.65
) -> ContextExpansionService:
    """Get or create singleton ContextExpansionService instance."""
    global _context_expansion_service
    if _context_expansion_service is None:
        _context_expansion_service = ContextExpansionService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_relevance=min_relevance
        )
    return _context_expansion_service


def expand_summary_with_context(
    summary_items: List[Dict],
    document_text: str,
    min_relevance: float = 0.65
) -> List[Dict]:
    """
    Convenience function to expand summary items with supporting context.
    
    Args:
        summary_items: List of summary item dicts with 'collapsed' and 'expanded' fields
        document_text: The full original document text
        min_relevance: Minimum similarity score for including context
        
    Returns:
        Enhanced summary items with 'context_expansion' field
    """
    try:
        service = get_context_expansion_service(min_relevance=min_relevance)
        return service.expand_summary_items(summary_items, document_text)
    except Exception as e:
        logger.error(f"❌ Context expansion failed: {e}")
        return summary_items


def expand_single_bullet(
    bullet_point: str,
    document_text: str,
    min_relevance: float = 0.65
) -> Dict:
    """
    Convenience function to expand a single bullet point.
    
    Args:
        bullet_point: The bullet point text to expand
        document_text: The full original document text
        min_relevance: Minimum similarity score
        
    Returns:
        Dict with 'bullet', 'context', and 'confidence' keys
    """
    try:
        service = get_context_expansion_service(min_relevance=min_relevance)
        expansion = service.expand_bullet_point(bullet_point, document_text)
        
        return {
            "bullet": bullet_point,
            "context": expansion.combined_context,
            "confidence": max([c.similarity_score for c in expansion.contexts]) if expansion.contexts else 0,
            "details": expansion.contexts[0].extracted_details if expansion.contexts else {}
        }
    except Exception as e:
        logger.error(f"❌ Single bullet expansion failed: {e}")
        return {
            "bullet": bullet_point,
            "context": "",
            "confidence": 0,
            "details": {}
        }
