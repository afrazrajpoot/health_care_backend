import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignatureExtractor:
    """
    Fast regex-based extractor for author signatures and signature blocks
    from medical documents.
    """
    
    # Comprehensive signature patterns (ordered by priority/reliability)
    SIGNATURE_PATTERNS = [
        # Electronic/Digital signatures (highest priority)
        r'electronically\s+signed\s+by[:\s]+([^\n]{3,100})',
        r'digitally\s+signed\s+by[:\s]+([^\n]{3,100})',
        r'digital\s+signature[:\s]+([^\n]{3,100})',
        r'e-signature[:\s]+([^\n]{3,100})',
        r'authenticated\s+by[:\s]+([^\n]{3,100})',
        
        # Formal closings with names (high priority)
        r'respectfully\s+submitted[,:\s]+([^\n]{3,100})',
        r'sincerely[,:\s]+([^\n]{3,100})',
        r'yours\s+(?:truly|sincerely)[,:\s]+([^\n]{3,100})',
        r'cordially[,:\s]+([^\n]{3,100})',
        r'regards[,:\s]+([^\n]{3,100})',
        
        # Explicit signature indicators
        r'signature[:\s]+([^\n]{3,100})',
        r'signed[:\s]+([^\n]{3,100})',
        r'signed\s+by[:\s]+([^\n]{3,100})',
        r'attestation[:\s]+([^\n]{3,100})',
        
        # Professional closings
        r'prepared\s+by[:\s]+([^\n]{3,100})',
        r'dictated\s+by[:\s]+([^\n]{3,100})',
        r'authored\s+by[:\s]+([^\n]{3,100})',
        r'reviewed\s+(?:and\s+)?signed\s+by[:\s]+([^\n]{3,100})',
        r'examined\s+(?:and\s+)?signed\s+by[:\s]+([^\n]{3,100})',
        
        # Report specific
        r'report\s+(?:prepared|authored|signed)\s+by[:\s]+([^\n]{3,100})',
        r'this\s+report\s+(?:is\s+)?(?:prepared|authored|signed)\s+by[:\s]+([^\n]{3,100})',
        
        # Provider identification
        r'performed\s+by[:\s]+([^\n]{3,100})',
        r'conducted\s+by[:\s]+([^\n]{3,100})',
        r'evaluated\s+by[:\s]+([^\n]{3,100})',
        r'assessed\s+by[:\s]+([^\n]{3,100})',
        
        # Date + Name patterns (common in signature blocks)
        r'date[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\n?\s*([^\n]{3,100})',
        
        # Standalone credentials at end (lower priority)
        r'\n([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+,?\s+(?:M\.?D\.?|D\.?O\.?|P\.?A\.?|N\.?P\.?|R\.?N\.?|Ph\.?D\.?)[^\n]*)\s*$',
    ]
    
    # Compiled patterns for performance
    _compiled_patterns = None
    
    @classmethod
    def _get_compiled_patterns(cls) -> List[re.Pattern]:
        """Lazy compilation of regex patterns for performance."""
        if cls._compiled_patterns is None:
            cls._compiled_patterns = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in cls.SIGNATURE_PATTERNS
            ]
        return cls._compiled_patterns
    
    @classmethod
    def extract_signature_blocks(cls, text: str, context_words: int = 500) -> List[Dict[str, any]]:
        """
        Extract signature blocks and surrounding context from text.
        
        Args:
            text: Full document text
            context_words: Number of words to extract after the signature indicator
                          (default 500 words for comprehensive context)
        
        Returns:
            List of dictionaries containing:
            - pattern: The regex pattern that matched
            - match_text: The exact text that matched the pattern
            - position: Character position in document
            - context: Extended context (next N words after match)
            - priority: Pattern priority (0 = highest)
        """
        if not text or len(text.strip()) < 20:
            return []
        
        results = []
        patterns = cls._get_compiled_patterns()
        
        for priority, pattern in enumerate(patterns):
            for match in pattern.finditer(text):
                match_start = match.start()
                match_end = match.end()
                match_text = match.group(0).strip()
                
                # Extract context after the match
                context_start = match_end
                context = cls._extract_word_context(text, context_start, context_words)
                
                results.append({
                    'pattern': pattern.pattern,
                    'match_text': match_text,
                    'captured_group': match.group(1).strip() if match.groups() else '',
                    'position': match_start,
                    'context': context,
                    'priority': priority,
                    'pattern_type': cls._get_pattern_type(pattern.pattern)
                })
        
        # Sort by priority (lower number = higher priority) and position
        results.sort(key=lambda x: (x['priority'], -x['position']))
        
        # Deduplicate overlapping matches (keep highest priority)
        deduplicated = cls._deduplicate_results(results)
        
        return deduplicated
    
    @staticmethod
    def _extract_word_context(text: str, start_pos: int, num_words: int) -> str:
        """
        Extract approximately N words from text starting at position.
        Fast word-based extraction without heavy parsing.
        """
        # Extract a reasonable chunk (words * avg 6 chars + spaces)
        chunk_size = num_words * 7
        chunk = text[start_pos:start_pos + chunk_size]
        
        # Split into words and take exactly num_words
        words = chunk.split()[:num_words]
        
        return ' '.join(words)
    
    @staticmethod
    def _get_pattern_type(pattern: str) -> str:
        """Classify pattern type for logging/debugging."""
        pattern_lower = pattern.lower()
        if 'electronic' in pattern_lower or 'digital' in pattern_lower:
            return 'electronic_signature'
        elif 'respectfully' in pattern_lower or 'sincerely' in pattern_lower:
            return 'formal_closing'
        elif 'prepared' in pattern_lower or 'authored' in pattern_lower:
            return 'authorship_statement'
        elif 'performed' in pattern_lower or 'conducted' in pattern_lower:
            return 'provider_identification'
        elif 'signature' in pattern_lower or 'signed' in pattern_lower:
            return 'signature_indicator'
        else:
            return 'other'
    
    @staticmethod
    def _deduplicate_results(results: List[Dict]) -> List[Dict]:
        """
        Remove overlapping matches, keeping highest priority ones.
        Two matches overlap if their positions are within 100 characters.
        """
        if not results:
            return []
        
        deduplicated = []
        used_positions = set()
        
        for result in results:
            position = result['position']
            
            # Check if this position overlaps with any used position
            is_overlap = any(
                abs(position - used_pos) < 100
                for used_pos in used_positions
            )
            
            if not is_overlap:
                deduplicated.append(result)
                used_positions.add(position)
        
        return deduplicated
    
    @classmethod
    def extract_author_from_text(cls, text: str) -> Optional[Dict[str, any]]:
        """
        Extract the most likely author from text using signature patterns.
        Returns the BEST match based on pattern priority and position.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with:
            - author: Extracted author name
            - confidence: high/medium/low
            - evidence: How the author was identified
            - context: Surrounding text
            - pattern_type: Type of pattern matched
        """
        signature_blocks = cls.extract_signature_blocks(text, context_words=500)
        
        if not signature_blocks:
            return None
        
        # Get the highest priority match
        best_match = signature_blocks[0]
        
        # Extract name from captured group or match text
        author_text = best_match['captured_group'] or best_match['match_text']
        
        # Clean up the author text
        author = cls._clean_author_name(author_text)
        
        if not author:
            return None
        
        # Determine confidence based on pattern type and position
        confidence = cls._determine_confidence(best_match)
        
        # Build full extracted text: matched text + context (for AI processing)
        full_extracted_text = f"{best_match['match_text']}\n{best_match['context']}"
        
        return {
            'author': author,
            'confidence': confidence,
            'evidence': best_match['match_text'],
            'context': best_match['context'],  # Full context (500 words)
            'full_extracted_text': full_extracted_text,  # Evidence + context combined for AI
            'pattern_type': best_match['pattern_type'],
            'position_in_document': best_match['position']
        }
    
    @staticmethod
    def _clean_author_name(text: str) -> Optional[str]:
        """
        Clean and validate extracted author name.
        Remove common noise and validate it looks like a name.
        """
        if not text:
            return None
        
        # Remove leading/trailing punctuation and whitespace
        text = text.strip(' \t\n\r,.:;')
        
        # Remove common noise words/phrases
        noise_patterns = [
            r'^(?:by|dr\.?|doctor|md|do)\s+',
            r'\s+(?:on|date|dated)\s+\d',
            r'^\s*[:\-_]+\s*',
        ]
        
        for noise in noise_patterns:
            text = re.sub(noise, '', text, flags=re.IGNORECASE)
        
        text = text.strip()
        
        # Validate: should contain at least one letter and reasonable length
        if not re.search(r'[A-Za-z]', text):
            return None
        
        if len(text) < 3 or len(text) > 100:
            return None
        
        # Extract name with credentials if present
        # Pattern: "FirstName LastName, Credentials" or "FirstName LastName Credentials"
        name_match = re.search(
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            r'(?:,?\s+([A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+)*?))?',
            text
        )
        
        if name_match:
            name = name_match.group(1).strip()
            credentials = name_match.group(2).strip() if name_match.group(2) else ''
            
            if credentials:
                return f"{name}, {credentials}"
            return name
        
        # Fallback: return first reasonable line
        first_line = text.split('\n')[0].strip()
        if 3 <= len(first_line) <= 100:
            return first_line
        
        return None
    
    @staticmethod
    def _determine_confidence(match: Dict) -> str:
        """
        Determine confidence level based on pattern type and characteristics.
        """
        pattern_type = match['pattern_type']
        priority = match['priority']
        
        # High confidence: electronic signatures and explicit authorship
        if pattern_type in ['electronic_signature', 'signature_indicator']:
            return 'high'
        
        # Medium confidence: formal closings and authorship statements
        if pattern_type in ['formal_closing', 'authorship_statement']:
            return 'medium'
        
        # Lower confidence for provider identification (could be examiner or referrer)
        if pattern_type == 'provider_identification':
            # Check position: if near end of document, higher confidence
            if match['position'] > len(match.get('full_text', '')) * 0.7:
                return 'medium'
            return 'low'
        
        return 'low'


# Convenience function for quick extraction
def extract_author_signature(text: str, context_words: int = 500) -> Optional[Dict[str, any]]:
    """
    Quick function to extract author signature from document text.
    
    Args:
        text: Full document text
        context_words: Number of words of context to extract (default 500)
    
    Returns:
        Dictionary with author info or None if not found
    
    Example:
        >>> result = extract_author_signature(document_text)
        >>> if result:
        >>>     print(f"Author: {result['author']}")
        >>>     print(f"Confidence: {result['confidence']}")
        >>>     print(f"Evidence: {result['evidence']}")
    """
    logger.info("=" * 60)
    logger.info("🔍 SIGNATURE EXTRACTOR - extract_author_signature called")
    logger.info(f"📄 Input text length: {len(text) if text else 0} chars")
    
    result = SignatureExtractor.extract_author_from_text(text)
    
    logger.info("=" * 60)
    logger.info("📋 SIGNATURE EXTRACTOR RESULT:")
    logger.info("=" * 60)
    if result:
        logger.info(f"  ✅ Author found: {result.get('author')}")
        logger.info(f"  🎯 Confidence: {result.get('confidence')}")
        logger.info(f"  📝 Evidence (matched pattern): {result.get('evidence')}")
        logger.info(f"  🏷️ Pattern type: {result.get('pattern_type')}")
        logger.info(f"  📍 Position: {result.get('position_in_document')}")
        logger.info("=" * 60)
        logger.info("📖 FULL EXTRACTED TEXT (evidence + next 500 words - sent to AI):")
        logger.info("=" * 60)
        logger.info(result.get('full_extracted_text', ''))
        logger.info("=" * 60)
    else:
        logger.info("  ❌ No author signature found")
    logger.info("=" * 60)
    
    return result


def extract_all_signature_blocks(text: str, context_words: int = 500) -> List[Dict[str, any]]:
    """
    Extract all signature blocks from document for analysis/debugging.
    
    Args:
        text: Full document text
        context_words: Number of words of context to extract (default 500)
    
    Returns:
        List of all signature blocks found, sorted by priority
    
    Example:
        >>> blocks = extract_all_signature_blocks(document_text)
        >>> for block in blocks:
        >>>     print(f"Match: {block['match_text']}")
        >>>     print(f"Context: {block['context'][:100]}...")
        >>>     print("---")
    """
    return SignatureExtractor.extract_signature_blocks(text, context_words)
