"""
Summary Helper Utilities
Shared functions for ensuring Date and Author extraction across all document extractors.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger("document_ai")


def ensure_date_and_author(summary: str, long_summary: str) -> str:
    """
    Programmatically ensure Date and Author are in the summary.
    Extracts from long_summary if missing from LLM output.
    
    Format: [Document Title] | [Author] | [Date] | ...rest
    
    Args:
        summary: The generated short summary from LLM
        long_summary: The long summary containing detailed information
    
    Returns:
        Updated summary with Date and Author if found, omitted if not
    """
    # Split summary by pipes
    parts = [p.strip() for p in summary.split('|')]
    
    if len(parts) < 3:
        logger.warning(f"⚠️ Summary has fewer than 3 parts, cannot validate Date/Author structure")
        return summary
    
    # Expected positions: [0]=Title, [1]=Author, [2]=Date
    title = parts[0]
    author = parts[1] if len(parts) > 1 else ""
    date = parts[2] if len(parts) > 2 else ""
    rest = parts[3:] if len(parts) > 3 else []
    
    # Check if Author is missing or is a key-value pair (means it was skipped)
    author_missing = not author or ':' in author or len(author.strip()) == 0
    
    # Check if Date is missing or is a key-value pair
    date_missing = not date or ':' in date or len(date.strip()) == 0
    
    # Extract Author from long_summary if missing
    if author_missing:
        extracted_author = extract_author_from_long_summary(long_summary)
        if extracted_author:
            logger.info(f"📝 Programmatically added Author: {extracted_author}")
            author = extracted_author
        else:
            logger.warning("⚠️ Author not found in long_summary, skipping Author field")
            author = None  # Will be removed from output
    
    # Extract Date from long_summary if missing
    if date_missing:
        extracted_date = extract_date_from_long_summary(long_summary)
        if extracted_date:
            logger.info(f"📅 Programmatically added Date: {extracted_date}")
            date = extracted_date
        else:
            logger.warning("⚠️ Date not found in long_summary, skipping Date field")
            date = None  # Will be removed from output
    
    # Rebuild summary, omitting None values
    rebuilt_parts = [title]
    if author:
        rebuilt_parts.append(author)
    if date:
        rebuilt_parts.append(date)
    rebuilt_parts.extend(rest)
    
    return ' | '.join(rebuilt_parts)


def extract_author_from_long_summary(long_summary: str) -> Optional[str]:
    """
    Extract Author/Signature from long_summary using regex patterns.
    Returns author name without "Dr." prefix.
    
    Args:
        long_summary: The long summary text to extract from
    
    Returns:
        Extracted author name or None if not found
    """
    # Pattern 1: Author: section with Signature subsection
    author_patterns = [
        r'Author:\s*(?:•\s*)?Signature:\s*([^\n\|]+?)(?:\n|$|\|)',  # Author: • Signature: Name
        r'Signature:\s*([^\n\|]+?)(?:\n|$|\|)',                      # Signature: Name
        r'From:\s*([^\n\|]+?)(?:\n|Organization:|Title:|$|\|)',      # From: Name
        r'Author:\s*([^\n\|]+?)(?:\n|$|\|)',                         # Author: Name
        r'Signed by:\s*([^\n\|]+?)(?:\n|$|\|)',                      # Signed by: Name
        r'Physician:\s*([^\n\|]+?)(?:\n|$|\|)',                      # Physician: Name
        r'Provider:\s*([^\n\|]+?)(?:\n|$|\|)',                       # Provider: Name
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, long_summary, re.IGNORECASE | re.MULTILINE)
        if match:
            author = match.group(1).strip()
            # Clean up common artifacts
            author = re.sub(r'\[.*?\]', '', author)  # Remove [extracted name/title]
            author = re.sub(r'\(.*?\)', '', author)  # Remove (parenthetical info)
            author = author.split(',')[0].strip()    # Take first part before comma
            author = author.split(';')[0].strip()    # Take first part before semicolon
            
            # Remove "Dr." prefix if present
            author = re.sub(r'^Dr\.\s*', '', author, flags=re.IGNORECASE)
            author = re.sub(r'^Doctor\s+', '', author, flags=re.IGNORECASE)
            
            # Validate: should be a person name (1-4 words, not a business name)
            words = author.split()
            if 1 <= len(words) <= 4 and not any(word.lower() in ['llc', 'inc', 'corp', 'ltd', 'office', 'clinic', 'hospital', 'medical', 'center'] for word in words):
                return author
    
    return None


def extract_date_from_long_summary(long_summary: str) -> Optional[str]:
    """
    Extract Document Date from long_summary using regex patterns.
    
    Args:
        long_summary: The long summary text to extract from
    
    Returns:
        Extracted date string or None if not found
    """
    # Pattern 1: Document Date: field
    date_patterns = [
        r'Document Date:\s*([^\n\|]+?)(?:\n|$|\|)',           # Document Date: MM/DD/YYYY
        r'Date:\s*([^\n\|]+?)(?:\n|Subject:|Purpose:|$|\|)', # Date: MM/DD/YYYY
        r'Report Date:\s*([^\n\|]+?)(?:\n|$|\|)',            # Report Date: MM/DD/YYYY
        r'Date of Service:\s*([^\n\|]+?)(?:\n|$|\|)',        # Date of Service: MM/DD/YYYY
        r'Service Date:\s*([^\n\|]+?)(?:\n|$|\|)',           # Service Date: MM/DD/YYYY
        r'Exam Date:\s*([^\n\|]+?)(?:\n|$|\|)',              # Exam Date: MM/DD/YYYY
        r'Visit Date:\s*([^\n\|]+?)(?:\n|$|\|)',             # Visit Date: MM/DD/YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, long_summary, re.IGNORECASE | re.MULTILINE)
        if match:
            date_str = match.group(1).strip()
            # Clean up common artifacts
            date_str = re.sub(r'\[.*?\]', '', date_str)  # Remove [from primary source...]
            date_str = date_str.split(',')[0].strip()    # Take first part before comma
            
            # Validate: should match common date formats
            date_formats = [
                r'\d{1,2}/\d{1,2}/\d{2,4}',           # MM/DD/YYYY or M/D/YY
                r'\d{4}-\d{2}-\d{2}',                  # YYYY-MM-DD
                r'\d{1,2}-\d{1,2}-\d{2,4}',           # MM-DD-YYYY
                r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}',     # January 15, 2024
                r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',       # 15 January 2024
            ]
            
            for date_format in date_formats:
                if re.search(date_format, date_str):
                    return date_str
    
    return None
