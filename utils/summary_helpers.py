"""
Summary Helper Utilities
Shared functions for ensuring Date and Author extraction across all document extractors.
Also includes long summary cleaning utilities.
"""
import logging
import re
from typing import Optional, List

logger = logging.getLogger("document_ai")


def clean_long_summary(long_summary: str) -> str:
    """
    Clean up the long summary by removing empty fields, placeholders, 
    instruction text, and formatting artifacts.
    
    This function removes:
    - Lines with empty values (None, Not specified, Not applicable, etc.)
    - Placeholder text and instructions
    - Mandatory notes/rules sections
    - Empty bullet points and list items
    - Consecutive empty lines
    
    Args:
        long_summary: The generated long summary from LLM
    
    Returns:
        Cleaned long summary with only populated fields
    """
    if not long_summary:
        return ""
    
    # Patterns to identify empty/placeholder values
    empty_value_patterns = [
        r':\s*None\s*$',
        r':\s*Not specified\s*$',
        r':\s*Not applicable\s*$',
        r':\s*N/A\s*$',
        r':\s*n/a\s*$',
        r':\s*None explicitly mentioned\s*$',
        r':\s*Not explicitly mentioned\s*$',
        r':\s*Not mentioned\s*$',
        r':\s*Not provided\s*$',
        r':\s*Not available\s*$',
        r':\s*Unknown\s*$',
        r':\s*\-\s*$',
        r':\s*$',  # Empty value after colon
        r'^-\s*None\s*$',
        r'^-\s*Not specified\s*$',
        r'^-\s*Not applicable\s*$',
        r'^-\s*N/A\s*$',
        r'^•\s*None\s*$',
        r'^•\s*Not specified\s*$',
        r'^•\s*$',  # Empty bullet point
        r'^-\s*$',  # Empty dash item
        r'^\*\s*$',  # Empty asterisk item
    ]
    
    # Patterns to remove entire sections/blocks
    remove_section_patterns = [
        r'⚠️\s*MANDATORY\s*EXTRACTION\s*(?:NOTES|RULES).*?(?=\n#|\n📋|\n---|\Z)',
        r'⚠️\s*CRITICAL\s*REMINDERS.*?(?=\n#|\n📋|\n---|\Z)',
        r'⚠️\s*FINAL\s*REMINDER.*?(?=\n#|\n📋|\n---|\Z)',
        r'\(donot include in output.*?\)',
        r'\(for LLM use only\)',
        r'━━━.*?EXTRACTION.*?━━━.*?(?=\n\n|\Z)',
        r'MANDATORY EXTRACTION NOTES:.*?(?=\n#|\n📋|\Z)',
        r'\d+\.\s*All radiological interpretations.*?(?=\n\d+\.|\n#|\Z)',
        r'\d+\.\s*No additional findings.*?(?=\n\d+\.|\n#|\Z)',
        r'\d+\.\s*Empty fields indicate.*?(?=\n\d+\.|\n#|\Z)',
    ]
    
    # First, remove entire instruction/note sections
    cleaned = long_summary
    for pattern in remove_section_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
    # Process line by line
    lines = cleaned.split('\n')
    cleaned_lines: List[str] = []
    skip_next_empty = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines after a removed line
        if skip_next_empty and not stripped:
            skip_next_empty = False
            continue
        
        # Check if line matches empty value patterns
        should_remove = False
        for pattern in empty_value_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                should_remove = True
                skip_next_empty = True
                break
        
        if should_remove:
            continue
        
        # Skip lines that are just placeholders in brackets
        if re.match(r'^-?\s*\[.*(?:extracted|from|if|or).*\]\s*$', stripped, re.IGNORECASE):
            continue
        
        # Skip instruction lines that shouldn't be in output
        instruction_keywords = [
            'donot include in output',
            'for llm use only',
            'extract only',
            'never assume',
            'never infer',
            'zero tolerance',
            'mandatory extraction',
        ]
        if any(kw in stripped.lower() for kw in instruction_keywords):
            continue
        
        cleaned_lines.append(line)
    
    # Join lines back
    result = '\n'.join(cleaned_lines)
    
    # Clean up multiple consecutive empty lines (reduce to max 2)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # Clean up empty sections (header followed by only dashes/empty lines)
    result = re.sub(r'(^#+\s*[^\n]+\n---+\n)(\s*\n)+(?=^#|\Z)', '', result, flags=re.MULTILINE)
    result = re.sub(r'(^---+\n)(\s*\n)+(?=^#|\Z)', '', result, flags=re.MULTILINE)
    
    # Remove trailing whitespace and empty lines
    result = result.strip()
    
    # Remove empty sub-sections that have no content
    result = _remove_empty_sections(result)
    
    logger.info(f"✅ Cleaned long summary: {len(long_summary)} → {len(result)} chars")
    return result


def _remove_empty_sections(text: str) -> str:
    """
    Remove sections that have a header but no meaningful content.
    """
    lines = text.split('\n')
    result_lines: List[str] = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a section header (starts with # or has --- underneath)
        is_header = line.strip().startswith('#') or line.strip().startswith('📋') or line.strip().startswith('🎯') or line.strip().startswith('🔧') or line.strip().startswith('📊') or line.strip().startswith('💡') or line.strip().startswith('👥') or line.strip().startswith('👤')
        
        if is_header:
            # Look ahead to see if this section has content
            section_lines = [line]
            j = i + 1
            
            # Skip separator lines (--- or similar)
            while j < len(lines) and (lines[j].strip().startswith('---') or lines[j].strip() == ''):
                section_lines.append(lines[j])
                j += 1
            
            # Check for content until next header or end
            has_content = False
            while j < len(lines):
                next_line = lines[j].strip()
                
                # Check if we hit another header
                if next_line.startswith('#') or any(next_line.startswith(e) for e in ['📋', '🎯', '🔧', '📊', '💡', '👥', '👤']):
                    break
                
                # Check if this line has actual content (not just empty or placeholder)
                if next_line and not next_line.startswith('---'):
                    # Check it's not just a label with no value
                    if ':' in next_line:
                        # Get value after colon
                        value = next_line.split(':', 1)[1].strip() if ':' in next_line else ''
                        if value and value.lower() not in ['none', 'not specified', 'not applicable', 'n/a', '-', '']:
                            has_content = True
                            break
                    elif next_line.startswith('-') or next_line.startswith('•') or next_line.startswith('*'):
                        # List item - check if it has content
                        item_content = re.sub(r'^[-•*]\s*', '', next_line).strip()
                        if item_content and item_content.lower() not in ['none', 'not specified', 'not applicable', 'n/a']:
                            has_content = True
                            break
                    else:
                        has_content = True
                        break
                
                section_lines.append(lines[j])
                j += 1
            
            if has_content:
                # Keep this section header and continue
                result_lines.append(line)
            else:
                # Skip the empty section
                i = j - 1  # Will be incremented at end of loop
                if i < 0:
                    i = 0
        else:
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)


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
