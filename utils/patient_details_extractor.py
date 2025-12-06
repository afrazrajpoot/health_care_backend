"""
Patient Details Extractor
Robust extraction of patient name, DOB, DOI, and claim number from layout JSON.
Handles variations in field names, formats, and ordering.
"""

import re
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger("patient_extractor")


class PatientDetailsExtractor:
    """
    Extract patient details from Document AI Layout Parser JSON.
    Handles field variations, format differences, and ordering inconsistencies.
    """
    
    # Field name variations (case-insensitive patterns)
    NAME_PATTERNS = [
        r'\bpatient\s*name\b',
        r'\bpatient\b',
        r'\bname\b',
        r'\bre\b',  # "Re:" or "RE:" commonly used for patient name in medical reports
        r'\bapplicant\b',
        r'\binjured\s*worker\b',
        r'\binjured\s*employee\b',
    ]
    
    DOB_PATTERNS = [
        r'\bd\s*o\s*b\b',  # Handles: DOB, D O B, D.O.B
        r'\bd\.?\s*o\.?\s*b\.?\b',  # Handles: D.O.B, DOB, D O B
        r'\bd\s*/\s*b\b',  # Handles: D/B, D / B
        r'\bdate\s*of\s*birth\b',  # Handles: Date of Birth, Dateofbirth
        r'\bbirth\s*date\b',  # Handles: Birth Date, Birthdate
        r'\bb\.?\s*d\.?\b',  # Handles: B.D, BD
        r'\bbirthdate\b',
        r'\bborn\b',  # Sometimes just "Born:"
    ]
    
    DOI_PATTERNS = [
        r'\bd\s*o\s*i\b',  # Handles: DOI, D O I, D.O.I
        r'\bd\.?\s*o\.?\s*i\.?\b',  # Handles: D.O.I, DOI, D O I
        r'\bd\s*/\s*i\b',  # Handles: D/I, D / I (Date of Injury abbreviation)
        r'\bdates?\s*of\s*injury\b',  # Handles: Date of Injury, Dates of Injury
        r'\binjury\s*dates?\b',  # Handles: Injury Date, Injury Dates
        r'\baccident\s*dates?\b',  # Handles: Accident Date, Accident Dates
        r'\bincident\s*dates?\b',  # Handles: Incident Date, Incident Dates
        r'\bloss\s*dates?\b',  # Handles: Loss Date, Loss Dates
        r'\bd\s*/\s*e\b',  # Handles: D/E, D / E (Date of Exam - sometimes used interchangeably)
    ]
    
    CLAIM_PATTERNS = [
        r'\bclaim\s*no\.?\s*\b',  # Handles: Claim No, Claim No., ClaimNo
        r'\bclaim\s*number\b',  # Handles: Claim Number, Claimnumber
        r'\bclaim\s*#\b',  # Handles: Claim #, Claim#
        r'\bclaim\b',  # Handles: Claim
        r'\bcl\.?\s*no\.?\b',  # Handles: Cl No, Cl. No., ClNo
        r'\bcl\.?\s*#\b',  # Handles: Cl #, Cl#
        r'\bcl\s*:\s*\b',  # Handles: Cl:, Cl : (abbreviated claim with colon)
        r'\bcl\b',  # Handles: Cl (standalone abbreviation)
        r'\bfile\s*no\.?\b',  # Handles: File No, FileNo
        r'\bcase\s*no\.?\b',  # Handles: Case No, CaseNo
        r'\bwc\s*#\b',  # Handles: WC #, WC#
        r'\bwcab\s*#\b',  # Handles: WCAB #, WCAB# (Workers' Comp Appeals Board)
        r'\binsured\s*id\b',  # Handles: Insured ID, InsuredID
        r'\bpol\.?\s*#\b',  # Handles: Pol #, Pol#
        r'\bpolicy\s*#\b',  # Handles: Policy #, Policy#
        r'\bacct\.?\s*#\b',  # Handles: Acct #, Acct#
        r'\baccount\s*#\b',  # Handles: Account #, Account#
        r'\bref\.?\s*no\.?\b',  # Handles: Ref No, RefNo
        r'\breference\s*no\.?\b',  # Handles: Reference No
    ]
    
    # Date format patterns
    DATE_FORMATS = [
        r'\b\d{1}[-/]\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # 4-part with leading zero: 0-4-05-1955 or 0/4/05/1955
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',    # YYYY-MM-DD
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Mon YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Mon DD, YYYY
        r'\b\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}\b',  # 4-part with spaces: 0 - 4 - 05 - 1955
        r'\b\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{2,4}\b',  # With spaces around dashes (standard 3-part)
    ]
    
    # Claim number patterns (WC claims often have specific formats)
    CLAIM_NUMBER_PATTERNS = [
        r'\b\d{5,}[-/]?\d+[-/]?(?:WC|wc)[-/]?\d*\b',  # 002456-654361-WC-01
        r'\b\d{5,}[-/]?\d+\b',  # General number format
        r'\b[A-Z]{2,}\d{5,}\b',  # State code + numbers
    ]
    
    def __init__(self):
        # Compile patterns with mixed case sensitivity
        # Full words: case-insensitive, Abbreviations: case-sensitive where needed
        self.name_regex = []
        for p in self.NAME_PATTERNS:
            self.name_regex.append(re.compile(p, re.IGNORECASE))
        
        self.dob_regex = []
        for p in self.DOB_PATTERNS:
            # Keep case-insensitive for flexibility
            self.dob_regex.append(re.compile(p, re.IGNORECASE))
        
        self.doi_regex = []
        for p in self.DOI_PATTERNS:
            # D/I and D/E should be case-sensitive to avoid false matches
            if r'd\s*/\s*i' in p.lower() or r'd\s*/\s*e' in p.lower():
                # Make case-insensitive but will validate format in extraction
                self.doi_regex.append(re.compile(p, re.IGNORECASE))
            else:
                self.doi_regex.append(re.compile(p, re.IGNORECASE))
        
        self.claim_regex = []
        for p in self.CLAIM_PATTERNS:
            # Cl (standalone) should be more strict to avoid false positives
            if p == r'\bcl\b':
                # Case-sensitive for "Cl" to avoid matching random "cl" in words
                self.claim_regex.append(re.compile(r'\b[Cc]l\b'))
            elif r'\bcl\.' in p or r'\bcl\s*:' in p or r'\bcl\s*#' in p:
                # Cl with punctuation can be case-insensitive
                self.claim_regex.append(re.compile(p, re.IGNORECASE))
            else:
                # Full words like "claim", "file", "case" etc. - case-insensitive
                self.claim_regex.append(re.compile(p, re.IGNORECASE))
        
        self.date_formats = [re.compile(p, re.IGNORECASE) for p in self.DATE_FORMATS]
        self.claim_number_formats = [re.compile(p) for p in self.CLAIM_NUMBER_PATTERNS]
    
    def extract_from_layout_json(self, document_dict: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Extract patient details from layout parser JSON.
        
        Args:
            document_dict: Full Document AI Layout Parser output
            
        Returns:
            Dictionary with patient_name, dob, doi, claim_number
        """
        result = {
            "patient_name": None,
            "dob": None,
            "doi": None,
            "claim_number": None
        }
        
        # Extract all text blocks with their context
        blocks = self._extract_all_blocks(document_dict)
        
        if not blocks:
            logger.warning("No blocks found in document_dict")
            return result
        
        logger.info(f"Extracted {len(blocks)} blocks for patient details extraction")
        
        # Log first few blocks for debugging
        if blocks:
            logger.debug("Sample blocks:")
            for i, block in enumerate(blocks[:10]):
                logger.debug(f"  Block {i}: source={block.get('source')}, text='{block.get('text')[:50]}'")
        
        # Extract each field
        result["patient_name"] = self._extract_name(blocks)
        result["dob"] = self._extract_dob(blocks)
        result["doi"] = self._extract_doi(blocks)
        result["claim_number"] = self._extract_claim_number(blocks)
        
        # Log results
        logger.info(f"Patient Details Extraction Results:")
        logger.info(f"  Name: {result['patient_name']}")
        logger.info(f"  DOB: {result['dob']}")
        logger.info(f"  DOI: {result['doi']}")
        logger.info(f"  Claim: {result['claim_number']}")
        
        return result
    
    def _extract_all_blocks(self, document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all text blocks from document_dict.
        Handles tables, paragraphs, and nested structures.
        """
        blocks = []
        
        # Method 1: Extract from document_layout blocks (with recursive nested block extraction)
        if document_dict.get('document_layout') and document_dict['document_layout'].get('blocks'):
            layout_blocks = document_dict['document_layout']['blocks']
            blocks.extend(self._extract_blocks_recursive(layout_blocks, 'layout'))
        
        # Method 2: Extract from tables
        if document_dict.get('document_layout') and document_dict['document_layout'].get('tables'):
            tables = document_dict['document_layout']['tables']
            for table_idx, table in enumerate(tables):
                if not table.get('body_rows'):
                    continue
                
                for row_idx, row in enumerate(table['body_rows']):
                    if not row.get('cells'):
                        continue
                    
                    row_texts = []
                    row_cells_data = []  # Store cell data for pairing
                    
                    for cell_idx, cell in enumerate(row['cells']):
                        if not cell.get('blocks'):
                            continue
                        
                        cell_text = []
                        for block in cell['blocks']:
                            text_block = block.get('text_block', {})
                            text = text_block.get('text', '').strip()
                            if text:
                                cell_text.append(text)
                                row_texts.append(text)
                                blocks.append({
                                    'text': text,
                                    'block_id': block.get('block_id', ''),
                                    'page': block.get('page_span', {}).get('page_start', 0),
                                    'table_idx': table_idx,
                                    'row_idx': row_idx,
                                    'cell_idx': cell_idx,
                                    'source': 'table'
                                })
                        
                        # Store combined cell text
                        if cell_text:
                            row_cells_data.append(' '.join(cell_text))
                    
                    # Create paired field-value blocks for 2-column tables
                    if len(row_cells_data) == 2:
                        # Common pattern: Field | Value
                        field_text = row_cells_data[0].strip()
                        value_text = row_cells_data[1].strip()
                        
                        # Create a combined block with colon separator for easier parsing
                        combined_text = f"{field_text}: {value_text}"
                        blocks.append({
                            'text': combined_text,
                            'block_id': f'table_row_{table_idx}_{row_idx}_paired',
                            'page': row.get('cells', [{}])[0].get('blocks', [{}])[0].get('page_span', {}).get('page_start', 0),
                            'source': 'table_row_paired'
                        })
                        
                        logger.debug(f"Created table row pair: '{field_text}' -> '{value_text}'")
                    
                    # Also store row-level context for adjacent cell matching (original logic)
                    if len(row_texts) >= 2:
                        blocks.append({
                            'text': ' | '.join(row_texts),
                            'block_id': f'row_{table_idx}_{row_idx}',
                            'page': row['cells'][0].get('blocks', [{}])[0].get('page_span', {}).get('page_start', 0),
                            'source': 'table_row'
                        })
        
        # Method 3: Extract from pages/paragraphs (if available)
        if document_dict.get('pages'):
            for page_idx, page in enumerate(document_dict['pages']):
                if page.get('paragraphs'):
                    for para in page['paragraphs']:
                        if para.get('layout') and para['layout'].get('text_anchor'):
                            text = para['layout']['text_anchor'].get('content', '').strip()
                            if text:
                                blocks.append({
                                    'text': text,
                                    'block_id': f'para_{page_idx}',
                                    'page': page_idx + 1,
                                    'source': 'paragraph'
                                })
        
        return blocks
    
    def _is_valid_field_format(self, text: str, matched_field: str) -> bool:
        """
        Validate that abbreviated field names have proper formatting.
        Ensures "D/I", "Cl", "Re", etc. are used as field labels, not random text.
        """
        text = text.strip()
        matched_field = matched_field.strip()
        text_lower = text.lower()
        matched_lower = matched_field.lower()
        
        # Very short abbreviations (2-3 chars) should have colon, #, or be standalone
        if len(matched_field) <= 3 and '/' in matched_field:
            # Abbreviations like "D/I", "D/E", "D/B" should be followed by colon or be the whole text
            if text == matched_field or text.startswith(matched_field + ':') or text.startswith(matched_field + ' :'):
                return True
            # Or be at the end with colon
            if text.endswith(':') and matched_field in text:
                return True
            return False
        
        # Standalone "Cl" should be the whole text or followed by :, #, or "No"
        if matched_lower == 'cl' and len(matched_field) == 2:
            # Check case-insensitive patterns
            if text_lower == matched_lower or text_lower in ['cl:', 'cl#', 'cl :', 'cl #']:
                return True
            if text_lower.startswith('cl no') or text_lower.startswith('cl:') or text_lower.startswith('cl#') or text_lower.startswith('cl '):
                return True
            return False
        
        # "Re" should be followed by colon (Re:, RE:, re:)
        if matched_lower == 're' and len(matched_field) == 2:
            # Must have colon after it to be valid (Re:, RE:, Re :, etc.)
            if ':' in text and (text_lower.startswith('re:') or text_lower.startswith('re :')):
                return True
            # Or be exactly "Re:" or "RE:"
            if text_lower in ['re:', 're :']:
                return True
            return False
        
        # Handle patterns with dots like "Claim No.:" or "Cl No.:"
        # Remove dots and check if it ends with colon
        if '.' in text and ':' in text:
            # Pattern like "Claim No.:" or "Cl No.:"
            # If field is in the text and text ends with colon, it's valid
            if matched_lower in text_lower and text.endswith(':'):
                return True
        
        # Full words are always valid
        if len(matched_field) > 3:
            return True
        
        return True
    
    def _extract_blocks_recursive(self, blocks_list: List[Dict], source: str, parent_id: str = '') -> List[Dict[str, Any]]:
        """
        Recursively extract all blocks including nested ones and tables.
        The JSON structure has blocks within text_block.blocks within blocks.
        Also handles table_block structures with body_rows and cells.
        """
        extracted = []
        
        for block in blocks_list:
            block_id = block.get('block_id', '')
            page_span = block.get('page_span', {})
            page = page_span.get('page_start', 0)
            
            # Handle text_block
            if 'text_block' in block:
                text_block = block['text_block']
                text = text_block.get('text', '').strip()
                
                if text:
                    extracted.append({
                        'text': text,
                        'block_id': block_id,
                        'page': page,
                        'source': source,
                        'parent_id': parent_id
                    })
                
                # Recursively extract nested blocks
                nested_blocks = text_block.get('blocks', [])
                if nested_blocks:
                    extracted.extend(self._extract_blocks_recursive(nested_blocks, source, block_id))
            
            # Handle table_block (NEW!)
            elif 'table_block' in block:
                table_block = block['table_block']
                
                # Extract from table rows
                if table_block.get('body_rows'):
                    for row_idx, row in enumerate(table_block['body_rows']):
                        if not row.get('cells'):
                            continue
                        
                        row_cells_data = []
                        
                        for cell_idx, cell in enumerate(row['cells']):
                            if not cell.get('blocks'):
                                continue
                            
                            cell_text_parts = []
                            for cell_block in cell['blocks']:
                                cell_text_block = cell_block.get('text_block', {})
                                cell_text_str = cell_text_block.get('text', '').strip()
                                if cell_text_str:
                                    cell_text_parts.append(cell_text_str)
                                    
                                    # Add individual cell block
                                    extracted.append({
                                        'text': cell_text_str,
                                        'block_id': cell_block.get('block_id', ''),
                                        'page': cell_block.get('page_span', {}).get('page_start', page),
                                        'table_row_idx': row_idx,
                                        'table_cell_idx': cell_idx,
                                        'source': f'{source}_table',
                                        'parent_id': block_id
                                    })
                            
                            # Store combined cell text
                            if cell_text_parts:
                                row_cells_data.append(' '.join(cell_text_parts))
                        
                        # Create paired field-value blocks for 2-column tables
                        if len(row_cells_data) == 2:
                            field_text = row_cells_data[0].strip()
                            value_text = row_cells_data[1].strip()
                            
                            # Remove trailing colon from field if present (avoid double colons)
                            if field_text.endswith(':'):
                                field_text = field_text[:-1].strip()
                            
                            # Create combined block with colon separator
                            combined_text = f"{field_text}: {value_text}"
                            extracted.append({
                                'text': combined_text,
                                'block_id': f'{block_id}_row_{row_idx}_paired',
                                'page': page,
                                'source': f'{source}_table_row_paired',
                                'parent_id': block_id
                            })
                            
                            logger.debug(f"Table row pair: '{field_text}' -> '{value_text}'")
        
        return extracted
    
    def _extract_name(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Extract patient name using field patterns and validation"""
        candidates = []
        
        for i, block in enumerate(blocks):
            text = block['text']
            
            # Debug: Check if this looks like a name field
            for regex in self.name_regex:
                match = regex.search(text)
                if match:
                    logger.debug(f"Name field candidate found: '{text}' (matched: '{match.group(0)}')")
                    break
            
            # Check if block contains name field with value in same block (colon or # separator)
            value_found_in_same_block = False
            
            # Try colon separator
            if ':' in text:
                for regex in self.name_regex:
                    match = regex.search(text)
                    if match:
                        is_valid = self._is_valid_field_format(text, match.group(0))
                        logger.debug(f"Checking name field with colon: '{text}', matched='{match.group(0)}', valid={is_valid}")
                        if is_valid:
                            # Extract value after colon
                            parts = text.split(':', 1)
                            if len(parts) == 2:
                                candidate = parts[1].strip()
                                if self._is_valid_name(candidate):
                                    logger.debug(f"Found name in same block (colon): '{candidate}'")
                                    candidates.append({
                                        'value': candidate,
                                        'confidence': 0.95,
                                        'source': 'same_block_colon'
                                    })
                                    value_found_in_same_block = True
                                    break
            
            # Try # separator (less common for names but possible)
            if not value_found_in_same_block and '#' in text:
                for regex in self.name_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # Extract value after #
                        parts = text.split('#', 1)
                        if len(parts) == 2:
                            candidate = parts[1].strip()
                            if self._is_valid_name(candidate):
                                candidates.append({
                                    'value': candidate,
                                    'confidence': 0.95,
                                    'source': 'same_block_hash'
                                })
                                value_found_in_same_block = True
                                break
            
            # Check if this block is a name field (look in next block if no value found in same block)
            is_name_field = False
            for regex in self.name_regex:
                match = regex.search(text)
                if match and self._is_valid_field_format(text, match.group(0)):
                    is_name_field = True
                    break
            
            if is_name_field and not value_found_in_same_block:
                # Look for value in next block (field: value pattern)
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    # Same page
                    if next_block['page'] == block['page']:
                        candidate = next_block['text']
                        if self._is_valid_name(candidate):
                            candidates.append({
                                'value': candidate,
                                'confidence': 0.9,
                                'source': 'adjacent_block'
                            })
            
            # Check if text looks like just a name (common in headers)
            # Pattern: "Lastname, Firstname" or "Firstname Lastname"
            if self._is_valid_name(text) and not is_name_field:
                # Higher confidence if it follows a pattern
                if ',' in text and len(text.split(',')) == 2:
                    # "Fierro, Xavier" pattern
                    candidates.append({
                        'value': text,
                        'confidence': 0.85,
                        'source': 'name_pattern'
                    })
            
            # Check if previous block was a name field (value field pattern)
            if i > 0:
                prev_block = blocks[i - 1]
                prev_is_field = any(regex.search(prev_block['text']) for regex in self.name_regex)
                if prev_is_field and prev_block['page'] == block['page']:
                    if self._is_valid_name(text):
                        candidates.append({
                            'value': text,
                            'confidence': 0.88,
                            'source': 'previous_field'
                        })
            
            # Check table row format (field | value in same row)
            if block['source'] == 'table_row' and '|' in text:
                parts = text.split('|')
                for j, part in enumerate(parts):
                    if any(regex.search(part) for regex in self.name_regex):
                        # Field found, check adjacent parts
                        if j + 1 < len(parts):
                            candidate = parts[j + 1].strip()
                            if self._is_valid_name(candidate):
                                candidates.append({
                                    'value': candidate,
                                    'confidence': 0.92,
                                    'source': 'table_row'
                                })
                        if j > 0:
                            candidate = parts[j - 1].strip()
                            if self._is_valid_name(candidate):
                                candidates.append({
                                    'value': candidate,
                                    'confidence': 0.90,
                                    'source': 'table_row_before'
                                })
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            logger.info(f"Name extracted: {best['value']} (confidence: {best['confidence']}, source: {best['source']})")
            return best['value']
        
        return None
    
    def _extract_dob(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Extract date of birth"""
        return self._extract_date_field(blocks, self.dob_regex, "DOB")
    
    def _extract_doi(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Extract date of injury"""
        return self._extract_date_field(blocks, self.doi_regex, "DOI")
    
    def _extract_date_field(self, blocks: List[Dict[str, Any]], field_regex: List, field_name: str) -> Optional[str]:
        """Generic date field extraction with validation"""
        candidates = []
        
        for i, block in enumerate(blocks):
            text = block['text']
            
            # Check if block contains date field with value in same block (colon or # separator)
            value_found_in_same_block = False
            
            # Try colon separator
            if ':' in text:
                for regex in field_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # Extract value after colon
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            date_match = self._extract_date_from_text(parts[1])
                            if date_match:
                                candidates.append({
                                    'value': date_match,
                                    'confidence': 0.95,
                                    'source': 'same_block_colon'
                                })
                                value_found_in_same_block = True
                                break
            
            # Try # separator (less common for dates but possible)
            if not value_found_in_same_block and '#' in text:
                for regex in field_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # Extract value after #
                        parts = text.split('#', 1)
                        if len(parts) == 2:
                            date_match = self._extract_date_from_text(parts[1])
                            if date_match:
                                candidates.append({
                                    'value': date_match,
                                    'confidence': 0.95,
                                    'source': 'same_block_hash'
                                })
                                value_found_in_same_block = True
                                break
            
            # Check if this block is a date field (look in next block if no value found in same block)
            is_field = False
            for regex in field_regex:
                match = regex.search(text)
                if match and self._is_valid_field_format(text, match.group(0)):
                    is_field = True
                    break
            
            if is_field and not value_found_in_same_block:
                # Look for date in next block
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    if next_block['page'] == block['page']:
                        date_match = self._extract_date_from_text(next_block['text'])
                        if date_match:
                            candidates.append({
                                'value': date_match,
                                'confidence': 0.9,
                                'source': 'adjacent_block'
                            })
            
            # Check previous block
            if i > 0:
                prev_block = blocks[i - 1]
                prev_is_field = any(regex.search(prev_block['text']) for regex in field_regex)
                if prev_is_field and prev_block['page'] == block['page']:
                    date_match = self._extract_date_from_text(text)
                    if date_match:
                        candidates.append({
                            'value': date_match,
                            'confidence': 0.88,
                            'source': 'previous_field'
                        })
            
            # Check table row format
            if block['source'] == 'table_row' and '|' in text:
                parts = text.split('|')
                for j, part in enumerate(parts):
                    if any(regex.search(part) for regex in field_regex):
                        if j + 1 < len(parts):
                            date_match = self._extract_date_from_text(parts[j + 1])
                            if date_match:
                                candidates.append({
                                    'value': date_match,
                                    'confidence': 0.92,
                                    'source': 'table_row'
                                })
                        if j > 0:
                            date_match = self._extract_date_from_text(parts[j - 1])
                            if date_match:
                                candidates.append({
                                    'value': date_match,
                                    'confidence': 0.90,
                                    'source': 'table_row_before'
                                })
        
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            logger.info(f"{field_name} extracted: {best['value']} (confidence: {best['confidence']}, source: {best['source']})")
            return best['value']
        
        return None
    
    def _extract_claim_number(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Extract claim number with format validation"""
        candidates = []
        
        for i, block in enumerate(blocks):
            text = block['text']
            
            # NEW: Check for embedded claim number in brackets or after # symbol
            # Pattern: "Something [Claim #123456-001]" or "Something (Claim: 123456)"
            for regex in self.claim_regex:
                match = regex.search(text)
                if match:
                    # Try to extract claim number from the same text after the keyword
                    # Look for patterns like: [Claim #NUMBER], (Claim #NUMBER), [Claim: NUMBER]
                    embedded_patterns = [
                        r'[\[\(].*?claim.*?[#:]?\s*(\d+[-/\dA-Z]+).*?[\]\)]',  # [Claim #123-456] or (Claim: 123)
                        r'claim\s*[#:]\s*(\d+[-/\dA-Z]+)',  # Claim #123-456 or Claim: 123
                    ]
                    
                    for pattern in embedded_patterns:
                        embedded_match = re.search(pattern, text, re.IGNORECASE)
                        if embedded_match:
                            claim_number = embedded_match.group(1).strip()
                            # Validate it's not a date
                            if not any(date_regex.search(claim_number) for date_regex in self.date_formats):
                                candidates.append({
                                    'value': claim_number,
                                    'confidence': 0.96,
                                    'source': 'embedded_in_text'
                                })
                                break
            
            # Check if block contains claim field with value in same block (colon or # separator)
            value_found_in_same_block = False
            
            # Try colon separator
            if ':' in text:
                for regex in self.claim_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # Extract value after colon
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            claim_match = self._extract_claim_from_text(parts[1])
                            if claim_match:
                                candidates.append({
                                    'value': claim_match,
                                    'confidence': 0.95,
                                    'source': 'same_block_colon'
                                })
                                value_found_in_same_block = True
                                break
            
            # Try # separator (e.g., "Cl# 123" or "Claim #123")
            if not value_found_in_same_block and '#' in text:
                for regex in self.claim_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # Extract value after #
                        parts = text.split('#', 1)
                        if len(parts) == 2:
                            claim_match = self._extract_claim_from_text(parts[1])
                            if claim_match:
                                candidates.append({
                                    'value': claim_match,
                                    'confidence': 0.95,
                                    'source': 'same_block_hash'
                                })
                                value_found_in_same_block = True
                                break
            
            # Check if this block is a claim field (look in next block if no value found in same block)
            is_claim_field = False
            for regex in self.claim_regex:
                match = regex.search(text)
                if match and self._is_valid_field_format(text, match.group(0)):
                    is_claim_field = True
                    break
            
            if is_claim_field and not value_found_in_same_block:
                # Look for value in next block
                if i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    if next_block['page'] == block['page']:
                        claim_match = self._extract_claim_from_text(next_block['text'])
                        if claim_match:
                            candidates.append({
                                'value': claim_match,
                                'confidence': 0.9,
                                'source': 'adjacent_block'
                            })
            
            # Check previous block
            if i > 0:
                prev_block = blocks[i - 1]
                prev_is_field = any(regex.search(prev_block['text']) for regex in self.claim_regex)
                if prev_is_field and prev_block['page'] == block['page']:
                    claim_match = self._extract_claim_from_text(text)
                    if claim_match:
                        candidates.append({
                            'value': claim_match,
                            'confidence': 0.88,
                            'source': 'previous_field'
                        })
            
            # Check table row format
            if block['source'] == 'table_row' and '|' in text:
                parts = text.split('|')
                for j, part in enumerate(parts):
                    if any(regex.search(part) for regex in self.claim_regex):
                        if j + 1 < len(parts):
                            claim_match = self._extract_claim_from_text(parts[j + 1])
                            if claim_match:
                                candidates.append({
                                    'value': claim_match,
                                    'confidence': 0.92,
                                    'source': 'table_row'
                                })
                        if j > 0:
                            claim_match = self._extract_claim_from_text(parts[j - 1])
                            if claim_match:
                                candidates.append({
                                    'value': claim_match,
                                    'confidence': 0.90,
                                    'source': 'table_row_before'
                                })
        
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            logger.info(f"Claim extracted: {best['value']} (confidence: {best['confidence']}, source: {best['source']})")
            return best['value']
        
        return None
    
    def _is_valid_name(self, text: str) -> bool:
        """Validate that text looks like a name"""
        if not text or len(text) < 2:
            return False
        
        # Remove common prefixes
        text = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+', '', text, flags=re.IGNORECASE)
        
        # Name should have letters
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Should not be too long (likely a sentence)
        if len(text) > 50:
            return False
        
        # Should not contain too many numbers (likely not a name)
        digit_count = sum(c.isdigit() for c in text)
        if digit_count > 3:
            return False
        
        # Should not match date patterns
        if any(regex.search(text) for regex in self.date_formats):
            return False
        
        # Should not match claim patterns
        if any(regex.search(text) for regex in self.claim_number_formats):
            return False
        
        return True
    
    def _extract_date_from_text(self, text: str) -> Optional[str]:
        """Extract and normalize date from text"""
        if not text:
            return None
        
        # Try to find date pattern
        for regex in self.date_formats:
            match = regex.search(text)
            if match:
                date_str = match.group(0)
                # Try to normalize date
                normalized = self._normalize_date(date_str)
                if normalized:
                    return normalized
        
        return None
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format"""
        # Clean up the date string (remove extra spaces, handle multiple separators)
        date_str = date_str.strip()
        
        # Handle dates with spaces around separators: "0-4-05-1955" or "0 - 4 - 05 - 1955"
        date_str = re.sub(r'\s*-\s*', '-', date_str)
        date_str = re.sub(r'\s*/\s*', '/', date_str)
        
        date_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
            "%b %d, %Y", "%B %d, %Y",
            "%m-%d-%Y", "%d-%m-%Y",  # Additional formats
            "%-m-%d-%Y", "%-d-%m-%Y",  # Without leading zeros
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # Try parsing dates with no leading zeros manually (e.g., "0-4-05-1955" or "4-5-1955")
        # Pattern: M-D-MM-YYYY or M-D-YY-YYYY
        date_parts = date_str.split('-')
        if len(date_parts) >= 3:
            try:
                # Try different interpretations
                # Format: 0-M-DD-YYYY or M-D-DD-YYYY (0-4-05-1955 = April 5, 1955)
                if len(date_parts) == 4:
                    # Check if first part is 0 (leading zero case)
                    if date_parts[0] == '0' or int(date_parts[0]) == 0:
                        # Skip the leading zero: 0-4-05-1955 → month=4, day=5, year=1955
                        month = int(date_parts[1])
                        day = int(date_parts[2])
                        year = int(date_parts[3])
                    else:
                        # Normal case: M-D-DD-YYYY
                        month = int(date_parts[0])
                        day = int(date_parts[1])
                        # Handle DD-YYYY where DD could be day and YYYY is year
                        if len(date_parts[2]) <= 2:
                            day = int(date_parts[2])
                            year = int(date_parts[3])
                        else:
                            year = int(date_parts[2])
                    
                    if year < 100:
                        year += 1900 if year > 30 else 2000
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
                # Format: M-D-YYYY or MM-DD-YYYY
                elif len(date_parts) == 3:
                    month = int(date_parts[0])
                    day = int(date_parts[1])
                    year = int(date_parts[2])
                    if year < 100:
                        year += 1900 if year > 30 else 2000
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass
        
        # Try with slashes (e.g., "0/4/05/1955" or "4/5/1955")
        date_parts = date_str.split('/')
        if len(date_parts) >= 3:
            try:
                # Format: 0/M/DD/YYYY or M/D/DD/YYYY
                if len(date_parts) == 4:
                    # Check if first part is 0 (leading zero case)
                    if date_parts[0] == '0' or int(date_parts[0]) == 0:
                        # Skip the leading zero: 0/4/05/1955 → month=4, day=5, year=1955
                        month = int(date_parts[1])
                        day = int(date_parts[2])
                        year = int(date_parts[3])
                    else:
                        # Normal case: M/D/DD/YYYY
                        month = int(date_parts[0])
                        day = int(date_parts[1])
                        # Handle DD/YYYY
                        if len(date_parts[2]) <= 2:
                            day = int(date_parts[2])
                            year = int(date_parts[3])
                        else:
                            year = int(date_parts[2])
                    
                    if year < 100:
                        year += 1900 if year > 30 else 2000
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
                # Format: M/D/YYYY or MM/DD/YYYY
                elif len(date_parts) == 3:
                    month = int(date_parts[0])
                    day = int(date_parts[1])
                    year = int(date_parts[2])
                    if year < 100:
                        year += 1900 if year > 30 else 2000
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass
        
        return None
    
    def _extract_claim_from_text(self, text: str) -> Optional[str]:
        """Extract claim number from text"""
        if not text:
            return None
        
        text = text.strip()
        
        # Try specific claim patterns first
        for regex in self.claim_number_formats:
            match = regex.search(text)
            if match:
                return match.group(0).strip()
        
        # Fallback: any sequence that looks like a claim number
        # Pattern 1: Continuous sequence (at least 5 chars with digits/letters/dashes/slashes)
        fallback_pattern = r'\b[\dA-Z][-/\dA-Z]{4,}\b'
        match = re.search(fallback_pattern, text)
        if match:
            candidate = match.group(0).strip()
            # Make sure it's not a date
            if not any(regex.search(candidate) for regex in self.date_formats):
                return candidate
        
        # Pattern 2: Alphanumeric with spaces (common format: "480 CB JH3 4455 K")
        # Must have at least one digit and one letter, and be at least 10 chars with spaces
        if len(text) >= 10 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
            # Remove extra whitespace, but keep single spaces
            normalized = ' '.join(text.split())
            # Check if it looks like a claim (mix of letters, numbers, spaces)
            # Must not be a date
            if not any(regex.search(normalized) for regex in self.date_formats):
                # Must not be all letters or all numbers
                has_letters = any(c.isalpha() for c in normalized)
                has_digits = any(c.isdigit() for c in normalized)
                if has_letters and has_digits:
                    return normalized
        
        return None


# Global instance
_extractor_instance = None

def get_patient_extractor() -> PatientDetailsExtractor:
    """Get singleton patient extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = PatientDetailsExtractor()
    return _extractor_instance
