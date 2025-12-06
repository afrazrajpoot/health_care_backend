"""
Patient Details Extractor
Robust extraction of patient name, DOB, DOI, and claim number from layout JSON.
Handles variations in field names, formats, and ordering.
"""

import re
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import CONFIG
import json
import os

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
        r'\bclaim\s*:\s*\b',  # Handles: Claim: (with colon - specific match)
        r'\bclaim\b',  # Handles: Claim (standalone)
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
    
    # ❌ EXCLUDED PATTERNS - DO NOT extract as claim numbers
    NON_CLAIM_PATTERNS = [
        r'\bBCN\s*:\s*',   # BCN: (Billing Control Number)
        r'\bDCN\s*:\s*',   # DCN: (Document Control Number)
        r'\bICN\s*:\s*',   # ICN: (Internal Control Number)
        r'\bPCN\s*:\s*',   # PCN: (Prescription Control Number)
        r'\bTRN\s*:\s*',   # TRN: (Transaction Number)
        r'\bRCN\s*:\s*',   # RCN: (Record Control Number)
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
    # ✅ STRICT PATTERNS - Only match typical claim number formats
    CLAIM_NUMBER_PATTERNS = [
        r'\b[A-Z]{2}\d{4,8}(?:-\d{3})?\b',  # JH34455-001, JH34455
        r'\b\d{5,7}[-/]\d{5,7}[-/](?:WC|wc)[-/]?\d{1,3}\b',  # 002456-654361-WC-01
        r'\b\d{4,6}[-/][A-Z]{2}[-/][A-Z]{2}\d[-/]\d{4}[-/][A-Z]\b',  # 480-CB-JH3-4455-K (space-separated converted)
        r'\b[A-Z]{2,4}\d{2,4}[-/]?\d{2,4}\b',  # WC1234-5678, WCAB12-34
        r'\b\d{4,6}\s+[A-Z]{2}\s+[A-Z]{2}\d\s+\d{4}\s+[A-Z]\b',  # 480 CB JH3 4455 K (with spaces)
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
        
        # Compile non-claim patterns to exclude (BCN, DCN, etc.)
        self.non_claim_regex = []
        for p in self.NON_CLAIM_PATTERNS:
            self.non_claim_regex.append(re.compile(p, re.IGNORECASE))
        
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
        
        # 🆕 LLM Validation: Verify extracted details using first 2000 words
        result = self._validate_with_llm(result, blocks)
        
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
        """
        Extract claim number with STRICT keyword validation.
        Only extracts numbers that are explicitly associated with claim keywords.
        """
        candidates = []
        
        for i, block in enumerate(blocks):
            text = block['text']
            
            # ✅ PRIORITY 1: Embedded claim number with keyword in brackets/parentheses
            # Pattern: "Something [Claim #123456-001]" or "Something (Claim: 123456)"
            for regex in self.claim_regex:
                match = regex.search(text)
                if match:
                    # Check this is NOT a non-claim pattern
                    is_non_claim = any(non_claim_regex.search(text) for non_claim_regex in self.non_claim_regex)
                    if is_non_claim:
                        continue
                    
                    # Try to extract claim number from the same text after the keyword
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
                                    'confidence': 0.98,  # Highest confidence - explicit keyword
                                    'source': 'embedded_with_keyword'
                                })
                                break
            
            # Check if block contains claim field with value in same block (colon or # separator)
            value_found_in_same_block = False
            
            # Try colon separator - ENHANCED: Only extract after valid claim keywords
            if ':' in text:
                for regex in self.claim_regex:
                    match = regex.search(text)
                    if match and self._is_valid_field_format(text, match.group(0)):
                        # ✅ NEW: Check this is NOT a non-claim pattern (BCN:, DCN:, etc.)
                        is_non_claim = any(non_claim_regex.search(text) for non_claim_regex in self.non_claim_regex)
                        if is_non_claim:
                            continue  # Skip this match, it's BCN:/DCN:/etc.
                        
                        # Find the position of the matched keyword
                        keyword_pos = match.start()
                        # Find the colon after this keyword
                        text_after_keyword = text[keyword_pos:]
                        colon_idx = text_after_keyword.find(':')
                        
                        if colon_idx != -1:
                            # Extract text after the colon following THIS keyword
                            value_text = text_after_keyword[colon_idx + 1:].strip()
                            
                            # Stop at next field or separator
                            # Look for next keyword boundary (another field name or end of text)
                            stop_idx = len(value_text)
                            for stop_pattern in [r'\b(?:Received|Date|BCN|DCN|ICN|PCN|TRN|RCN)\s*:', r'\s{3,}']:  # Stop at next field or 3+ spaces
                                stop_match = re.search(stop_pattern, value_text, re.IGNORECASE)
                                if stop_match:
                                    stop_idx = min(stop_idx, stop_match.start())
                            
                            value_text = value_text[:stop_idx].strip()
                            
                            claim_match = self._extract_claim_from_text(value_text)
                            if claim_match:
                                candidates.append({
                                    'value': claim_match,
                                    'confidence': 0.95,
                                    'source': 'same_block_colon_specific'
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
    
    def _extract_claim_from_text(self, text: str, require_keyword: bool = False) -> Optional[str]:
        """
        Extract claim number from text.
        
        Args:
            text: Text to extract from
            require_keyword: If True, only extract if claim keyword is present (stricter validation)
        
        Returns:
            Extracted claim number or None
        """
        if not text:
            return None
        
        text = text.strip()
        
        # If require_keyword is True, check that claim keyword exists in the text
        if require_keyword:
            has_keyword = any(regex.search(text) for regex in self.claim_regex)
            if not has_keyword:
                return None  # No claim keyword found, don't extract
        
        # Try specific claim patterns first (formats with letters/dashes common in claims)
        for regex in self.claim_number_formats:
            match = regex.search(text)
            if match:
                candidate = match.group(0).strip()
                # Make sure it's not a date
                if not any(regex.search(candidate) for regex in self.date_formats):
                    return candidate
        
        # ❌ REMOVED: Fallback patterns that extract any long number sequence
        # This was causing false positives like "1500550266046699" 
        # Now we ONLY extract if it matches specific claim formats or has keyword context
        
        return None
    
    def _validate_with_llm(self, extracted_data: Dict[str, Optional[str]], blocks: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Validate extracted patient details using Azure OpenAI LLM.
        Uses first 2000 words of document to verify accuracy.
        
        Args:
            extracted_data: Initial extraction results
            blocks: All text blocks from document
            
        Returns:
            Validated/corrected extraction results
        """
        try:
            
            # Get first 2000 words from document
            all_text = " ".join([block['text'] for block in blocks[:50]])  # First 50 blocks ~2000 words
            words = all_text.split()[:2000]
            context_text = " ".join(words)
            
            # Skip validation if Azure OpenAI is not configured
            if not CONFIG.get("azure_openai_api_key") or not CONFIG.get("azure_openai_endpoint"):
                logger.warning("⚠️ Azure OpenAI not configured, skipping LLM validation")
                return extracted_data
            
            # Initialize Azure OpenAI client
            llm = AzureChatOpenAI(
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                deployment_name=CONFIG.get("azure_openai_deployment"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.0,
                timeout=60
            )
            
            # Create validation prompt
            system_message = """You are a medical document analysis expert. Verify the accuracy of extracted patient details.

INSTRUCTIONS:
1. Verify each extracted detail against the document text
2. For Claim Number: ONLY accept if it appears with keywords like "Claim:", "Claim #", "Claim No", "Cl:", "File No", etc.
3. DO NOT accept random numbers like "1500550266046699" or "BCN: 1014518231" as claim numbers
4. Patient Name should be an actual person's name, not a sentence fragment
5. Dates should be in valid formats

Return ONLY a JSON object with corrected values. Use "Not specified" if a field is truly not found or incorrect.

Required JSON format:
{
  "patient_name": "corrected or original value",
  "dob": "YYYY-MM-DD or Not specified",
  "doi": "YYYY-MM-DD or Not specified",
  "claim_number": "corrected or original value or Not specified",
  "validation_notes": "brief explanation of any corrections made"
}"""

            human_message = f"""EXTRACTED DETAILS:
- Patient Name: {extracted_data.get('patient_name', 'Not found')}
- Date of Birth (DOB): {extracted_data.get('dob', 'Not found')}
- Date of Injury (DOI): {extracted_data.get('doi', 'Not found')}
- Claim Number: {extracted_data.get('claim_number', 'Not found')}

DOCUMENT TEXT (first 2000 words):
{context_text}

Validate these details and return the JSON."""

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ]
            
            response = llm.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response (might be wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                json_str = json_match.group(0) if json_match else None
            
            if json_str:
                validated_data = json.loads(json_str)
                
                logger.info("✅ LLM Validation completed:")
                logger.info(f"  Original Claim: {extracted_data.get('claim_number')} → Validated: {validated_data.get('claim_number')}")
                logger.info(f"  Original Name: {extracted_data.get('patient_name')} → Validated: {validated_data.get('patient_name')}")
                logger.info(f"  Notes: {validated_data.get('validation_notes', 'No corrections')}")
                
                # Update with validated data (but keep original if LLM returns "Not specified" and we had something)
                result = {}
                for key in ['patient_name', 'dob', 'doi', 'claim_number']:
                    validated_value = validated_data.get(key)
                    original_value = extracted_data.get(key)
                    
                    if validated_value and validated_value.lower() not in ['not specified', 'not found', 'none']:
                        result[key] = validated_value
                    elif original_value:
                        # Keep original if LLM didn't find better value
                        result[key] = original_value
                    else:
                        result[key] = None
                
                return result
            else:
                logger.warning("⚠️ Could not parse LLM response, using original extraction")
                return extracted_data
                
        except Exception as e:
            logger.error(f"❌ LLM validation failed: {str(e)}")
            logger.debug("Using original extraction results")
            return extracted_data


# Global instance
_extractor_instance = None

def get_patient_extractor() -> PatientDetailsExtractor:
    """Get singleton patient extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = PatientDetailsExtractor()
    return _extractor_instance
