# document_helpers.py

"""
Helper utilities for document processing.
Includes MIME type detection, patient details formatting, and result merging.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from models.schemas import ExtractionResult
from utils.multi_report_detector import detect_multiple_reports

logger = logging.getLogger("document_ai")


# MIME type mapping for supported file formats
MIME_TYPE_MAPPING = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def get_mime_type(filepath: str) -> str:
    """Get MIME type based on file extension"""
    file_ext = Path(filepath).suffix.lower()
    return MIME_TYPE_MAPPING.get(file_ext, "application/octet-stream")


def format_patient_details_text(patient_details: Dict[str, Any]) -> str:
    """
    Format patient details dictionary into a text block.
    
    Args:
        patient_details: Dictionary with patient information
    
    Returns:
        Formatted string with patient details
    """
    if not patient_details:
        return ""
    
    text = "--- PATIENT DETAILS ---\n"
    text += f"Patient Name: {patient_details.get('patient_name', 'N/A')}\n"
    text += f"Date of Birth: {patient_details.get('dob', 'N/A')}\n"
    text += f"Date of Injury: {patient_details.get('doi', 'N/A')}\n"
    text += f"Claim Number: {patient_details.get('claim_number', 'N/A')}\n"
    if patient_details.get('date_of_report'):
        text += f"Date of Report: {patient_details.get('date_of_report', 'N/A')}\n"
    if patient_details.get('author'):
        text += f"Author/Signed By: {patient_details.get('author', 'N/A')}\n"
    text += "--- END PATIENT DETAILS ---\n\n"
    
    return text


def log_patient_details(patient_details: Dict[str, Any], header: str = "PATIENT DETAILS"):
    """Log patient details with formatting"""
    if not patient_details:
        return
    
    logger.info("=" * 80)
    logger.info(f"👤 {header}:")
    logger.info("=" * 80)
    logger.info(f"  Patient Name: {patient_details.get('patient_name', 'Not found')}")
    logger.info(f"  DOB: {patient_details.get('dob', 'Not found')}")
    logger.info(f"  DOI: {patient_details.get('doi', 'Not found')}")
    logger.info(f"  Claim Number: {patient_details.get('claim_number', 'Not found')}")
    if patient_details.get('date_of_report'):
        logger.info(f"  Date of Report: {patient_details.get('date_of_report', 'Not found')}")
    if patient_details.get('author'):
        logger.info(f"  Author/Signed By: {patient_details.get('author', 'Not found')}")
    logger.info("=" * 80)


def merge_extraction_results(
    results: List[ExtractionResult],
    original_file: str = ""
) -> ExtractionResult:
    """
    Merge results from multiple document chunks.
    
    Args:
        results: List of ExtractionResult objects from each chunk
        original_file: Path to the original file (for logging)
    
    Returns:
        Merged ExtractionResult
    """
    if not results:
        return ExtractionResult(success=False, error="No successful chunks")
    
    merged_text = ""
    merged_raw_text = ""
    
    logger.info(f"🔗 Starting merge of {len(results)} summarizer chunks...")
    
    for i, result in enumerate(results):
        chunk_num = i + 1
        logger.info(f"📦 Processing chunk {chunk_num}:")
        logger.info(f"   - Has text: {bool(result.text)}")
        logger.info(f"   - Pages in chunk: {result.pages}")
        
        if result.text:
            if i > 0:
                merged_text += f"\n\n{'='*80}\nCHUNK {i + 1}\n{'='*80}\n\n"
            merged_text += result.text
        
        if hasattr(result, 'raw_text') and result.raw_text:
            merged_raw_text += result.raw_text + "\n\n"
    
    total_pages = sum(r.pages for r in results)
    
    # Extract patient_details from the first chunk (only first chunk has patient details)
    patient_details = {}
    if results and hasattr(results[0], 'metadata') and results[0].metadata:
        patient_details = results[0].metadata.get('patient_details', {})
        if patient_details:
            log_patient_details(patient_details, "MERGED PATIENT DETAILS (from first chunk)")
            
            # Prepend patient details to merged_raw_text
            patient_details_text = format_patient_details_text(patient_details)
            merged_raw_text = patient_details_text + merged_raw_text
            
            logger.info("🔍 DEBUG: Patient details prepended to merged_raw_text")
            logger.info(f"🔍 First 500 chars of merged_raw_text:\n{merged_raw_text[:500]}")
    
    logger.info(f"🔗 Merge complete:")
    logger.info(f"   - Total pages: {total_pages}")
    logger.info(f"   - Total text-merged---------------: {(merged_raw_text)}")
    
    # Log the complete merged summarizer output
    logger.info("=" * 80)
    logger.info("🤖 COMPLETE MERGED SUMMARIZER OUTPUT (ALL CHUNKS COMBINED):")
    logger.info("=" * 80)
    logger.info(f"Total chunks processed: {len(results)}")
    logger.info(f"Total pages: {total_pages}")
    logger.info(f"Total characters: {len(merged_text)}")
    logger.info("=" * 80)
    logger.info(merged_text)
    logger.info("=" * 80)
    
    # Run multi-report detection on merged text
    multi_report_result = detect_multiple_reports(merged_text)
    
    merged_result = ExtractionResult(
        text=merged_raw_text,
        raw_text=merged_raw_text,
        llm_text=merged_raw_text,
        page_zones={},
        pages=total_pages,
        entities=[],
        tables=[],
        formFields=[],
        symbols=[],
        confidence=1.0,
        success=True,
        metadata={"patient_details": patient_details} if patient_details else {},
        is_multiple_reports=multi_report_result.get("is_multiple", False),
        multi_report_info=multi_report_result
    )
    
    return merged_result


def extract_summary_from_result(result: Any) -> str:
    """
    Extract summary text from Document AI result.
    Tries multiple locations where summary might be stored.
    
    Args:
        result: Document AI processing result
    
    Returns:
        Extracted summary text
    """
    summary_text = ""
    
    # Check if document has chunked_document field
    if hasattr(result, 'chunked_document') and result.chunked_document:
        logger.info(f"📦 Found chunked document with {len(result.chunked_document.chunks)} chunks")
        summary_parts = []
        for chunk in result.chunked_document.chunks:
            if hasattr(chunk, 'content') and chunk.content:
                summary_parts.append(chunk.content)
        summary_text = "\n\n".join(summary_parts)
    
    # Try to get entity-based summary (for newer API versions)
    if not summary_text and hasattr(result, 'entities'):
        for entity in result.entities:
            if entity.type_ == 'summary' or 'summary' in entity.type_.lower():
                summary_text = entity.mention_text
                break
    
    # Fallback: Check for summary in document text
    if not summary_text:
        full_text = result.text or ""
        page_count = len(result.pages) if result.pages else 0
        
        # If text is less than ~500 chars per page, it's likely a summary
        avg_chars_per_page = len(full_text) / page_count if page_count > 0 else 0
        
        if avg_chars_per_page < 500 or page_count == 0:
            summary_text = full_text
            logger.info(f"✅ Detected summary in text field (avg {avg_chars_per_page:.0f} chars/page)")
        else:
            logger.warning(f"⚠️ Got full OCR text instead of summary (avg {avg_chars_per_page:.0f} chars/page)")
            logger.warning("⚠️ Processor may not be configured as a summarizer")
            summary_text = full_text
    
    return summary_text
