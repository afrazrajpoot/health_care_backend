"""
Document Save Service - Handles saving documents to database and Redis cache
Extracted from webhook_service.py for better modularity
"""
from datetime import datetime
from typing import Dict, Any
from models.schemas import ExtractionResult
from utils.logger import logger
import json


async def save_document(
    db_service,
    processed_data: dict,
    lookup_result: dict,
    generate_filename_func,
    save_to_redis_cache_func
) -> dict:
    """
    Save document to database and Redis cache.
    
    Args:
        db_service: Database service instance
        processed_data: Processed document data from process_document_data
        lookup_result: Patient lookup result
        generate_filename_func: Function to generate structured filename
        save_to_redis_cache_func: Function to save to Redis cache
        
    Returns:
        dict with status, document_id, parse_count_decremented, filename, cache_success
    """
    # ✅ Check if both DOB and DOI are not specified
    document_analysis = processed_data["document_analysis"]
    dob_not_specified = (
        not hasattr(document_analysis, 'dob') or 
        not document_analysis.dob or 
        str(document_analysis.dob).lower() in ["not specified", "none", ""]
    )
    
    doi_not_specified = (
        not lookup_result.get("doi_to_save") or 
        str(lookup_result["doi_to_save"]).lower() in ["not specified", "none", ""]
    )
    
    # If both DOB and DOI are not specified, save as fail document
    if dob_not_specified and doi_not_specified:
        # ✅ Get the actual parsed text from the result data
        parsed_text = processed_data["result_data"].get("text", "")
        
        # ✅ Also get the brief summary and other analysis data
        brief_summary = processed_data.get("brief_summary", "")
        
        # ✅ Combine text and summary for comprehensive document text
        full_document_text = f"ORIGINAL TEXT:\n{parsed_text}\n\nSUMMARY:\n{brief_summary}"
        
        fail_doc_id = await db_service.save_fail_doc(
            reason="Both DOB and DOI are not specified",
            db=document_analysis.dob if hasattr(document_analysis, 'dob') else None,
            claim_number=document_analysis.claim_number if hasattr(document_analysis, 'claim_number') and document_analysis.claim_number else "Not specified",
            patient_name=lookup_result.get("patient_name_to_use"),
            physician_id=processed_data.get("physician_id"),
            gcs_file_link=processed_data.get("gcs_url"),
            file_name=processed_data.get("filename"),
            file_hash=processed_data.get("file_hash"),
            blob_path=processed_data.get("blob_path"),
            mode=processed_data.get("mode"),
            # ✅ SAVE THE PARSED TEXT AND SUMMARY
            document_text=full_document_text,
            doi=lookup_result.get("doi_to_save"),
            ai_summarizer_text=brief_summary
        )

        
        # ✅ Decrement parse count even for failed documents since they consumed resources
        parse_decremented = await db_service.decrement_parse_count(processed_data["physician_id"])
        
        return {
            "status": "failed",
            "document_id": fail_doc_id,
            "parse_count_decremented": parse_decremented,
            "filename": processed_data["filename"],
            "cache_success": False,
            "failure_reason": "Both DOB and DOI are not specified"
        }
    
    # Continue with normal document saving process...
    # Create ExtractionResult
    extraction_result = ExtractionResult(
        text=processed_data["result_data"].get("text", ""),
        pages=processed_data["result_data"].get("pages", 0),
        entities=processed_data["result_data"].get("entities", []),
        tables=processed_data["result_data"].get("tables", []),
        formFields=processed_data["result_data"].get("formFields", []),
        confidence=processed_data["result_data"].get("confidence", 0.0),
        success=processed_data["result_data"].get("success", False),
        gcs_file_link=processed_data["result_data"].get("gcs_file_link", processed_data["gcs_url"]),
        fileInfo=processed_data["result_data"].get("fileInfo", {}),
        summary=processed_data["brief_summary"],
        document_id=processed_data["result_data"].get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    
    # Prepare the additional required parameters with default values
    document_analysis = processed_data["document_analysis"]
    
    # Get RD (Report Date) - use from analysis
    rd = None
    if hasattr(document_analysis, 'rd') and document_analysis.rd and str(document_analysis.rd).lower() != "not specified":
        try:
            date_str = str(document_analysis.rd).strip()
            logger.info(f"📅 Parsing report date: {date_str}")
            
            # Try multiple date formats
            date_formats = [
                "%Y-%m-%d",      # 2025-11-25
                "%m-%d-%Y",      # 11-25-2025
                "%m/%d/%Y",      # 11/25/2025
                "%m/%d/%y",      # 11/25/25
                "%d-%m-%Y",      # 25-11-2025
                "%d/%m/%Y",      # 25/11/2025
                "%Y/%m/%d",      # 2025/11/25
            ]
            
            parsed = False
            for fmt in date_formats:
                try:
                    rd = datetime.strptime(date_str, fmt)
                    # Set time to noon to avoid timezone shifting issues
                    rd = rd.replace(hour=12, minute=0, second=0, microsecond=0)
                    logger.info(f"✅ Parsed report date with format {fmt}: {rd}")
                    parsed = True
                    break
                except ValueError:
                    continue
            
            if not parsed:
                logger.warning(f"⚠️ Could not parse report date: {date_str}, keeping as None")
                rd = None
                
        except Exception as e:
            logger.warning(f"⚠️ Error parsing report date: {e}")
            rd = None  # Don't fallback to current date - leave as None
    
    # Create summary snapshots
    summary_snapshots = []
    if hasattr(document_analysis, 'body_parts_analysis') and document_analysis.body_parts_analysis:
        for body_part_analysis in document_analysis.body_parts_analysis:
            snapshot = {
                "mode": processed_data["mode"],
                "body_part": body_part_analysis.body_part if processed_data["mode"] == "wc" else None,
                "condition": None if processed_data["mode"] == "wc" else body_part_analysis.body_part,
                "dx": body_part_analysis.diagnosis,
                "key_concern": body_part_analysis.key_concern,
                "next_step": body_part_analysis.extracted_recommendation or None,
                "ur_decision": document_analysis.ur_decision or None,
                "recommended": body_part_analysis.extracted_recommendation or None,
                "ai_outcome": document_analysis.ai_outcome or None,
                "consulting_doctor": document_analysis.consulting_doctor or None,
                "key_findings": body_part_analysis.clinical_summary or None,
                "treatment_approach": body_part_analysis.treatment_plan or None,
                "clinical_summary": body_part_analysis.clinical_summary or None,
                "referral_doctor": document_analysis.referral_doctor or None,
                "adls_affected": body_part_analysis.adls_affected or None,
            }
            summary_snapshots.append(snapshot)
    else:
        # Create a single snapshot if no body parts analysis
        snapshot = {
            "mode": processed_data["mode"],
            "body_part": document_analysis.body_part if processed_data["mode"] == "wc" else None,
            "condition": None if processed_data["mode"] == "wc" else document_analysis.body_part,
            "dx": document_analysis.diagnosis,
            "key_concern": document_analysis.key_concern,
            "next_step": document_analysis.extracted_recommendation or None,
            "ur_decision": document_analysis.ur_decision or None,
            "recommended": document_analysis.extracted_recommendation or None,
            "ai_outcome": document_analysis.ai_outcome or None,
            "consulting_doctor": document_analysis.consulting_doctor or None,
            "key_findings": document_analysis.diagnosis or None,
            "treatment_approach": document_analysis.extracted_recommendation or None,
            "clinical_summary": f"{document_analysis.diagnosis} - {document_analysis.key_concern}" or None,
            "referral_doctor": document_analysis.referral_doctor or None,
            "adls_affected": document_analysis.adls_affected or None,
        }
        summary_snapshots.append(snapshot)
    
    # ✅ FIXED: Get the ACTUAL long and short summaries from ReportAnalyzer
    report_analyzer_result = processed_data.get("report_analyzer_result", {})
    
    # Use the actual summaries from ReportAnalyzer if available
    if report_analyzer_result and isinstance(report_analyzer_result, dict):
        long_summary = report_analyzer_result.get("long_summary", "")
        short_summary = report_analyzer_result.get("short_summary", "")
    else:
        # Fallback to the existing data
        long_summary = processed_data.get("text_for_analysis", "")
        short_summary = processed_data["brief_summary"]
    
    whats_new = {
        "long_summary": long_summary,
        "short_summary": short_summary
    }
    
    # Create ADL data
    adl_data = {
        "mode": processed_data["mode"],
        "adls_affected": document_analysis.adls_affected if hasattr(document_analysis, 'adls_affected') else None,
        "work_restrictions": document_analysis.work_restrictions if hasattr(document_analysis, 'work_restrictions') else None
    }
    
    # Create document summary
    document_summary = {
        "type": document_analysis.document_type if hasattr(document_analysis, 'document_type') else "Unknown",
        "created_at": datetime.now(),
        "summary": " | ".join(document_analysis.summary_points) if hasattr(document_analysis, 'summary_points') and document_analysis.summary_points else processed_data["brief_summary"]
    }
    
    # Generate structured filename: patientName_typeOfReport_dateOfReport
    original_filename = processed_data["filename"]
    document_type = document_analysis.document_type if hasattr(document_analysis, 'document_type') else "Document"
    structured_filename = generate_filename_func(
        patient_name=lookup_result["patient_name_to_use"],
        document_type=document_type,
        report_date=rd,
        original_filename=original_filename
    )
    
    # Save document to database with all required parameters
    document_id = await db_service.save_document_analysis(
        extraction_result=extraction_result,
        file_name=structured_filename,  # New structured filename
        file_size=processed_data["file_size"],
        mime_type=processed_data["mime_type"],
        processing_time_ms=processed_data["processing_time_ms"],
        blob_path=processed_data["blob_path"],
        file_hash=processed_data["file_hash"],
        gcs_file_link=processed_data["gcs_url"],
        patient_name=lookup_result["patient_name_to_use"],
        claim_number=document_analysis.claim_number if hasattr(document_analysis, 'claim_number') and document_analysis.claim_number else "Not specified",  # Required field with default
        dob=document_analysis.dob if hasattr(document_analysis, 'dob') else None,
        doi=lookup_result["doi_to_save"],
        status=lookup_result["document_status"],
        brief_summary=processed_data["brief_summary"],
        physician_id=processed_data["physician_id"],
        mode=processed_data["mode"],
        # Add the missing required parameters
        rd=rd,
        summary_snapshots=summary_snapshots,
        whats_new=whats_new,
        adl_data=adl_data,
        document_summary=document_summary,
        original_name=original_filename,  # Store original filename
        ai_summarizer_text=processed_data.get("raw_text")
    )
    
    # SAVE TO REDIS CACHE (only for successful documents, not for fail documents)
    cache_success = False
    if document_id:
        # Prepare data for caching
        cache_data = {
            "patient_name": lookup_result["patient_name_to_use"],
            "claim_number": document_analysis.claim_number if hasattr(document_analysis, 'claim_number') and document_analysis.claim_number else "Not specified",
            "dob": document_analysis.dob if hasattr(document_analysis, 'dob') else None,
            "doi": lookup_result["doi_to_save"],
            "physician_id": processed_data["physician_id"],
            "status": lookup_result["document_status"],
            "mode": processed_data["mode"],
            "brief_summary": processed_data["brief_summary"],
            "filename": processed_data["filename"],
            "document_analysis": document_analysis.dict() if hasattr(document_analysis, 'dict') else str(document_analysis),
            "summary_snapshots": summary_snapshots,
            "created_at": datetime.now().isoformat()
        }
        
        cache_success = await save_to_redis_cache_func(document_id, cache_data)
    
    # Decrement parse count
    parse_decremented = await db_service.decrement_parse_count(processed_data["physician_id"])
    
    return {
        "status": lookup_result["document_status"],
        "document_id": document_id,
        "parse_count_decremented": parse_decremented,
        "filename": processed_data["filename"],
        "cache_success": cache_success
    }
