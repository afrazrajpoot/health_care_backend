"""
Fail Document Update Service - Handles updating and reprocessing failed documents
Extracted from webhook_service.py for better modularity
"""
from datetime import datetime
from typing import Any, List, Optional, Dict
from fastapi import HTTPException
from pydantic import BaseModel, Field
from models.data_models import DocumentAnalysis
from services.report_analyzer import ReportAnalyzer
from utils.logger import logger
from utils.document_detector import detect_document_type
import asyncio
import json
import re

# --- Report Field Matrix: Defines allowed fields per document type ---
REPORT_FIELD_MATRIX: Dict[str, Dict] = {
    # Med-Legal Reports (QME family)
    "QME": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations",
            "rationale",
            "mmi_status",
            "work_status"
        }
    },
    "AME": {"inherit": "QME"},
    "PQME": {"inherit": "QME"},
    "IME": {"inherit": "QME"},

    # Consult / Clinical Reports
    "CONSULT": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations"
        }
    },
    "PAIN MANAGEMENT": {"inherit": "CONSULT"},
    "PROGRESS NOTE": {"inherit": "CONSULT"},
    "OFFICE VISIT": {"inherit": "CONSULT"},
    "CLINIC NOTE": {"inherit": "CONSULT"},

    # Imaging Reports
    "MRI": {
        "allowed": {"findings", "impressions"}
    },
    "CT": {"inherit": "MRI"},
    "X-RAY": {"inherit": "MRI"},
    "XRAY": {"inherit": "MRI"},
    "ULTRASOUND": {"inherit": "MRI"},
    "EMG": {"inherit": "MRI"},
    "PET SCAN": {"inherit": "MRI"},
    "BONE SCAN": {"inherit": "MRI"},
    "DEXA SCAN": {"inherit": "MRI"},

    # Utilization Review
    "UR": {
        "allowed": {"recommendations", "rationale"}
    },
    "IMR": {"inherit": "UR"},
    "PEER REVIEW": {"inherit": "UR"},

    # Therapy Reports
    "PHYSICAL THERAPY": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "recommendations"
        }
    },
    "THERAPY NOTE": {"inherit": "PHYSICAL THERAPY"},
    "OCCUPATIONAL THERAPY": {"inherit": "PHYSICAL THERAPY"},
    "CHIROPRACTIC": {"inherit": "PHYSICAL THERAPY"},

    # Surgical / Operative Reports
    "SURGERY REPORT": {
        "allowed": {"findings"}
    },
    "OPERATIVE NOTE": {"inherit": "SURGERY REPORT"},
    "POST-OP": {"inherit": "SURGERY REPORT"},

    # PR-2 Reports
    "PR-2": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations",
            "work_status"
        }
    },
    "PR2": {"inherit": "PR-2"},

    # Labs & Diagnostics
    "LABS": {
        "allowed": {"findings"}
    },
    "PATHOLOGY": {"inherit": "LABS"},

    # Legal / Administrative Reports
    "ATTORNEY QUESTIONS": {
        "allowed": {"questions", "mmi_status", "work_status"}
    },
    "ADJUSTER QUESTIONS": {"inherit": "ATTORNEY QUESTIONS"},
    "NURSE CASE MANAGER": {"inherit": "ATTORNEY QUESTIONS"},

    # Default fallback
    "DEFAULT": {
        "allowed": {"findings", "recommendations"}
    }
}

# --- Pydantic Models for Structured Summary ---

class DocumentMetadata(BaseModel):
    document_type: str = Field(description="Type of the document extracted from context", default="Unknown")

class MedicationItem(BaseModel):
    name: str = Field(description="Exact medication name")
    dosage: str = Field(description="Exact dosage")
    frequency: Optional[str] = Field(description="Frequency if specified")
    status: Optional[str] = Field(description="prescribed/continued/discontinued")
    unclear_requires_verification: Optional[bool] = Field(description="Set to true only if details are unclear")

class TestResultItem(BaseModel):
    test_name: str = Field(description="Exact test name")
    result: str = Field(description="Exact result value")
    date: Optional[str] = Field(description="Date of test if mentioned")
    interpretation: Optional[str] = Field(description="Interpretation only if explicitly stated")

class FollowUpItem(BaseModel):
    plan: Optional[str] = Field(description="Exact follow-up plan")
    timeframe: Optional[str] = Field(description="Timeframe if mentioned")

class AuthorizationStatusItem(BaseModel):
    status: Optional[str] = Field(description="approved/denied/modified/pending")
    details: Optional[str] = Field(description="Specific details")
    conditions: List[str] = Field(default_factory=list, description="Any conditions mentioned")

class VitalSignsItem(BaseModel):
    parameter: str = Field(description="Name of the vital sign")
    value: str = Field(description="Value of the vital sign with units")

class StructuredSummaryResponse(BaseModel):
    # Standard Output Fields (Global Schema) - shown first when relevant
    document_metadata: DocumentMetadata
    primary_diagnosis: Optional[List[str]] = Field(default=None, description="Exact diagnoses as stated")
    clinical_findings: Optional[List[str]] = Field(default=None, description="Specific findings mentioned")
    medications: Optional[List[MedicationItem]] = Field(default=None)
    procedures: Optional[List[str]] = Field(default=None, description="Exact procedures mentioned")
    test_results: Optional[List[TestResultItem]] = Field(default=None)
    recommendations: Optional[List[str]] = Field(default=None, description="Exact recommendations")
    follow_up: Optional[FollowUpItem] = Field(default=None, description="Follow up plan details")
    authorization_status: Optional[AuthorizationStatusItem] = Field(default=None, description="Authorization status details")
    allergies: Optional[List[str]] = Field(default=None, description="Exact allergies listed")
    vital_signs: Optional[Dict[str, str]] = Field(default=None, description="Key value pairs of vital signs (e.g., 'BP': '120/80')")
    unclear_items: Optional[List[str]] = Field(default=None, description="Any information that requires verification")
    additional_notes: Optional[List[str]] = Field(default=None, description="Any other relevant information not fitting above categories")
    
    # Report Type-Specific Fields (from REPORT_FIELD_MATRIX)
    mechanism_of_injury: Optional[str] = Field(default=None, description="Exact mechanism of injury as stated")
    # findings: Optional[List[str]] = Field(default=None, description="Specific findings from the report")
    physical_exam: Optional[Dict[str, str]] = Field(default=None, description="Physical examination findings")
    rationale: Optional[str] = Field(default=None, description="Rationale for decision or recommendation")
    mmi_status: Optional[str] = Field(default=None, description="Maximum Medical Improvement status")
    work_status: Optional[str] = Field(default=None, description="Work restrictions or status")
    impressions: Optional[List[str]] = Field(default=None, description="Impressions from imaging or diagnostic reports")
    questions: Optional[List[Dict[str, str]]] = Field(default=None, description="Questions and answers from legal/administrative documents")

async def generate_concise_brief_summary(raw_summary_text: str, document_type: str = "Medical Document", llm_executor=None) -> dict:
    """
    Uses LLM to transform the raw summarizer output into a structured JSON format.
    Extracts ONLY explicitly stated information without fabrication or interpretation.
    
    Args:
        raw_summary_text: The raw summary text to process
        document_type: Type of document for context
        llm_executor: ThreadPoolExecutor for running LLM operations
        
    Returns:
        Dictionary with structured summary data
    """
    if not raw_summary_text or len(raw_summary_text) < 10:
        return {"status": "unavailable", "message": "Summary not available"}

    try:
        logger.info("🤖 Generating structured JSON summary using LLM and PydanticOutputParser...")
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import AzureChatOpenAI
        from langchain_core.output_parsers import PydanticOutputParser
        from config.settings import CONFIG

        parser = PydanticOutputParser(pydantic_object=StructuredSummaryResponse)

        llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,  # Zero temperature for maximum factual accuracy
            timeout=30,
            model_kwargs={"response_format": {"type": "json_object"}} 
        )

        system_template = """You are a medical documentation assistant that extracts information into structured JSON format.

CRITICAL EXTRACTION RULES:

1. **No Fabrication**: Extract ONLY information explicitly stated in the raw summary - ZERO fabrication or inference
2. **Exact Wording**: Use EXACT wording from source - do not paraphrase medical terms or values
3. **No Assumptions**: Do not generate or infer any information not present in the document
4. **Field Selection**: Only return fields that:
   - Are allowed for the detected report type (per REPORT_FIELD_MATRIX), AND
   - Actually have data present in the source document
5. **Omit Empty Fields**: DO NOT include any field if that information is not present in the source
   - Use `null` for optional fields with no data, or omit them entirely
6. **No Duplication**: Do not return the same information in multiple fields
   - For example, do not return recommendations in both matrix fields and global fields
7. **Standard Fields First**: Use standard global fields when they apply:
   - primary_diagnosis, clinical_findings, medications, procedures, test_results
   - recommendations, follow_up, authorization_status, vital_signs, allergies
   - unclear_items, additional_notes
8. **Matrix Fields**: Use report-type-specific fields only when allowed for this document type:
   - mechanism_of_injury, findings, physical_exam, rationale, mmi_status
   - work_status, impressions, questions
9. **Unknown Information**: If new information doesn't match any defined field, place it in additional_notes
10. **Data Integrity**: This is a pure extraction system, not generative:
    - Use only existing content from the report
    - No hallucinations, no AI-created data
    - Structure and filter strictly using the schema

{format_instructions}
"""

        user_template = f"""Document Type: {document_type}

Raw Summary Input:
{raw_summary_text}

**IMPORTANT**: Based on the document type "{document_type}", only extract fields that are:
1. Allowed for this report type according to REPORT_FIELD_MATRIX
2. Actually present in the raw summary above

Do not include fields with no data. Do not fabricate or infer information.

Extract into structured JSON following all rules above."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # Inject format instructions
        prompt_with_instructions = prompt.partial(format_instructions=parser.get_format_instructions())
        
        # Chain now includes parser
        chain = prompt_with_instructions | llm | parser
        
        # Run in executor to avoid blocking
        # Note: parser is synchronous, but the whole chain execution via invoke can be wrapped
        if llm_executor:
            response = await asyncio.get_event_loop().run_in_executor(
                llm_executor, 
                lambda: chain.invoke({})
            )
        else:
            response = await asyncio.to_thread(lambda: chain.invoke({}))
        
        # Response is already a Pydantic object
        structured_summary = response.model_dump()
        
        # Validation: Check if summary extracted meaningful data
        meaningful_keys = [k for k in structured_summary.keys() 
                          if k not in ['document_metadata'] 
                          and structured_summary[k]]
        
        if len(meaningful_keys) == 0 and len(raw_summary_text) > 200:
            logger.warning("⚠️ No meaningful data extracted from substantial input - using fallback")
            return {
                "status": "extraction_incomplete",
                "raw_content": raw_summary_text,
                "message": "Automated extraction incomplete, raw content preserved"
            }
        
        # Validation: Check for fabricated generic content
        def contains_fabricated_content(data, source_text):
            """Recursively check if any values contain fabricated generic phrases"""
            generic_phrases = [
                "patient was seen",
                "routine care provided", 
                "standard treatment given",
                "typical findings noted",
                "normal results",
                "as expected"
            ]
            
            source_lower = source_text.lower()
            
            def check_value(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    for phrase in generic_phrases:
                        if phrase in val_lower and phrase not in source_lower:
                            return True
                elif isinstance(val, (list, tuple)):
                    return any(check_value(v) for v in val)
                elif isinstance(val, dict):
                    return any(check_value(v) for v in val.values())
                return False
            
            return check_value(data)
        
        if contains_fabricated_content(structured_summary, raw_summary_text):
            logger.warning("⚠️ Detected potentially fabricated content - using raw summary")
            return {
                "status": "fabrication_detected",
                "raw_content": raw_summary_text,
                "message": "Potential fabricated content detected, raw content provided for verification"
            }
        
        # Add extraction metadata
        structured_summary["_extraction_metadata"] = {
            "source_length": len(raw_summary_text),
            "extracted_keys": list(structured_summary.keys()),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Generated structured summary with {len(meaningful_keys)} data sections")
        return structured_summary

    except Exception as e:
        logger.error(f"❌ Failed to generate structured summary: {e}")
        # Fallback: Return raw text in error structure
        return {
            "status": "error",
            "error_message": str(e),
            "raw_content": raw_summary_text,
            "message": "Error during extraction, raw content preserved"
        }


async def update_fail_document(
    fail_doc: Any,
    updated_fields: dict,
    user_id: str,
    db_service: Any,
    patient_lookup,
    save_document_func,
    create_tasks_func,
    llm_executor=None
) -> dict:
    """
    Updates and processes a failed document using the complete webhook-like logic.
    
    Args:
        fail_doc: The failed document object from database
        updated_fields: Fields to update (patient_name, dob, doi, claim_number, author, document_text)
        user_id: User ID making the update
        db_service: Database service instance
        patient_lookup: Patient lookup service instance
        save_document_func: Function to save document
        create_tasks_func: Function to create tasks
        llm_executor: ThreadPoolExecutor for LLM operations
        
    Returns:
        dict with save result
    """
    # Use updated values if provided, otherwise fallback to fail_doc values
    document_text = updated_fields.get("document_text") or fail_doc.documentText
    dob_str = updated_fields.get("dob") or fail_doc.dob
    doi = updated_fields.get("doi") or fail_doc.doi
    claim_number = updated_fields.get("claim_number") or fail_doc.claimNumber
    patient_name = updated_fields.get("patient_name") or fail_doc.patientName
    author = updated_fields.get("author") or fail_doc.author
    physician_id = fail_doc.physicianId
    filename = fail_doc.fileName
    gcs_url = fail_doc.gcsFileLink
    blob_path = fail_doc.blobPath
    file_hash = fail_doc.fileHash
    mode = "wc"  # Default mode
    
    # ✅ Use aiSummarizerText if available (actual Document AI Summarizer output)
    # This is preferred over documentText for summary generation
    ai_summarizer_text = getattr(fail_doc, 'aiSummarizerText', None)
    if ai_summarizer_text and len(ai_summarizer_text) > 50:
        logger.info(f"📋 Using aiSummarizerText for processing ({len(ai_summarizer_text)} chars)")
        # Use aiSummarizerText as the primary text for document type detection and summary generation
        summarizer_output = ai_summarizer_text
    else:
        logger.info(f"📋 aiSummarizerText not available, using documentText")
        summarizer_output = document_text

    # Construct webhook-like data
    result_data = {
        "text": document_text,
        "pages": 0,
        "entities": [],
        "tables": [],
        "formFields": [],
        "confidence": 0.0,
        "success": False,
        "gcs_file_link": gcs_url,
        "fileInfo": {},
        "comprehensive_analysis": None,
        "document_id": f"update_fail_{fail_doc.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    try:
        # Step 1: Detect document type and check if valid for summary card
        logger.info(f"🔍 Detecting document type for failed document: {fail_doc.id}")
        
        doc_type_result = await asyncio.to_thread(
            lambda: detect_document_type(summarizer_output=summarizer_output, raw_text=document_text)
        )
        
        detected_doc_type = doc_type_result.get('doc_type', 'Unknown')
        is_valid_for_summary_card = doc_type_result.get('is_valid_for_summary_card', True)  # Default True for safety
        summary_card_reasoning = doc_type_result.get('summary_card_reasoning', '')
        
        logger.info(f"📋 Detected Document Type: {detected_doc_type}")
        logger.info(f"🎯 Summary Card Eligibility: {is_valid_for_summary_card}")
        logger.info(f"   Reasoning: {summary_card_reasoning[:100]}..." if len(summary_card_reasoning) > 100 else f"   Reasoning: {summary_card_reasoning}")
        
        # Initialize variables
        long_summary = ""
        short_summary = ""
        report_result = {}
        
        # Step 2: Process document based on summary card eligibility
        if is_valid_for_summary_card:
            # ✅ FULL EXTRACTION: Document requires physician review - generate summaries
            logger.info("📋 Document requires Summary Card - running full LLM extraction...")
            
            report_analyzer = ReportAnalyzer(mode)
            report_result = await asyncio.to_thread(
                report_analyzer.extract_document,
                summarizer_output,  # Use aiSummarizerText (Document AI Summarizer output)
                document_text,  # Use documentText as raw_text parameter
                doc_type_result  # Pass pre-detected doc_type
            )
            
            # ✅ STORE THE ACTUAL REPORT ANALYZER RESULT
            long_summary = report_result.get("long_summary", "")
            short_summary = report_result.get("short_summary", "")
            logger.info(f"✅ ReportAnalyzer completed, author field: {author}")
            
            # ✅ If author provided by user, replace or inject it into long_summary as "Signature:" field
            if author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
                # Replace existing signature line or add new one
                signature_pattern = r'•\s*Signature:.*?(?=\n•|\n\n|$)'
                signature_line = f"• Signature: {author.strip()}"
                
                if re.search(signature_pattern, long_summary, re.IGNORECASE | re.DOTALL):
                    # Replace existing signature
                    long_summary = re.sub(signature_pattern, signature_line, long_summary, flags=re.IGNORECASE | re.DOTALL)
                    logger.info(f"✅ Replaced existing signature with: {author}")
                else:
                    # Add new signature line
                    long_summary = long_summary + f"\n\n{signature_line}"
                    logger.info(f"✅ Injected author into long_summary: {author}")
                
                # Update the report_result dictionary to reflect the modified long_summary
                report_result["long_summary"] = long_summary
                
                # ✅ Also update the author field in short_summary header
                if isinstance(short_summary, dict) and 'header' in short_summary:
                    short_summary['header']['author'] = author.strip()
                    report_result["short_summary"] = short_summary
                    logger.info(f"✅ Updated short_summary header author: {author}")
            
            logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
            logger.info(f"✅ Generated short summary: {type(short_summary)}")
        else:
            # ⏭️ TASK-ONLY MODE: Document is administrative - skip expensive LLM extraction
            logger.info("📌 Document is TASK-ONLY (administrative) - skipping LLM extraction for summaries")
            logger.info(f"   Document type: {detected_doc_type}")
            logger.info(f"   Reason: {summary_card_reasoning}")
            
            # Create minimal summary for task generation (just use raw_text as reference)
            long_summary = f"[TASK-ONLY DOCUMENT]\nType: {detected_doc_type}\nReason: {summary_card_reasoning}\n\nThis document is administrative and does not require physician clinical review. Tasks have been generated for staff action."
            short_summary = ""  # No short summary for task-only docs
            report_result = {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "doc_type": detected_doc_type,
                "is_task_only": True,
                "task_only_reason": summary_card_reasoning
            }
            logger.info("⏭️ Skipped ReportAnalyzer - will proceed to task generation only")
        
        # Helper to convert structured short_summary dict to string
        raw_brief_summary_text = "Summary not available"
        
        # Handle task-only documents differently
        if not is_valid_for_summary_card:
            raw_brief_summary_text = f"{detected_doc_type} - Administrative document for staff action"
        elif short_summary:
            if isinstance(short_summary, dict):
                # Try to extract meaningful text from structured summary
                try:
                    # 1. Try to get items texts
                    items = short_summary.get('summary', {}).get('items', [])
                    text_parts = []
                    for item in items:
                        if isinstance(item, dict):
                            # Prefer expanded text, fall back to collapsed
                            part = item.get('expanded') or item.get('collapsed')
                            if part:
                                text_parts.append(part)
                    
                    if text_parts:
                        raw_brief_summary_text = " ".join(text_parts)
                    elif short_summary.get('header', {}).get('title'):
                         # Fallback to Title if no items
                         raw_brief_summary_text = f"Report: {short_summary['header']['title']}"
                    else:
                        # Fallback to JSON string as last resort
                        raw_brief_summary_text = json.dumps(short_summary)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse structured short_summary: {e}")
                    raw_brief_summary_text = str(short_summary)
            else:
                raw_brief_summary_text = str(short_summary)
        
        # ✅ Process the raw summary through the AI Condenser (only for summary card eligible docs)
        if is_valid_for_summary_card and raw_brief_summary_text != "Summary not available":
            brief_summary_text = await generate_concise_brief_summary(
                raw_brief_summary_text, 
                detected_doc_type,
                llm_executor
            )
        else:
            brief_summary_text = raw_brief_summary_text
        
        # Prepare fields for DocumentAnalysis
        da_patient_name = patient_name or "Not specified"
        da_claim_number = claim_number or "Not specified"
        da_dob = dob_str or "0000-00-00" 
        da_doi = doi or "0000-00-00"
        
        # For task-only documents (not valid for summary card), we bypass author/clinic member checks
        # by treating the consulting_doctor as Not specified, ensuring the flow continues for task generation
        if is_valid_for_summary_card:
            da_author = author or "Not specified"
        else:
            da_author = "Not specified"
            if author:
                logger.info(f"ℹ️ Task-only document: Skipping consulting_doctor assignment for author '{author}' to allow processing")
        
        # Manually construct DocumentAnalysis
        document_analysis = DocumentAnalysis(
            patient_name=da_patient_name,
            claim_number=da_claim_number,
            dob=da_dob,
            doi=da_doi,
            status="Not specified",
            rd="0000-00-00", 
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="See summary",
            key_concern="Medical evaluation",
            extracted_recommendation="See summary",
            extracted_decision="Not specified",
            ur_decision="",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor=da_author,
            all_doctors=[],
            referral_doctor="Not specified",
            ai_outcome="Review required",
            document_type=detected_doc_type,
            summary_points=[],
            brief_summary=brief_summary_text,
            date_reasoning=None,
            is_task_needed=False,
            formatted_summary=brief_summary_text,
            extraction_confidence=1.0 if short_summary else 0.0,
            verified=True,
            verification_notes=["Analysis from basic ReportAnalyzer (Update Fail Doc)"]
        )
        
        brief_summary = document_analysis.brief_summary
        
        # Override with updated fields from the user
        if updated_fields.get("patient_name") and str(updated_fields["patient_name"]).lower() != "not specified":
            document_analysis.patient_name = updated_fields["patient_name"]
            logger.info(f"✅ Overridden patient_name: {updated_fields['patient_name']}")
        
        if updated_fields.get("dob") and str(updated_fields["dob"]).lower() != "not specified":
            document_analysis.dob = updated_fields["dob"]
            logger.info(f"✅ Overridden DOB: {updated_fields['dob']}")
        
        if updated_fields.get("doi") and str(updated_fields["doi"]).lower() != "not specified":
            document_analysis.doi = updated_fields["doi"]
            logger.info(f"✅ Overridden DOI: {updated_fields['doi']}")
        
        if updated_fields.get("claim_number") and str(updated_fields["claim_number"]).lower() != "not specified":
            document_analysis.claim_number = updated_fields["claim_number"]
            logger.info(f"✅ Overridden claim_number: {updated_fields['claim_number']}")
        
        # ✅ Override consulting_doctor (author) if provided by user
        # Only override if document is valid for summary card (medical document)
        if is_valid_for_summary_card and author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
            document_analysis.consulting_doctor = author.strip()
            logger.info(f"✅ Overridden consulting_doctor (author): {author}")

        logger.info(f"Author detected: {author}")

        # Prepare processed_data similar to process_document_data
        processed_data = {
            "document_analysis": document_analysis,
            "brief_summary": brief_summary,
            "text_for_analysis": document_text,
            "raw_text": document_text,  # ✅ Add raw_text for task generation
            "report_analyzer_result": report_result,
            "patient_name": document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else None,
            "claim_number": document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None,
            "dob": document_analysis.dob if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else None,
            "has_patient_name": bool(document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"),
            "has_claim_number": bool(document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"),
            "physician_id": physician_id,
            "user_id": user_id,
            "filename": filename,
            "gcs_url": gcs_url,
            "blob_path": blob_path,
            "file_size": 0,
            "mime_type": "application/octet-stream",
            "processing_time_ms": 0,
            "file_hash": file_hash,
            "result_data": result_data,
            "document_id": str(fail_doc.id),
            "mode": mode,
            "is_valid_for_summary_card": is_valid_for_summary_card,
            "is_task_only": not is_valid_for_summary_card,
            "doc_type_result": doc_type_result
        }

        # Step 2: Perform patient lookup with enhanced fuzzy matching
        logger.info("🔍 Performing patient lookup for updated failed document...")
        lookup_result = await patient_lookup.perform_patient_lookup(db_service, processed_data)
        
        # Step 3: Save document to database
        logger.info("💾 Saving updated document to database...")
        save_result = await save_document_func(db_service, processed_data, lookup_result)
        
        # Step 4: Create tasks if needed
        # ✅ FIX: Pass the actual Document AI Summarizer output (summarizer_output) as document_analysis
        # The task generator expects raw text content, not a DocumentAnalysis object
        tasks_created = 0
        if save_result["document_id"] and save_result["status"] != "failed":
            tasks_created = await create_tasks_func(
                summarizer_output,  # ✅ Pass Document AI Summarizer output (same as direct processing)
                save_result["document_id"],
                processed_data["physician_id"],
                processed_data["filename"],
                processed_data  # Pass full processed_data for patient_name and document_type
            )
            save_result["tasks_created"] = tasks_created

        # Step 5: Delete the FailDoc only if successful
        if save_result["status"] != "failed" and save_result["document_id"]:
            await db_service.delete_fail_doc(fail_doc.id)
            logger.info(f"🗑️ Deleted fail doc {fail_doc.id} after successful update")
            logger.info(f"📡 Success event processed for document: {save_result['document_id']}")
        else:
            logger.warning(f"⚠️ Failed document update unsuccessful, keeping fail doc {fail_doc.id}")
            # Optionally update the fail doc with the new failure reason
            if save_result.get("failure_reason"):
                logger.info(f"📝 Updating fail doc reason: {save_result['failure_reason']}")

        return save_result

    except Exception as e:
        logger.error(f"❌ Failed to update fail document {fail_doc.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update fail document processing failed: {str(e)}")


async def update_multiple_fail_documents(
    fail_docs_data: List[dict],
    user_id: str,
    db_service: Any,
    patient_lookup,
    save_document_func,
    create_tasks_func,
    llm_executor=None
) -> dict:
    """
    Updates and processes multiple failed documents in batch.
    
    Args:
        fail_docs_data: List of dictionaries containing:
            - fail_doc: The failed document object
            - updated_fields: Fields to update
        user_id: User ID making the update
        db_service: Database service instance
        patient_lookup: Patient lookup service instance
        save_document_func: Function to save document
        create_tasks_func: Function to create tasks
        llm_executor: ThreadPoolExecutor for LLM operations
        
    Returns:
        dict with overall results and individual document results
    """
    results = {
        "total_documents": len(fail_docs_data),
        "successful": 0,
        "failed": 0,
        "documents": []
    }
    
    # Process documents sequentially
    for doc_data in fail_docs_data:
        fail_doc = doc_data.get("fail_doc")
        updated_fields = doc_data.get("updated_fields", {})
        
        if not fail_doc:
            logger.error("❌ Missing fail_doc in batch data")
            results["failed"] += 1
            results["documents"].append({
                "fail_doc_id": "unknown",
                "status": "failed",
                "error": "Missing fail_doc object"
            })
            continue
        
        try:
            # Process individual document
            document_result = await update_fail_document(
                fail_doc=fail_doc,
                updated_fields=updated_fields,
                user_id=user_id,
                db_service=db_service,
                patient_lookup=patient_lookup,
                save_document_func=save_document_func,
                create_tasks_func=create_tasks_func,
                llm_executor=llm_executor
            )
            
            results["successful"] += 1
            results["documents"].append({
                "fail_doc_id": fail_doc.id,
                "status": "success",
                "document_id": document_result.get("document_id"),
                "tasks_created": document_result.get("tasks_created", 0)
            })
            
            logger.info(f"✅ Successfully processed fail document {fail_doc.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process fail document {fail_doc.id}: {str(e)}")
            results["failed"] += 1
            results["documents"].append({
                "fail_doc_id": fail_doc.id,
                "status": "failed",
                "error": str(e)
            })
    
    return results

