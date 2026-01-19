"""
Fail Document Update Service - Handles updating and reprocessing failed documents
Extracted from webhook_service.py for better modularity
"""
from datetime import datetime
from typing import Any, List
from fastapi import HTTPException
from models.data_models import DocumentAnalysis
from services.report_analyzer import ReportAnalyzer
from utils.logger import logger
from utils.document_detector import detect_document_type
import asyncio
import json
import re


async def generate_concise_brief_summary(raw_summary_text: str, document_type: str = "Medical Document", llm_executor=None) -> str:
    """
    Uses LLM to transform the raw summarizer output into a concise, accurate professional summary.
    Focuses on factual extraction without adding interpretations or missing critical details.
    
    Args:
        raw_summary_text: The raw summary text to process
        document_type: Type of document for context
        llm_executor: ThreadPoolExecutor for running LLM operations
        
    Returns:
        Concise brief summary string
    """
    if not raw_summary_text or len(raw_summary_text) < 10:
        return "Summary not available"

    try:
        logger.info("🤖 Generating concise brief summary using LLM...")
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import AzureChatOpenAI
        from config.settings import CONFIG

        llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,  # Zero temperature for maximum factual accuracy
            timeout=30
        )

        system_template = """You are a medical documentation assistant specialized in accurate information extraction.

CRITICAL RULES:
1. Extract ONLY information explicitly stated in the raw summary - DO NOT infer, assume, or add any details
2. If critical information is present, include it even if it makes the summary longer
3. Preserve ALL specific medical details: diagnoses, medications (with dosages), test results, dates, measurements
4. Use the EXACT medical terminology from the source - do not paraphrase medical terms
5. If information is uncertain or not clearly stated, omit it rather than guessing
6. Do not include generic statements like "patient was treated" without specifying what treatment

STRUCTURE (only include sections with available information):
- Primary diagnosis/condition with any relevant clinical findings
- Key interventions, procedures, or medications (include specific names and dosages if mentioned)
- Critical test results or measurements if present
- Current status, follow-up plan, or next steps

OUTPUT FORMAT:
- Write in clear, concise paragraphs (NOT bullet points)
- No headers, no "Here is the summary" preamble
- Aim for 3-5 sentences, but extend if necessary to capture all critical information
- Prioritize completeness and accuracy over brevity

If the raw summary lacks substantive medical information, state "Limited clinical information available in source document" rather than fabricating content."""

        user_template = f"""Document Type: {document_type}

Raw Summary Input:
{raw_summary_text}

Extract and present the concise summary following the rules above:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        chain = prompt | llm
        
        # Run in executor to avoid blocking
        if llm_executor:
            response = await asyncio.get_event_loop().run_in_executor(
                llm_executor, 
                lambda: chain.invoke({})
            )
        else:
            response = await asyncio.to_thread(lambda: chain.invoke({}))
        
        clean_summary = response.content.strip()
        
        # Validation: Check if summary is suspiciously short given substantial input
        if len(raw_summary_text) > 200 and len(clean_summary) < 50:
            logger.warning("⚠️ Generated summary may be incomplete - falling back to raw summary")
            return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text
        
        # Validation: Check for generic/hallucinated content patterns
        hallucination_indicators = [
            "patient was seen",
            "routine care provided",
            "standard treatment given",
            "typical findings noted"
        ]
        if any(indicator in clean_summary.lower() for indicator in hallucination_indicators):
            if not any(indicator in raw_summary_text.lower() for indicator in hallucination_indicators):
                logger.warning("⚠️ Detected potentially fabricated generic content - using raw summary")
                return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text
        
        logger.info(f"✅ Generated concise summary ({len(clean_summary)} chars)")
        return clean_summary

    except Exception as e:
        logger.error(f"❌ Failed to generate concise brief summary: {e}")
        # Fallback: Return truncated original text with better length handling
        return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text


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
        da_author = author or "Not specified"
        
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
        if author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
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

