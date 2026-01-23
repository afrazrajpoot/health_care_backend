"""
Fail Document Update Service - Optimized for Critical Point Extraction
Takes structured short summary as input and extracts only critical points for physicians.
"""
from datetime import datetime
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.document_detector import detect_document_type
import json
import re
import asyncio
from utils.logger import logger
from config.settings import CONFIG
from fastapi import HTTPException
from models.data_models import DocumentAnalysis
from services.report_analyzer import ReportAnalyzer


# ============== Pydantic Models for Critical Summary ==============

class CriticalSummaryItem(BaseModel):
    """A critical summary item with field and prioritized content"""
    field: str = Field(description="Field name from the structured summary")
    critical_points: List[str] = Field(
        description="List of the MOST critical points from this field (2-4 items max), as complete sentences"
    )
    priority: int = Field(
        description="Priority level: 1=Critical, 2=Important, 3=Supporting",
        ge=1,
        le=3
    )


class CriticalClinicalSummary(BaseModel):
    """Critical summary extracted from structured short summary"""
    items: List[CriticalSummaryItem] = Field(
        description="List of critical summary items, ordered by priority"
    )


# ============== Critical Field Priority Mapping ==============

CRITICAL_FIELD_PRIORITIES = {
    # Highest Priority (1) - Critical for clinical decisions
    "mmi_declaration": 1,
    "mmi_status": 1,
    "impairment_rating": 1,
    "permanent_restrictions": 1,
    "work_status": 1,
    "work_restrictions": 1,
    "disability_status": 1,
    "causation_opinion": 1,
    "causation_analysis": 1,
    "final_decision": 1,
    "decision": 1,
    "authorization_status": 1,
    "treatment_authorization": 1,
    "critical_findings": 1,
    "urgent_findings": 1,
    "complications": 1,
    
    # High Priority (2) - Important for treatment
    "diagnoses": 2,
    "assessment": 2,
    "key_findings": 2,
    "significant_findings": 2,
    "treatment_plan": 2,
    "recommendations": 2,
    "treatment_response": 2,
    "medication_changes": 2,
    "new_medications": 2,
    "surgical_procedures": 2,
    "procedure_performed": 2,
    "follow_up_plan": 2,
    
    # Medium Priority (3) - Supporting information
    "findings": 3,
    "physical_exam": 3,
    "objective_findings": 3,
    "subjective_complaints": 3,
    "reported_history": 3,
    "current_treatment": 3,
    "medications": 3,
    "vital_signs": 3,
    "clinical_course": 3,
    "historical_context": 3,
}


def get_field_priority(field_name: str) -> int:
    """
    Get priority level for a field name.
    """
    field_lower = field_name.lower().replace(" ", "_")
    
    # Check exact match
    if field_lower in CRITICAL_FIELD_PRIORITIES:
        return CRITICAL_FIELD_PRIORITIES[field_lower]
    
    # Check partial matches
    for critical_field, priority in CRITICAL_FIELD_PRIORITIES.items():
        if critical_field in field_lower or field_lower in critical_field:
            return priority
    
    # Default priority for unknown fields
    return 3


# ============== Main Critical Summary Generator ==============

async def generate_critical_summary(
    structured_short_summary: Dict[str, Any],
    document_type: str,
    llm: Optional[AzureChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Generate a critical-focused summary from structured short summary.
    Uses AI to extract ONLY the most critical points for physicians.
    
    Args:
        structured_short_summary: Output from generate_structured_short_summary
            Format: {"summary": {"items": [{"field": "...", "collapsed": "...", "expanded": "..."}]}}
        document_type: Document type for context
        llm: Optional AzureChatOpenAI instance
        
    Returns:
        Dict matching CriticalClinicalSummary structure
    """
    logger.info(f"🎯 Generating critical summary from structured summary for {document_type}")
    
    # Validate input
    if not structured_short_summary:
        return create_critical_fallback_summary(document_type)
    
    summary_items = structured_short_summary.get("summary", {}).get("items", [])
    
    if not summary_items:
        logger.warning("⚠️ No summary items found in structured summary")
        return create_critical_fallback_summary(document_type)
    
    logger.info(f"📊 Processing {len(summary_items)} fields from structured summary")
    
    # Create LLM if not provided
    if llm is None:
        try:
            llm = AzureChatOpenAI(
                azure_deployment=CONFIG.get("azure_openai_deployment"),
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.1,
                max_tokens=4000,
                timeout=60,
                request_timeout=60,
            )
        except Exception as e:
            logger.error(f"❌ Failed to create LLM: {e}")
            return generate_critical_summary_without_llm(structured_short_summary, document_type)
    
    # Convert structured summary to text for LLM processing
    summary_text = format_summary_for_llm(structured_short_summary, document_type)
    
    # Create Pydantic parser
    parser = PydanticOutputParser(pydantic_object=CriticalClinicalSummary)
    
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a PHYSICIAN'S CRITICAL POINT EXTRACTOR.

Your SOLE TASK: From a detailed medical summary, extract ONLY the MOST CRITICAL points that physicians need for immediate clinical decisions.

🔴 **CRITICAL FILTERING RULES:**

**EXTRACT ONLY:**
✅ MMI status and impairment ratings
✅ Work restrictions/disability status
✅ Treatment authorization decisions (approved/denied)
✅ New or changed diagnoses
✅ Significant abnormal findings
✅ Surgical procedures performed
✅ Critical medication changes
✅ Urgent follow-up needs

**FILTER OUT:**
❌ Normal/negative findings (unless they negate something critical)
❌ Routine administrative details
❌ Generic statements without specifics
❌ Historical information without current relevance
❌ Minor variations or measurements without clinical impact

🧠 **CLINICAL PRIORITIZATION:**
For each field, ask: "Is this information CRITICAL for physician decision-making RIGHT NOW?"
If NO → Exclude or summarize to most critical 1-2 points.

📋 **OUTPUT REQUIREMENTS:**

1. **FIELD PRIORITIZATION:**
   - Priority 1: MMI, work status, authorization decisions, critical findings
   - Priority 2: Diagnoses, treatment plans, key recommendations
   - Priority 3: Supporting findings, exam details, historical context

2. **CONTENT EXTRACTION:**
   - 2-4 critical points MAX per field
   - Each point must be a COMPLETE sentence
   - Focus on ACTIONABLE information
   - Preserve exact medical terminology
   - Maintain past-tense attribution

3. **QUALITY OVER QUANTITY:**
   - Better to have 2 critical points than 10 trivial ones
   - No bullet fragments (e.g., "5mm" → "5mm disc protrusion was documented")
   - Group related critical findings

**EXAMPLES OF GOOD EXTRACTION:**

Input field "findings" with many bullet points:
• L4-L5 disc herniation was noted
• 5mm central protrusion was measured
• Mild spinal stenosis was documented
• No significant degenerative changes
• Normal alignment
• No prior studies available

Critical extraction (Priority 2):
• L4-L5 disc herniation with 5mm central protrusion was documented
• Mild spinal stenosis was noted at the same level

Input field "work_status" with various details:
• Off work status was documented
• No lifting greater than 10 pounds for 4 weeks
• Sedentary work only
• Follow-up in 2 weeks

Critical extraction (Priority 1):
• Off work with temporary total disability status was documented
• No lifting greater than 10 pounds for 4 weeks was specified

** Important: Do not return duplicate or overlapping information across fields. **
like this: as both are some with only key name changed
FOLLOW UP PLAN
• Resubmission of the injection request was documented as a follow-up action.
PLAN
• Resubmission of the request for a T10-T11 injection was referenced in the treatment plan.

**DOCUMENT TYPE CONTEXT:** {document_type}

{format_instructions}

Output ONLY the JSON object.
""")

    user_prompt = HumanMessagePromptTemplate.from_template("""
**DOCUMENT TYPE:** {document_type}

**STRUCTURED SUMMARY TO FILTER:**
{summary_text}

**TASK:** Extract ONLY the most critical points for physician review.

**FIELD-BY-FIELD EXTRACTION GUIDANCE:**

1. **FOR EACH FIELD IN THE SUMMARY:**
   - Identify the 2-4 MOST critical points
   - Exclude routine/normal information
   - Focus on what would change clinical decisions

2. **CRITICALITY ASSESSMENT:**
   - Priority 1: Immediate action needed (MMI, work status, authorizations)
   - Priority 2: Important for treatment planning (diagnoses, key findings)
   - Priority 3: Supporting context (exam details, history)

3. **SENTENCE QUALITY:**
   - Complete sentences only
   - Past-tense attribution maintained
   - Clear, specific clinical information

4. **MEDICATION SPECIAL CASE:**
   - For medications: Include only NEW prescriptions or DOSE CHANGES
   - Format: "Medication [dose] [frequency] was prescribed/started/changed"
   - Exclude stable chronic medications

**OUTPUT:** JSON with critical summary items, ordered by priority.
""")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Critical extraction attempt {attempt + 1}/{max_retries}")
            
            chain = chat_prompt | llm
            response = chain.invoke({
                "document_type": document_type,
                "summary_text": summary_text,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Extract JSON
            response_content = response.content.strip()
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                response_content = response_content[start_idx:end_idx+1]
            
            # Clean and parse
            response_content = "".join(ch for ch in response_content if ch >= ' ' or ch in '\n\r\t')
            
            # Parse with Pydantic
            critical_summary = parser.parse(response_content)
            
            # Post-process: Ensure items are ordered by priority
            critical_summary.items.sort(key=lambda x: (x.priority, x.field))
            
            logger.info(f"✅ Generated critical summary with {len(critical_summary.items)} prioritized items")
            return critical_summary.model_dump()
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error("❌ All parsing attempts failed, using fallback")
                return generate_critical_summary_without_llm(structured_short_summary, document_type)
                
        except Exception as e:
            logger.warning(f"⚠️ Critical extraction failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error("❌ All extraction attempts failed, using fallback")
                return generate_critical_summary_without_llm(structured_short_summary, document_type)
    
    return generate_critical_summary_without_llm(structured_short_summary, document_type)


def format_summary_for_llm(structured_summary: Dict[str, Any], doc_type: str) -> str:
    """
    Format structured summary into text for LLM processing.
    """
    summary_items = structured_summary.get("summary", {}).get("items", [])
    
    if not summary_items:
        return "No summary items available."
    
    formatted_sections = []
    
    for item in summary_items:
        field = item.get("field", "Unknown")
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        
        # Format this field's content
        field_text = f"FIELD: {field.upper()}"
        
        if collapsed:
            field_text += f"\nSummary: {collapsed}"
        
        if expanded:
            # Clean and format bullets
            bullets = []
            for line in expanded.split('\n'):
                line = line.strip()
                if line.startswith(('•', '-', '*')):
                    bullet_text = line[1:].strip()
                    if bullet_text:
                        bullets.append(bullet_text)
                elif line and len(line.split()) >= 3:
                    bullets.append(line)
            
            if bullets:
                field_text += f"\nDetails ({len(bullets)} items):"
                for i, bullet in enumerate(bullets[:15], 1):  # Limit to 15 bullets per field
                    field_text += f"\n  {i}. {bullet}"
                if len(bullets) > 15:
                    field_text += f"\n  ... and {len(bullets) - 15} more items"
        
        formatted_sections.append(field_text)
    
    # Add document type context
    header = f"DOCUMENT TYPE: {doc_type}\nTOTAL FIELDS: {len(summary_items)}\n\n"
    
    return header + "\n\n".join(formatted_sections)


def generate_critical_summary_without_llm(
    structured_summary: Dict[str, Any],
    doc_type: str
) -> Dict[str, Any]:
    """
    Fallback critical summary generation without LLM.
    Uses rule-based extraction of critical points.
    """
    logger.info("🔄 Using rule-based critical extraction (LLM fallback)")
    
    summary_items = structured_summary.get("summary", {}).get("items", [])
    
    if not summary_items:
        return create_critical_fallback_summary(doc_type)
    
    critical_items = []
    
    # Keywords that indicate critical content
    critical_keywords = {
        "high": ["mmi", "permanent", "impairment", "disability", "denied", "authorized", 
                "approved", "restriction", "severe", "acute", "critical", "urgent",
                "complication", "surgery", "procedure", "operation", "cancer", "fracture",
                "infection", "bleeding", "emergency", "admit", "hospitalize"],
        "medium": ["diagnosis", "finding", "abnormal", "positive", "elevated", "decreased",
                  "changed", "new", "started", "increased", "decreased", "stopped",
                  "recommend", "refer", "follow-up", "appointment", "test", "scan"],
    }
    
    for item in summary_items:
        field = item.get("field", "")
        collapsed = item.get("collapsed", "").lower()
        expanded = item.get("expanded", "").lower()
        
        if not field:
            continue
        
        # Get field priority
        priority = get_field_priority(field)
        
        # Extract critical points from expanded content
        critical_points = extract_critical_points_from_text(item, critical_keywords, priority)
        
        if critical_points:
            critical_items.append({
                "field": field,
                "critical_points": critical_points,
                "priority": priority
            })
    
    # Sort by priority
    critical_items.sort(key=lambda x: (x["priority"], x["field"]))
    
    # Limit to top items
    if len(critical_items) > 10:
        critical_items = critical_items[:10]
        logger.info(f"📏 Limited to top 10 critical items")
    
    return {
        "items": critical_items
    }


def extract_critical_points_from_text(
    item: Dict[str, str],
    critical_keywords: Dict[str, List[str]],
    field_priority: int
) -> List[str]:
    """
    Extract critical points from summary item using keyword matching.
    """
    field = item.get("field", "")
    expanded = item.get("expanded", "")
    
    if not expanded:
        # Use collapsed text if no expanded
        collapsed = item.get("collapsed", "")
        if collapsed and len(collapsed.split()) >= 4:
            return [ensure_complete_sentence(collapsed)]
        return []
    
    # Parse bullet points
    bullets = []
    for line in expanded.split('\n'):
        line = line.strip()
        if line.startswith(('•', '-', '*')):
            bullet_text = line[1:].strip()
            if bullet_text:
                bullets.append(bullet_text)
    
    if not bullets:
        return []
    
    # Score bullets based on critical keywords
    scored_bullets = []
    for bullet in bullets:
        bullet_lower = bullet.lower()
        
        # Calculate criticality score
        score = 0
        
        # High-priority keywords
        for keyword in critical_keywords["high"]:
            if keyword in bullet_lower:
                score += 3
                break
        
        # Medium-priority keywords
        for keyword in critical_keywords["medium"]:
            if keyword in bullet_lower:
                score += 1
        
        # Field priority bonus
        score += (4 - field_priority)  # Higher priority fields get more weight
        
        # Length bonus (longer sentences often have more detail)
        word_count = len(bullet.split())
        if word_count >= 8:
            score += 1
        
        scored_bullets.append((score, bullet))
    
    # Sort by score and take top 2-4
    scored_bullets.sort(key=lambda x: x[0], reverse=True)
    
    # Determine how many to take based on field priority
    max_points = 4 if field_priority == 1 else 3 if field_priority == 2 else 2
    
    critical_points = []
    for score, bullet in scored_bullets[:max_points]:
        if score > 0:  # Only include bullets with some critical content
            # Ensure complete sentence
            complete_bullet = ensure_complete_sentence(bullet)
            if complete_bullet:
                critical_points.append(complete_bullet)
    
    return critical_points


def ensure_complete_sentence(text: str) -> str:
    """
    Ensure text is a complete sentence.
    """
    if not text:
        return text
    
    text = text.strip()
    
    # Add period if missing
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Check if it looks like a complete sentence
    words = text.split()
    if len(words) < 3:
        return ""  # Too short to be meaningful
    
    # Check for verb-like words
    has_verb = any(word.lower() in ['was', 'were', 'documented', 'noted', 'reported', 
                                   'found', 'observed', 'measured', 'prescribed'] 
                   for word in words)
    
    if not has_verb:
        # Try to add attribution
        if ":" in text:
            # Likely a label: value format
            return text
        else:
            # Add basic attribution
            return f"The report documented {text.lower()}"
    
    return text


def create_critical_fallback_summary(doc_type: str) -> Dict[str, Any]:
    """
    Create a fallback critical summary.
    """
    return {
        "items": [
            {
                "field": "critical_summary",
                "critical_points": [
                    f"Critical information extraction from {doc_type} failed.",
                    "Physician review of original document is recommended.",
                    "Key clinical findings require manual assessment."
                ],
                "priority": 1
            }
        ]
    }


# ============== Integration with Existing Functions ==============

async def generate_concise_brief_summary(
    structured_short_summary: Dict[str, Any],
    document_type: str = "Medical Document",
    llm_executor=None  # Kept for backward compatibility but not used
) -> Dict[str, Any]:
    """
    Main entry point - generates critical summary from structured short summary.
    Maintains compatibility with existing code.
    
    Note: llm_executor parameter is kept for backward compatibility but is not used.
    The generate_critical_summary function creates its own LLM instance.
    """
    logger.info(f"🚀 Generating concise brief summary for {document_type}")
    
    try:
        # Generate critical summary - let it create its own LLM
        critical_summary = await generate_critical_summary(
            structured_short_summary=structured_short_summary,
            document_type=document_type,
            llm=None  # Let generate_critical_summary create its own LLM
        )
        
        # Convert to simple key-value format if needed for compatibility
        if "items" in critical_summary:
            # Convert to simple dict format for backward compatibility
            simple_dict = {}
            for item in critical_summary["items"]:
                field = item.get("field", "")
                points = item.get("critical_points", [])
                if field and points:
                    simple_dict[field] = points
            
            # Add priority indicator
            if simple_dict:
                simple_dict["_metadata"] = {
                    "type": "critical_summary",
                    "document_type": document_type,
                    "item_count": len(critical_summary["items"])
                }
            
            logger.info(f"✅ Generated critical summary with {len(critical_summary['items'])} items")
            return simple_dict
        
        return critical_summary
        
    except Exception as e:
        logger.error(f"❌ Critical summary generation failed: {e}")
        return generate_critical_summary_without_llm(structured_short_summary, document_type)


# ============== Quality Validation ==============

def validate_critical_summary_quality(
    critical_summary: Dict[str, Any],
    structured_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that critical summary properly filters and prioritizes content.
    """
    validation = {
        "passed": True,
        "warnings": [],
        "reduction_ratio": 0.0,
        "critical_item_count": 0,
        "total_item_count": 0
    }
    
    # Get counts from structured summary
    structured_items = structured_summary.get("summary", {}).get("items", [])
    validation["total_item_count"] = len(structured_items)
    
    # Get counts from critical summary
    if "items" in critical_summary:
        critical_items = critical_summary["items"]
        validation["critical_item_count"] = len(critical_items)
        
        # Count total points
        total_points = sum(len(item.get("critical_points", [])) for item in critical_items)
        
        # Calculate reduction ratio
        if validation["total_item_count"] > 0:
            # Estimate original points (average 5 per field)
            estimated_original_points = validation["total_item_count"] * 5
            if estimated_original_points > 0:
                validation["reduction_ratio"] = total_points / estimated_original_points
        
        # Check prioritization
        priority_counts = {1: 0, 2: 0, 3: 0}
        for item in critical_items:
            priority = item.get("priority", 3)
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        validation["priority_distribution"] = priority_counts
        
        # Check sentence completeness
        incomplete_sentences = []
        for item in critical_items:
            for point in item.get("critical_points", []):
                words = point.split()
                if len(words) < 4:
                    incomplete_sentences.append(point[:50] + "...")
        
        if incomplete_sentences:
            validation["warnings"].append(f"Found {len(incomplete_sentences)} potentially incomplete sentences")
    
    # Check if summary is too sparse
    if validation["critical_item_count"] == 0:
        validation["passed"] = False
        validation["warnings"].append("No critical items extracted")
    
    # Check if reduction is too aggressive
    if validation["reduction_ratio"] < 0.1 and validation["total_item_count"] > 5:
        validation["warnings"].append("Extremely aggressive filtering - may have lost important context")
    
    return validation


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
        
        # ✅ Process the structured summary through the reducer (only for summary card eligible docs)
        if is_valid_for_summary_card and isinstance(short_summary, dict) and short_summary.get('summary', {}).get('items'):
            brief_summary_text = await generate_concise_brief_summary(
                short_summary,  # Pass the structured summary directly
                detected_doc_type,
                llm_executor
            )
        elif is_valid_for_summary_card and raw_brief_summary_text != "Summary not available":
            # Fallback: create minimal structure from raw text
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}
        else:
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}
        
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
