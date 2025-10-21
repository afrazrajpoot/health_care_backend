from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re
import logging
import json

from config.settings import CONFIG

logger = logging.getLogger("document_ai")

# Pydantic models
class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema"""
    patient_name: str = Field(..., description="Full name of the patient")
    claim_number: str = Field(..., description="Claim number or case ID. Use 'Not specified' if not found")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    doi: str = Field(..., description="Date of injury in YYYY-MM-DD format")
    status: str = Field(..., description="Current status: normal, urgent, critical, etc.")
    rd: str = Field(..., description="Report date in YYYY-MM-DD format")
    diagnosis: str = Field(..., description="Primary diagnosis and key findings (comma-separated if multiple, 5-10 words total)")
    key_concern: str = Field(..., description="Main clinical concern in 2-3 words")
    next_step: str = Field(..., description="Recommended next steps in 2-3 words")
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
    document_type: str = Field(..., description="Type of document")
    summary_points: List[str] = Field(..., description="3-5 key points, each 2-3 words")

class BriefSummary(BaseModel):
    """Structured brief summary of the report"""
    brief_summary: str = Field(..., description="A concise 1-2 sentence summary of the entire report")

class ReportAnalyzer:
    """Service for extracting structured data from medical documents"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.whats_new_parser = JsonOutputParser()  # Raw JSON parser for dynamic flat dict output
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)

    def detect_document_type_preview(self, text: str) -> str:
        """
        Quick preview-based document type detection using keywords.
        Falls back to 'unknown' if no match.
        """
        try:
            text_lower = text.lower()
            if any(word in text_lower for word in ["mri", "ct scan", "x-ray", "imaging", "scan"]):
                return "imaging_report"
            elif any(word in text_lower for word in ["qme", "qualified medical evaluator", "independent medical exam"]):
                return "qme_report"
            elif any(word in text_lower for word in ["claim", "attorney", "legal", "denied", "approved"]):
                return "legal_document"
            elif any(word in text_lower for word in ["utilization review", "ur decision", "work restrictions"]):
                return "ur_decision"
            elif any(word in text_lower for word in ["patient", "diagnosis", "dob", "doi"]):
                return "medical_report"
            else:
                return "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Document type detection failed: {str(e)}")
            return "unknown"

    def create_extraction_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Extract structured information from the following medical document.
        
        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        EXTRACT THE FOLLOWING INFORMATION IN POINT FORM :
        
        - Patient name (full name)
        - Claim number or case ID (look for any identifiers like claim numbers, case IDs, etc. If not found, use "Not specified")
        - Date of birth (DOB) in YYYY-MM-DD format
        - Date of injury (DOI) in YYYY-MM-DD format  
        - Current status (normal, urgent, critical)
        - Primary diagnosis and 2-3 key findings (comma-separated if multiple)
        - Key clinical concerns 
        - Recommended next steps 
        - Activities of daily living affected
        - Work restrictions 
        - Document type
        - 3-5 key summary points 
        -Report date (RD) in YYYY-MM-DD format
        
        CRITICAL INSTRUCTIONS:
        - For diagnosis, include primary diagnosis plus 2-3 key findings (e.g., "Normal MRI, no mass lesion, clear sinuses"), comma-separated, up to 10 words total.
        - For key_concern, next_step, adls_affected, work_restrictions: Keep to 2-3 words each.
        - For summary_points, provide 3-5 bullet points, each 2-3 words.
        - If claim number is not explicitly found, use "Not specified".
        - Do NOT invent claim numbers.
        - If information is missing, use "Not specified".
        
        {format_instructions}
        
        Return ONLY valid JSON. No additional text.
        """
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def create_brief_summary_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document summarization expert. Generate a brief summary of the following medical document.
        
        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        CRITICAL INSTRUCTIONS:
        - Create a concise 1-2 sentence summary capturing the essence of the report.
        - Focus on key findings, diagnosis, and recommendations.
        - Keep it professional and objective.
        - If information is missing, use "Not specified".
        
        {format_instructions}
        
        Return ONLY valid JSON. No additional text.
        """
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.brief_summary_parser.get_format_instructions()},
        )

    def create_whats_new_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document comparison expert. Compile the patient's complete medical history and current document into a historical progression format.
        
        PATIENT'S MEDICAL HISTORY (most recent first, ordered by reportDate descending; extract dates from each entry's reportDate):
        {previous_analyses}
        
        CURRENT DOCUMENT ANALYSIS (extract report date from analysis text, fallback to provided {current_date} if missing):
        {current_analysis}
        
        CRITICAL INSTRUCTIONS:
        - STRICTLY obey document dates: EXTRACT reportDate MM/DD from each previous entry in {previous_analyses}, and EXTRACT report date MM/DD from {current_analysis} (e.g., from 'date:', 'DOI:', 'rd:', or summary date field; fallback to {current_date} only if no date found in analysis). Treat the extracted report date (rd) from current_analysis as its document date.
        - Collect ALL entries (previous + current) with their extracted dates, then sort them chronologically by date ascending (oldest first). If a previous reportDate is later than the current rd, that previous entry must appear AFTER current in the chronological chain.
        - Use arrow notation "→" to indicate progression or continuation in strict chronological order by extracted dates (oldest to newest), NOT by input order. ALWAYS sort the chain by extracted dates ascending, regardless of whether the current document is older or newer than previous ones.
        - Include EVERY diagnosis, treatment, finding, etc., without skipping any data.
        - If patient has multiple diagnoses, include ALL of them in the chain.
        - For first document (no previous), show all pure findings WITHOUT arrows.
        - Show ALL continuing items as they appear in sequence based on SORTED extracted dates.
        - Categorize ALL items into these specific categories:
        * diagnostic: Diagnosis changes, medical findings
        * qme: Qualified Medical Evaluator reports, independent medical exams
        * raf: Risk Adjustment Factor reports, claim adjustments
        * ur_decision: Utilization Review decisions, work restrictions, treatment approvals
        * legal: Legal developments, attorney letters, claim updates, whether approved or denied along with reason.
        
        - For EACH category, provide a concise description (3-5 words) with date in MM/DD format: STRICTLY use EXTRACTED reportDate for previous documents and EXTRACTED report date (rd) for current (do NOT use {current_date} unless extraction fails). Chain across the sorted sequence for that category.
        - Include SPECIFIC FINDINGS like all diagnoses, test results, restrictions - list multiples separated by commas.
        - Include ALL categories with data from any document.
        - Only include categories that have actual data. Do not include entries with 'None' or empty.
        - Use format: "Previous Item → Current Item (MM/DD)" for progression across documents, "Item (MM/DD)" for first-time or standalone items, where MM/DD is EXTRACTED from respective analyses (reportDate for prev, report date/rd for current). For chains with multiple: "Item1 (date1) → Item2 (date2) → ... → ItemN (dateN)".
        - Build a full history chain showing evolution over time, ensuring all dates align precisely with EXTRACTED document dates and the sequence is sorted chronologically (oldest to newest), with arrows connecting in order—differ where progression occurs (e.g., 05/15 → 09/12 → 10/05 if dates sort that way). If current document date is earlier (e.g., 05/15) and previous is later (e.g., 09/12), chain as "Item from current (05/15) → Item from previous (09/12)".
        - For diagnostic category, use the full diagnosis string from each analysis.diagnosis (which includes comma-separated key findings), chaining them in sorted date order.
        - OUTPUT MUST BE A FLAT JSON OBJECT: {{"category": "description string", ...}}. Do NOT nest values as objects or arrays—keep all values as simple strings.
        
        IMPORTANT: Include ALL historical data in the chain. Do not skip or omit any information. ALWAYS extract and use dates from the analyses text first (reportDate for prev, report date/rd for current)—ignore {current_date} unless explicitly no date in analysis. Focus on historical progression format, with all changes and dates matching the document-specific dates, sorted chronologically. The document labeled "current" may actually be older than "previous" documents - ALWAYS sort by extracted dates ascending (oldest first), not by labels. Ensure the arrow chain reflects the true timeline, e.g., if dates are 09/12 (previous) and 05/15 (current), sort to 05/15 → 09/12.
        
        EXAMPLES FOR FIRST DOCUMENT (using extracted report date):
        - First MRI report (analysis has "Date: 10/02"): {{"diagnostic": "Normal MRI, no mass lesion, clear sinuses (10/02)"}}
        - First QME report (analysis has "reportDate: 10/02"): {{"qme": "QME evaluation, restrictions (10/02)"}}
        - First legal document: {{"legal": "Claim QM12345 approved (10/02)"}}
        
        EXAMPLES FOR HISTORICAL CHAIN (extracted dates sorted: prev reportDate 09/01, current rd 10/02):
        - Diagnosis progression: {{"diagnostic": "lumbar strain (09/01) → lumbar strain, disc bulge, no compression (10/02)"}}
        - Multiple work restrictions: {{"ur_decision": "light duty (09/01) → no heavy lifting, no bending (10/02)"}}
        - Legal updates: {{"legal": "Claim QM12345 filed (09/01) → Claim QM12345 approved (10/02)"}}
        
        EXAMPLES WHEN PREVIOUS IS LATEST (extracted dates sorted: current rd 05/15, prev reportDate 09/12):
        - {{"diagnostic": "Lumbar Disc Herniation, Left L5 Radiculopathy, L4-L5 disc protrusion (05/15) → Lumbar Disc Herniation, Resolving L5 Radiculopathy (09/12)"}}
        
        EXAMPLES OF WHAT TO INCLUDE:
        - ✅ DO: Chain all diagnoses even if continuing: "strain (09/01) → strain, bulge, no compression (10/02)"
        - ✅ DO: List multiples: "PT, meds, restrictions (10/02)"
        - ✅ DO: Show all history: "denied (09/01) → approved (10/02)"
        - ❌ DON'T: Use same date on both sides unless extracted dates match exactly
        - ✅ DO: Sort dates chronologically: if current rd 05/15 and prev reportDate 09/12, chain as "item (05/15) → item (09/12)"
        - ✅ DO: If current is older, it appears first in chain: current rd 05/15, prev 09/12 = "item (05/15) → item (09/12)"
        
        {format_instructions}
        
        Return ONLY valid JSON. No additional text.
        """
        return PromptTemplate(
            template=template,
            input_variables=["previous_analyses", "current_analysis", "current_date"],
            partial_variables={"format_instructions": self.whats_new_parser.get_format_instructions()},
        )

    def extract_document_data(self, document_text: str) -> DocumentAnalysis:
        try:
            prompt = self.create_extraction_prompt()
            chain = prompt | self.llm | self.parser
            current_date = datetime.now().strftime("%Y-%m-%d")
            result = chain.invoke({
                "document_text": document_text[:15000],
                "current_date": current_date
            })
            logger.info(f"✅ Extracted data: Patient={result['patient_name']}, Claim={result['claim_number']}, Diagnosis={result['diagnosis']}")
            return DocumentAnalysis(**result)
        except Exception as e:
            logger.error(f"❌ Document analysis failed: {str(e)}")
            return self.create_fallback_analysis()

    def generate_brief_summary(self, document_text: str) -> str:
        """
        Generate a brief AI-powered summary of the document.
        """
        try:
            prompt = self.create_brief_summary_prompt()
            chain = prompt | self.llm | self.brief_summary_parser
            current_date = datetime.now().strftime("%Y-%m-%d")
            result = chain.invoke({
                "document_text": document_text[:15000],
                "current_date": current_date
            })
            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated brief summary: {brief_summary}")
            return brief_summary
        except Exception as e:
            logger.error(f"❌ Brief summary generation failed: {str(e)}")
            return "Brief summary unavailable"

    def create_fallback_analysis(self) -> DocumentAnalysis:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=datetime.now().strftime("%Y-%m-%d"),
            doi=datetime.now().strftime("%Y-%m-%d"),
            status="normal",
            rd=datetime.now().strftime("%Y-%m-%d"),
            diagnosis="Not specified",
            key_concern="Not specified",
            next_step="Not specified",
            adls_affected="Not specified",
            work_restrictions="Not specified",
            document_type="Medical Document",
            summary_points=["Processing completed", "Analysis unavailable"]
        )
    
    def compare_with_previous_documents(
        self, 
        current_analysis: DocumentAnalysis, 
        previous_documents: List[Dict[str, Any]]  
    ) -> Dict[str, str]:
        """
        Use LLM to compare previous documents with current analysis and generate 'What's New'.
        PRESERVES all previous whats_new data and adds new changes.
        """
        mm_dd = datetime.now().strftime("%m/%d")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"DEBUG: previous_documents = {previous_documents}")
        logger.info(f"DEBUG: Has previous? {bool(previous_documents)}")
        
        # ✅ FIX: Ensure current_analysis is a DocumentAnalysis instance
        if isinstance(current_analysis, dict):
            try:
                current_analysis = DocumentAnalysis(**current_analysis)
                logger.info(f"✅ Converted dict to DocumentAnalysis for current_analysis")
            except Exception as e:
                logger.error(f"❌ Failed to convert dict to DocumentAnalysis: {str(e)}")
                # Fallback to empty analysis if conversion fails
                current_analysis = self.create_fallback_analysis()
        elif not isinstance(current_analysis, DocumentAnalysis):
            logger.warning(f"⚠️ current_analysis is unexpected type {type(current_analysis)}; using fallback")
            current_analysis = self.create_fallback_analysis()
        
        # Collect ALL previous whats_new data from all documents
        all_previous_whats_new = {}
        if previous_documents:
            # Sort by creation date, oldest first to see progression
            sorted_prev = sorted(
                previous_documents, 
                key=lambda d: d.get('createdAt') or d.get('created_at') or datetime.min
            )
            
            # Accumulate all previous whats_new entries
            for doc in sorted_prev:
                whats_new = doc.get('whatsNew') or doc.get('whats_new') or {}
                for category, value in whats_new.items():
                    if value and isinstance(value, str) and value.strip() and value.lower() != 'none':  # Only add non-empty string values, skip 'none'
                        all_previous_whats_new[category] = value
        
        logger.info(f"DEBUG: Accumulated previous whats_new: {all_previous_whats_new}")
        
        # ✅ KEY FIX: If no previous documents, skip LLM entirely and create initial whats_new directly (no error risk)
        if not previous_documents:
            logger.info("✅ No previous documents: Creating initial whats_new directly.")
            initial_whats_new = self._create_initial_whats_new(current_analysis, mm_dd)
            # Merge with empty previous (just returns initial)
            merged_result = {**all_previous_whats_new, **initial_whats_new}
            logger.info(f"✅ Final merged 'What's New' (initial): {merged_result}")
            return merged_result
        
        # For previous data: Proceed with LLM, but with robust parsing
        # Prepare previous analyses as a formatted string INCLUDING all accumulated whats_new data
        previous_analyses = "ACCUMULATED HISTORY FROM PREVIOUS DOCUMENTS:\n"
        for category, value in all_previous_whats_new.items():
            previous_analyses += f"- {category}: {value}\n"
        
        # Also include the most recent document details for context
        most_recent = sorted(
            previous_documents, 
            key=lambda d: d.get('createdAt') or d.get('created_at') or datetime.min,
            reverse=True
        )[0]
        previous_analyses += f"\nMOST RECENT DOCUMENT SUMMARY:\n"
        previous_analyses += f"- Diagnosis: {most_recent.get('diagnosis', 'N/A')}\n"
        previous_analyses += f"- Document Type: {most_recent.get('document_type', 'N/A')}\n"
        previous_analyses += f"- Key Concerns: {most_recent.get('key_concern', 'N/A')}\n"
        previous_analyses += f"- Work Restrictions: {most_recent.get('work_restrictions', 'N/A')}\n"
        
        # Current analysis as detailed formatted string
        current_analysis_str = f"""
        CURRENT DOCUMENT ANALYSIS (Date: {current_date}):
        - Document Type: {current_analysis.document_type}
        - Diagnosis: {current_analysis.diagnosis}
        - Key Concerns: {current_analysis.key_concern}
        - Work Restrictions: {current_analysis.work_restrictions}
        - Next Steps: {current_analysis.next_step}
        - Claim Number: {current_analysis.claim_number}
        - Status: {current_analysis.status}
        - ADLs Affected: {current_analysis.adls_affected}
        - Summary Points: {', '.join(current_analysis.summary_points)}
        - Report Date (RD): {current_analysis.rd}
        """
        
        try:
            prompt = self.create_whats_new_prompt()
            chain = prompt | self.llm | self.whats_new_parser
            raw_result = chain.invoke({
                "previous_analyses": previous_analyses,
                "current_analysis": current_analysis_str,
                "current_date": current_date
            })
            
            # ✅ ROBUST PARSING FIX: Manually parse JSON and flatten nested values
            if isinstance(raw_result, str):
                try:
                    result = json.loads(raw_result)
                except json.JSONDecodeError as je:
                    logger.error(f"❌ Invalid JSON from LLM: {raw_result[:200]}... Error: {str(je)}")
                    # Fallback to default structure
                    return self._create_default_whats_new(current_analysis)
            else:
                result = raw_result  # Assume it's already a dict from parser
            
            if not isinstance(result, dict):
                logger.error(f"❌ AI returned non-dict result: {result}")
                return self._create_default_whats_new(current_analysis)
            
            # Flatten any nested dicts/strings (handles cases like {"diagnostic": {"value": "str"}} -> {"diagnostic": "str"})
            flattened_result = {}
            for category, value in result.items():
                if isinstance(value, dict):
                    # Extract first string value or str(value)
                    nested_str = None
                    for k, v in value.items():
                        if isinstance(v, str):
                            nested_str = v
                            break
                    if nested_str:
                        flattened_result[category] = nested_str
                    else:
                        flattened_result[category] = str(value)  # Fallback to str repr
                elif isinstance(value, str) and value.strip().lower() != 'none':
                    flattened_result[category] = value
                # Skip empty/non-strings
            
            logger.info(f"✅ Parsed LLM result (flattened): {flattened_result}")
            
            # ✅ ENSURE NON-EMPTY RESULT: If flattened result is empty, use default
            if not flattened_result:
                logger.warning("⚠️ Flattened result is empty; using default structure")
                flattened_result = self._create_default_whats_new(current_analysis)
            
            # Merge previous whats_new with new changes - PRESERVE ALL HISTORY
            merged_result = all_previous_whats_new.copy()  # Start with all previous data
            for category, value in flattened_result.items():
                if value and value.strip():
                    # Update with latest value (this will overwrite previous with updated info)
                    merged_result[category] = value
            
            # ✅ FINAL SAFETY CHECK: Ensure result is never empty
            if not merged_result:
                logger.warning("⚠️ Merged result is empty; creating default structure")
                merged_result = self._create_default_whats_new(current_analysis)
            
            logger.info(f"✅ MERGED 'What's New' (preserving history): {merged_result}")
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ AI comparison failed: {str(e)}")
            # Fallback: Return accumulated previous data OR create default if empty
            if all_previous_whats_new:
                logger.info(f"✅ Falling back to previous whats_new only: {all_previous_whats_new}")
                return all_previous_whats_new
            else:
                logger.info("✅ No previous data available; creating default whats_new")
                return self._create_default_whats_new(current_analysis)

    def _create_initial_whats_new(self, current_analysis: DocumentAnalysis, mm_dd: str) -> Dict[str, str]:
        """Create initial whats_new data for first document"""
        initial_whats_new = {}
        
        # Based on document type, create appropriate initial entries
        doc_type_lower = current_analysis.document_type.lower()
        
        if 'mri' in doc_type_lower or 'imaging' in doc_type_lower or 'scan' in doc_type_lower:
            if current_analysis.diagnosis and current_analysis.diagnosis.lower() not in ['not specified', 'none']:
                initial_whats_new['diagnostic'] = f"{current_analysis.diagnosis} ({mm_dd})"
        elif 'qme' in doc_type_lower or 'evaluator' in doc_type_lower:
            initial_whats_new['qme'] = f"QME evaluation ({mm_dd})"
        elif 'legal' in doc_type_lower or 'attorney' in doc_type_lower or 'claim' in doc_type_lower:
            if current_analysis.claim_number and current_analysis.claim_number.lower() != 'not specified':
                initial_whats_new['legal'] = f"Claim {current_analysis.claim_number} ({mm_dd})"
        
        # Default fallback for any document type if no specific match
        if not initial_whats_new and current_analysis.diagnosis and current_analysis.diagnosis.lower() not in ['not specified', 'none']:
            initial_whats_new['diagnostic'] = f"{current_analysis.diagnosis} ({mm_dd})"
        
        # Add work restrictions if specified
        if current_analysis.work_restrictions and current_analysis.work_restrictions.lower() not in ['not specified', 'none', '']:
            initial_whats_new['ur_decision'] = f"{current_analysis.work_restrictions} ({mm_dd})"
        
        # ✅ Ensure we always return at least one entry
        if not initial_whats_new:
            initial_whats_new['processing'] = f"Document processed ({mm_dd})"
        
        logger.info(f"✅ Created initial whats_new: {initial_whats_new}")
        return initial_whats_new

    def _create_default_whats_new(self, current_analysis: DocumentAnalysis) -> Dict[str, str]:
        """Create a default whats_new structure when comparison fails or returns empty"""
        mm_dd = datetime.now().strftime("%m/%d")
        
        default_whats_new = {
            "initial_processing": f"Document processed ({mm_dd})"
        }
        
        # Add document-type specific default entries
        doc_type_lower = current_analysis.document_type.lower()
        
        if any(word in doc_type_lower for word in ["mri", "ct", "x-ray", "imaging", "scan"]):
            default_whats_new["diagnostic"] = f"Imaging report - {current_analysis.diagnosis} ({mm_dd})" if current_analysis.diagnosis and current_analysis.diagnosis.lower() != "not specified" else f"Imaging report ({mm/dd})"
        elif "qme" in doc_type_lower:
            default_whats_new["qme"] = f"QME evaluation ({mm_dd})"
        elif any(word in doc_type_lower for word in ["legal", "attorney", "claim"]):
            default_whats_new["legal"] = f"Legal document processed ({mm_dd})"
        else:
            default_whats_new["medical"] = f"Medical report - {current_analysis.diagnosis} ({mm_dd})" if current_analysis.diagnosis and current_analysis.diagnosis.lower() != "not specified" else f"Medical report ({mm_dd})"
        
        # Add work restrictions if available
        if (current_analysis.work_restrictions and 
            current_analysis.work_restrictions.lower() not in ['not specified', 'none', '']):
            default_whats_new["ur_decision"] = f"Work restrictions: {current_analysis.work_restrictions} ({mm_dd})"
        
        logger.info(f"✅ Created default whats_new: {default_whats_new}")
        return default_whats_new

    def format_whats_new_as_highlights(self, whats_new_dict: Dict[str, str], current_date: str) -> List[str]:
        """
        Formats the merged 'whats_new' dict into bullet-point highlights.
        Preserves arrows for changes; uses plain descriptions for new items.
        Maps internal keys to sample categories (e.g., 'diagnostic' → 'New diagnostics').
        """
        mm_dd = datetime.strptime(current_date, "%Y-%m-%d").strftime("%m/%d")
        category_mapping = {
            "diagnostic": "New diagnostics",
            "qme": "New consults",
            "raf": "New authorizations/denials",
            "ur_decision": "New authorizations/denials",
            "legal": "Other"
        }
        
        highlights = []
        for internal_key, value in whats_new_dict.items():
            if not value.strip():
                continue
            user_friendly_key = category_mapping.get(internal_key, "Other")
            # Ensure date is appended if missing
            if not re.search(r'\(\d{2}/\d{2}\)', value):
                value += f" ({mm_dd})"
            highlights.append(f"• **{user_friendly_key}**: {value}")
        
        if not highlights:
            highlights = ["• No new changes since last visit."]
        
        return highlights