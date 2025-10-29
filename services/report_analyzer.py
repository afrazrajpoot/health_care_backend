from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from typing import List, Dict, Any
from datetime import datetime
import json
import logging

from config.settings import CONFIG

logger = logging.getLogger("document_ai")

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
        self.whats_new_parser = JsonOutputParser()

    def compare_with_previous_documents(
        self, 
        current_raw_text: str,
        previous_documents: List[Dict[str, Any]] = None  # Optional, ignored for current-only generation
    ) -> Dict[str, str]:
        """
        Use LLM to analyze raw current text and generate 'What's New' from current document only.
        Ignores previous_documents; focuses solely on current raw text.
        LLM directly extracts diagnosis, recommendations, outcomes, etc., from raw text.
        """
        mm_dd = datetime.now().strftime("%m/%d")
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"DEBUG: Processing raw text length: {len(current_raw_text)}")
          
        # Current as raw text (truncate if too long)
        max_text_len = 8000
        current_raw_truncated = current_raw_text[:max_text_len] + "..." if len(current_raw_text) > max_text_len else current_raw_text
        
        current_raw_str = f"""
        CURRENT DOCUMENT RAW TEXT (Date: {current_date}):
        {current_raw_truncated}
        """
        
        try:
            prompt = self.create_whats_new_prompt()
            chain = prompt | self.llm | self.whats_new_parser
            raw_result = chain.invoke({
                "current_raw_text": current_raw_str,
                "current_date": current_date
            })
            
            # Robust parsing: Manually parse JSON and flatten nested values
            if isinstance(raw_result, str):
                try:
                    result = json.loads(raw_result)
                except json.JSONDecodeError as je:
                    logger.error(f"❌ Invalid JSON from LLM: {raw_result[:200]}... Error: {str(je)}")
                    return self._fallback_whats_new(mm_dd)
            else:
                result = raw_result
            
            if not isinstance(result, dict):
                logger.error(f"❌ AI returned non-dict result: {result}")
                return self._fallback_whats_new(mm_dd)
            
            # Flatten any nested dicts/strings
            flattened_result = {}
            for category, value in result.items():
                if isinstance(value, dict):
                    nested_str = None
                    for k, v in value.items():
                        if isinstance(v, str):
                            nested_str = v
                            break
                    if nested_str:
                        flattened_result[category] = nested_str
                    else:
                        flattened_result[category] = str(value)
                elif isinstance(value, str) and value.strip().lower() != 'none':
                    flattened_result[category] = value
            
            logger.info(f"✅ Parsed LLM result (flattened): {flattened_result}")
            
            # Final safety check: Ensure result is never empty
            if not flattened_result:
                flattened_result = self._fallback_whats_new(mm_dd)
            logger.info(f"✅ Generated 'What's New' from current: {flattened_result}")
            return flattened_result
            
        except Exception as e:
            logger.error(f"❌ AI generation failed: {str(e)}")
            return self._fallback_whats_new(mm_dd)

    def _fallback_whats_new(self, mm_dd: str) -> Dict[str, str]:
        """Simple fallback without LLM."""
        return {'initial': f"Document processed ({mm_dd})"}

    def create_whats_new_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Analyze the raw document text and generate key findings in a historical progression format for this single document.
        
        CURRENT DOCUMENT RAW TEXT (extract report date from text, fallback to provided {current_date} if missing):
        {current_raw_text}
        
        CRITICAL INSTRUCTIONS:
        - EXTRACT report date MM/DD from {current_raw_text} (e.g., from 'Report Date:', 'Date:', 'RD:', or any date field; fallback to {current_date} only if no date found).
        - Show all pure findings WITHOUT arrows since this is a single document.
        - Include EVERY diagnosis, treatment, finding, recommendation, outcome, etc., from the text.
        - If multiple diagnoses, include ALL of them.
        - Categorize ALL items into these specific categories:
        * diagnosis: Diagnosis changes, medical findings (include outcomes if mentioned)
        * qme: Qualified Medical Evaluator reports, independent medical exams (include recommendations)
        * raf: Risk Adjustment Factor reports, claim adjustments
        * ur_decision: Utilization Review decisions, work restrictions, treatment approvals/denials (include outcomes)
        * legal: Legal developments, attorney letters, claim updates, whether approved or denied along with reason.
        * recommendations: Next steps, treatments, or extracted recommendations from the document
        * outcome: Predicted or actual outcomes, resolutions from the document
        
        - For EACH category, provide a concise description (3-5 words) with date in MM/DD format: STRICTLY use EXTRACTED report date (do NOT use {current_date} unless extraction fails).
        - Include SPECIFIC FINDINGS like all diagnoses, test results, restrictions, recommendations, outcomes - list multiples separated by commas.
        - Include ALL categories with data from the document, especially recommendations and outcomes.
        - Only include categories that have actual data. Do not include entries with 'None' or empty.
        - Use format: "Item (MM/DD)" for standalone items, where MM/DD is EXTRACTED from the document.
        - Focus on key elements: diagnosis, recommendations, outcome, work restrictions, etc.
        - OUTPUT MUST BE A FLAT JSON OBJECT: {{"category": "description string", ...}}. Do NOT nest values as objects or arrays—keep all values as simple strings.
        
        IMPORTANT: Extract and use date from the raw text first—ignore {current_date} unless no date in text. Focus on current document findings, with recommendations, outcomes, and dates matching the document-specific date.
        
        EXAMPLES FOR SINGLE DOCUMENT (using extracted report date):
        - MRI report (text has "Report Date: 10/02"): {{"diagnosis": "Normal MRI, no mass lesion, clear sinuses, stable outcome (10/02)", "recommendations": "Follow-up in 6 months (10/02)"}}
        - QME report (text has "Date: 10/02"): {{"qme": "QME evaluation, restrictions, recommended PT (10/02)", "outcome": "Improved mobility (10/02)"}}
        - Legal document: {{"legal": "Claim QM12345 approved, positive outcome (10/02)"}}
        - UR decision: {{"ur_decision": "Light duty approved, no heavy lifting (10/02)", "recommendations": "PT and meds (10/02)"}}
        
        EXAMPLES OF WHAT TO INCLUDE:
        - ✅ DO: List multiples: "PT, meds, restrictions, positive outcome (10/02)"
        - ✅ DO: Always include recommendations and outcomes where present in raw text.
        
        {format_instructions}
        
        Return ONLY valid JSON. No additional text.
        """
        return PromptTemplate(
            template=template,
            input_variables=["current_raw_text", "current_date"],
            partial_variables={"format_instructions": self.whats_new_parser.get_format_instructions()},
        )

    def format_whats_new_as_highlights(self, whats_new_dict: Dict[str, str], current_date: str) -> List[str]:
        """
        Formats the 'whats_new' dict into bullet-point highlights.
        Maps internal keys to user-friendly categories.
        """
        mm_dd = datetime.strptime(current_date, "%Y-%m-%d").strftime("%m/%d")
        category_mapping = {
            "diagnosis": "New diagnosis",
            "qme": "New consults",
            "raf": "New authorizations/denials",
            "ur_decision": "New authorizations/denials",
            "legal": "Other",
            "recommendations": "New recommendations",
            "outcome": "New outcomes"
        }
        
        highlights = []
        for internal_key, value in whats_new_dict.items():
            if not value.strip():
                continue
            user_friendly_key = category_mapping.get(internal_key, "Other")
            # Ensure date is appended if missing
            if not any(d in value for d in ['(', '[']):  # Simple check for date pattern
                value += f" ({mm_dd})"
            highlights.append(f"• **{user_friendly_key}**: {value}")
        
        if not highlights:
            highlights = ["• No new changes since last visit."]
        
        return highlights

# ✅ USAGE: No change needed in calling code - still call compare_with_previous_documents(raw_text, previous_documents=[])
# It ignores previous_documents and generates from current only.