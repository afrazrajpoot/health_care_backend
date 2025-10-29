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
        - First, EXTRACT the report date (MM/DD) from {current_raw_text}. Look for patterns like:
        'Report Date:', 'Date of Service:', 'Exam Date:', 'Evaluation Date:', or 'Dated:'
        → If no date found, fallback to {current_date}.
        - Output pure findings only for THIS document (no historical arrows or comparisons).
        - Include EVERY diagnosis, treatment, finding, recommendation, and outcome mentioned.
        - Only include categories that are RELEVANT (i.e., mentioned in the text).

        ✅ ALWAYS INCLUDE if found:
        * diagnosis — medical findings, conditions, or changes in condition.
        * treatment_plan — therapy, medications, surgeries, injections, follow-ups, PT, etc.

        ✅ INCLUDE THESE ONLY IF RELEVANT / MENTIONED IN THE DOCUMENT:
        * qme — if the document mentions Qualified Medical Evaluator (QME), Independent Medical Exam (IME), or similar.
        * ur_decision — if the document includes Utilization Review (UR), authorization, denial, approval, or work restrictions.
        * legal — if the document discusses attorney letters, legal claim updates, approvals/denials.
        * raf — if the document mentions Risk Adjustment Factor, claim scoring, or rating.
        * recommendations — if there are any explicit or implicit next steps, follow-ups, or instructions.
        * outcome — if there is any result, improvement, worsening, or resolution mentioned.

        ✅ FORMATTING RULES:
        - Use the extracted report date (MM/DD) after each finding.
        - Output a **flat JSON object** — e.g.:
        {{"diagnosis": "Lumbar strain, persistent pain (10/02)", "treatment_plan": "Continue PT, use NSAIDs (10/02)", "qme": "Independent medical evaluation scheduled (10/02)"}}
        - If a category isn’t mentioned in the document, OMIT it completely (do not include empty or None values).
        - Each value should be a short but information-rich description (5–15 words).
        - Use simple language, no lists or arrays.
        - Maintain the date reference consistently across all entries.

        EXAMPLES:
        - If the document is a UR Denial: 
        {{"ur_decision": "Denied for lack of clinical justification (10/07)", "recommendations": "Consider alternative conservative management (10/07)"}}
        - If the document is a QME report:
        {{"qme": "QME evaluation performed, work restrictions applied (10/12)", "outcome": "Partial improvement (10/12)"}}
        - If it's a treatment note:
        {{"diagnosis": "Chronic lumbar pain (10/05)", "treatment_plan": "Continue PT twice weekly (10/05)", "recommendations": "Re-evaluate in 2 weeks (10/05)"}}

        OUTPUT REQUIREMENTS:
        - Use extracted report date.
        - Output ONLY valid JSON — no commentary or explanation.
        - Include diagnosis and treatment_plan whenever available.
        - Include other categories ONLY IF the document text explicitly or contextually contains relevant information.

        {format_instructions}
        """
        return PromptTemplate(
            template=template,
            input_variables=["current_raw_text", "current_date"],
            partial_variables={
                "format_instructions": self.whats_new_parser.get_format_instructions()
            },
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