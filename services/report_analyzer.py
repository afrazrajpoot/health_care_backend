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
        previous_documents: List[Dict[str, Any]] = None
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
            
            # Ensure ALL categories are present with "not specified" for missing ones
            required_categories = [
                "diagnosis", "treatment_plan", "qme", "ur_decision", 
                "legal", "raf", "recommendations", "outcome"
            ]
            
            final_result = {}
            for category in required_categories:
                if category in result:
                    value = result[category]
                    if isinstance(value, dict):
                        # Extract first non-empty string value from nested dict
                        nested_str = None
                        for k, v in value.items():
                            if isinstance(v, str) and v.strip() and v.strip().lower() not in ['none', 'n/a', 'not mentioned', 'not applicable', 'not specified']:
                                nested_str = v
                                break
                        final_result[category] = nested_str if nested_str else "not specified"
                    elif isinstance(value, str) and value.strip() and value.strip().lower() not in ['none', 'n/a', 'not mentioned', 'not applicable', 'not specified']:
                        final_result[category] = value
                    else:
                        final_result[category] = "not specified"
                else:
                    final_result[category] = "not specified"
            
            logger.info(f"✅ Parsed LLM result (with not specified): {final_result}")
            
            # Final safety check: Ensure result is never empty
            if not final_result or all(v == "not specified" for v in final_result.values()):
                final_result = self._fallback_whats_new(mm_dd)
            logger.info(f"✅ Generated 'What's New' from current: {final_result}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ AI generation failed: {str(e)}")
            return self._fallback_whats_new(mm_dd)

    def _fallback_whats_new(self, mm_dd: str) -> Dict[str, str]:
        """Simple fallback without LLM."""
        return {
            'diagnosis': 'not specified',
            'treatment_plan': 'not specified', 
            'qme': 'not specified',
            'ur_decision': 'not specified',
            'legal': 'not specified',
            'raf': 'not specified',
            'recommendations': 'not specified',
            'outcome': 'not specified'
        }

    def create_whats_new_prompt(self) -> PromptTemplate:
        template = """
You are a medical document analysis expert. Your task is to READ the document text carefully and EXTRACT ONLY the information that is EXPLICITLY MENTIONED in the text.

DOCUMENT TEXT TO ANALYZE:
{current_raw_text}

CRITICAL RULES - READ CAREFULLY:
1. **ONLY extract information that is ACTUALLY PRESENT in the document text above**
2. **DO NOT add categories or information that is NOT mentioned in the text**
3. **DO NOT make assumptions or infer information**
4. **If something is not mentioned, use "not specified" for that category**
5. **You MUST include ALL categories in the output, using "not specified" for missing information**

STEP-BY-STEP PROCESS:
1. First, extract the report date from the document text:
   - Look for: "Report Date:", "Date of Service:", "Exam Date:", "Date:", etc.
   - Format as MM/DD (e.g., 10/15)
   - If no date found, use: {current_date} formatted as MM/DD

2. Read the ENTIRE document and identify what information is ACTUALLY present:
   - Is there a diagnosis or medical condition mentioned? → Extract it
   - Is there a treatment plan or medication mentioned? → Extract it
   - Is there a QME/IME evaluation mentioned? → Extract it
   - Is there a UR decision/authorization/denial mentioned? → Extract it
   - Is there legal information mentioned? → Extract it
   - Are there recommendations mentioned? → Extract it
   - Is there an outcome or result mentioned? → Extract it

3. For EACH category below, provide the information if found in text, otherwise use "not specified"

REQUIRED CATEGORIES (you MUST include all of these):
- **diagnosis**: Medical conditions, symptoms, findings, changes in health status. If none mentioned: "not specified"
- **treatment_plan**: Medications, therapies, surgeries, procedures, PT, follow-ups. If none mentioned: "not specified"  
- **qme**: QME evaluations, IME exams, independent assessments. If none mentioned: "not specified"
- **ur_decision**: Utilization Review decisions, authorizations, denials, approvals. If none mentioned: "not specified"
- **legal**: Legal matters, attorney communications, claim updates. If none mentioned: "not specified"
- **raf**: Risk adjustment factors, claim scoring, ratings. If none mentioned: "not specified"
- **recommendations**: Next steps, follow-up instructions, suggestions. If none mentioned: "not specified"
- **outcome**: Results, improvements, deterioration, resolution. If none mentioned: "not specified"

OUTPUT FORMAT:
- Return a JSON object with ALL categories listed above
- For categories with information: "Brief summary of finding (MM/DD)"
- For categories without information: "not specified"
- Example: {{"diagnosis": "Chronic lower back pain, radiculopathy (10/15)", "treatment_plan": "Continue physical therapy 3x weekly (10/15)", "qme": "not specified", "ur_decision": "not specified", "legal": "not specified", "raf": "not specified", "recommendations": "Follow up in 4 weeks (10/15)", "outcome": "not specified"}}

EXAMPLES OF CORRECT BEHAVIOR:

Example 1 - Document mentions diagnosis and treatment only:
Input: "Patient diagnosed with lumbar strain. Prescribed ibuprofen 600mg TID. Report date 10/05/2024"
Output: {{
  "diagnosis": "Lumbar strain diagnosed (10/05)",
  "treatment_plan": "Ibuprofen 600mg three times daily (10/05)", 
  "qme": "not specified",
  "ur_decision": "not specified",
  "legal": "not specified",
  "raf": "not specified",
  "recommendations": "not specified",
  "outcome": "not specified"
}}

Example 2 - Document mentions UR decision only:
Input: "UR Decision: Request for MRI denied due to insufficient medical necessity. Date 10/12/2024"
Output: {{
  "diagnosis": "not specified",
  "treatment_plan": "not specified",
  "qme": "not specified", 
  "ur_decision": "MRI request denied, insufficient medical necessity (10/12)",
  "legal": "not specified",
  "raf": "not specified",
  "recommendations": "not specified",
  "outcome": "not specified"
}}

Example 3 - Document with multiple findings:
Input: "Patient seen for follow-up. Lower back pain improving with PT. Continue current treatment plan. Recommend MRI if no improvement. Date 10/20/2024"
Output: {{
  "diagnosis": "Lower back pain improving (10/20)",
  "treatment_plan": "Continue physical therapy (10/20)",
  "qme": "not specified",
  "ur_decision": "not specified",
  "legal": "not specified", 
  "raf": "not specified",
  "recommendations": "MRI recommended if no improvement (10/20)",
  "outcome": "Patient showing improvement (10/20)"
}}

REMEMBER:
- You MUST include ALL 8 categories in the output
- Use "not specified" for any category not mentioned in the document
- Only include what is ACTUALLY WRITTEN in the document text
- Be accurate and truthful - no hallucinations!

Now analyze the document above and extract information for ALL categories.

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
        Only includes categories that are not "not specified".
        """
        mm_dd = datetime.strptime(current_date, "%Y-%m-%d").strftime("%m/%d")
        category_mapping = {
            "diagnosis": "New diagnosis",
            "treatment_plan": "New treatment",
            "qme": "New consults",
            "raf": "New authorizations/denials",
            "ur_decision": "New authorizations/denials",
            "legal": "Other",
            "recommendations": "New recommendations",
            "outcome": "New outcomes"
        }
        
        highlights = []
        for internal_key, value in whats_new_dict.items():
            if not value or value.strip().lower() == "not specified":
                continue
            user_friendly_key = category_mapping.get(internal_key, "Other")
            # Ensure date is appended if missing
            if not any(d in value for d in ['(', '[']):
                value += f" ({mm_dd})"
            highlights.append(f"• **{user_friendly_key}**: {value}")
        
        if not highlights:
            highlights = ["• No new changes detected in this document."]
        
        return highlights