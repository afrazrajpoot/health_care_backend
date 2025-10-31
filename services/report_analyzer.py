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
        self.bullet_parser = JsonOutputParser()

    def compare_with_previous_documents(
        self, 
        current_raw_text: str,
     
    ) -> List[str]:
        """
        Use LLM to extract key findings directly from document text and format as bullet points.
        Only extracts information explicitly mentioned in the text.
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        mm_dd = datetime.now().strftime("%m/%d")
        logger.info(f"DEBUG: Processing raw text length: {len(current_raw_text)}")
          
        # Current as raw text (truncate if too long)
        max_text_len = 8000
        current_raw_truncated = current_raw_text[:max_text_len] + "..." if len(current_raw_text) > max_text_len else current_raw_text
        
        try:
            prompt = self.create_extraction_prompt()
            chain = prompt | self.llm | self.bullet_parser
            raw_result = chain.invoke({
                "current_raw_text": current_raw_truncated,
                "mm_dd": mm_dd
            })
            
            # Parse and validate result
            if isinstance(raw_result, str):
                try:
                    result = json.loads(raw_result)
                except json.JSONDecodeError as je:
                    logger.error(f"❌ Invalid JSON from LLM: {raw_result[:200]}... Error: {str(je)}")
                    return self._fallback_bullet_points()
            else:
                result = raw_result
            
            # Extract bullet points from result
            bullet_points = self._extract_bullet_points(result)
            
            logger.info(f"✅ Generated {len(bullet_points)} bullet points from document")
            return bullet_points
            
        except Exception as e:
            logger.error(f"❌ AI extraction failed: {str(e)}")
            return self._fallback_bullet_points()

    def _extract_bullet_points(self, result: Dict) -> List[str]:
        """Extract bullet points from parsed result"""
        bullet_points = []
        
        if isinstance(result, dict) and "bullet_points" in result:
            bullets = result["bullet_points"]
            if isinstance(bullets, list):
                for bullet in bullets:
                    if bullet and bullet.strip() and bullet.strip().lower() != "no significant findings":
                        bullet_points.append(bullet.strip())
        
        if not bullet_points:
            bullet_points = self._fallback_bullet_points()
            
        return bullet_points

    def _fallback_bullet_points(self) -> List[str]:
        """Simple fallback without LLM."""
        return ["• No significant new findings identified in current document"]

    def create_extraction_prompt(self) -> PromptTemplate:
        template = """
    You are a medical document analyst. Your task is to ANALYZE the entire document text and generate a PRECISE SUMMARY in exactly 3 bullet points of NEW CHANGES SINCE LAST VISIT. Extract and summarize ONLY medical information explicitly present in the text—do not include any patient personal details (e.g., names, IDs, DOB, contact info) or non-medical data. Focus strictly on medical facts: diagnoses, symptoms, treatments, findings, progress, or decisions indicating changes.

    DOCUMENT TEXT TO ANALYZE:
    {current_raw_text}

    CRITICAL RULES - READ CAREFULLY:
    1. **EXCLUDE PERSONAL DATA**: No patient names, demographics, identifiers—only pure medical content
    2. **FOCUS ON NEW CHANGES**: Look for phrases like "since last visit", "improved", "worsened", "new", "continued", "updated"—summarize only what's changed or newly noted
    3. **ANALYZE ALL PARTS** of the document: Cover the whole document by grouping into 3 major medical changes or themes
    4. **SUMMARIZE PRECISELY** using keywords and short phrases—no full sentences, max 10-15 words per bullet
    5. **BE CONCISE AND CLINICAL**: Use medical terminology from the text; group related changes with semicolons if needed
    6. **INCLUDE DATES/LOCATIONS** only if they relate to medical changes (e.g., MM/DD/YY for test dates)
    7. **EXACTLY 3 BULLETS**: One bullet per major change/theme; prioritize most relevant new medical info to fit exactly 3 total
    8. Each bullet starts with "•"
    9. If no new changes since last visit: ["• No significant new changes since last visit"]

    OUTPUT REQUIREMENTS:
    - Return ONLY JSON with "bullet_points" array (exactly 3 items)
    - Bullets should capture the essence of each new medical change concisely, covering the whole document

    EXAMPLE OF SUMMARY STYLE (for format only—do NOT use this content; derive solely from {current_raw_text}):
    {{
    "bullet_points": [
        "• Symptoms: mild improvement in low back pain since last visit",
        "• Treatment: added naproxen 500mg BID; continued PT",
        "• Findings: MRI shows reduced inflammation; follow-up advanced to 4 weeks"
    ]
    }}

    NOW ANALYZE AND SUMMARIZE THE PROVIDED DOCUMENT:
    - Read the full text carefully, ignoring non-medical parts
    - Break down the entire document into exactly 3 key new medical changes
    - Generate precise bullet points covering all major aspects
    - Ensure everything is directly from the text and medical-only

    {format_instructions}
    """
        return PromptTemplate(
            template=template,
            input_variables=["current_raw_text", "mm_dd"],
            partial_variables={
                "format_instructions": self.bullet_parser.get_format_instructions()
            },
        )
    def format_whats_new_as_highlights(self, bullet_points: List[str]) -> List[str]:
        """
        Formats the bullet points for display (minimal processing since LLM already formatted them).
        """
        if not bullet_points:
            return ["• No significant new findings identified in current document"]
        
        return bullet_points