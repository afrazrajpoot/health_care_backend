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
    You are a medical document analyst. Your task is to EXTRACT key medical information using KEYWORDS and SHORT PHRASES from the document text.

    DOCUMENT TEXT TO ANALYZE:
    {current_raw_text}

    CRITICAL RULES - READ CAREFULLY:
    1. **EXTRACT ONLY** information that is ACTUALLY PRESENT in the document text
    2. **USE KEYWORDS AND SHORT PHRASES** - no full sentences
    3. **BE CONCISE** - maximum 5-7 words per bullet point
    4. **USE MEDICAL TERMINOLOGY** found in the document
    5. **FOCUS ON NEW/CHANGED INFORMATION** since last visit

    KEY CATEGORIES TO EXTRACT (IF PRESENT):
    - **Diagnosis**: Conditions, findings, test results
    - **Symptoms**: Pain, limitations, functional issues  
    - **Treatment**: Medications, therapies, procedures
    - **Imaging**: MRI, X-ray, CT findings
    - **Recommendations**: Next steps, referrals, follow-ups
    - **Work Status**: Disability, restrictions, modifications
    - **UR Decisions**: Approvals, denials, authorizations
    - **Progress**: Improvements, deteriorations, changes

    OUTPUT REQUIREMENTS:
    - Return JSON with "bullet_points" array
    - Each bullet point should start with "•"
    - Use keyword format: "• Category: keyword1, keyword2, keyword3 ({mm_dd})"
    - Maximum 6-8 bullet points, focus on most clinically relevant
    - If no significant findings: ["• No significant new findings"]

    EXAMPLES OF KEYWORD FORMAT:

    Example 1 - Document with findings:
    Document: "Patient diagnosed with lumbar radiculopathy. MRI shows L4-L5 disc herniation. Prescribed physical therapy 3x weekly and naproxen 500mg BID. Follow up in 6 weeks."
    Output: {{
    "bullet_points": [
        "• Diagnosis: Lumbar radiculopathy, L4-L5 disc herniation ({mm_dd})",
        "• Treatment: Physical therapy 3x/week, naproxen 500mg BID ({mm_dd})",
        "• Follow-up: 6 weeks ({mm_dd})"
    ]
    }}

    Example 2 - Document with symptoms and imaging:
    Document: "Patient reports severe low back pain radiating to left leg, numbness in left foot. MRI shows L4-L5 disc protrusion compressing L5 nerve root."
    Output: {{
    "bullet_points": [
        "• Symptoms: Severe LBP, left leg radiation, left foot numbness ({mm_dd})",
        "• Imaging: L4-L5 disc protrusion, L5 nerve root compression ({mm_dd})"
    ]
    }}

    Example 3 - Treatment authorization:
    Document: "UR Decision: Request for epidural steroid injection approved. Physical therapy authorized for 12 visits."
    Output: {{
    "bullet_points": [
        "• UR Approved: Epidural steroid injection ({mm_dd})",
        "• Treatment: Physical therapy - 12 visits ({mm_dd})"
    ]
    }}

    Example 4 - Work status update:
    Document: "Work status: Temporary Partial Disability. Restrictions: no lifting >10 lbs, avoid repetitive bending."
    Output: {{
    "bullet_points": [
        "• Work Status: TPD ({mm_dd})",
        "• Restrictions: No lifting >10 lbs, avoid bending ({mm_dd})"
    ]
    }}

    Example 5 - Minimal findings:
    Document: "Routine follow-up. Mild improvement in symptoms. Continue current treatment."
    Output: {{
    "bullet_points": [
        "• Status: Mild symptomatic improvement ({mm_dd})",
        "• Plan: Continue current treatment ({mm_dd})"
    ]
    }}

    NOW EXTRACT FROM THE PROVIDED DOCUMENT:
    - Scan the document for key medical information
    - Extract using keywords and short phrases only
    - Group related information together
    - Focus on what's new or changed
    - Be concise and clinical

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