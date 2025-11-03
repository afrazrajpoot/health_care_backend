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
            You are a medical document extractor for our AI Healthcare Platform. 
            Your job is to extract ONLY the explicitly mentioned key information from the text below 
            and summarize it into concise one-line or two-line bullet summaries following the Kebilo Categorization Map (v2).

            🧠 GOAL:
            Identify the document type and extract ONLY the required fields per its category.
            Each output must be a short, data-rich summary (≤15 words) following this pattern:

            [DOCUMENT TYPE or PHYSICIAN/SENDER] [BODY PART or TOPIC] [DATE] = [KEY FINDING, ACTION, or DECISION]

            ---

            📄 DOCUMENT TEXT:
            {current_raw_text}

            ---

            ⚙️ EXTRACTION RULES (STRICT):

            1️⃣ Detect document type:
            - Clinical: MRI, CT, X-ray, Ultrasound, EMG, Labs  
            - Progress/Follow-up: PR-2  
            - Consult/Specialist: Consult, Ortho, Neuro, Pain Mgmt  
            - Initial Evaluation: DFR  
            - Final/Impairment: PR-4  
            - Authorization Workflow: RFA, UR, Authorization, Peer-to-Peer, IMR  
            - Administrative: Adjuster, Attorney, NCM, Signature/Fax requests  
            - Med-Legal: QME, AME, IME  
            - General Medicine: Office/PCP visit, New/Annual/Wellness, Specialist Consult (non-WC), ED/Urgent Care, Hospital Discharge, Imaging, Labs/Path/Screening, Pharmacy/PA, External Facility/Home Health/DME, Care Manager/CCM, Patient Message, or Referral


            2️⃣ Extract ONLY the following fields depending on document type:

            - **Clinical** → body part, date, key finding/severity  
            Example: MRI R shoulder 5/10/25 = partial rotator cuff tear  

            - **PR-2** → date, physician, body part, status, next plan  
            Example: Dr Calhoun PR-2 11/1/25 = R knee improved; continue PT; RFA pending  

            - **Consult** → date, physician, specialty, plan/recommendation  
            Example: Dr Johnson Ortho 9/12/25 = PT + ESI plan; f/u 6w  

            - **DFR** → DOI, diagnosis, plan/work status  
            Example: DFR 8/3/25 = DOI 7/30/25; R ankle sprain; PT + brace  

            - **PR-4** → date, MMI status, plan  
            Example: PR-4 10/5/25 = MMI; ongoing PT  

            - **RFA** → date, service requested, body part  
            Example: RFA 9/1/25 = PT 6v R knee requested  

            - **UR** → date, service denied, reason  
            Example: UR 9/8/25 = PT denied; no functional improvement  

            - **Authorization** → date, service approved, body part  
            Example: Auth 9/25/25 = MRI L shoulder approved  

            - **Administrative Letters** → date, sender or side, core request/issue  
            Example: Adjuster 9/10/25 = request latest PR-2 + MRI report  

            - **QME / IME / AME** → date, physician/specialty, recommendations or plan  
            Example: Dr Smith Ortho QME 9/10/25 = PT + ortho f/u; meds PRN  

            - **General Medicine** → date, purpose or issue  
            Example: Referral 10/1/25 = ortho eval for R knee pain  

            3️⃣ Do NOT infer, guess, or expand missing information.
            Only extract what is **explicitly stated**.

            4️⃣ Use date format MM/DD.  
            If no explicit date exists, use: {mm_dd}

            5️⃣ Ignore:
            - Patient names, ages, vitals, complaints, ROS, disclaimers, signatures, addresses, greetings.  
            - Historical narrative or unrelated paragraphs.

            6️⃣ OUTPUT FORMAT:
            Return ONLY JSON:
            {{
            "bullet_points": [
                "• [concise summary 1]",
                "• [concise summary 2]",
                "• [concise summary 3]"
            ]
            }}

            ❌ DO NOT:
            - Add explanations, section names, or headings
            - Include inferred or assumed data
            - Write paragraphs or multi-line sentences

            ✅ EXAMPLES OF CORRECT OUTPUT:
            {{
            "bullet_points": [
                "• MRI L knee 5/10/25 = medial meniscus tear",
                "• Dr Johnson Ortho 9/12/25 = PT + ESI plan; f/u 6w",
                "• UR 9/8/25 = PT denied; no functional improvement"
            ]
            }}

            Now extract and summarize according to the Kebilo v2 categorization.

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