"""
PR-2 Progress Report extractor with few-shot prompting and DoctorDetector integration.
"""
import logging
import json
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with few-shot prompting:
    - Stage 1: Extract clinical data using chunked processing with few-shot examples
    - Stage 2: Doctor detection via DoctorDetector
    - Stage 3: Build professional summary with few-shot guidance
    - Stage 4: Verify and correct
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        # Initialize recursive text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        logger.info("✅ PR2ExtractorChained initialized with few-shot prompting")

    def extract(
        self, 
        text: str, 
        doc_type: str, 
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract with few-shot prompting and DoctorDetector integration.
        """
        # Stage 1: Extract clinical data using few-shot chunked processing
        raw_result = self._extract_clinical_data(text, doc_type, fallback_date)
        
        # Stage 2: Doctor detection via DoctorDetector
        physician_name = self._detect_physician(text, page_zones)
        raw_result["physician_name"] = physician_name
        
        # Stage 3: Build initial result with professional summary
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        verified_result = self.verifier.verify_and_fix(initial_result)
        
        return verified_result

    def _extract_clinical_data(
        self, 
        text: str, 
        doc_type: str, 
        fallback_date: str
    ) -> Dict:
        """
        Stage 1: Extract clinical data using few-shot chunked processing.
        """
        logger.info(f"🔍 Stage 1: Splitting PR-2 document into chunks (text length: {len(text)})")
        
        chunks = self.splitter.split_text(text)
        logger.info(f"📦 Created {len(chunks)} chunks for processing")
        
        # Few-shot examples for PR-2 extraction
        few_shot_examples = [
            {
                "input": """PROGRESS NOTE: Patient with right shoulder pain. STATUS: Improved with PT. 
                TREATMENT: Continue PT 2x/week, add NSAIDs. WORK: Modified duty - no lifting >10 lbs. 
                PLAN: Follow-up in 4 weeks.""",
                "output": {
                    "report_date": "10/15/2024",
                    "body_part": "R shoulder",
                    "current_status": "Improved with PT",
                    "treatment_recommendations": "Continue PT 2x/week, add NSAIDs",
                    "work_status": "Modified duty - no lifting >10 lbs",
                    "next_plan": "Follow-up in 4 weeks"
                }
            },
            {
                "input": """PR-2 REPORT: Lumbar spine evaluation. STATUS: Stable, minimal pain. 
                TREATMENT: Continue home exercises. WORK: Full duty, no restrictions. 
                PLAN: PRN follow-up if symptoms worsen.""",
                "output": {
                    "report_date": "11/20/2024",
                    "body_part": "Lumbar spine",
                    "current_status": "Stable, minimal pain",
                    "treatment_recommendations": "Continue home exercises",
                    "work_status": "Full duty, no restrictions",
                    "next_plan": "PRN follow-up if symptoms worsen"
                }
            },
            {
                "input": """PROGRESS REPORT: Right knee post-op. STATUS: Worsened swelling. 
                TREATMENT: Ice, elevation, continue medications. WORK: TTD. 
                PLAN: Re-eval next week, consider imaging.""",
                "output": {
                    "report_date": "12/05/2024",
                    "body_part": "R knee",
                    "current_status": "Worsened swelling",
                    "treatment_recommendations": "Ice, elevation, continue medications",
                    "work_status": "TTD",
                    "next_plan": "Re-eval next week, consider imaging"
                }
            }
        ]

        # System prompt with extraction rules
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical data extraction specialist. Extract structured clinical information from PR-2 Progress Reports.

EXTRACTION RULES:
- Extract ONLY information present in the text
- Return empty string "" for missing fields
- Be precise and clinical in terminology
- For work status: extract specific restrictions and limitations
- For treatments: include specific procedures, therapies, medications
- For status: use terms like "improved", "stable", "worsened", "resolved"
- DO NOT extract physician names - handled separately

OUTPUT FORMAT: JSON with these exact fields:
- report_date, body_part, current_status
- treatment_recommendations, work_status, next_plan
""")

        # User prompt with few-shot examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW EXTRACT FROM THIS PR-2 TEXT:
{text}

Return valid JSON only.
""")

        # Create the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])

        try:
            partial_results = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                
                # Format few-shot examples as string
                examples_str = "\n\n".join([
                    f"Input: {ex['input']}\nOutput: {json.dumps(ex['output'])}" 
                    for ex in few_shot_examples
                ])
                
                chain = chat_prompt | self.llm | self.parser
                partial = chain.invoke({
                    "examples": examples_str,
                    "text": chunk
                })
                partial_results.append(partial)
                logger.debug(f"✅ Chunk {i+1} processed")
            
            # Merge partial extractions
            merged_result = self._merge_partial_extractions(partial_results, fallback_date)
            logger.info(f"✅ Stage 1: Few-shot extraction completed - {len(partial_results)} chunks merged")
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ PR-2 few-shot extraction failed: {e}")
            return self._get_fallback_result(fallback_date)

    def _merge_partial_extractions(self, partials: List[Dict], fallback_date: str) -> Dict:
        """Merge extractions from multiple chunks."""
        if not partials:
            return self._get_fallback_result(fallback_date)
        
        merged = self._get_fallback_result(fallback_date)
        
        # String fields: take most complete value across chunks
        string_fields = [
            "report_date", "body_part", "current_status",
            "treatment_recommendations", "work_status", "next_plan"
        ]
        
        for field in string_fields:
            candidates = []
            for partial in partials:
                value = partial.get(field, "")
                if isinstance(value, str) and value.strip() and value.strip() != fallback_date:
                    candidates.append(value.strip())
            
            if candidates:
                candidates.sort(key=len, reverse=True)
                merged[field] = candidates[0]

        # Special handling for date
        if not merged["report_date"]:
            merged["report_date"] = fallback_date

        logger.info(f"📊 Merge completed: body_part='{merged['body_part']}', status='{merged['current_status']}'")
        return merged

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure."""
        return {
            "report_date": fallback_date,
            "body_part": "",
            "current_status": "",
            "treatment_recommendations": "",
            "work_status": "",
            "next_plan": "",
        }

    def _detect_physician(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Stage 2: Detect physician using DoctorDetector."""
        logger.info("🔍 Stage 2: Running DoctorDetector...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Physician detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid physician found")
            return ""

    def _build_initial_result(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> ExtractionResult:
        """Stage 3: Build initial result with professional summary."""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_professional_summary(cleaned, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name", ""),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (physician: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data."""
        cleaned = {}
        
        # Date validation
        date = result.get("report_date", "")
        cleaned["report_date"] = date if date and date != "empty" else fallback_date

        # Physician (from DoctorDetector)
        physician = result.get("physician_name", "")
        cleaned["physician_name"] = physician.strip()

        # Body part validation
        body_part = result.get("body_part", "")
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        # String fields validation
        string_fields = [
            "current_status",
            "treatment_recommendations",
            "work_status",
            "next_plan",
        ]
        
        negative_phrases = [
            "no treatment", "no changes", "no new", "no follow-up",
            "fully resolved", "resolved", "no restrictions",
            "released to full duty", "full duty", "unrestricted",
            "not indicated", "not recommended", "not needed"
        ]
        
        for field in string_fields:
            raw_value = result.get(field, "")
            
            if isinstance(raw_value, list):
                v = ", ".join([str(item).strip() for item in raw_value if item])
            elif isinstance(raw_value, str):
                v = raw_value.strip()
            else:
                v = str(raw_value).strip() if raw_value else ""
            
            v_lower = v.lower()
            
            if not v or v_lower in ["", "empty", "none", "n/a", "not mentioned"]:
                cleaned[field] = ""
                continue
            
            if any(neg_phrase in v_lower for neg_phrase in negative_phrases):
                cleaned[field] = ""
                continue
            
            cleaned[field] = v

        return cleaned

    def _build_professional_summary(
        self,
        data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """Build professional 50-60 word summary using few-shot approach."""
        
        # Few-shot examples for summary generation
        summary_examples = [
            {
                "data": {
                    "date": "10/15/2024",
                    "physician": "Dr. Smith",
                    "body_part": "R shoulder",
                    "status": "Improved with PT",
                    "treatment": "Continue PT 2x/week, add NSAIDs",
                    "work_status": "Modified duty - no lifting >10 lbs",
                    "next_plan": "Follow-up in 4 weeks"
                },
                "summary": "PR-2 Progress Report dated 10/15/2024 by Dr. Smith for R shoulder. Clinical status: Improved with PT. Work status: Modified duty with no lifting over 10 lbs. Treatment: Continue PT 2x/week and add NSAIDs. Plan: Follow-up evaluation in 4 weeks to assess progress."
            },
            {
                "data": {
                    "date": "11/20/2024",
                    "physician": "",
                    "body_part": "Lumbar spine",
                    "status": "Stable, minimal pain",
                    "treatment": "Continue home exercises",
                    "work_status": "Full duty, no restrictions",
                    "next_plan": "PRN follow-up"
                },
                "summary": "PR-2 Progress Report dated 11/20/2024 for lumbar spine. Patient reports stable condition with minimal pain. Cleared for full duty work without restrictions. Treatment plan: continue home exercise program. Follow-up as needed if symptoms change."
            }
        ]

        # System prompt for summary generation
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical summarizer creating concise PR-2 Progress Report summaries for physicians.

RULES:
- Keep summary 50-60 words
- Focus on clinical progress and recommendations
- Use professional medical language
- Include: date, body part, status, key treatments, work status, follow-up plan
- Be concise but informative
- Maintain clinical accuracy
""")

        # User prompt with data and examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW CREATE SUMMARY FROM THIS DATA:
Date: {date}
Physician: {physician}
Body Part: {body_part}
Status: {status}
Treatment: {treatment}
Work Status: {work_status}
Next Plan: {next_plan}

Create a professional 50-60 word summary:
""")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])

        try:
            # Prepare data
            date = data.get("report_date", fallback_date)
            physician = data.get("physician_name", "")
            body_part = data.get("body_part", "")
            status = data.get("current_status", "")
            treatment = data.get("treatment_recommendations", "")
            work_status = data.get("work_status", "")
            next_plan = data.get("next_plan", "")

            # Format few-shot examples
            examples_str = "\n\n".join([
                f"Data: {json.dumps(ex['data'])}\nSummary: {ex['summary']}" 
                for ex in summary_examples
            ])

            chain = chat_prompt | self.llm
            response = chain.invoke({
                "examples": examples_str,
                "date": date,
                "physician": physician,
                "body_part": body_part,
                "status": status,
                "treatment": treatment,
                "work_status": work_status,
                "next_plan": next_plan
            })
            
            summary = response.content.strip()
            
            # Ensure appropriate length
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:65]) + "..."
            elif len(words) < 45:
                # Add context if too short
                if body_part and not status:
                    summary += f" {body_part} condition reviewed."
                if not summary.endswith('.'):
                    summary += '.'
            
            logger.info(f"📊 Generated PR-2 summary: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"❌ PR-2 summary generation failed: {e}")
            return self._build_manual_summary(data, doc_type, fallback_date)

    def _build_manual_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Fallback manual summary construction."""
        date = data.get("report_date", fallback_date)
        physician = data.get("physician_name", "")
        body_part = data.get("body_part", "")
        status = data.get("current_status", "")
        treatment = data.get("treatment_recommendations", "")
        work_status = data.get("work_status", "")
        next_plan = data.get("next_plan", "")

        parts = [f"PR-2 Progress Report dated {date}"]
        
        if physician:
            parts.append(f"by {physician}")
        
        if body_part:
            parts.append(f"for {body_part}")

        if status:
            parts.append(f"Status: {status}")

        if work_status:
            parts.append(f"Work: {work_status[:40]}")

        if treatment:
            parts.append(f"Treatment: {treatment[:40]}")

        if next_plan:
            parts.append(f"Plan: {next_plan[:30]}")

        summary = " ".join(parts)
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:65]) + "..."
        
        return summary