"""
Specialist Consult extractor with few-shot prompting and DoctorDetector integration.
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


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with few-shot prompting:
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
        logger.info("✅ ConsultExtractorChained initialized with few-shot prompting")

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
        consulting_physician = self._detect_consultant(text, page_zones)
        raw_result["physician_name"] = consulting_physician
        
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
        logger.info(f"🔍 Stage 1: Splitting Consult document into chunks (text length: {len(text)})")
        
        chunks = self.splitter.split_text(text)
        logger.info(f"📦 Created {len(chunks)} chunks for processing")
        
        # Few-shot examples for consultation extraction
        few_shot_examples = [
            {
                "input": """ORTHOPEDIC CONSULTATION
                DATE: 10/15/2024
                PATIENT: Right shoulder pain
                ASSESSMENT: Rotator cuff tendinopathy with impingement
                PLAN: Physical therapy 2x/week, subacromial injection
                WORK STATUS: Modified duty - no overhead lifting
                FOLLOW-UP: 6 weeks""",
                "output": {
                    "consult_date": "10/15/2024",
                    "specialty": "Ortho",
                    "body_part": "R shoulder",
                    "findings": "Rotator cuff tendinopathy with impingement",
                    "treatment_recommendations": "Physical therapy 2x/week, subacromial injection",
                    "recommendations": "Follow-up in 6 weeks",
                    "work_status": "Modified duty - no overhead lifting"
                }
            },
            {
                "input": """NEUROLOGY CONSULT
                DATE: 11/20/2024
                CHIEF COMPLAINT: Low back pain with radiculopathy
                IMPRESSION: L5 radiculopathy, likely discogenic
                RECOMMENDATIONS: MRI lumbar spine, gabapentin 300mg TID
                WORK STATUS: TTD, no bending/lifting
                FOLLOW-UP: After MRI results""",
                "output": {
                    "consult_date": "11/20/2024",
                    "specialty": "Neuro",
                    "body_part": "Lumbar spine",
                    "findings": "L5 radiculopathy, likely discogenic",
                    "treatment_recommendations": "Gabapentin 300mg TID",
                    "recommendations": "MRI lumbar spine, follow-up after results",
                    "work_status": "TTD, no bending/lifting"
                }
            },
            {
                "input": """PAIN MANAGEMENT CONSULTATION
                DATE: 12/05/2024
                EVALUATION: Chronic cervical pain
                DIAGNOSIS: Cervical spondylosis C5-C6
                TREATMENT: Cervical ESI recommended, continue NSAIDs
                WORK: Sedentary duty only
                PLAN: ESI scheduled for next week""",
                "output": {
                    "consult_date": "12/05/2024",
                    "specialty": "Pain",
                    "body_part": "Cervical spine",
                    "findings": "Cervical spondylosis C5-C6",
                    "treatment_recommendations": "Cervical ESI, continue NSAIDs",
                    "recommendations": "ESI scheduled for next week",
                    "work_status": "Sedentary duty only"
                }
            },
            {
                "input": """FOLLOW-UP CONSULT
                DATE: 01/10/2025
                STATUS: Improved with current treatment
                ASSESSMENT: Condition resolving
                RECOMMENDATIONS: Continue current regimen
                WORK: Released to full duty
                PLAN: Discharge to PCP""",
                "output": {
                    "consult_date": "01/10/2025",
                    "specialty": "",
                    "body_part": "",
                    "findings": "Condition resolving, improved with treatment",
                    "treatment_recommendations": "Continue current regimen",
                    "recommendations": "Discharge to PCP",
                    "work_status": "Released to full duty"
                }
            }
        ]

        # System prompt with extraction rules
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical consultation data extraction specialist. Extract structured clinical information from specialist consultation reports.

EXTRACTION RULES:
- Extract ONLY information present in the text
- Return empty string "" for missing fields
- Be precise and clinical in terminology
- For specialty: use abbreviated forms (Ortho, Neuro, Pain, PM&R, Psych, etc.)
- For findings: focus on key diagnostic impressions
- For treatments: include specific procedures, therapies, medications
- For work status: extract specific restrictions and limitations
- DO NOT extract physician names - handled separately

OUTPUT FORMAT: JSON with these exact fields:
- consult_date, specialty, body_part, findings
- treatment_recommendations, recommendations, work_status
""")

        # User prompt with few-shot examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW EXTRACT FROM THIS CONSULTATION TEXT:
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
            logger.error(f"❌ Consult few-shot extraction failed: {e}")
            return self._get_fallback_result(fallback_date)

    def _merge_partial_extractions(self, partials: List[Dict], fallback_date: str) -> Dict:
        """Merge extractions from multiple chunks."""
        if not partials:
            return self._get_fallback_result(fallback_date)
        
        merged = self._get_fallback_result(fallback_date)
        
        # String fields: take most complete value across chunks
        string_fields = [
            "consult_date", "specialty", "body_part", "findings",
            "treatment_recommendations", "recommendations", "work_status"
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
        if not merged["consult_date"]:
            merged["consult_date"] = fallback_date

        # Special handling for findings: combine complementary findings
        if len(partials) > 1:
            finding_candidates = []
            for partial in partials:
                candidate = partial.get("findings", "").strip()
                if candidate and candidate not in finding_candidates:
                    finding_candidates.append(candidate)
            
            if len(finding_candidates) > 1:
                combined = f"{finding_candidates[0]}; {finding_candidates[1][:50]}"
                if len(combined) <= 100:
                    merged["findings"] = combined

        logger.info(f"📊 Merge completed: specialty='{merged['specialty']}', findings='{merged['findings'][:50]}...'")
        return merged

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure."""
        return {
            "consult_date": fallback_date,
            "specialty": "",
            "body_part": "",
            "findings": "",
            "treatment_recommendations": "",
            "recommendations": "",
            "work_status": "",
        }

    def _detect_consultant(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Stage 2: Detect consultant using DoctorDetector."""
        logger.info("🔍 Stage 2: Running DoctorDetector...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Consultant detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid consultant found")
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
            document_date=cleaned.get("consult_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name", ""),
            specialty=cleaned.get("specialty"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (consultant: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data."""
        cleaned = {}
        
        # Date validation
        date = result.get("consult_date", "")
        cleaned["consult_date"] = date if date and date != "empty" else fallback_date

        # Physician (from DoctorDetector)
        physician = result.get("physician_name", "")
        cleaned["physician_name"] = physician.strip()

        # Specialty validation
        specialty = result.get("specialty", "")
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""

        # Body part validation
        body_part = result.get("body_part", "")
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        # String fields validation
        string_fields = [
            "findings",
            "treatment_recommendations",
            "recommendations",
            "work_status",
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
        
        # Few-shot examples for consultation summary generation
        summary_examples = [
            {
                "data": {
                    "date": "10/15/2024",
                    "physician": "Dr. Smith",
                    "specialty": "Ortho",
                    "body_part": "R shoulder",
                    "findings": "Rotator cuff tendinopathy with impingement",
                    "treatment": "Physical therapy 2x/week, subacromial injection",
                    "recommendations": "Follow-up in 6 weeks",
                    "work_status": "Modified duty - no overhead lifting"
                },
                "summary": "Orthopedic Consultation dated 10/15/2024 by Dr. Smith for right shoulder. Findings: rotator cuff tendinopathy with impingement. Treatment: physical therapy twice weekly and subacromial injection. Work status: modified duty with no overhead lifting. Plan: follow-up evaluation in 6 weeks to assess treatment response."
            },
            {
                "data": {
                    "date": "11/20/2024",
                    "physician": "Dr. Chen",
                    "specialty": "Neuro",
                    "body_part": "Lumbar spine",
                    "findings": "L5 radiculopathy, likely discogenic",
                    "treatment": "Gabapentin 300mg TID",
                    "recommendations": "MRI lumbar spine, follow-up after results",
                    "work_status": "TTD, no bending/lifting"
                },
                "summary": "Neurology Consultation dated 11/20/2024 by Dr. Chen for lumbar spine. Diagnosis: L5 radiculopathy likely discogenic. Treatment: gabapentin 300mg three times daily. Work restrictions: temporary total disability with no bending or lifting. Additional imaging with MRI recommended for further evaluation."
            },
            {
                "data": {
                    "date": "01/10/2025",
                    "physician": "",
                    "specialty": "",
                    "body_part": "",
                    "findings": "Condition resolving, improved with treatment",
                    "treatment": "Continue current regimen",
                    "recommendations": "Discharge to PCP",
                    "work_status": "Released to full duty"
                },
                "summary": "Follow-up Consultation dated 01/10/2025. Patient shows significant improvement with current treatment regimen. Condition resolving appropriately. Cleared for return to full duty work without restrictions. Plan: discharge to primary care physician for ongoing management with no further specialty follow-up needed."
            }
        ]

        # System prompt for summary generation
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical summarizer creating concise specialist consultation summaries for physicians.

RULES:
- Keep summary 50-60 words
- Focus on key findings and clinical recommendations
- Use professional medical terminology
- Include: date, specialty, body part, key findings, treatments, work status, follow-up plan
- Be concise but clinically accurate
- Highlight specialist recommendations and restrictions
- For discharge cases, clearly state resolution and clearance
""")

        # User prompt with data and examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW CREATE SUMMARY FROM THIS CONSULTATION DATA:
Date: {date}
Physician: {physician}
Specialty: {specialty}
Body Part: {body_part}
Findings: {findings}
Treatment: {treatment}
Recommendations: {recommendations}
Work Status: {work_status}

Create a professional 50-60 word consultation summary:
""")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])

        try:
            # Prepare data
            date = data.get("consult_date", fallback_date)
            physician = data.get("physician_name", "")
            specialty = data.get("specialty", "")
            body_part = data.get("body_part", "")
            findings = data.get("findings", "")
            treatment = data.get("treatment_recommendations", "")
            recommendations = data.get("recommendations", "")
            work_status = data.get("work_status", "")

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
                "specialty": specialty,
                "body_part": body_part,
                "findings": findings,
                "treatment": treatment,
                "recommendations": recommendations,
                "work_status": work_status
            })
            
            summary = response.content.strip()
            
            # Ensure appropriate length
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:65]) + "..."
            elif len(words) < 45:
                # Add context if too short
                if body_part and not findings:
                    summary += f" {body_part} condition assessed."
                if not summary.endswith('.'):
                    summary += '.'
            
            logger.info(f"📊 Generated consultation summary: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Consultation summary generation failed: {e}")
            return self._build_manual_summary(data, doc_type, fallback_date)

    def _build_manual_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Fallback manual summary construction."""
        date = data.get("consult_date", fallback_date)
        physician = data.get("physician_name", "")
        specialty = data.get("specialty", "")
        body_part = data.get("body_part", "")
        findings = data.get("findings", "")
        treatment = data.get("treatment_recommendations", "")
        recommendations = data.get("recommendations", "")
        work_status = data.get("work_status", "")

        parts = [f"Consultation Report dated {date}"]
        
        if physician:
            parts.append(f"by {physician}")
        
        if specialty:
            parts.append(f"({specialty})")

        if body_part:
            parts.append(f"for {body_part}")

        if findings:
            parts.append(f"Findings: {findings[:50]}")

        if treatment:
            parts.append(f"Treatment: {treatment[:40]}")

        if work_status:
            parts.append(f"Work: {work_status[:30]}")

        if recommendations:
            parts.append(f"Plan: {recommendations[:30]}")

        summary = " ".join(parts)
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:65]) + "..."
        
        return summary

    def _extract_physician_last_name(self, physician_name: str) -> str:
        """Extract last name from physician name string."""
        if not physician_name:
            return ""

        # Remove common titles and suffixes
        clean_name = (
            physician_name
            .replace("Dr.", "")
            .replace("MD", "")
            .replace("DO", "")
            .replace("M.D.", "")
            .replace("D.O.", "")
            .replace("MBBS", "")
            .replace("MBChB", "")
            .replace(",", "")
            .strip()
        )

        # Get the last word as last name
        parts = clean_name.split()
        if parts:
            last_name = parts[-1]
            logger.info(f"  🔍 Extracted last name: '{last_name}' from '{physician_name}'")
            return last_name
        return ""