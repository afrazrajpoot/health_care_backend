"""
Imaging reports extractor with few-shot prompting and DoctorDetector integration.
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


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with few-shot prompting:
    - Stage 1: Extract imaging data using chunked processing with few-shot examples
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
        logger.info("✅ ImagingExtractorChained initialized with few-shot prompting")

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
        logger.info(f"🔍 Starting extraction for {doc_type} report")
        
        # Stage 1: Extract imaging data using few-shot chunked processing
        raw_result = self._extract_clinical_data(text, doc_type, fallback_date)
        
        # Stage 2: Doctor detection via DoctorDetector
        radiologist_name = self._detect_radiologist(text, page_zones)
        raw_result["consulting_doctor"] = radiologist_name
        
        # Stage 3: Build initial result with professional summary
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        verified_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info(f"✅ Extraction completed for {doc_type}")
        return verified_result

    def _extract_clinical_data(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Stage 1: Extract imaging data using few-shot chunked processing.
        """
        logger.info(f"🔍 Stage 1: Splitting {doc_type} document into chunks (text length: {len(text)})")
        
        chunks = self.splitter.split_text(text)
        logger.info(f"📦 Created {len(chunks)} chunks for processing")
        
        # Few-shot examples for imaging extraction
        few_shot_examples = [
            {
                "input": """MRI RIGHT SHOULDER WITHOUT CONTRAST
                INDICATION: Shoulder pain
                FINDINGS: There is a full-thickness tear of the supraspinatus tendon.
                IMPRESSION: Full-thickness rotator cuff tear.""",
                "output": {
                    "study_date": "10/15/2024",
                    "document_type": "MRI",
                    "body_part": "R shoulder",
                    "contrast_used": "without contrast",
                    "primary_finding": "Full-thickness rotator cuff tear",
                    "impression_status": "abnormal"
                }
            },
            {
                "input": """CT LUMBAR SPINE WITH CONTRAST
                CLINICAL HISTORY: Low back pain
                FINDINGS: Mild degenerative changes at L4-L5. No acute fracture.
                IMPRESSION: Mild degenerative disc disease.""",
                "output": {
                    "study_date": "11/20/2024",
                    "document_type": "CT",
                    "body_part": "Lumbar spine",
                    "contrast_used": "with contrast",
                    "primary_finding": "Mild degenerative disc disease",
                    "impression_status": "abnormal"
                }
            },
            {
                "input": """X-RAY RIGHT KNEE
                INDICATION: Knee pain after fall
                FINDINGS: No fracture or dislocation. Joint spaces preserved.
                IMPRESSION: No acute bony abnormality.""",
                "output": {
                    "study_date": "12/05/2024",
                    "document_type": "X-ray",
                    "body_part": "R knee",
                    "contrast_used": "",
                    "primary_finding": "No acute bony abnormality",
                    "impression_status": "normal"
                }
            },
            {
                "input": """ULTRASOUND LEFT WRIST
                CLINICAL: Carpal tunnel symptoms
                FINDINGS: Moderate median nerve enlargement.
                IMPRESSION: Findings consistent with carpal tunnel syndrome.""",
                "output": {
                    "study_date": "01/10/2025",
                    "document_type": "Ultrasound",
                    "body_part": "L wrist",
                    "contrast_used": "",
                    "primary_finding": "Findings consistent with carpal tunnel syndrome",
                    "impression_status": "abnormal"
                }
            }
        ]

        # System prompt with extraction rules
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical imaging data extraction specialist. Extract structured clinical information from imaging reports.

EXTRACTION RULES:
- Extract ONLY information present in the text
- Return empty string "" for missing fields
- Be precise and clinical in terminology
- For findings: focus on clinically significant abnormalities
- For normal studies: use "normal" or "no acute findings"
- For contrast: specify "with contrast" or "without contrast" when stated
- For impression_status: use "normal", "abnormal", "post-op", or "inconclusive"
- DO NOT extract physician names - handled separately

OUTPUT FORMAT: JSON with these exact fields:
- study_date, document_type, body_part, contrast_used
- primary_finding, impression_status
""")

        # User prompt with few-shot examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW EXTRACT FROM THIS IMAGING REPORT TEXT:
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
            logger.info(f"  - study_date: {merged_result.get('study_date', 'Not found')}")
            logger.info(f"  - body_part: {merged_result.get('body_part', 'Not found')}")
            logger.info(f"  - primary_finding: {merged_result.get('primary_finding', 'Not found')}")
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ Imaging few-shot extraction failed: {e}")
            return self._get_fallback_result(fallback_date)

    def _merge_partial_extractions(self, partials: List[Dict], fallback_date: str) -> Dict:
        """Merge extractions from multiple chunks."""
        if not partials:
            return self._get_fallback_result(fallback_date)
        
        merged = self._get_fallback_result(fallback_date)
        
        # String fields: take most complete value across chunks
        string_fields = [
            "study_date", "document_type", "body_part", "contrast_used",
            "primary_finding", "impression_status"
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
        if not merged["study_date"]:
            merged["study_date"] = fallback_date

        # Special handling for findings: combine complementary findings
        if len(partials) > 1:
            finding_candidates = []
            for partial in partials:
                candidate = partial.get("primary_finding", "").strip()
                if candidate and candidate not in finding_candidates:
                    finding_candidates.append(candidate)
            
            if len(finding_candidates) > 1:
                combined = f"{finding_candidates[0]}; {finding_candidates[1][:50]}"
                if len(combined) <= 100:
                    merged["primary_finding"] = combined

        logger.info(f"📊 Merge completed: body_part='{merged['body_part']}', finding='{merged['primary_finding'][:50]}...'")
        return merged

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure."""
        return {
            "study_date": fallback_date,
            "document_type": "",
            "body_part": "",
            "contrast_used": "",
            "primary_finding": "",
            "impression_status": "",
        }

    def _detect_radiologist(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Stage 2: Detect radiologist using DoctorDetector."""
        logger.info("🔍 Stage 2: Running DoctorDetector...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Radiologist detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid radiologist found")
            return ""

    def _build_initial_result(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> ExtractionResult:
        """Stage 3: Build initial result with professional summary."""
        logger.info("🎯 Stage 3: Building initial result with validation")
        
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_professional_summary(cleaned, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("study_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("consulting_doctor", ""),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (radiologist: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data."""
        cleaned = {}
        
        # Date validation
        date = result.get("study_date", "")
        cleaned["study_date"] = date if date and date != "empty" else fallback_date
        logger.info(f"  📅 Date cleaned: {cleaned['study_date']}")

        # Radiologist (from DoctorDetector)
        radiologist = result.get("consulting_doctor", "")
        cleaned["consulting_doctor"] = radiologist.strip()
        logger.info(f"  👨‍⚕️ Radiologist: {radiologist if radiologist else 'None'}")

        # Body part validation
        body_part = result.get("body_part", "")
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""
        logger.info(f"  🦴 Body Part: {cleaned['body_part']}")

        # Primary finding validation
        primary_finding = result.get("primary_finding", "")
        cleaned["primary_finding"] = primary_finding if primary_finding and primary_finding != "empty" else ""
        logger.info(f"  🔍 Primary Finding: {cleaned['primary_finding']}")

        # Contrast validation
        contrast = result.get("contrast_used", "")
        cleaned["contrast_used"] = contrast if contrast and contrast != "empty" else ""
        logger.info(f"  💉 Contrast: {cleaned['contrast_used']}")

        # Impression status validation
        status = result.get("impression_status", "")
        cleaned["impression_status"] = status if status and status != "empty" else ""
        logger.info(f"  📊 Status: {cleaned['impression_status']}")

        return cleaned

    def _build_professional_summary(
        self,
        data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """Build professional 50-60 word summary using few-shot approach."""
        
        # Few-shot examples for imaging summary generation
        summary_examples = [
            {
                "data": {
                    "date": "10/15/2024",
                    "physician": "Dr. Smith",
                    "doc_type": "MRI",
                    "body_part": "R shoulder",
                    "contrast": "without contrast",
                    "status": "abnormal",
                    "finding": "Full-thickness rotator cuff tear"
                },
                "summary": "MRI Report dated 10/15/2024 by Dr. Smith for right shoulder without contrast. Impression: abnormal. Findings: full-thickness rotator cuff tear identified. This represents a significant tendon injury requiring orthopedic evaluation and potential surgical consideration for symptomatic patients."
            },
            {
                "data": {
                    "date": "11/20/2024",
                    "physician": "",
                    "doc_type": "X-ray",
                    "body_part": "R knee",
                    "contrast": "",
                    "status": "normal",
                    "finding": "No acute bony abnormality"
                },
                "summary": "X-ray Report dated 11/20/2024 for right knee. Impression: normal. Findings: no acute fracture or dislocation identified, joint spaces preserved. The study demonstrates no evidence of acute traumatic injury or significant degenerative changes at this time."
            },
            {
                "data": {
                    "date": "12/05/2024",
                    "physician": "Dr. Chen",
                    "doc_type": "CT",
                    "body_part": "Lumbar spine",
                    "contrast": "with contrast",
                    "status": "abnormal",
                    "finding": "Mild degenerative disc disease L4-L5"
                },
                "summary": "CT Report dated 12/05/2024 by Dr. Chen for lumbar spine with contrast. Impression: abnormal. Findings: mild degenerative disc disease at L4-L5 level. These age-appropriate changes may contribute to mechanical back pain but show no evidence of acute disc herniation or neural compression."
            }
        ]

        # System prompt for summary generation
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical imaging summarizer creating concise imaging report summaries for physicians.

RULES:
- Keep summary 50-60 words
- Focus on key findings and clinical significance
- Use professional radiology terminology
- Include: modality, date, body part, contrast, impression status, key findings
- Be concise but clinically accurate
- Highlight clinically significant abnormalities
- For normal studies, clearly state absence of pathology
""")

        # User prompt with data and examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW CREATE SUMMARY FROM THIS IMAGING DATA:
Date: {date}
Physician: {physician}
Modality: {doc_type}
Body Part: {body_part}
Contrast: {contrast}
Status: {status}
Finding: {finding}

Create a professional 50-60 word imaging summary:
""")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])

        try:
            # Prepare data
            date = data.get("study_date", fallback_date)
            physician = data.get("consulting_doctor", "")
            body_part = data.get("body_part", "")
            contrast = data.get("contrast_used", "")
            status = data.get("impression_status", "")
            finding = data.get("primary_finding", "")

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
                "doc_type": doc_type,
                "body_part": body_part,
                "contrast": contrast,
                "status": status,
                "finding": finding
            })
            
            summary = response.content.strip()
            
            # Ensure appropriate length
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:65]) + "..."
            elif len(words) < 45:
                # Add context if too short
                if body_part and not status:
                    summary += f" {body_part} evaluation completed."
                if not summary.endswith('.'):
                    summary += '.'
            
            logger.info(f"📊 Generated imaging summary: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Imaging summary generation failed: {e}")
            return self._build_manual_summary(data, doc_type, fallback_date)

    def _build_manual_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Fallback manual summary construction."""
        date = data.get("study_date", fallback_date)
        physician = data.get("consulting_doctor", "")
        body_part = data.get("body_part", "")
        contrast = data.get("contrast_used", "")
        status = data.get("impression_status", "")
        finding = data.get("primary_finding", "")

        parts = [f"{doc_type} Report dated {date}"]
        
        if physician:
            parts.append(f"by {physician}")
        
        if body_part:
            body_str = f"for {body_part}"
            if contrast:
                body_str += f" ({contrast})"
            parts.append(body_str)

        if status:
            parts.append(f"Impression: {status}")

        if finding:
            parts.append(f"Findings: {finding[:50]}")

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