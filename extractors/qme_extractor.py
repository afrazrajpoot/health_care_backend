"""
QME/AME/IME specialized extractor with few-shot prompting and chunked processing
"""
import logging
import re
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


class QMEExtractorChained:
    """
    Enhanced QME extractor with few-shot prompting and chunked processing:
    Stage 1: Extract raw data from full document via chunked processing
    Stage 2: Detect examiner via DoctorDetector (zone-aware)
    Stage 3: Build professional 50-60 word summary
    Stage 4: Verify and correct
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        # Initialize recursive text splitter for large documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        logger.info("✅ QMEExtractorChained initialized with few-shot prompting")

    def extract(
        self, 
        text: str, 
        doc_type: str, 
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract with DoctorDetector integration and verification chain.
        """
        if page_zones:
            logger.info(f"✅ QME extractor received page_zones with {len(page_zones)} pages")
        else:
            logger.warning("⚠️ QME extractor did NOT receive page_zones")
        
        # Stage 1: Extract clinical data using chunked processing
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        
        # Stage 2: Detect examiner via DoctorDetector
        examiner_name = self._detect_examiner(text, page_zones)
        raw_result["examiner_name"] = examiner_name
        
        # Stage 3: Build initial result with professional summary
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        return final_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data using chunked processing with few-shot prompts"""
        logger.info(f"🔍 Stage 1: Splitting document into chunks (text length: {len(text)})")
        
        chunks = self.splitter.split_text(text)
        logger.info(f"📦 Created {len(chunks)} chunks for processing")
        
        # Few-shot examples for better prompting
        few_shot_examples = [
            {
                "input": """QME REPORT: Patient with right shoulder pain. EXAM: Right shoulder tenderness. 
                DIAGNOSIS: Rotator cuff tear. MMI: Reached. IMPAIRMENT: 15% WPI. 
                WORK STATUS: No overhead lifting. TREATMENT: PT 2x/week.""",
                "output": {
                    "document_date": "10/15/2024",
                    "examiner_name": "Dr. John Smith",
                    "referral_physician": "",
                    "specialty": "Ortho",
                    "body_parts_evaluated": ["R shoulder"],
                    "diagnoses_confirmed": ["Rotator cuff tear"],
                    "MMI_status": "MMI reached",
                    "impairment_summary": "15% WPI",
                    "causation_opinion": "",
                    "treatment_recommendations": "PT 2x/week",
                    "medication_recommendations": "",
                    "work_restrictions": "No overhead lifting",
                    "future_medical_recommendations": ""
                }
            },
            {
                "input": """AME EVALUATION: Lumbar spine injury. FINDINGS: L4-5 disc herniation. 
                MMI: Not reached. IMPAIRMENT: Deferred. WORK: TTD. TREATMENT: ESI recommended.""",
                "output": {
                    "document_date": "11/20/2024",
                    "examiner_name": "Dr. Sarah Chen, MD",
                    "referral_physician": "",
                    "specialty": "Pain Management",
                    "body_parts_evaluated": ["Lumbar spine"],
                    "diagnoses_confirmed": ["L4-5 disc herniation"],
                    "MMI_status": "MMI not reached",
                    "impairment_summary": "Deferred",
                    "causation_opinion": "",
                    "treatment_recommendations": "ESI recommended",
                    "medication_recommendations": "",
                    "work_restrictions": "TTD",
                    "future_medical_recommendations": ""
                }
            }
        ]

        # System prompt with instructions
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical data extraction specialist. Extract structured clinical information from QME/AME/IME reports.

EXTRACTION RULES:
- Extract ONLY information present in the text
- Return empty string "" for missing fields
- Be precise and clinical
- For work restrictions: extract specific limitations
- For treatments: be specific about procedures
- For medications: include drug names and dosages

OUTPUT FORMAT: JSON with these exact fields:
- document_date, examiner_name, referral_physician, specialty
- body_parts_evaluated (list), diagnoses_confirmed (list)
- MMI_status, impairment_summary, causation_opinion
- treatment_recommendations, medication_recommendations
- work_restrictions, future_medical_recommendations
""")

        # User prompt with few-shot examples
        user_prompt = HumanMessagePromptTemplate.from_template("""
EXAMPLES:
{examples}

NOW EXTRACT FROM THIS TEXT:
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
            logger.info(f"✅ Chunked extraction completed: {len(partial_results)} chunks merged")
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ Chunked extraction failed: {e}")
            return self._get_fallback_result(fallback_date)

    def _merge_partial_extractions(self, partials: List[Dict], fallback_date: str) -> Dict:
        """Merge extractions from multiple chunks into a single comprehensive result."""
        if not partials:
            return self._get_fallback_result(fallback_date)
        
        merged = self._get_fallback_result(fallback_date)
        
        # List fields: union unique values across chunks
        list_fields = ["body_parts_evaluated", "diagnoses_confirmed"]
        for partial in partials:
            for field in list_fields:
                value = partial.get(field, [])
                if isinstance(value, list):
                    merged[field].extend([v.strip() for v in value if v and v.strip()])
                elif isinstance(value, str) and value.strip():
                    items = [v.strip() for v in value.split(",") if v.strip()]
                    merged[field].extend(items)
        
        # Deduplicate list fields
        for field in list_fields:
            seen = set()
            deduped = []
            for item in merged[field]:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            merged[field] = deduped

        # String fields: take most complete value
        string_fields = [
            "document_date", "examiner_name", "referral_physician", "specialty",
            "MMI_status", "impairment_summary", "causation_opinion",
            "treatment_recommendations", "medication_recommendations",
            "work_restrictions", "future_medical_recommendations"
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

        # Handle physician names with validation
        examiner_candidates = []
        referral_candidates = []
        
        for partial in partials:
            examiner = partial.get("examiner_name", "")
            if examiner and examiner.strip():
                validated = self._validate_physician_full_name(examiner)
                if validated:
                    examiner_candidates.append(validated)
            
            referral = partial.get("referral_physician", "")
            if referral and referral.strip():
                validated = self._validate_physician_full_name(referral)
                if validated:
                    referral_candidates.append(validated)
        
        if examiner_candidates:
            examiner_candidates.sort(key=len, reverse=True)
            merged["examiner_name"] = examiner_candidates[0]
        
        if referral_candidates:
            referral_candidates.sort(key=len, reverse=True)
            merged["referral_physician"] = referral_candidates[0]

        logger.info(f"📊 Merge completed: {len(merged['body_parts_evaluated'])} body parts")
        return merged

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return a minimal fallback result structure."""
        return {
            "document_date": fallback_date,
            "examiner_name": "",
            "referral_physician": "",
            "specialty": "",
            "body_parts_evaluated": [],
            "diagnoses_confirmed": [],
            "MMI_status": "",
            "impairment_summary": "",
            "causation_opinion": "",
            "treatment_recommendations": "",
            "medication_recommendations": "",
            "work_restrictions": "",
            "future_medical_recommendations": "",
        }

    def _detect_examiner(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Stage 2: Detect QME/AME examiner using DoctorDetector."""
        logger.info("🔍 Stage 2: Running DoctorDetector...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Examiner detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning(f"⚠️ No valid examiner found")
            return ""

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result with professional summary."""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        
        # Apply referral doctor fallback logic
        final_examiner = self._apply_referral_fallback(cleaned)
        cleaned["final_examiner"] = final_examiner
        
        # Generate professional summary using few-shot approach
        summary_line = self._build_professional_summary(cleaned, doc_type, fallback_date)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("document_date", fallback_date),
            summary_line=summary_line,
            examiner_name=final_examiner,
            specialty=cleaned.get("specialty"),
            body_parts=cleaned.get("body_parts_evaluated", []),
            raw_data=cleaned,
        )

    def _apply_referral_fallback(self, data: Dict) -> str:
        """Apply referral doctor fallback logic."""
        examiner_md = data.get("examiner_name", "")
        referral_md = data.get("referral_physician", "")
        
        if examiner_md and examiner_md != "":
            logger.info(f"✅ Using QME/AME examiner: {examiner_md}")
            return examiner_md
        
        if referral_md and referral_md != "":
            logger.info(f"🔄 Using referral doctor: {referral_md}")
            return referral_md
        
        logger.info("❌ No qualified doctors found")
        return ""

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data."""
        cleaned = {}
        
        # Date validation
        date = result.get("document_date", "")
        cleaned["document_date"] = date if date and date != "empty" else fallback_date

        # Physician validation
        examiner = result.get("examiner_name", "")
        referral_physician = result.get("referral_physician", "")
        cleaned["examiner_name"] = self._validate_physician_full_name(examiner)
        cleaned["referral_physician"] = self._validate_physician_full_name(referral_physician)

        # Specialty validation
        specialty = result.get("specialty", "")
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""

        # Body parts validation
        body_parts = result.get("body_parts_evaluated", [])
        if isinstance(body_parts, str):
            body_parts = [bp.strip() for bp in body_parts.split(",") if bp.strip()]
        elif isinstance(body_parts, list):
            body_parts = [bp.strip() for bp in body_parts if bp and isinstance(bp, str)]
        else:
            body_parts = []
        cleaned["body_parts_evaluated"] = [bp for bp in body_parts if bp and bp != "empty"]

        # Diagnoses validation
        diagnoses = result.get("diagnoses_confirmed", [])
        if isinstance(diagnoses, str):
            diagnoses = [dx.strip() for dx in diagnoses.split(",") if dx.strip()]
        elif isinstance(diagnoses, list):
            diagnoses = [dx.strip() for dx in diagnoses if dx and isinstance(dx, str)]
        else:
            diagnoses = []
        cleaned["diagnoses_confirmed"] = [dx for dx in diagnoses if dx and dx != "empty"]

        # String fields validation
        string_fields = [
            "MMI_status", "impairment_summary", "causation_opinion",
            "work_restrictions", "treatment_recommendations", 
            "medication_recommendations", "future_medical_recommendations"
        ]
        
        negative_phrases = [
            "no additional treatment", "no future medical", "no treatment",
            "no recommendations", "no restrictions", "no limitations",
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

    def _validate_physician_full_name(self, name: str) -> str:
        """Validate physician name has proper credentials."""
        if not name or name.lower() in ["not specified", "not found", "none", "n/a", ""]:
            return ""
        
        name_lower = name.lower()
        
        reject_terms = {
            "admin", "administrator", "case manager", "coordinator", "manager",
            "therapist", "technician", "assistant", "technologist"
        }
        
        if any(term in name_lower for term in reject_terms):
            return ""
        
        if not self.medical_credential_pattern.search(name_lower):
            return ""
        
        words = name.split()
        if len(words) < 2:
            return ""
        
        has_proper_structure = (
            (len(words) >= 3 and any(title in words[0].lower() for title in ["dr", "dr."])) or
            (len(words) >= 2 and any(title in words[-1].lower() for title in ["md", "do", "m.d.", "d.o."]))
        )
        
        if not has_proper_structure:
            if len(words) >= 2 and self.medical_credential_pattern.search(name_lower):
                return name
            return ""
        
        return name

    def _build_professional_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build professional 50-60 word summary using few-shot approach."""
        
        # System prompt for summary generation
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical summarizer creating concise QME/AME report summaries for physicians.

RULES:
- Keep summary 50-60 words
- Focus on key findings and recommendations
- Use professional clinical language
- Include: date, physician, body parts, key findings, recommendations
- Be concise but informative
""")

        # User prompt with data
        user_prompt = HumanMessagePromptTemplate.from_template("""
Create a professional 50-60 word summary from this QME/AME data:

Date: {date}
Physician: {physician}
Specialty: {specialty}
Body Parts: {body_parts}
Diagnoses: {diagnoses}
MMI Status: {mmi_status}
Impairment: {impairment}
Work Restrictions: {restrictions}
Treatment: {treatment}
Medications: {medications}
Future Care: {future_care}

Summary (50-60 words):
""")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])

        try:
            # Prepare data for summary
            date = data.get("document_date", fallback_date)
            physician = data.get("final_examiner", "")
            specialty = data.get("specialty", "")
            body_parts = ", ".join(data.get("body_parts_evaluated", [])[:3])
            diagnoses = ", ".join(data.get("diagnoses_confirmed", [])[:2])
            mmi_status = data.get("MMI_status", "")
            impairment = data.get("impairment_summary", "")
            restrictions = data.get("work_restrictions", "")
            treatment = data.get("treatment_recommendations", "")
            medications = data.get("medication_recommendations", "")
            future_care = data.get("future_medical_recommendations", "")

            chain = chat_prompt | self.llm
            response = chain.invoke({
                "date": date,
                "physician": physician,
                "specialty": specialty,
                "body_parts": body_parts,
                "diagnoses": diagnoses,
                "mmi_status": mmi_status,
                "impairment": impairment,
                "restrictions": restrictions,
                "treatment": treatment,
                "medications": medications,
                "future_care": future_care
            })
            
            summary = response.content.strip()
            
            # Ensure appropriate length
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:65]) + "..."
            elif len(words) < 45:
                # Add context if too short
                if not summary.endswith('.'):
                    summary += '.'
                if mmi_status:
                    summary += f" {mmi_status}."
            
            logger.info(f"📊 Generated summary: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            # Fallback manual summary
            return self._build_manual_summary(data, doc_type, fallback_date)

    def _build_manual_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Fallback manual summary construction."""
        date = data.get("document_date", fallback_date)
        physician = data.get("final_examiner", "")
        specialty = data.get("specialty", "")
        body_parts = data.get("body_parts_evaluated", [])
        mmi = data.get("MMI_status", "")
        impairment = data.get("impairment_summary", "")
        restrictions = data.get("work_restrictions", "")
        treatment = data.get("treatment_recommendations", "")

        parts = [f"{doc_type} Report dated {date}"]
        
        if physician:
            specialty_abbrev = self._abbreviate_specialty(specialty) if specialty else 'QME'
            parts.append(f"by {physician}, {specialty_abbrev}")

        if body_parts:
            body_str = ", ".join(body_parts[:3])
            parts.append(f"for {body_str}")

        findings = []
        if mmi:
            findings.append(mmi)
        if impairment:
            findings.append(impairment)

        if findings:
            parts.append(f"= {'; '.join(findings)}")

        if restrictions:
            parts.append(f"Work: {restrictions[:40]}")

        if treatment:
            parts.append(f"Treatment: {treatment[:40]}")

        summary = " ".join(parts)
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:65]) + "..."
        
        return summary

    def _abbreviate_specialty(self, specialty: str) -> str:
        """Abbreviate medical specialties."""
        abbreviations = {
            "Orthopedic Surgery": "Ortho",
            "Orthopedics": "Ortho",
            "Neurology": "Neuro",
            "Pain Management": "Pain",
            "Psychiatry": "Psych",
            "Psychology": "Psych",
            "Physical Medicine & Rehabilitation": "PM&R",
            "Physical Medicine and Rehabilitation": "PM&R",
            "Internal Medicine": "IM",
            "Occupational Medicine": "Occ Med",
        }
        return abbreviations.get(specialty, specialty[:12])