# service/report_analyzer.py
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from models.schemas import ComprehensiveAnalysis, PatientInfo
from services.rule_engine import RuleEngine
from config.settings import CONFIG
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("document_ai")


class ReportAnalyzer:
    """Service for comprehensive medical/legal report analysis (LLM-assisted for summaries).
       Note: Alerts/actions are *not* created by LLM — they are produced by RuleEngine.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.1,
            timeout=120
        )
        self.rule_engine = RuleEngine()
        # Set up parser bound to your Pydantic model
        self.parser = JsonOutputParser(pydantic_object=ComprehensiveAnalysis)



    def get_current_datetime(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def detect_document_type_preview(self, document_text: str) -> str:
        text_lower = document_text.lower()
        if any(term in text_lower for term in ['mri', 'ct scan', 'x-ray', 'ultrasound', 'mammography', 'radiolog']):
            return "Medical Imaging Report"
        if any(term in text_lower for term in ['lab result', 'pathology', 'blood test', 'urinalysis', 'biopsy']):
            return "Laboratory/Pathology Report"
        if any(term in text_lower for term in ['progress report', 'pr-2', 'follow-up', 'treatment progress']):
            return "Progress Report"
        if any(term in text_lower for term in ['independent medical examination', 'ime', 'medical evaluation']):
            return "Independent Medical Examination"
        if any(term in text_lower for term in ['request for authorization', 'rfa', 'pre-authorization', 'treatment request']):
            return "Request for Authorization (RFA)"
        if any(term in text_lower for term in ['denied', 'denial', 'not authorized', 'coverage denied']):
            return "Denial/Coverage Decision"
        if any(term in text_lower for term in ['ttd', 'temporary total disability', 'work restriction', 'return to work']):
            return "Work Status Document"
        if any(term in text_lower for term in ['legal opinion', 'attorney', 'litigation', 'deposition']):
            return "Legal Document"
        if any(term in text_lower for term in ['patient', 'diagnosis', 'treatment', 'medical', 'physician']):
            return "Medical Report"
        return "Unknown Document Type"

    def create_analysis_prompt(self) -> PromptTemplate:
        template = (
            "You are a medical/legal document analysis expert specializing in healthcare document classification and analysis.\n\n"
            "INPUT DOCUMENT TEXT:\n"
            "{document_text}\n\n"
            "CURRENT DATE/TIME: {current_datetime}\n\n"
            "TASK:\n"
            "Analyze this document and provide a structured response.\n\n"
            "{format_instructions}\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "- Output ONLY the JSON object with NO markdown, NO code block fences, NO commentary, and NO extra explanation.\n"
            "- The first character of your response MUST be '{{' and the last character MUST be '}}'.\n"
            "- Do NOT include any backticks or the word 'json' anywhere in your response.\n"
            "- Your response must strictly follow the above schema.\n\n"
            "SUMMARY FIELD REQUIREMENTS:\n"
            "- Return the `summary` field as an array of four strings in this exact order: \n"
            "  1) 'Clinical Overview — <purpose of the document, key decision or review date, and patient context>'\n"
            "  2) 'Treatment Decision — <authorization outcome, approved/denied services, medication and dosage, or therapy details>'\n"
            "  3) 'Reviewer — <clinician(s), facility, or department responsible, including credentials when present>'\n"
            "  4) 'Next Steps — <clear follow-up actions, deadlines, or contact instructions directly quoted or paraphrased from the document; write 'None stated' if absent>'\n"
            "- Each string must be concise (<= 180 characters) and contain concrete facts from the document.\n"
            "- Always include any explicit decision date (e.g., Decision Date, Date of Service) in the Clinical Overview line.\n"
            "- Mention prescribing or treating providers with their title within the Reviewer line whenever the document includes them.\n"
            "- Avoid generic language such as 'document summary' or 'confidential'; focus on actionable details."
        )
        return PromptTemplate(
            input_variables=["document_text", "current_datetime"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template=template
        )


    @staticmethod
    def clean_llm_json(raw_text: str) -> str:
        # Remove code fences and optional 'json'
        cleaned = re.sub(r"^```[\s]?json[\s]?|^``````$", "", raw_text, flags=re.MULTILINE).strip()
        # Remove preamble before the first curly brace
        cleaned = re.sub(r"^[^{]*({)", r"\1", cleaned, flags=re.DOTALL)
        return cleaned

    
    def validate_analysis_data(self, parsed_data: Dict[str, Any]) -> ComprehensiveAnalysis:
        try:
            patient_data = parsed_data.get("report_json", {})
            patient_info = PatientInfo(**patient_data)
            # IMPORTANT: do not rely on LLM-generated alerts; keep the list empty here
            summary = parsed_data.get("summary", [])
            # Ensure summary is a list
            if isinstance(summary, str):
                summary = [summary]
            analysis = ComprehensiveAnalysis(
                original_report=parsed_data.get("original_report", ""),
                report_json=patient_info,
                summary=summary,
                work_status_alert=[]  # rule engine will create deterministic alerts
            )
            return analysis
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise ValueError(f"Invalid analysis structure: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating analysis: {str(e)}")
            raise

    def analyze_document(self, document_text: str) -> ComprehensiveAnalysis:
        try:
            logger.info("Starting LLM-assisted comprehensive analysis...")
            if not document_text or len(document_text.strip()) < 50:
                raise ValueError("Document content appears too short or empty for meaningful analysis")

            prompt = self.create_analysis_prompt()
            formatted = prompt.format(
                document_text=document_text,
                current_datetime=self.get_current_datetime()
            )

            response = self.llm.invoke(formatted)
            response_text = response.content

            # Clean up possible markdown/code fences
            cleaned_text = ReportAnalyzer.clean_llm_json(response_text)

            parsed = self.parser.parse(cleaned_text)

            # Validate the parsed data
            analysis = self.validate_analysis_data(parsed)

            # Coerce into Pydantic model no matter what
            if isinstance(parsed, dict):
                analysis = ComprehensiveAnalysis(**parsed)
            else:
                analysis = parsed

            # Make work_status_alert deterministic (RuleEngine will fill it later if needed)
            if not analysis.work_status_alert:
                analysis.work_status_alert = []

            logger.info(
                f"Analysis OK: patient={analysis.report_json.patient_name}, summary_items={len(analysis.summary)}"
            )
            return analysis

        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise ValueError(f"Invalid analysis structure: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise



    def compare_summaries(self, previous_summary: List[str], current_summary: List[str]) -> List[str]:
        return self.rule_engine.compute_whats_new(previous_summary or [], current_summary or [])
