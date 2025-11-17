"""
Specialist Consult Enhanced Extractor - Full Context with Context-Awareness & Document Analysis

Optimized for accuracy using Gemini-style full-document processing with contextual guidance
8 CRITICAL FIELDS FOCUSED (NO HALLUCINATION, ONLY EXPLICIT INFORMATION)
"""

import logging
import re
import time
from typing import Dict, Optional, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector
from utils.document_context_analyzer import DocumentContextAnalyzer

logger = logging.getLogger("document_ai")


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction using DocumentContextAnalyzer guidance
    - Chain-of-thought reasoning for specialist recommendation evaluation
    - 8-FIELD FOCUSED extraction for zero hallucination
    - Optimized for Workers' Compensation consultation analysis
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.doctor_detector = DoctorDetector(llm)
        self.context_analyzer = DocumentContextAnalyzer(llm)
        self.verifier = ExtractionVerifier(llm)

    def extract(self, text: str, doc_type: str, fallback_date: str, page_zones: Optional[Dict] = None, context_analysis: Optional[Dict] = None) -> ExtractionResult:
        """Extract specialist consult report with full context processing. Accepts optional page_zones and context_analysis for compatibility with context-aware routing."""
        logger.info("=" * 80)
        logger.info("👨‍⚕️ STARTING CONSULT EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        start_time = time.time()
        try:
            # Step 1: Use provided context_analysis or analyze if not provided
            if context_analysis is None:
                logger.info("🔍 Analyzing document context for guidance...")
                context_analysis = self.context_analyzer.analyze(text, doc_type)
                logger.info(f"📊 Context Analysis Complete:")
                logger.info(f"   Document Type: {context_analysis.get('document_type', 'Unknown')}")
                logger.info(f"   Confidence: {context_analysis.get('confidence', 'medium')}")
                logger.info(f"   Key Sections: {context_analysis.get('key_sections', [])}")
                logger.info(f"   Critical Keywords: {context_analysis.get('critical_keywords', [])}")
            # Step 2: Extract raw data with full context and context_analysis guidance
            raw_data = self._extract_raw_data(text, doc_type, fallback_date, context_analysis, page_zones)
            # Step 3: Build initial result
            result = self._build_initial_result(raw_data, doc_type, fallback_date, context_analysis)
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context consultation extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted consultation data from complete {len(text):,} char document")
            logger.info("=" * 80)
            logger.info("✅ CONSULT EXTRACTION COMPLETE (FULL CONTEXT)")
            logger.info("=" * 80)
            return result
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            raise

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str, context_analysis: Dict, page_zones: Optional[Dict] = None) -> Dict:
        """Extract raw consultation data using LLM with full context and robust zone-aware physician detection"""
        # Detect consulting physician using zone-aware logic
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        physician_name = detection_result.get("doctor_name")
        physician_confidence = detection_result.get("confidence")
        if not physician_name:
            physician_name = context_analysis.get("identified_professionals", {}).get("primary_provider", "")
        
        # Build comprehensive context guidance for LLM
        context_str = f"""
DOCUMENT CONTEXT ANALYSIS (from DocumentContextAnalyzer):
- Document Type: {context_analysis.get('document_type', 'Unknown')}
- Confidence Level: {context_analysis.get('confidence', 'medium')}
- Key Sections: {', '.join(context_analysis.get('key_sections', ['Assessment', 'Plan']))}
- Critical Keywords: {', '.join(context_analysis.get('critical_keywords', [])[:10])}

CONSULTING PHYSICIAN: {physician_name or 'Not identified'}
IDENTIFIED PROFESSIONALS:
  - Primary Provider: {context_analysis.get('identified_professionals', {}).get('primary_provider', 'Not identified')}
  - Referring Provider: {context_analysis.get('identified_professionals', {}).get('referring_provider', 'Not identified')}

DIAGNOSTIC FOCUS:
- Primary Diagnoses Focus: {', '.join(context_analysis.get('diagnostic_focus', {}).get('primary_diagnoses', [])[:3]) if context_analysis.get('diagnostic_focus') else 'Not specified'}

CRITICAL EXTRACTION RULES:
1. Extract ONLY explicitly stated information - NO assumptions
2. For work restrictions: Use EXACT wording from document
3. For recommendations: Extract ONLY what is explicitly recommended in Plan/Assessment
4. For medications: Include dosage ONLY if explicitly stated
5. For diagnoses: Include ALL secondary diagnoses with qualifying language preserved
"""
        
        # Build system prompt with 8-field focus
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation consultation report specialist analyzing a COMPLETE Specialist Consultation Report.

PRIMARY PURPOSE: Extract the 8 critical fields that define current medical status, diagnostic findings, and treatment plan changes for claims administration.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE consultation report at once, allowing you to:
- Understand complete referral context and clinical questions
- Correlate subjective complaints with objective examination findings
- Assess specialist's diagnostic reasoning and treatment justifications
- Identify ALL treatment recommendations with medical necessity rationale
- Extract complete work capacity and restriction changes

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If NOT explicitly mentioned, return EMPTY string "" or empty list []
   - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
   - DO NOT add typical findings, standard recommendations, or common restrictions

2. **DIAGNOSES - INCLUDE ALL WITH EXACT TERMINOLOGY**
   - Extract PRIMARY diagnosis with ICD-10 code if stated
   - Extract ALL SECONDARY diagnoses - DO NOT miss any
   - Include diagnostic certainty: "probable", "confirmed", "consistent with", "rule out"
   - Include causation statements if explicitly stated

3. **WORK RESTRICTIONS - EXACT WORDING ONLY**
   - Use EXACT phrases from document
   - If "no overhead work" stated, extract "no overhead work" (NOT "no overhead reaching")
   - If "lifting restrictions" stated WITHOUT specifics, extract "lifting restrictions" (NOT "no lifting >10 lbs")

4. **RECOMMENDATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY recommendations explicitly stated in Assessment/Plan sections
   - DO NOT extract treatments "considered but not recommended"
   - Include ALL explicitly recommended procedures, injections, medications, therapy
   - DO NOT add standard care not mentioned

5. **EMPTY FIELDS ARE BETTER THAN GUESSED FIELDS**
   - Leave fields empty if information not found
   - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""

CONSULTATION EXTRACTION FOCUS - 8 CRITICAL FIELDS FOR CLAIMS ADMINISTRATION:

FIELD 1: HEADER & CONTEXT - Report Identity & Authority
FIELD 2: CHIEF COMPLAINT - Patient's Primary Issue
FIELD 3: DIAGNOSIS & ASSESSMENT - Medical Conclusion (HIGHEST PRIORITY)
FIELD 4: HISTORY OF PRESENT ILLNESS - Symptoms & Severity Context
FIELD 5: PRIOR TREATMENT & EFFICACY - Failure of Conservative Care
FIELD 6: OBJECTIVE FINDINGS - Verifiable Evidence (Exam & Imaging)
FIELD 7: PLAN - RECOMMENDED TREATMENTS (MOST CRITICAL FOR AUTHORIZATION)
FIELD 8: WORK STATUS & IMPAIRMENT - Legal/Administrative Status

Now analyze this COMPLETE specialist consultation report and extract the 8 critical fields:
""")
        
        # Build user prompt
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE SPECIALIST CONSULTATION REPORT TEXT:

{full_document_text}

Extract into STRUCTURED JSON focusing on 8 CRITICAL FIELDS:

{{
  "field_1_header_context": {{
    "consulting_physician": {{
      "name": "{primary_physician}",
      "specialty": "",
      "credentials": "",
      "facility": ""
    }},
    "consultation_date": "",
    "referring_physician": "",
    "patient_name": "",
    "patient_dob": "",
    "date_of_injury": "",
    "claim_number": ""
  }},
  
  "field_2_chief_complaint": {{
    "primary_complaint": "",
    "location": "",
    "duration": "",
    "radiation_pattern": ""
  }},
  
  "field_3_diagnosis_assessment": {{
    "primary_diagnosis": "",
    "icd10_code": "",
    "secondary_diagnoses": [],
    "diagnostic_certainty": "",
    "causation_statement": ""
  }},
  
  "field_4_history_present_illness": {{
    "pain_quality": "",
    "pain_location": "",
    "radiation": "",
    "aggravating_factors": [],
    "alleviating_factors": [],
    "functional_deficits": []
  }},
  
  "field_5_prior_treatment_efficacy": {{
    "treatments_received": [],
    "level_of_relief": {{
      "physical_therapy": "",
      "medications": "",
      "injections": "",
      "chiropractic": ""
    }},
    "treatment_failure_statement": ""
  }},
  
  "field_6_objective_findings": {{
    "physical_exam": {{
      "rom_measurements": "",
      "strength_testing": "",
      "special_tests_positive": [],
      "palpation_findings": "",
      "inspection_findings": ""
    }},
    "imaging_review": {{
      "mri_findings": "",
      "xray_findings": "",
      "ct_findings": "",
      "correlation_with_symptoms": ""
    }}
  }},
  
  "field_7_plan_recommendations": {{
    "specific_interventions": {{
      "injections_requested": [],
      "procedures_requested": [],
      "surgery_recommended": [],
      "diagnostics_ordered": []
    }},
    "medication_changes": {{
      "new_medications": [],
      "dosage_adjustments": [],
      "discontinued_medications": []
    }},
    "therapy_recommendations": {{
      "therapy_type": "",
      "frequency": "",
      "duration": "",
      "focus_areas": []
    }}
  }},
  
  "field_8_work_status_impairment": {{
    "current_work_status": "",
    "work_restrictions": [],
    "restriction_duration": "",
    "return_to_work_plan": ""
  }},
  
  "critical_findings": []
}}

⚠️ MANDATORY EXTRACTION RULES:
1. Field 3 (Diagnosis): Use EXACT diagnostic terminology, include ALL secondary diagnoses
2. Field 5 (Prior Treatment): Include quantified relief ("50% improvement", "no relief")
3. Field 6 (Objective): Extract EXACT measurements and imaging findings with anatomical specificity
4. Field 7 (Plan): Extract ONLY explicitly recommended treatments - THE MOST CRITICAL
5. Field 8 (Work Restrictions): Use EXACT wording (don't add weight/time limits not stated)
6. Empty fields are acceptable if not stated in report
""")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])
        
        # Create chain
        chain = prompt | self.llm | self.parser
        
        logger.info(f"📄 Document size: {len(text):,} chars (~{len(text) // 4:,} tokens)")
        logger.info("🔍 Processing ENTIRE consultation report in single context window with context_analysis guidance...")
        logger.info("🤖 Invoking LLM for full-context consultation extraction...")
        
        # Invoke LLM
        try:
            raw_data = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_str,
                "primary_physician": physician_name or ""
            })
            
            logger.info("✅ Extracted consultation data from complete document")
            return raw_data
        
        except Exception as e:
            logger.error(f"❌ LLM extraction failed: {str(e)}")
            return self._get_fallback_result(fallback_date)

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str, context_analysis: Dict = None) -> ExtractionResult:
        """Build initial result from extracted consultation data"""
        
        if context_analysis is None:
            context_analysis = {}
        
        logger.info("🔨 Building initial consultation extraction result...")
        
        try:
            # Extract from 8-field structure
            header_context = raw_data.get("field_1_header_context", {})
            consulting_physician = header_context.get("consulting_physician", {})
            diagnostic_impression = raw_data.get("field_3_diagnosis_assessment", {})
            
            # Build comprehensive consultation summary
            summary_line = self._build_consultation_narrative_summary(raw_data, doc_type, fallback_date, context_analysis)
            
            # CRITICAL: Ensure summary_line is STRING
            if not isinstance(summary_line, str):
                summary_line = str(summary_line) if summary_line else "Consultation summary not available"
            
            result = ExtractionResult(
                document_type=doc_type,
                document_date=header_context.get("consultation_date", fallback_date),
                summary_line=summary_line,  # MUST be STRING
                examiner_name=consulting_physician.get("name", ""),
                specialty=consulting_physician.get("specialty", ""),
                body_parts=[diagnostic_impression.get("primary_diagnosis", "")],
                raw_data=raw_data,
            )
            
            logger.info(f"✅ Initial consultation result built (Physician: {result.examiner_name})")
            return result
        
        except Exception as e:
            logger.error(f"❌ Error building initial result: {str(e)}")
            raise

    def _build_consultation_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str, context_analysis: Dict = None) -> str:
        """
        Build comprehensive narrative summary for consultation report.
        CRITICAL: Must return STRING with ALL critical elements, NO AI additions
        """
        
        if context_analysis is None:
            context_analysis = {}
        
        try:
            # Extract all data from 8-field structure
            header_context = data.get("field_1_header_context", {})
            chief_complaint = data.get("field_2_chief_complaint", {})
            diagnostic_impression = data.get("field_3_diagnosis_assessment", {})
            treatment_recommendations = data.get("field_7_plan_recommendations", {})
            work_status_recommendations = data.get("field_8_work_status_impairment", {})
            critical_findings = data.get("critical_findings", [])
            
            # Build narrative sections
            narrative_parts = []
            
            # Section 0: PHYSICIAN CONTEXT
            consulting_physician = header_context.get("consulting_physician", {})
            physician_name = self._safe_str(consulting_physician.get("name", ""))
            specialty = self._safe_str(consulting_physician.get("specialty", ""))
            consultation_date = self._safe_str(header_context.get("consultation_date", fallback_date))
            
            if physician_name and physician_name.strip():
                context_parts = [physician_name.strip()]
                if specialty and specialty.strip():
                    context_parts.append(f"({specialty.strip()})")
                context_parts.append(f"on {consultation_date if consultation_date else fallback_date}")
                context_line = " ".join(context_parts)
                narrative_parts.append(f"**Consult:** {context_line}")
            
            # Section 1: COMPLETE DIAGNOSIS ASSESSMENT (CRITICAL - INCLUDE ALL)
            diagnosis_text = self._build_complete_diagnosis_narrative(diagnostic_impression)
            if diagnosis_text and isinstance(diagnosis_text, str) and diagnosis_text.strip():
                narrative_parts.append(f"**Diagnosis:** {diagnosis_text.strip()}")
            
            # Section 2: CHIEF COMPLAINT
            complaint = self._safe_str(chief_complaint.get("primary_complaint", ""))
            location = self._safe_str(chief_complaint.get("location", ""))
            duration = self._safe_str(chief_complaint.get("duration", ""))
            
            if complaint and complaint.strip():
                complaint_parts = [complaint.strip()]
                if location and location.strip():
                    complaint_parts.append(f"({location.strip()})")
                if duration and duration.strip():
                    complaint_parts.append(f"- {duration.strip()}")
                complaint_text = " ".join(complaint_parts)
                narrative_parts.append(f"**CC:** {complaint_text}")
            
            # Section 3: TREATMENT RECOMMENDATIONS (CRITICAL - EXACT ONLY)
            recommendation_text = self._build_exact_recommendations_narrative(
                treatment_recommendations,
                critical_findings
            )
            if recommendation_text and isinstance(recommendation_text, str) and recommendation_text.strip():
                narrative_parts.append(f"**Plan:** {recommendation_text.strip()}")
            
            # Section 4: WORK STATUS & RESTRICTIONS (EXACT WORDING)
            work_text = self._build_exact_work_status_narrative(work_status_recommendations)
            if work_text and isinstance(work_text, str) and work_text.strip():
                narrative_parts.append(f"**Work Status:** {work_text.strip()}")
            
            # Filter and join
            valid_parts = [str(part) for part in narrative_parts if part and isinstance(part, str) and part.strip()]
            full_narrative = "\n\n".join(valid_parts)
            
            logger.info(f"📝 Consultation Narrative summary generated: {len(full_narrative)} characters")
            return full_narrative if full_narrative and isinstance(full_narrative, str) else "Consultation summary not available"
        
        except Exception as e:
            logger.error(f"❌ Error building consultation narrative: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _build_complete_diagnosis_narrative(self, diagnostic_impression: Dict) -> str:
        """
        Build COMPLETE diagnosis narrative including ALL diagnoses.
        CRITICAL: Do not miss any secondary diagnoses or details.
        """
        try:
            diagnosis_parts = []
            
            # Primary diagnosis with ALL details
            primary_dx = self._safe_str(diagnostic_impression.get("primary_diagnosis", ""))
            certainty = self._safe_str(diagnostic_impression.get("diagnostic_certainty", ""))
            icd10 = self._safe_str(diagnostic_impression.get("icd10_code", ""))
            
            if primary_dx and primary_dx.strip():
                dx_text = primary_dx.strip()
                
                # Add certainty qualifier if present
                if certainty and certainty.strip() and certainty.strip().lower() not in ["definitive", "confirmed"]:
                    dx_text = f"{dx_text} ({certainty.strip()})"
                
                # Add ICD-10 if present
                if icd10 and icd10.strip():
                    dx_text = f"{dx_text} [{icd10.strip()}]"
                
                diagnosis_parts.append(dx_text)
            
            # CRITICAL: Include ALL secondary diagnoses (don't miss any)
            secondary_diagnoses = diagnostic_impression.get("secondary_diagnoses", [])
            if secondary_diagnoses and isinstance(secondary_diagnoses, list):
                for i, sec_dx in enumerate(secondary_diagnoses):
                    sec_dx_str = self._safe_str(sec_dx)
                    if sec_dx_str and sec_dx_str.strip():
                        # Mark as secondary
                        diagnosis_parts.append(f"Secondary: {sec_dx_str.strip()}")
            
            # Clinical correlation/causation (important context)
            causation = self._safe_str(diagnostic_impression.get("causation_statement", ""))
            if causation and causation.strip():
                diagnosis_parts.append(f"Causation: {causation.strip()}")
            
            # Join all diagnosis elements
            valid_parts = [str(part).strip() for part in diagnosis_parts if part and isinstance(part, str) and part.strip()]
            result = "; ".join(valid_parts) if valid_parts else "Diagnosis not specified"
            
            return str(result)
        except Exception as e:
            logger.error(f"Error in _build_complete_diagnosis_narrative: {str(e)}")
            return "Diagnosis extraction error"

    def _build_exact_recommendations_narrative(self, treatment_recommendations: Dict, critical_recommendations: List = None) -> str:
        """
        Build treatment recommendations narrative with EXACT wording from document.
        CRITICAL: Extract ONLY what is explicitly recommended - NO AI additions.
        """
        try:
            if critical_recommendations is None:
                critical_recommendations = []
            
            recommendation_items = []
            
            if not treatment_recommendations:
                return ""
            
            specific_interventions = treatment_recommendations.get("specific_interventions", {})
            if specific_interventions:
                # Injections (EXACT wording)
                injections = specific_interventions.get("injections_requested", [])
                if injections and isinstance(injections, list):
                    for inj in injections:
                        inj_str = self._safe_str(inj)
                        if inj_str and inj_str.strip():
                            recommendation_items.append(f"Injection: {inj_str.strip()}")
                
                # Procedures (EXACT wording)
                procedures = specific_interventions.get("procedures_requested", [])
                if procedures and isinstance(procedures, list):
                    for proc in procedures:
                        proc_str = self._safe_str(proc)
                        if proc_str and proc_str.strip():
                            recommendation_items.append(proc_str.strip())
                
                # Surgery (EXACT wording)
                surgery = specific_interventions.get("surgery_recommended", [])
                if surgery and isinstance(surgery, list):
                    for surg in surgery:
                        surg_str = self._safe_str(surg)
                        if surg_str and surg_str.strip():
                            recommendation_items.append(f"Surgery: {surg_str.strip()}")
                
                # Diagnostics (EXACT wording)
                diagnostics = specific_interventions.get("diagnostics_ordered", [])
                if diagnostics and isinstance(diagnostics, list):
                    for diag in diagnostics:
                        diag_str = self._safe_str(diag)
                        if diag_str and diag_str.strip():
                            recommendation_items.append(f"Order: {diag_str.strip()}")
            
            # Medication changes (EXACT names and doses)
            medication_changes = treatment_recommendations.get("medication_changes", {})
            if medication_changes:
                # New medications
                new_meds = medication_changes.get("new_medications", [])
                if new_meds and isinstance(new_meds, list):
                    for med in new_meds:
                        if isinstance(med, dict):
                            med_name = self._safe_str(med.get("medication", ""))
                            med_dose = self._safe_str(med.get("dose", ""))
                            if med_name and med_name.strip():
                                if med_dose and med_dose.strip():
                                    recommendation_items.append(f"Start {med_name.strip()} {med_dose.strip()}")
                                else:
                                    recommendation_items.append(f"Start {med_name.strip()}")
                        else:
                            med_str = self._safe_str(med)
                            if med_str and med_str.strip():
                                recommendation_items.append(f"Start {med_str.strip()}")
                
                # Dosage adjustments
                dose_changes = medication_changes.get("dosage_adjustments", [])
                if dose_changes and isinstance(dose_changes, list):
                    for change in dose_changes:
                        change_str = self._safe_str(change)
                        if change_str and change_str.strip():
                            recommendation_items.append(f"Adjust: {change_str.strip()}")
            
            # Therapy recommendations (EXACT details)
            therapy = treatment_recommendations.get("therapy_recommendations", {})
            if therapy:
                therapy_type = self._safe_str(therapy.get("therapy_type", ""))
                frequency = self._safe_str(therapy.get("frequency", ""))
                duration = self._safe_str(therapy.get("duration", ""))
                focus_areas = therapy.get("focus_areas", [])
                
                if therapy_type and therapy_type.strip():
                    therapy_parts = [therapy_type.strip()]
                    if frequency and frequency.strip():
                        therapy_parts.append(frequency.strip())
                    if duration and duration.strip():
                        therapy_parts.append(f"for {duration.strip()}")
                    if focus_areas and isinstance(focus_areas, list):
                        focus_str = ", ".join([self._safe_str(f).strip() for f in focus_areas[:2] if f])
                        if focus_str:
                            therapy_parts.append(f"({focus_str})")
                    
                    therapy_text = " ".join(therapy_parts)
                    recommendation_items.append(therapy_text)
            
            # Join recommendations
            result = ", ".join(recommendation_items) if recommendation_items else ""
            return str(result)
        except Exception as e:
            logger.error(f"Error in _build_exact_recommendations_narrative: {str(e)}")
            return ""

    def _build_exact_work_status_narrative(self, work_status_recommendations: Dict) -> str:
        """
        Build work status narrative with EXACT wording from document.
        CRITICAL: Use exact restrictions - NO weight/time limits unless stated.
        """
        try:
            work_parts = []
            
            if not work_status_recommendations:
                return ""
            
            # Current work status (EXACT terminology)
            current_status = self._safe_str(work_status_recommendations.get("current_work_status", ""))
            if current_status and current_status.strip():
                work_parts.append(current_status.strip())
            
            # Work restrictions (EXACT wording - CRITICAL)
            restrictions = work_status_recommendations.get("work_restrictions", [])
            if restrictions and isinstance(restrictions, list):
                for restriction in restrictions:
                    restriction_str = self._safe_str(restriction)
                    if restriction_str and restriction_str.strip():
                        # Add each restriction exactly as stated
                        work_parts.append(restriction_str.strip())
            
            # Restriction duration (if stated)
            duration = self._safe_str(work_status_recommendations.get("restriction_duration", ""))
            if duration and duration.strip():
                work_parts.append(f"for {duration.strip()}")
            
            # Return-to-work plan (if stated)
            return_plan = self._safe_str(work_status_recommendations.get("return_to_work_plan", ""))
            if return_plan and return_plan.strip():
                # Shorten if too long
                plan_text = return_plan.strip()
                if len(plan_text) > 80:
                    plan_text = plan_text[:77] + "..."
                work_parts.append(f"Plan: {plan_text}")
            
            result = ", ".join(work_parts) if work_parts else ""
            return str(result)
        except Exception as e:
            logger.error(f"Error in _build_exact_work_status_narrative: {str(e)}")
            return ""

    def _safe_str(self, value, default="") -> str:
        """Convert any value to string safely - MUST return STRING"""
        try:
            if value is None:
                return str(default)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                # For dicts, convert to string representation
                return str(value)
            if isinstance(value, list):
                # For lists, join with comma
                flat = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        flat.append(str(item))
                    elif item:
                        flat.append(str(item))
                return ", ".join(flat) if flat else str(default)
            return str(value)
        except Exception as e:
            logger.error(f"Error in _safe_str: {str(e)}")
            return str(default)

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return fallback result structure matching 8-field structure"""
        return {
            "field_1_header_context": {
                "consulting_physician": {
                    "name": "",
                    "specialty": "",
                    "credentials": "",
                    "facility": ""
                },
                "consultation_date": fallback_date,
                "referring_physician": "",
                "patient_name": "",
                "patient_dob": "",
                "date_of_injury": "",
                "claim_number": ""
            },
            "field_2_chief_complaint": {
                "primary_complaint": "",
                "location": "",
                "duration": "",
                "radiation_pattern": ""
            },
            "field_3_diagnosis_assessment": {
                "primary_diagnosis": "",
                "icd10_code": "",
                "secondary_diagnoses": [],
                "diagnostic_certainty": "",
                "causation_statement": ""
            },
            "field_4_history_present_illness": {
                "pain_quality": "",
                "pain_location": "",
                "radiation": "",
                "aggravating_factors": [],
                "alleviating_factors": [],
                "functional_deficits": []
            },
            "field_5_prior_treatment_efficacy": {
                "treatments_received": [],
                "level_of_relief": {
                    "physical_therapy": "",
                    "medications": "",
                    "injections": "",
                    "chiropractic": ""
                },
                "treatment_failure_statement": ""
            },
            "field_6_objective_findings": {
                "physical_exam": {
                    "rom_measurements": "",
                    "strength_testing": "",
                    "special_tests_positive": [],
                    "palpation_findings": "",
                    "inspection_findings": ""
                },
                "imaging_review": {
                    "mri_findings": "",
                    "xray_findings": "",
                    "ct_findings": "",
                    "correlation_with_symptoms": ""
                }
            },
            "field_7_plan_recommendations": {
                "specific_interventions": {
                    "injections_requested": [],
                    "procedures_requested": [],
                    "surgery_recommended": [],
                    "diagnostics_ordered": []
                },
                "medication_changes": {
                    "new_medications": [],
                    "dosage_adjustments": [],
                    "discontinued_medications": []
                },
                "therapy_recommendations": {
                    "therapy_type": "",
                    "frequency": "",
                    "duration": "",
                    "focus_areas": []
                }
            },
            "field_8_work_status_impairment": {
                "current_work_status": "",
                "work_restrictions": [],
                "restriction_duration": "",
                "return_to_work_plan": ""
            },
            "critical_findings": []
        }
