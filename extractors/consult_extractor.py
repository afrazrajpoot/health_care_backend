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

    def extract(self, text: str, doc_type: str, fallback_date: str, page_zones: Optional[Dict] = None, context_analysis: Optional[Dict] = None) -> Dict:
        """
        Extract specialist consult report with full context processing. 
        Returns dictionary with long_summary and short_summary like QME extractor.
        """
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

            # Step 3: Build comprehensive long summary from ALL raw data
            long_summary = self._build_comprehensive_long_summary(raw_data, doc_type, fallback_date)

            # Step 4: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)

            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context consultation extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted consultation data from complete {len(text):,} char document")
            logger.info("=" * 80)
            logger.info("✅ CONSULT EXTRACTION COMPLETE (FULL CONTEXT)")
            logger.info("=" * 80)

            # Return dictionary with both summaries like QME extractor
            return {
                "long_summary": long_summary,
                "short_summary": short_summary
            }

        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            # Return fallback result structure
            return {
                "long_summary": f"Consultation extraction failed: {str(e)}",
                "short_summary": "Consultation summary not available"
            }

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

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted raw data with detailed headings.
        Similar to QME extractor structure.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted consultation data...")
        
        sections = []
        
        # Section 1: CONSULTATION OVERVIEW
        sections.append("📋 CONSULTATION OVERVIEW")
        sections.append("-" * 50)
        
        header_context = raw_data.get("field_1_header_context", {})
        consulting_physician = header_context.get("consulting_physician", {})
        
        physician_name = consulting_physician.get("name", "")
        specialty = consulting_physician.get("specialty", "")
        consultation_date = header_context.get("consultation_date", fallback_date)
        referring_physician = header_context.get("referring_physician", "")
        
        overview_lines = [
            f"Document Type: {doc_type}",
            f"Consultation Date: {consultation_date}",
            f"Consulting Physician: {physician_name}",
            f"Specialty: {specialty}" if specialty else "Specialty: Not specified",
            f"Referring Physician: {referring_physician}" if referring_physician else "Referring Physician: Not specified"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PATIENT INFORMATION
        sections.append("\n👤 PATIENT INFORMATION")
        sections.append("-" * 50)
        
        patient_lines = [
            f"Name: {header_context.get('patient_name', 'Not specified')}",
            f"Date of Birth: {header_context.get('patient_dob', 'Not specified')}",
            f"Date of Injury: {header_context.get('date_of_injury', 'Not specified')}",
            f"Claim Number: {header_context.get('claim_number', 'Not specified')}"
        ]
        sections.append("\n".join(patient_lines))
        
        # Section 3: CHIEF COMPLAINT
        sections.append("\n🎯 CHIEF COMPLAINT")
        sections.append("-" * 50)
        
        chief_complaint = raw_data.get("field_2_chief_complaint", {})
        complaint_lines = [
            f"Primary Complaint: {chief_complaint.get('primary_complaint', 'Not specified')}",
            f"Location: {chief_complaint.get('location', 'Not specified')}",
            f"Duration: {chief_complaint.get('duration', 'Not specified')}",
            f"Radiation Pattern: {chief_complaint.get('radiation_pattern', 'Not specified')}"
        ]
        sections.append("\n".join(complaint_lines))
        
        # Section 4: DIAGNOSIS & ASSESSMENT
        sections.append("\n🏥 DIAGNOSIS & ASSESSMENT")
        sections.append("-" * 50)
        
        diagnosis_assessment = raw_data.get("field_3_diagnosis_assessment", {})
        diagnosis_lines = []
        
        # Primary diagnosis
        primary_dx = diagnosis_assessment.get("primary_diagnosis", "")
        icd10_code = diagnosis_assessment.get("icd10_code", "")
        certainty = diagnosis_assessment.get("diagnostic_certainty", "")
        
        if primary_dx:
            dx_text = primary_dx
            if icd10_code:
                dx_text += f" [{icd10_code}]"
            if certainty:
                dx_text += f" ({certainty})"
            diagnosis_lines.append(f"Primary Diagnosis: {dx_text}")
        
        # Secondary diagnoses
        secondary_dx = diagnosis_assessment.get("secondary_diagnoses", [])
        if secondary_dx:
            diagnosis_lines.append("\nSecondary Diagnoses:")
            for dx in secondary_dx[:5]:  # Limit to 5 secondary diagnoses
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", dx.get("name", ""))
                    if dx_name and dx_name.strip():
                        diagnosis_lines.append(f"  • {dx_name}")
                elif dx and str(dx).strip():
                    diagnosis_lines.append(f"  • {dx}")
        
        # Causation statement
        causation = diagnosis_assessment.get("causation_statement", "")
        if causation:
            diagnosis_lines.append(f"\nCausation: {causation}")
        
        sections.append("\n".join(diagnosis_lines) if diagnosis_lines else "No diagnoses extracted")
        
        # Section 5: CLINICAL HISTORY & SYMPTOMS
        sections.append("\n🔬 CLINICAL HISTORY & SYMPTOMS")
        sections.append("-" * 50)
        
        history = raw_data.get("field_4_history_present_illness", {})
        history_lines = []
        
        if history.get("pain_quality"):
            history_lines.append(f"Pain Quality: {history['pain_quality']}")
        if history.get("pain_location"):
            history_lines.append(f"Pain Location: {history['pain_location']}")
        if history.get("radiation"):
            history_lines.append(f"Radiation: {history['radiation']}")
        
        # Aggravating factors
        aggravating = history.get("aggravating_factors", [])
        if aggravating:
            history_lines.append("\nAggravating Factors:")
            for factor in aggravating[:5]:
                if isinstance(factor, dict):
                    desc = factor.get("factor", "")
                    if desc:
                        history_lines.append(f"  • {desc}")
                elif factor:
                    history_lines.append(f"  • {factor}")
        
        # Alleviating factors
        alleviating = history.get("alleviating_factors", [])
        if alleviating:
            history_lines.append("\nAlleviating Factors:")
            for factor in alleviating[:5]:
                if isinstance(factor, dict):
                    desc = factor.get("factor", "")
                    if desc:
                        history_lines.append(f"  • {desc}")
                elif factor:
                    history_lines.append(f"  • {factor}")
        
        sections.append("\n".join(history_lines) if history_lines else "No clinical history details extracted")
        
        # Section 6: PRIOR TREATMENT & EFFICACY
        sections.append("\n💊 PRIOR TREATMENT & EFFICACY")
        sections.append("-" * 50)
        
        prior_treatment = raw_data.get("field_5_prior_treatment_efficacy", {})
        treatment_lines = []
        
        # Treatments received
        treatments = prior_treatment.get("treatments_received", [])
        if treatments:
            treatment_lines.append("Prior Treatments Received:")
            for tx in treatments[:8]:
                if isinstance(tx, dict):
                    tx_name = tx.get("treatment", "")
                    tx_duration = tx.get("duration", "")
                    if tx_name:
                        if tx_duration:
                            treatment_lines.append(f"  • {tx_name} ({tx_duration})")
                        else:
                            treatment_lines.append(f"  • {tx_name}")
                elif tx:
                    treatment_lines.append(f"  • {tx}")
        
        # Level of relief
        relief = prior_treatment.get("level_of_relief", {})
        if relief:
            treatment_lines.append("\nLevel of Relief:")
            for key, value in relief.items():
                if value and value not in ["", "Not specified"]:
                    treatment_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        # Treatment failure
        failure = prior_treatment.get("treatment_failure_statement", "")
        if failure:
            treatment_lines.append(f"\nTreatment Failure Statement: {failure}")
        
        sections.append("\n".join(treatment_lines) if treatment_lines else "No prior treatment information extracted")
        
        # Section 7: OBJECTIVE FINDINGS
        sections.append("\n📊 OBJECTIVE FINDINGS")
        sections.append("-" * 50)
        
        objective_findings = raw_data.get("field_6_objective_findings", {})
        objective_lines = []
        
        # Physical exam
        physical_exam = objective_findings.get("physical_exam", {})
        if physical_exam:
            objective_lines.append("Physical Examination:")
            for key, value in physical_exam.items():
                if value and value not in ["", "Not specified"]:
                    objective_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        # Imaging review
        imaging = objective_findings.get("imaging_review", {})
        if imaging:
            objective_lines.append("\nImaging Review:")
            for key, value in imaging.items():
                if value and value not in ["", "Not specified"]:
                    objective_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        sections.append("\n".join(objective_lines) if objective_lines else "No objective findings extracted")
        
        # Section 8: TREATMENT RECOMMENDATIONS (MOST CRITICAL)
        sections.append("\n🎯 TREATMENT RECOMMENDATIONS")
        sections.append("-" * 50)
        
        plan_recommendations = raw_data.get("field_7_plan_recommendations", {})
        plan_lines = []
        
        # Specific interventions
        interventions = plan_recommendations.get("specific_interventions", {})
        if interventions:
            # Injections
            injections = interventions.get("injections_requested", [])
            if injections:
                plan_lines.append("Injections Requested:")
                for inj in injections[:5]:
                    if isinstance(inj, dict):
                        inj_name = inj.get("injection", "")
                        inj_location = inj.get("location", "")
                        if inj_name:
                            if inj_location:
                                plan_lines.append(f"  • {inj_name} - {inj_location}")
                            else:
                                plan_lines.append(f"  • {inj_name}")
                    elif inj:
                        plan_lines.append(f"  • {inj}")
            
            # Procedures
            procedures = interventions.get("procedures_requested", [])
            if procedures:
                plan_lines.append("\nProcedures Requested:")
                for proc in procedures[:5]:
                    if isinstance(proc, dict):
                        proc_name = proc.get("procedure", "")
                        proc_reason = proc.get("reason", "")
                        if proc_name:
                            if proc_reason:
                                plan_lines.append(f"  • {proc_name} - {proc_reason}")
                            else:
                                plan_lines.append(f"  • {proc_name}")
                    elif proc:
                        plan_lines.append(f"  • {proc}")
            
            # Surgery
            surgery = interventions.get("surgery_recommended", [])
            if surgery:
                plan_lines.append("\nSurgery Recommended:")
                for surg in surgery[:3]:
                    if isinstance(surg, dict):
                        surg_name = surg.get("procedure", "")
                        surg_urgency = surg.get("urgency", "")
                        if surg_name:
                            if surg_urgency:
                                plan_lines.append(f"  • {surg_name} ({surg_urgency})")
                            else:
                                plan_lines.append(f"  • {surg_name}")
                    elif surg:
                        plan_lines.append(f"  • {surg}")
            
            # Diagnostics
            diagnostics = interventions.get("diagnostics_ordered", [])
            if diagnostics:
                plan_lines.append("\nDiagnostics Ordered:")
                for diag in diagnostics[:5]:
                    if isinstance(diag, dict):
                        diag_name = diag.get("test", "")
                        diag_reason = diag.get("reason", "")
                        if diag_name:
                            if diag_reason:
                                plan_lines.append(f"  • {diag_name} - {diag_reason}")
                            else:
                                plan_lines.append(f"  • {diag_name}")
                    elif diag:
                        plan_lines.append(f"  • {diag}")
        
        # Medication changes
        med_changes = plan_recommendations.get("medication_changes", {})
        if med_changes:
            plan_lines.append("\nMedication Changes:")
            
            # New medications
            new_meds = med_changes.get("new_medications", [])
            if new_meds:
                plan_lines.append("  New Medications:")
                for med in new_meds[:5]:
                    if isinstance(med, dict):
                        med_name = med.get("medication", "")
                        med_dose = med.get("dose", "")
                        if med_name:
                            if med_dose:
                                plan_lines.append(f"    • {med_name} - {med_dose}")
                            else:
                                plan_lines.append(f"    • {med_name}")
                    elif med:
                        plan_lines.append(f"    • {med}")
            
            # Dosage adjustments
            dose_adj = med_changes.get("dosage_adjustments", [])
            if dose_adj:
                plan_lines.append("  Dosage Adjustments:")
                for adj in dose_adj[:3]:
                    if isinstance(adj, dict):
                        adj_name = adj.get("medication", "")
                        adj_details = adj.get("adjustment", "")
                        if adj_name:
                            if adj_details:
                                plan_lines.append(f"    • {adj_name} - {adj_details}")
                            else:
                                plan_lines.append(f"    • {adj_name}")
                    elif adj:
                        plan_lines.append(f"    • {adj}")
        
        # Therapy recommendations
        therapy = plan_recommendations.get("therapy_recommendations", {})
        if therapy:
            plan_lines.append("\nTherapy Recommendations:")
            therapy_type = therapy.get("therapy_type", "")
            frequency = therapy.get("frequency", "")
            duration = therapy.get("duration", "")
            focus_areas = therapy.get("focus_areas", [])
            
            if therapy_type:
                therapy_info = therapy_type
                if frequency:
                    therapy_info += f" - {frequency}"
                if duration:
                    therapy_info += f" for {duration}"
                plan_lines.append(f"  • {therapy_info}")
            
            if focus_areas:
                plan_lines.append("  Focus Areas:")
                for area in focus_areas[:3]:
                    if area:
                        plan_lines.append(f"    • {area}")
        
        sections.append("\n".join(plan_lines) if plan_lines else "No treatment recommendations extracted")
        
        # Section 9: WORK STATUS & IMPAIRMENT
        sections.append("\n💼 WORK STATUS & IMPAIRMENT")
        sections.append("-" * 50)
        
        work_status = raw_data.get("field_8_work_status_impairment", {})
        work_lines = []
        
        current_status = work_status.get("current_work_status", "")
        if current_status:
            work_lines.append(f"Current Work Status: {current_status}")
        
        restrictions = work_status.get("work_restrictions", [])
        if restrictions:
            work_lines.append("\nWork Restrictions:")
            for restriction in restrictions[:10]:
                if isinstance(restriction, dict):
                    desc = restriction.get("restriction", "")
                    duration = restriction.get("duration", "")
                    if desc:
                        if duration:
                            work_lines.append(f"  • {desc} ({duration})")
                        else:
                            work_lines.append(f"  • {desc}")
                elif restriction:
                    work_lines.append(f"  • {restriction}")
        
        restriction_duration = work_status.get("restriction_duration", "")
        if restriction_duration:
            work_lines.append(f"\nRestriction Duration: {restriction_duration}")
        
        return_plan = work_status.get("return_to_work_plan", "")
        if return_plan:
            work_lines.append(f"Return to Work Plan: {return_plan}")
        
        sections.append("\n".join(work_lines) if work_lines else "No work status information extracted")
        
        # Section 10: CRITICAL FINDINGS
        sections.append("\n🚨 CRITICAL FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            for finding in critical_findings[:8]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("description", "")
                    finding_priority = finding.get("priority", "")
                    if finding_desc:
                        if finding_priority:
                            sections.append(f"• [{finding_priority}] {finding_desc}")
                        else:
                            sections.append(f"• {finding_desc}")
                elif finding:
                    sections.append(f"• {finding}")
        else:
            sections.append("No critical findings explicitly listed")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Long summary built: {len(long_summary)} characters")
        
        return long_summary
    def _clean_pipes_from_summary(self, short_summary: str) -> str:
            """
            Clean empty pipes from short summary to avoid consecutive pipes or trailing pipes.
            
            Args:
                short_summary: The pipe-delimited short summary string
                
            Returns:
                Cleaned summary with proper pipe formatting
            """
            if not short_summary or '|' not in short_summary:
                return short_summary
            
            # Split by pipe and clean each part
            parts = short_summary.split('|')
            cleaned_parts = []
            
            for part in parts:
                # Remove whitespace and check if part has meaningful content
                stripped_part = part.strip()
                # Keep part if it has actual content (not just empty or whitespace)
                if stripped_part:
                    cleaned_parts.append(stripped_part)
            
            # Join back with pipes - only include parts with actual content
            cleaned_summary = ' . '.join(cleaned_parts)
            
            logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
            return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str) -> str:
        """
        Generate a precise 30–60 word consultation summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word consultation structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a medical-legal consultation specialist.

    TASK:
    Create a concise, factual consultation summary using ONLY information explicitly stated in the long summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:

    [Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Key Findings:[value] | Medication:[value] | Treatments:[value] | Clinical Assessment:[value] | Plan:[value] | MMI Status:[value] | Work Status:[value] | Critical Finding:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
    - NEVER fabricate: no invented dates, meds, findings, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.
    - First three fields (Report Title, Author, Date) appear without keys
    - All other fields use key-value format: Key:[value]

    CONTENT PRIORITY (only if provided in the long summary):
    1. Report Title  
    2. Author  
    3. Visit Date  
    4. Body parts  
    5. Diagnosis  
    6. Key objective findings  
    7. Medications  
    8. Treatments provided  
    9. Clinical assessment  
    10. Plan/next steps  
    11. MMI status  
    12. Work status  
    13. Critical finding

    ABSOLUTELY FORBIDDEN:
    - assumptions, interpretations, invented medications, or inferred diagnoses
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")

    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    CONSULTATION LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured consultation summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "long_summary": long_summary
            })
            summary = response.content.strip()

            # Normalize whitespace only - no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()

            # Validate word count
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Consultation summary out of range ({wc} words). Fixing...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary contained {wc} words. Rewrite it to be **between 30 and 60** words. "
                        "Do NOT add fabricated data. Preserve all factual elements. Maintain the key-value pipe-delimited format: [Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after fix

            logger.info(f"✅ Consultation summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Consultation summary generation failed: {e}")
            return "Summary unavailable due to processing error."
 
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Consulting Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Consulting Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'diagnosis': r'Primary Diagnosis:\s*([^\n]+)',
            'complaint': r'Primary Complaint:\s*([^\n]+)',
            'recommendations': r'TREATMENT RECOMMENDATIONS(.*?)(?:\n\n|\n[A-Z]|$)',
            'restrictions': r'Work Restrictions:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} consultation")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"for {extracted['diagnosis'][:80]}")
        elif 'complaint' in extracted:
            parts.append(f"for {extracted['complaint'][:80]}")
        
        # Add recommendations
        if 'recommendations' in extracted:
            # Take first line of recommendations
            first_rec = extracted['recommendations'].split('\n')[0].replace('•', '').strip()[:60]
            if first_rec:
                parts.append(f"Recommendations: {first_rec}")
        
        # Add work restrictions
        if 'restrictions' in extracted:
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with follow-up planned", "for ongoing management", "and progress evaluation"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary

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