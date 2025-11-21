"""
PR-2 Progress Report Enhanced Extractor - Full Context
Optimized for accuracy using Gemini-style full-document processing
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

logger = logging.getLogger("document_ai")


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning for clinical progress tracking
    - Optimized for PR-2 specific clinical workflow patterns
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex patterns for PR-2 specific content
        self.progress_patterns = {
            'status': re.compile(r'\b(improved|stable|worsened|resolved|unchanged|progressing|regressing)\b', re.IGNORECASE),
            'work_status': re.compile(r'\b(ttd|modified duty|full duty|light duty|no restrictions|work restrictions)\b', re.IGNORECASE),
            'treatment': re.compile(r'\b(pt|physical therapy|injection|medication|therapy|exercise)\b', re.IGNORECASE)
        }
        
        logger.info("✅ PR2ExtractorChained initialized (Full Context)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract PR-2 data with FULL CONTEXT.
        Returns dictionary with long_summary and short_summary like QME extractor.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (PR-2 Progress Report)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING PR-2 EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Check document size
            text_length = len(text)
            token_estimate = text_length // 4
            logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
            
            if token_estimate > 120000:
                logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
                logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
            
            # Stage 1: Extract with FULL CONTEXT
            raw_data = self._extract_full_context(
                text=text,
                doc_type=doc_type,
                fallback_date=fallback_date
            )
            
            # Stage 3: Build comprehensive long summary from ALL raw data
            long_summary = self._build_comprehensive_long_summary(raw_data, doc_type, fallback_date)
            
            # Stage 4: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context PR-2 extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted progress data from complete {len(text):,} char document")
            
            logger.info("=" * 80)
            logger.info("✅ PR-2 EXTRACTION COMPLETE (FULL CONTEXT)")
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
                "long_summary": f"PR-2 extraction failed: {str(e)}",
                "short_summary": "PR-2 summary not available"
            }

    def _extract_full_context(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Extract with FULL document context.
        Optimized for PR-2 Progress Report specific patterns and clinical workflow.
        """
        logger.info("🔍 Processing ENTIRE PR-2 document in single context window...")
        
        # Final PR-2 System Prompt - Complete Workers' Compensation Focus
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation medical documentation specialist analyzing a COMPLETE PR-2 Progress Report.

PRIMARY PURPOSE: Extract critical information for workers' compensation claims administrators to:
1. Assess work capacity and disability status
2. Process treatment authorization requests
3. Track patient progress and treatment effectiveness
4. Plan future care and return-to-work timeline

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE PR-2 document at once, allowing you to:
- Track treatment effectiveness from baseline to current status
- Identify ALL treatment authorization requests
- Connect objective findings with work capacity changes
- Extract complete medication regimen and changes

⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
- If NOT explicitly mentioned, return EMPTY string "" or empty list []
- DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps
- DO NOT add typical values, standard dosages, or common restrictions

2. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
- Extract ONLY medications explicitly listed as "current", "continuing", or "prescribed"
- Include dosage, frequency, and route ONLY if explicitly stated
- DO NOT extract discontinued, past, or recommended future medications (unless clearly marked as "new prescription")
- If document says "Patient on Gabapentin", extract ONLY: {{"name": "Gabapentin", "dose": ""}}
- If document says "Continue Gabapentin 300mg TID", extract: {{"name": "Gabapentin", "dose": "300mg TID"}}

3. **WORK RESTRICTIONS - EXACT WORDING ONLY**
- Use EXACT phrases from document
- If document says "no lifting", extract "no lifting" (NOT "no lifting >10 lbs")
- If document says "modified duty", extract "modified duty" (NOT "modified duty with restrictions")
- DO NOT add weight limits, time limits, or specifics not explicitly stated

4. **EMPTY FIELDS ARE BETTER THAN GUESSED FIELDS**
- If you cannot find information, leave field empty
- DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""

5. **NO CLINICAL ASSUMPTIONS OR PREDICTIONS**
- DO NOT infer patient compliance from attendance records
- DO NOT predict treatment effectiveness from partial data
- DO NOT extrapolate progress from single data points

6. **TREATING PHYSICIAN/AUTHOR DETECTION**:
   - Identify the author who signed the report as the "treating_physician" name (e.g., from signature block, "Dictated by:", or closing statement).
   - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
   - Extract specialty and facility only if explicitly stated near the signature.
   - If no clear signer is found, leave "name" empty.

PR-2 EXTRACTION FOCUS - 4 CORE WORKERS' COMPENSATION AREAS:

I. WORK STATUS AND IMPAIRMENT (HIGHEST PRIORITY FOR WC CLAIMS)
II. TREATMENT AUTHORIZATION REQUESTS (MOST TIME-SENSITIVE)
III. PATIENT PROGRESS AND CURRENT STATUS
IV. NEXT STEPS AND PLANNING

Now analyze this COMPLETE PR-2 Progress Report and extract ALL critical Workers' Compensation information:
""")

        # Final PR-2 User Prompt - Complete Workers' Compensation Structure
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE PR-2 PROGRESS REPORT TEXT:

{full_document_text}

Extract into WORKERS' COMPENSATION structured JSON:

{{
"report_metadata": {{
    "report_type": "PR-2 Progress Report",
    "report_date": "",
    "visit_date": "",
    "time_since_injury": "",
    "time_since_last_visit": "",
    "reason_for_report": []
}},

"patient_visit_info": {{
    "patient_name": "",
    "patient_dob": "",
    "patient_age": "",
    "date_of_injury": "",
    "occupation": "",
    "employer": "",
    "claims_administrator": "",
    "treating_physician": {{
    "name": "",
    "specialty": "",
    "facility": ""
    }}
}},

"chief_complaint": {{
    "primary_complaint": "",
    "location": "",
    "description": ""
}},

"subjective_assessment": {{
    "pain_score_current": "",
    "pain_score_previous": "",
    "symptom_changes": "",
    "functional_status_patient_reported": "",
    "patient_compliance": ""
}},

"objective_status": {{
    "physical_exam_findings": "",
    "rom_measurements": "",
    "strength_testing": "",
    "gait_assessment": "",
    "neurological_findings": "",
    "functional_limitations_observed": []
}},

"diagnosis_icd10": {{
    "primary_diagnosis": "",
    "icd10_code": "",
    "secondary_diagnoses": []
}},

"current_medications": [],

"medication_changes": {{
    "new_medications": [],
    "dosage_changes": [],
    "discontinued_medications": []
}},

"prior_treatment": {{
    "completed_treatments": [],
    "therapy_sessions_completed": "",
    "procedures_performed": [],
    "imaging_studies_completed": []
}},

"treatment_effectiveness": {{
    "patient_response": "",
    "objective_improvements": [],
    "functional_gains": "",
    "barriers_to_progress": ""
}},

"treatment_authorization_request": {{
    "primary_request": "",
    "secondary_requests": [],
    "requested_frequency": "",
    "requested_duration": "",
    "body_part_location": "",
    "medical_necessity_rationale": "",
    "objective_evidence_supporting_request": ""
}},

"work_status": {{
    "current_status": "",
    "status_effective_date": "",
    "work_limitations": [],
    "work_status_rationale": "",
    "changes_from_previous_status": "",
    "expected_return_to_work_date": ""
}},

"follow_up_plan": {{
    "next_appointment_date": "",
    "purpose_of_next_visit": "",
    "specialist_referrals_requested": [],
    "mmi_ps_anticipated_date": "",
    "return_sooner_if": ""
}},

"critical_findings": []
}}

⚠️ MANDATORY EXTRACTION RULES:
1. "current_medications": Extract ALL current medications with exact doses from document
2. "work_limitations": Use EXACT wording (don't add weight/time limits not stated)
3. "treatment_authorization_request": The MOST CRITICAL field - be specific
4. "critical_findings": Only 3-5 most actionable items for claims administrator
5. Empty fields are acceptable if information not stated in document
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context PR-2 extraction...")
            
            # Single LLM call with FULL document context
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context PR-2 extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted progress data from complete {len(text):,} char document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context PR-2 extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted raw data with detailed headings.
        Similar to QME extractor structure.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted PR-2 data...")
        
        sections = []
        
        # Section 1: REPORT OVERVIEW
        sections.append("📋 REPORT OVERVIEW")
        sections.append("-" * 50)
        
        report_metadata = raw_data.get("report_metadata", {})
        patient_visit_info = raw_data.get("patient_visit_info", {})
        treating_physician = patient_visit_info.get("treating_physician", {})
        
        physician_name = treating_physician.get("name", "")
        specialty = treating_physician.get("specialty", "")
        report_date = report_metadata.get("report_date", fallback_date)
        visit_date = report_metadata.get("visit_date", "")
        
        overview_lines = [
            f"Document Type: {doc_type}",
            f"Report Date: {report_date}",
            f"Visit Date: {visit_date}" if visit_date else "Visit Date: Not specified",
            f"Treating Physician: {physician_name}",
            f"Specialty: {specialty}" if specialty else "Specialty: Not specified",
            f"Time Since Injury: {report_metadata.get('time_since_injury', 'Not specified')}",
            f"Time Since Last Visit: {report_metadata.get('time_since_last_visit', 'Not specified')}"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PATIENT INFORMATION
        sections.append("\n👤 PATIENT INFORMATION")
        sections.append("-" * 50)
        
        patient_lines = [
            f"Name: {patient_visit_info.get('patient_name', 'Not specified')}",
            f"Date of Birth: {patient_visit_info.get('patient_dob', 'Not specified')}",
            f"Age: {patient_visit_info.get('patient_age', 'Not specified')}",
            f"Date of Injury: {patient_visit_info.get('date_of_injury', 'Not specified')}",
            f"Occupation: {patient_visit_info.get('occupation', 'Not specified')}",
            f"Employer: {patient_visit_info.get('employer', 'Not specified')}",
            f"Claims Administrator: {patient_visit_info.get('claims_administrator', 'Not specified')}"
        ]
        sections.append("\n".join(patient_lines))
        
        # Section 3: CHIEF COMPLAINT
        sections.append("\n🎯 CHIEF COMPLAINT")
        sections.append("-" * 50)
        
        chief_complaint = raw_data.get("chief_complaint", {})
        complaint_lines = [
            f"Primary Complaint: {chief_complaint.get('primary_complaint', 'Not specified')}",
            f"Location: {chief_complaint.get('location', 'Not specified')}",
            f"Description: {chief_complaint.get('description', 'Not specified')}"
        ]
        sections.append("\n".join(complaint_lines))
        
        # Section 4: SUBJECTIVE ASSESSMENT
        sections.append("\n💬 SUBJECTIVE ASSESSMENT")
        sections.append("-" * 50)
        
        subjective = raw_data.get("subjective_assessment", {})
        subjective_lines = [
            f"Current Pain Score: {subjective.get('pain_score_current', 'Not specified')}",
            f"Previous Pain Score: {subjective.get('pain_score_previous', 'Not specified')}",
            f"Symptom Changes: {subjective.get('symptom_changes', 'Not specified')}",
            f"Functional Status (Patient Reported): {subjective.get('functional_status_patient_reported', 'Not specified')}",
            f"Patient Compliance: {subjective.get('patient_compliance', 'Not specified')}"
        ]
        sections.append("\n".join(subjective_lines))
        
        # Section 5: OBJECTIVE FINDINGS
        sections.append("\n🔬 OBJECTIVE FINDINGS")
        sections.append("-" * 50)
        
        objective = raw_data.get("objective_status", {})
        objective_lines = [
            f"Physical Exam Findings: {objective.get('physical_exam_findings', 'Not specified')}",
            f"ROM Measurements: {objective.get('rom_measurements', 'Not specified')}",
            f"Strength Testing: {objective.get('strength_testing', 'Not specified')}",
            f"Gait Assessment: {objective.get('gait_assessment', 'Not specified')}",
            f"Neurological Findings: {objective.get('neurological_findings', 'Not specified')}"
        ]
        
        # Functional limitations
        functional_limitations = objective.get("functional_limitations_observed", [])
        if functional_limitations:
            objective_lines.append("\nFunctional Limitations Observed:")
            for limitation in functional_limitations[:5]:
                if isinstance(limitation, dict):
                    desc = limitation.get("limitation", "")
                    if desc:
                        objective_lines.append(f"  • {desc}")
                elif limitation:
                    objective_lines.append(f"  • {limitation}")
        
        sections.append("\n".join(objective_lines))
        
        # Section 6: DIAGNOSIS
        sections.append("\n🏥 DIAGNOSIS")
        sections.append("-" * 50)
        
        diagnosis = raw_data.get("diagnosis_icd10", {})
        diagnosis_lines = []
        
        primary_dx = diagnosis.get("primary_diagnosis", "")
        icd10_code = diagnosis.get("icd10_code", "")
        
        if primary_dx:
            dx_text = primary_dx
            if icd10_code:
                dx_text += f" [{icd10_code}]"
            diagnosis_lines.append(f"Primary Diagnosis: {dx_text}")
        
        secondary_dx = diagnosis.get("secondary_diagnoses", [])
        if secondary_dx:
            diagnosis_lines.append("\nSecondary Diagnoses:")
            for dx in secondary_dx[:3]:  # Limit to 3 secondary diagnoses
                if isinstance(dx, dict):
                    dx_name = dx.get("diagnosis", "")
                    if dx_name:
                        diagnosis_lines.append(f"  • {dx_name}")
                elif dx and str(dx).strip():
                    diagnosis_lines.append(f"  • {dx}")
        
        sections.append("\n".join(diagnosis_lines) if diagnosis_lines else "No diagnoses extracted")
        
        # Section 7: MEDICATIONS
        sections.append("\n💊 MEDICATIONS")
        sections.append("-" * 50)
        
        medication_lines = []
        
        # Current medications
        current_meds = raw_data.get("current_medications", [])
        if current_meds:
            medication_lines.append("Current Medications:")
            for med in current_meds[:8]:  # Limit to 8 medications
                if isinstance(med, dict):
                    med_name = med.get("name", "")
                    med_dose = med.get("dose", "")
                    if med_name:
                        if med_dose:
                            medication_lines.append(f"  • {med_name} - {med_dose}")
                        else:
                            medication_lines.append(f"  • {med_name}")
                elif med:
                    medication_lines.append(f"  • {med}")
        else:
            medication_lines.append("No current medications listed")
        
        # Medication changes
        med_changes = raw_data.get("medication_changes", {})
        if med_changes:
            # New medications
            new_meds = med_changes.get("new_medications", [])
            if new_meds:
                medication_lines.append("\nNew Medications:")
                for med in new_meds[:3]:
                    if isinstance(med, dict):
                        med_name = med.get("name", "")
                        med_dose = med.get("dose", "")
                        if med_name:
                            if med_dose:
                                medication_lines.append(f"  • {med_name} - {med_dose}")
                            else:
                                medication_lines.append(f"  • {med_name}")
                    elif med:
                        medication_lines.append(f"  • {med}")
            
            # Dosage changes
            dose_changes = med_changes.get("dosage_changes", [])
            if dose_changes:
                medication_lines.append("\nDosage Changes:")
                for change in dose_changes[:3]:
                    if isinstance(change, dict):
                        med_name = change.get("medication", "")
                        change_details = change.get("change", "")
                        if med_name:
                            if change_details:
                                medication_lines.append(f"  • {med_name} - {change_details}")
                            else:
                                medication_lines.append(f"  • {med_name}")
                    elif change:
                        medication_lines.append(f"  • {change}")
        
        sections.append("\n".join(medication_lines))
        
        # Section 8: TREATMENT EFFECTIVENESS
        sections.append("\n📈 TREATMENT EFFECTIVENESS")
        sections.append("-" * 50)
        
        effectiveness = raw_data.get("treatment_effectiveness", {})
        effectiveness_lines = []
        
        if effectiveness.get("patient_response"):
            effectiveness_lines.append(f"Patient Response: {effectiveness['patient_response']}")
        
        if effectiveness.get("functional_gains"):
            effectiveness_lines.append(f"Functional Gains: {effectiveness['functional_gains']}")
        
        # Objective improvements
        obj_improvements = effectiveness.get("objective_improvements", [])
        if obj_improvements:
            effectiveness_lines.append("\nObjective Improvements:")
            for improvement in obj_improvements[:5]:
                if isinstance(improvement, dict):
                    desc = improvement.get("improvement", "")
                    if desc:
                        effectiveness_lines.append(f"  • {desc}")
                elif improvement:
                    effectiveness_lines.append(f"  • {improvement}")
        
        if effectiveness.get("barriers_to_progress"):
            effectiveness_lines.append(f"\nBarriers to Progress: {effectiveness['barriers_to_progress']}")
        
        sections.append("\n".join(effectiveness_lines) if effectiveness_lines else "No treatment effectiveness details extracted")
        
        # Section 9: TREATMENT AUTHORIZATION REQUEST (MOST CRITICAL)
        sections.append("\n✅ TREATMENT AUTHORIZATION REQUEST")
        sections.append("-" * 50)
        
        treatment_auth = raw_data.get("treatment_authorization_request", {})
        auth_lines = []
        
        primary_request = treatment_auth.get("primary_request", "")
        if primary_request:
            auth_lines.append(f"Primary Request: {primary_request}")
        
        secondary_requests = treatment_auth.get("secondary_requests", [])
        if secondary_requests:
            auth_lines.append("\nSecondary Requests:")
            for request in secondary_requests[:3]:
                if isinstance(request, dict):
                    req_desc = request.get("request", "")
                    if req_desc:
                        auth_lines.append(f"  • {req_desc}")
                elif request:
                    auth_lines.append(f"  • {request}")
        
        if treatment_auth.get("requested_frequency"):
            auth_lines.append(f"Requested Frequency: {treatment_auth['requested_frequency']}")
        
        if treatment_auth.get("requested_duration"):
            auth_lines.append(f"Requested Duration: {treatment_auth['requested_duration']}")
        
        if treatment_auth.get("medical_necessity_rationale"):
            auth_lines.append(f"Medical Necessity Rationale: {treatment_auth['medical_necessity_rationale']}")
        
        sections.append("\n".join(auth_lines) if auth_lines else "No treatment authorization requests")
        
        # Section 10: WORK STATUS (HIGHEST PRIORITY)
        sections.append("\n💼 WORK STATUS")
        sections.append("-" * 50)
        
        work_status = raw_data.get("work_status", {})
        work_lines = []
        
        current_status = work_status.get("current_status", "")
        if current_status:
            work_lines.append(f"Current Status: {current_status}")
        
        status_effective = work_status.get("status_effective_date", "")
        if status_effective:
            work_lines.append(f"Status Effective Date: {status_effective}")
        
        # Work limitations (EXACT wording)
        work_limitations = work_status.get("work_limitations", [])
        if work_limitations:
            work_lines.append("\nWork Limitations:")
            for limitation in work_limitations[:8]:
                if isinstance(limitation, dict):
                    desc = limitation.get("limitation", "")
                    if desc:
                        work_lines.append(f"  • {desc}")
                elif limitation:
                    work_lines.append(f"  • {limitation}")
        
        if work_status.get("work_status_rationale"):
            work_lines.append(f"\nWork Status Rationale: {work_status['work_status_rationale']}")
        
        if work_status.get("changes_from_previous_status"):
            work_lines.append(f"Changes from Previous Status: {work_status['changes_from_previous_status']}")
        
        if work_status.get("expected_return_to_work_date"):
            work_lines.append(f"Expected Return to Work Date: {work_status['expected_return_to_work_date']}")
        
        sections.append("\n".join(work_lines) if work_lines else "No work status information extracted")
        
        # Section 11: FOLLOW-UP PLAN
        sections.append("\n📅 FOLLOW-UP PLAN")
        sections.append("-" * 50)
        
        follow_up = raw_data.get("follow_up_plan", {})
        follow_lines = []
        
        if follow_up.get("next_appointment_date"):
            follow_lines.append(f"Next Appointment Date: {follow_up['next_appointment_date']}")
        
        if follow_up.get("purpose_of_next_visit"):
            follow_lines.append(f"Purpose of Next Visit: {follow_up['purpose_of_next_visit']}")
        
        # Specialist referrals
        specialist_referrals = follow_up.get("specialist_referrals_requested", [])
        if specialist_referrals:
            follow_lines.append("\nSpecialist Referrals Requested:")
            for referral in specialist_referrals[:3]:
                if isinstance(referral, dict):
                    specialty = referral.get("specialty", "")
                    if specialty:
                        follow_lines.append(f"  • {specialty}")
                elif referral:
                    follow_lines.append(f"  • {referral}")
        
        if follow_up.get("mmi_ps_anticipated_date"):
            follow_lines.append(f"MMI/P&S Anticipated Date: {follow_up['mmi_ps_anticipated_date']}")
        
        if follow_up.get("return_sooner_if"):
            follow_lines.append(f"Return Sooner If: {follow_up['return_sooner_if']}")
        
        sections.append("\n".join(follow_lines) if follow_lines else "No specific follow-up plan")
        
        # Section 12: CRITICAL FINDINGS
        sections.append("\n🚨 CRITICAL FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            for finding in critical_findings[:5]:
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
        Generate a precise 30–60 word PR-2 structured summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word PR-2 structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a Workers' Compensation medical-legal extraction specialist.

    TASK:
    Create a concise, accurate PR-2 Progress Report summary using ONLY the information explicitly present in the long summary.

    STRICT OUTPUT FORMAT (include fields only when data exists):
    [Title] | [Author] | [Date] | Work Status:[value] | Restrictions:[value] | Meds:[value] | Physical Exam:[value] | Treatment Progress:[value] | Auth Requests:[value] | Follow-up:[value] | Critical Finding:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
    - NEVER fabricate: no invented dates, meds, restrictions, exam findings, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.
    - Use the shortest, clearest key names:
    • Title = Report title (without key)
    • Author = MD/DO/PA/NP or signer (without key)  
    • Date = Visit or exam date (without key)
    • Work Status:[value] = current status (if given)  
    • Restrictions:[value] = physical restrictions (if given)  
    • Meds:[value] = medications explicitly listed (if given)
    • Physical Exam:[value] = objective exam findings only (if given)
    • Treatment Progress:[value] = progress or response (if given)
    • Auth Requests:[value] = items requested for authorization (if given)
    • Follow-up:[value] = next appointment or instruction (if given)
    • Critical Finding:[value] = one most clinically important finding (if given)

    CONTENT PRIORITY (only if provided in the long summary):
    1. Report Title  
    2. Author  
    3. Visit Date  
    4. Work status & restrictions  
    5. Medications  
    6. Physical examination details  
    7. Treatment progress  
    8. Authorization requests  
    9. Follow-up plan  
    10. Critical finding

    ABSOLUTELY FORBIDDEN:
    - assumptions, interpretations, invented medications, or inferred diagnoses
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")

    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    LONG SUMMARY:

    {long_summary}

    Now generate a 30–60 word PR-2 structured summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})
            summary = response.content.strip()

            # Clean output - only whitespace, no pipe cleaning
            summary = re.sub(r'\s+', ' ', summary).strip()

            # Word count check
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ PR-2 summary out of range ({wc} words). Attempting auto-fix.")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous output contained {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy and key-value pipe-delimited format. Do NOT add fabricated content."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r'\s+', ' ', fixed.content.strip())
                # No pipe cleaning after fix

            logger.info(f"✅ PR-2 summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ PR-2 short summary generation failed: {e}")
            return "Summary unavailable due to processing error."
   
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Treating Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Treating Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'work_status': r'Current Status:\s*([^\n]+)',
            'restrictions': r'Work Limitations:(.*?)(?:\n\n|\n[A-Z]|$)',
            'progress': r'Patient Response:\s*([^\n]+)',
            'requests': r'Primary Request:\s*([^\n]+)',
            'followup': r'Next Appointment Date:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} progress report")
        
        # Add work status
        if 'work_status' in extracted:
            parts.append(f"Work status: {extracted['work_status'][:40]}")
        
        # Add restrictions
        if 'restrictions' in extracted:
            # Take first line of restrictions
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        # Add progress
        if 'progress' in extracted:
            parts.append(f"Progress: {extracted['progress'][:60]}")
        
        # Add requests
        if 'requests' in extracted:
            parts.append(f"Request: {extracted['requests'][:60]}")
        
        # Add follow-up
        if 'followup' in extracted:
            parts.append(f"Follow-up: {extracted['followup']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with ongoing clinical management", "for workers' compensation case", "and functional improvement tracking"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return fallback result structure matching NEW WC structure"""
        return {
            "report_metadata": {
                "report_type": "PR-2 Progress Report",
                "report_date": fallback_date,
                "visit_date": "",
                "time_since_injury": "",
                "time_since_last_visit": "",
                "reason_for_report": []
            },
            "patient_visit_info": {
                "patient_name": "",
                "patient_dob": "",
                "patient_age": "",
                "date_of_injury": "",
                "occupation": "",
                "employer": "",
                "claims_administrator": "",
                "treating_physician": {
                    "name": "",
                    "specialty": "",
                    "facility": ""
                }
            },
            "chief_complaint": {
                "primary_complaint": "",
                "location": "",
                "description": ""
            },
            "subjective_assessment": {
                "pain_score_current": "",
                "pain_score_previous": "",
                "symptom_changes": "",
                "functional_status_patient_reported": "",
                "patient_compliance": ""
            },
            "objective_status": {
                "physical_exam_findings": "",
                "rom_measurements": "",
                "strength_testing": "",
                "gait_assessment": "",
                "neurological_findings": "",
                "functional_limitations_observed": []
            },
            "diagnosis_icd10": {
                "primary_diagnosis": "",
                "icd10_code": "",
                "secondary_diagnoses": []
            },
            "current_medications": [],
            "medication_changes": {
                "new_medications": [],
                "dosage_changes": [],
                "discontinued_medications": []
            },
            "prior_treatment": {
                "completed_treatments": [],
                "therapy_sessions_completed": "",
                "procedures_performed": [],
                "imaging_studies_completed": []
            },
            "treatment_effectiveness": {
                "patient_response": "",
                "objective_improvements": [],
                "functional_gains": "",
                "barriers_to_progress": ""
            },
            "treatment_authorization_request": {
                "primary_request": "",
                "secondary_requests": [],
                "requested_frequency": "",
                "requested_duration": "",
                "body_part_location": "",
                "medical_necessity_rationale": "",
                "objective_evidence_supporting_request": ""
            },
            "work_status": {
                "current_status": "",
                "status_effective_date": "",
                "work_limitations": [],
                "work_status_rationale": "",
                "changes_from_previous_status": "",
                "expected_return_to_work_date": ""
            },
            "follow_up_plan": {
                "next_appointment_date": "",
                "purpose_of_next_visit": "",
                "specialist_referrals_requested": [],
                "mmi_ps_anticipated_date": "",
                "return_sooner_if": ""
            },
            "critical_findings": []
        }