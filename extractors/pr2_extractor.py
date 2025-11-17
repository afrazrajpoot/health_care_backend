"""
PR-2 Progress Report Enhanced Extractor - Full Context with Context-Awareness
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction using DocumentContextAnalyzer guidance
    - Chain-of-thought reasoning for clinical progress tracking
    - Optimized for PR-2 specific clinical workflow patterns
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex patterns for PR-2 specific content
        self.progress_patterns = {
            'status': re.compile(r'\b(improved|stable|worsened|resolved|unchanged|progressing|regressing)\b', re.IGNORECASE),
            'work_status': re.compile(r'\b(ttd|modified duty|full duty|light duty|no restrictions|work restrictions)\b', re.IGNORECASE),
            'treatment': re.compile(r'\b(pt|physical therapy|injection|medication|therapy|exercise)\b', re.IGNORECASE)
        }
        
        logger.info("✅ PR2ExtractorChained initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,  # NEW: from DocumentContextAnalyzer
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract PR-2 data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (PR-2 Progress Report)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING PR-2 EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        if context_analysis:
            primary_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Primary Physician: {primary_physician.get('name', 'Unknown')}")
            logger.info(f"   Confidence: {primary_physician.get('confidence', 'Unknown')}")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Critical Locations: {list(critical_locations.keys())}")
        else:
            logger.warning("⚠️ No context analysis provided - proceeding without guidance")
            logger.info("📋 STARTING PR-2 EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
            logger.info("=" * 80)
        
        # Check document size
        text_length = len(text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Extract with FULL CONTEXT and contextual guidance
        raw_result = self._extract_full_context_with_guidance(
            text=text,
            doc_type=doc_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )
        
        # Stage 2: Override physician if context identified one with high confidence
        if context_analysis:
            context_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            if context_physician.get("name") and context_physician.get("confidence") in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified physician: {context_physician.get('name')}")
                raw_result["treating_physician_name"] = context_physician.get("name")
        
        # Stage 3: Fallback to DoctorDetector if no physician identified
        if not raw_result.get("treating_physician_name"):
            logger.info("🔍 No physician from context/extraction, using DoctorDetector...")
            physician_name = self._detect_treating_physician(text, page_zones)
            raw_result["treating_physician_name"] = physician_name
        
        # Stage 4: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 5: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info("=" * 80)
        logger.info("✅ PR-2 EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)
        
        return final_result

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        Optimized for PR-2 Progress Report specific patterns and clinical workflow.
        """
        logger.info("🔍 Processing ENTIRE PR-2 document in single context window with guidance...")
        
        # Extract guidance from context analysis
        primary_physician = ""
        focus_sections = []
        clinical_timeline = {}
        physician_reasoning = ""
        ambiguities = []
        
        if context_analysis:
            phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            primary_physician = phys_analysis.get("name", "")
            physician_reasoning = phys_analysis.get("reasoning", "")
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            clinical_timeline = context_analysis.get("clinical_timeline", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
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

        CONTEXTUAL GUIDANCE PROVIDED:
        {context_guidance}

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

        PR-2 EXTRACTION FOCUS - 4 CORE WORKERS' COMPENSATION AREAS:

        ═══════════════════════════════════════════════════════════════════
        I. WORK STATUS AND IMPAIRMENT (HIGHEST PRIORITY FOR WC CLAIMS)
        ═══════════════════════════════════════════════════════════════════

        This is the PRIMARY focus for claims administrators.

        **Key Elements to Extract:**

        A. CURRENT WORK STATUS (Use EXACT terminology from document)
        - "Temporarily Totally Disabled (TTD)"
        - "Temporarily Partially Disabled (TPD)"
        - "Permanent and Stationary (P&S)" / "Maximum Medical Improvement (MMI)"
        - "Released to Full Duty"
        - "Released to Modified Duty"
        - "Off work"

        B. WORK LIMITATIONS/RESTRICTIONS (EXACT wording)
        Extract SPECIFIC functional limitations as stated:
        - Lifting: "No lifting >20 lbs" (if stated), NOT "lifting restrictions"
        - Standing: "No prolonged standing" (exact phrase), NOT "limited standing"
        - Sitting: "Sitting as tolerated" (if stated)
        - Repetitive tasks: "No repetitive bending/twisting" (exact phrase)
        - Hours: "4-hour work day" (if stated)
        
        ❌ WRONG: Document says "lifting restrictions" → DO NOT extract "no lifting >10 lbs"
        ✅ CORRECT: Document says "no lifting over 25 pounds" → Extract "no lifting >25 lbs"

        C. WORK STATUS RATIONALE
        - WHY has work status changed or remained the same?
        - Link to objective findings: "Unable to return to work due to persistent 7/10 pain and limited ROM"
        - Link to functional capacity: "Cleared for modified duty due to improved strength (4/5)"

        D. NEW OR CHANGED LIMITATIONS
        - What changed since last report?
        - "Lifting increased from 10 lbs to 20 lbs"
        - "Standing tolerance improved from 2 hours to 4 hours"

        ═══════════════════════════════════════════════════════════════════
        II. TREATMENT AUTHORIZATION REQUESTS (MOST TIME-SENSITIVE)
        ═══════════════════════════════════════════════════════════════════

        This directly impacts claims processing and must be extracted with precision.

        **Key Elements to Extract:**

        A. SPECIFIC TREATMENT REQUEST
        - WHAT is being requested? Be exact.
            * "Continue physical therapy 2x/week for 4 additional weeks"
            * "MRI lumbar spine without contrast"
            * "ESI L5-S1 for persistent radicular symptoms"
            * "Increase Gabapentin from 300mg to 600mg TID"
        
        Include:
        - Treatment type
        - Frequency (if applicable)
        - Duration (if applicable)
        - Body part/location (if applicable)

        B. MEDICAL NECESSITY/RATIONALE
        - WHY is this treatment needed?
        - Objective evidence: "MRI needed due to failed conservative care and positive straight leg raise"
        - Progress evidence: "Continue PT due to 50% improvement in ROM after 6 sessions"
        - Functional impact: "Injection recommended to enable return to work"

        C. PRIOR AUTHORIZATION STATUS
        - Was this treatment previously authorized?
        - Is this a continuation or new request?

        ═══════════════════════════════════════════════════════════════════
        III. PATIENT PROGRESS AND CURRENT STATUS
        ═══════════════════════════════════════════════════════════════════

        Provides medical context for work capacity and treatment decisions.

        **Key Elements to Extract:**

        A. SUBJECTIVE IMPROVEMENT (Patient-Reported)
        - Pain scores: "Pain decreased from 7/10 to 4/10" (exact numbers if stated)
        - Symptom changes: "Patient reports improved sleep", "Numbness resolved"
        - Functional improvements: "Able to walk 2 blocks vs. unable previously"

        B. OBJECTIVE FINDINGS (Examination Results)
        - ROM measurements: "L knee flexion 110° (improved from 90°)"
        - Strength testing: "Quad strength 4/5 (improved from 3/5)"
        - Gait: "Antalgic gait improved, now walks without limp"
        - Physical exam: "Decreased swelling, full ROM achieved"

        C. CURRENT MEDICATIONS (Complete List with Doses)
        Extract ONLY medications explicitly listed as "current" or "continuing":
        
        Format: {{"name": "Drug name", "dose": "amount + frequency", "purpose": "indication if stated"}}
        
        Examples:
        - "Gabapentin 300mg three times daily for neuropathic pain"
        - "Meloxicam 15mg once daily"
        - "Tramadol 50mg as needed for breakthrough pain"
        
        ❌ DO NOT include:
        - Discontinued medications
        - Past medications
        - Medications mentioned in history only

        D. MEDICATION CHANGES
        - New medications prescribed
        - Dosage adjustments
        - Discontinued medications (with reason if stated)

        ═══════════════════════════════════════════════════════════════════
        IV. NEXT STEPS AND PLANNING
        ═══════════════════════════════════════════════════════════════════

        Dictates future claim management and timeline.

        **Key Elements to Extract:**

        A. FOLLOW-UP DATE
        - When is next scheduled appointment?
        - "Return in 4 weeks" or specific date if stated

        B. SPECIALIST REFERRALS
        - Requested referrals: "Referral to Orthopedic Surgery for surgical evaluation"
        - Purpose of referral

        C. MMI/P&S STATUS AND TIMING
        - Is patient approaching MMI/P&S?
        - "Anticipate P&S status in 6-8 weeks if current progress continues"
        - Will a PR-3/PR-4 report be needed?

        D. CRITICAL FINDINGS (TOP 3-5 MOST IMPORTANT ITEMS)
        Extract ONLY the most actionable items for claims administrators:
        
        Priority items:
        - Treatment authorization requests
        - Work status changes
        - Significant progress indicators
        - Failed treatments requiring escalation
        - Specialist referral needs
        
        DO NOT include:
        - Routine administrative details
        - Standard examination procedures
        - Minor symptom fluctuations

        ⚠️ EXTRACTION EXAMPLES:

        Example 1 - Work Restrictions (EXACT wording):
        ✅ CORRECT: Document states "Patient may lift up to 25 pounds occasionally, no repetitive bending"
        Extract: ["No lifting >25 lbs occasionally", "No repetitive bending"]

        ❌ WRONG: Document states "Patient has lifting restrictions"
        DO NOT extract: ["No lifting >10 lbs"] (specific limit not stated)

        Example 2 - Medication (EXACT details):
        ✅ CORRECT: Document states "Current Medications: Gabapentin 300mg TID, Meloxicam 15mg daily"
        Extract: [
        {{"name": "Gabapentin", "dose": "300mg TID"}},
        {{"name": "Meloxicam", "dose": "15mg daily"}}
        ]

        ❌ WRONG: Document states "Patient taking Gabapentin"
        DO NOT extract: {{"name": "Gabapentin", "dose": "300mg TID"}} (dose not stated)

        Example 3 - Authorization Request:
        ✅ CORRECT: Document states "Request authorization for 6 additional PT sessions, 2x/week for 3 weeks, to continue strengthening program"
        Extract: {{
        "primary_request": "Continue physical therapy 2x/week for 3 weeks (6 sessions)",
        "rationale": "Continue strengthening program"
        }}

        ⚠️ FINAL REMINDER:
        - PRIMARY FOCUS: Work capacity, treatment authorization, patient progress, future planning
        - MEDICATIONS: Include complete current medication list with exact doses
        - WORK RESTRICTIONS: Use EXACT wording from document
        - If information is NOT in document, return EMPTY ("" or [])
        - ZERO tolerance for assumptions, inferences, or extrapolations

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
            "name": "{primary_physician}",
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

        # Build context guidance summary
        context_guidance_text = f"""
PRIMARY TREATING PHYSICIAN: {primary_physician or 'Not identified in context'}
REASONING: {physician_reasoning or 'See document for identification'}

FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'Subjective, Objective, Assessment, Plan (SOAP format)'}

CLINICAL TIMELINE CONTEXT:
{self._format_clinical_timeline(clinical_timeline)}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context PR-2 extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_guidance_text,
                "primary_physician": primary_physician or "Extract from document",
                "physician_reasoning": physician_reasoning or "Use signature and documentation sections",
                "focus_sections": ', '.join(focus_sections) if focus_sections else "SOAP format sections",
                "clinical_timeline": str(clinical_timeline),
                "ambiguities": str(ambiguities)
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

    def _format_clinical_timeline(self, timeline: Dict) -> str:
        """Format clinical timeline for context guidance"""
        if not timeline:
            return "No clinical timeline available"
        
        formatted = []
        for event_type, events in timeline.items():
            if events:
                formatted.append(f"{event_type.upper()}:")
                for event in events[:3]:  # Show top 3 events per type
                    formatted.append(f"  - {event.get('date', 'Unknown')}: {event.get('description', '')}")
        
        return "\n".join(formatted) if formatted else "No significant timeline events"

    def _detect_treating_physician(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Fallback: Detect treating physician using DoctorDetector"""
        logger.info("🔍 Fallback: Running DoctorDetector for treating physician...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Treating Physician detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid treating physician found")
            return ""

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result from extracted PR-2 data"""
        logger.info("🔨 Building initial PR-2 extraction result...")
        
        # Extract core progress information
        clinical_progress = raw_data.get("clinical_progress", {})
        report_metadata = raw_data.get("report_metadata", {})
        
        # Build comprehensive PR-2 summary
        summary_line = self._build_progress_narrative_summary(raw_data, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=report_metadata.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=raw_data.get("treating_physician_name", ""),
            specialty=clinical_progress.get("treating_physician", {}).get("specialty", ""),
            body_parts=[clinical_progress.get("body_part_assessed", "")] if clinical_progress.get("body_part_assessed") else [],
            raw_data=raw_data,
        )
        
        logger.info(f"✅ Initial PR-2 result built (Physician: {result.examiner_name})")
        return result

    def _build_progress_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build a clean, readable narrative summary for PR-2 progress tracking, with clear sections and simple text formatting (not JSON/dict).
        """
        def list_to_lines(lst):
            if not lst:
                return ""
            return "\n".join(f"- {item}" for item in lst if item)

        # Section 1: Work Status and Impairment
        ws = data.get('1. Work Status and Impairment', {})
        ws_lines = []
        if ws:
            if ws.get('Work Status'):
                ws_lines.append(f"Work Status: {ws['Work Status']}")
            if ws.get('New/Changed Limitations'):
                ws_lines.append("New/Changed Limitations:")
                ws_lines.append(list_to_lines(ws['New/Changed Limitations']))

        # Section 2: Treatment Authorization Requests
        ta = data.get('2. Treatment Authorization Requests', {})
        ta_lines = []
        if ta:
            if ta.get('Specific Request'):
                ta_lines.append(f"Specific Request: {ta['Specific Request']}")
            if ta.get('Medical Necessity/Rationale'):
                ta_lines.append(f"Medical Necessity/Rationale: {ta['Medical Necessity/Rationale']}")
            if ta.get('Objective Evidence Supporting Request'):
                ta_lines.append(f"Objective Evidence Supporting Request: {ta['Objective Evidence Supporting Request']}")

        # Section 3: Patient Progress and Current Status
        pr = data.get('3. Patient Progress and Current Status', {})
        pr_lines = []
        if pr:
            if pr.get('Subjective Improvement'):
                pr_lines.append(f"Subjective Improvement: {pr['Subjective Improvement']}")
            obj = pr.get('Objective Findings', {})
            if obj:
                pr_lines.append("Objective Findings:")
                for k, v in obj.items():
                    if isinstance(v, dict):
                        pr_lines.append(f"- {k}: " + ", ".join(f"{sk} {sv}" for sk, sv in v.items() if sv))
                    elif v:
                        pr_lines.append(f"- {k}: {v}")
            if pr.get('Functional Gains'):
                pr_lines.append(f"Functional Gains: {pr['Functional Gains']}")

        # Section 4: Next Steps and Planning
        np = data.get('4. Next Steps and Planning', {})
        np_lines = []
        if np:
            if np.get('Follow-up Date'):
                np_lines.append(f"Follow-up Date: {np['Follow-up Date']}")

        # Compose the final narrative
        sections = []
        if ws_lines:
            sections.append("**1. Work Status and Impairment**\n" + "\n".join(ws_lines))
        if ta_lines:
            sections.append("**2. Treatment Authorization Requests**\n" + "\n".join(ta_lines))
        if pr_lines:
            sections.append("**3. Patient Progress and Current Status**\n" + "\n".join(pr_lines))
        if np_lines:
            sections.append("**4. Next Steps and Planning**\n" + "\n".join(np_lines))

        narrative = "\n\n".join(sections)
        logger.info(f"📝 PR-2 Narrative summary generated: {len(narrative)} characters (plain text)")
        return narrative

    def _build_clinical_status_narrative_new(self, subjective: Dict, objective: Dict, effectiveness: Dict) -> str:
        """Build clinical status narrative from NEW structure"""
        status_parts = []
        
        # Pain scores
        pain_current = subjective.get("pain_score_current", "")
        pain_previous = subjective.get("pain_score_previous", "")
        if pain_current and pain_previous:
            status_parts.append(f"Pain improved from {pain_previous} to {pain_current}")
        elif pain_current:
            status_parts.append(f"Pain: {pain_current}")
        
        # Symptom changes
        symptom_changes = subjective.get("symptom_changes", "")
        if symptom_changes:
            status_parts.append(symptom_changes)
        
        # Patient response
        patient_response = effectiveness.get("patient_response", "")
        if patient_response:
            status_parts.append(patient_response)
        
        # Objective improvements
        obj_improvements = effectiveness.get("objective_improvements", [])
        if obj_improvements and isinstance(obj_improvements, list):
            status_parts.extend([str(imp) for imp in obj_improvements[:2] if imp])
        
        # Functional gains
        functional_gains = effectiveness.get("functional_gains", "")
        if functional_gains:
            status_parts.append(f"Function: {functional_gains}")
        
        return ", ".join(status_parts[:5]) if status_parts else "Status unchanged"

    def _build_medications_narrative_new(self, current_medications: list, medication_changes: Dict) -> str:
        """Build medications narrative - CRITICAL for summary"""
        med_parts = []
        
        # Current medications
        if current_medications and isinstance(current_medications, list):
            for med in current_medications[:5]:  # Max 5 in summary
                if isinstance(med, dict):
                    med_name = med.get("name", "")
                    med_dose = med.get("dose", "")
                    if med_name:
                        if med_dose:
                            med_parts.append(f"{med_name} {med_dose}")
                        else:
                            med_parts.append(med_name)
                elif isinstance(med, str) and med:
                    med_parts.append(med)
        
        # Medication changes (additions/adjustments)
        new_meds = medication_changes.get("new_medications", [])
        if new_meds and isinstance(new_meds, list):
            for new_med in new_meds[:2]:
                if isinstance(new_med, dict):
                    name = new_med.get("name", "")
                    if name:
                        med_parts.append(f"NEW: {name}")
                elif new_med:
                    med_parts.append(f"NEW: {new_med}")
        
        return ", ".join(med_parts) if med_parts else ""

    def _build_authorization_narrative_new(self, treatment_auth: Dict) -> str:
        """Build treatment authorization request narrative"""
        auth_parts = []
        
        # Primary request
        primary = treatment_auth.get("primary_request", "")
        if primary:
            auth_parts.append(primary)
        
        # Secondary requests
        secondary = treatment_auth.get("secondary_requests", [])
        if secondary and isinstance(secondary, list):
            for req in secondary[:2]:
                if req:
                    auth_parts.append(str(req))
        
        # Rationale if no requests
        if not auth_parts:
            rationale = treatment_auth.get("medical_necessity_rationale", "")
            if rationale:
                return f"Continue current care ({rationale})"
        
        return "; ".join(auth_parts) if auth_parts else ""

    def _build_work_status_narrative_new(self, work_status: Dict) -> str:
        """Build work status narrative from NEW structure"""
        parts = []
        
        # Current status
        current = work_status.get("current_status", "")
        if current:
            parts.append(current)
        
        # Work limitations (EXACT wording)
        limitations = work_status.get("work_limitations", [])
        if limitations and isinstance(limitations, list):
            flat_limitations = []
            for limit in limitations:
                if isinstance(limit, list):
                    flat_limitations.extend([str(x) for x in limit if x])
                elif limit:
                    flat_limitations.append(str(limit))
            
            if flat_limitations:
                parts.extend(flat_limitations[:3])  # Max 3 limitations
        
        # Changes from previous
        changes = work_status.get("changes_from_previous_status", "")
        if changes:
            parts.append(f"Change: {changes}")
        
        return ", ".join(parts) if parts else "Work status unchanged"

    def _build_followup_narrative_new(self, follow_up: Dict) -> str:
        """Build follow-up plan narrative from NEW structure"""
        plan_items = []
        
        # Next appointment
        next_appt = follow_up.get("next_appointment_date", "")
        if next_appt:
            plan_items.append(f"Follow-up: {next_appt}")
        
        # Purpose
        purpose = follow_up.get("purpose_of_next_visit", "")
        if purpose:
            plan_items.append(purpose)
        
        # Specialist referrals
        referrals = follow_up.get("specialist_referrals_requested", [])
        if referrals and isinstance(referrals, list):
            for ref in referrals[:2]:
                if isinstance(ref, dict):
                    specialty = ref.get("specialty", "")
                    if specialty:
                        plan_items.append(f"Refer to {specialty}")
                elif ref:
                    plan_items.append(f"Refer to {ref}")
        
        # MMI/P&S anticipated
        mmi_date = follow_up.get("mmi_ps_anticipated_date", "")
        if mmi_date:
            plan_items.append(f"Anticipated MMI: {mmi_date}")
        
        return "; ".join(plan_items) if plan_items else "Routine follow-up"

    
    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure matching NEW WC structure"""
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
