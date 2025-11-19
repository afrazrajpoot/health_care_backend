"""
ClinicalNoteExtractor - Enhanced Extractor for Clinical Progress Notes and Therapy Reports
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
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


class ClinicalNoteExtractor:
    """
    Enhanced Clinical Note extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different clinical specialties
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Progress Notes, PT/OT/Chiro/Acupuncture, Pain Management, Psychiatry, Nursing Notes
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.note_type_patterns = {
            'progress_note': re.compile(r'\b(progress note|follow[- ]?up|office visit|clinic note)\b', re.IGNORECASE),
            'physical_therapy': re.compile(r'\b(physical therapy|PT|therapeutic exercise|range of motion|ROM)\b', re.IGNORECASE),
            'occupational_therapy': re.compile(r'\b(occupational therapy|OT|ADL|activities of daily living|functional capacity)\b', re.IGNORECASE),
            'chiropractic': re.compile(r'\b(chiropractic|chiropractor|adjustment|manipulation|subluxation)\b', re.IGNORECASE),
            'acupuncture': re.compile(r'\b(acupuncture|needle|meridian|qi|energy flow)\b', re.IGNORECASE),
            'pain_management': re.compile(r'\b(pain management|pain clinic|chronic pain|pain scale|analgesic)\b', re.IGNORECASE),
            'psychiatry': re.compile(r'\b(psychiatry|psychiatric|mental status|affect|mood|psychotropic)\b', re.IGNORECASE),
            'psychology': re.compile(r'\b(psychology|psychological|therapy session|counseling|behavioral)\b', re.IGNORECASE),
            'nursing': re.compile(r'\b(nursing note|nurse visit|vital signs|nursing assessment|medication administration)\b', re.IGNORECASE)
        }
        
        # Clinical assessment patterns
        self.assessment_patterns = {
            'pain_scale': re.compile(r'\b(pain|discomfort)\s*(scale|level|score)?\s*[:\-]?\s*(\d+/10|\d+\s*out\s*of\s*10)', re.IGNORECASE),
            'rom_measurements': re.compile(r'\b(ROM|range of motion)\s*[:\-]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'functional_status': re.compile(r'\b(able to|unable to|independent|assist|assistance|with help)\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'treatment_codes': re.compile(r'\b(CPT[:\s]*(\d{4,5})|(9716[01234]|9753[05]|9775[05]))', re.IGNORECASE)
        }
        
        logger.info("✅ ClinicalNoteExtractor initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract Clinical Note data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Progress Note, PT, OT, Chiro, Acupuncture, Pain Management, Psychiatry, Nursing)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING CLINICAL NOTE EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Auto-detect specific note type if not specified
        detected_type = self._detect_note_type(text, doc_type)
        logger.info(f"📋 Clinical Note Type: {detected_type} (original: {doc_type})")
        
        # Log context guidance if available
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Critical Locations: {list(critical_locations.keys())}")
        else:
            logger.warning("⚠️ No context analysis provided - proceeding without guidance")
        
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
            doc_type=detected_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )

        # Stage 2: Build long summary from ALL raw data
        long_summary = self._build_comprehensive_long_summary(raw_result, detected_type, fallback_date)

        # Stage 3: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)

        logger.info("=" * 80)
        logger.info("✅ CLINICAL NOTE EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_note_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific clinical note type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for note_type, pattern in self.note_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[note_type] = len(matches)
        
        # Boost scores for treatment-specific terminology
        if self.assessment_patterns['treatment_codes'].search(text):
            for note_type in ['physical_therapy', 'occupational_therapy', 'chiropractic']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 2
        
        if self.assessment_patterns['pain_scale'].search(text):
            for note_type in ['pain_management', 'progress_note']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected note type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect note type, using: {original_type}")
        return original_type or "Clinical Note"

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        """
        logger.info("🔍 Processing ENTIRE clinical note in single context window with guidance...")
        
        # Extract guidance from context analysis
        focus_sections = []
        critical_locations = {}
        ambiguities = []
        
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert clinical documentation specialist analyzing a COMPLETE {doc_type} with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE clinical note at once, allowing you to:
- Understand the complete clinical encounter from subjective complaints to treatment plan
- Track progress across multiple visits and treatment sessions
- Identify patterns in symptoms, functional limitations, and treatment response
- Provide comprehensive extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the note, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate clinical information
   - DO NOT fill in "typical" or "common" clinical values
   - DO NOT use clinical knowledge to "complete" incomplete information
   
2. **SUBJECTIVE COMPLAINTS - PATIENT'S EXACT WORDS**
   - Extract patient complaints using EXACT wording from note
   - DO NOT interpret or rephrase patient statements
   - Include pain descriptions, functional limitations, and concerns verbatim
   
3. **OBJECTIVE FINDINGS - MEASURED VALUES ONLY**
   - Extract ROM measurements, strength grades, pain scales ONLY if explicitly stated
   - Include units and specific values EXACTLY as written
   - DO NOT calculate or estimate ranges
   
4. **TREATMENTS & MODALITIES - SPECIFIC DETAILS ONLY**
   - Extract treatment techniques, modalities, exercises ONLY if explicitly listed
   - Include durations, frequencies, parameters ONLY if specified
   - DO NOT add typical treatment protocols
   
5. **ASSESSMENT & PLAN - CLINICIAN'S EXACT WORDING**
   - Extract clinical assessment using EXACT phrasing from note
   - Include treatment plan details verbatim
   - DO NOT interpret clinical reasoning

EXTRACTION FOCUS - 8 CRITICAL CLINICAL NOTE CATEGORIES:

I. NOTE IDENTITY & ENCOUNTER CONTEXT
- Note type, dates, encounter information
- Provider details and credentials
- Visit type and duration

II. SUBJECTIVE FINDINGS (PATIENT'S PERSPECTIVE)
- Chief complaint and history of present illness
- Pain characteristics: location, intensity, quality, timing
- Functional limitations and impact on daily activities
- Patient's goals and expectations
- Relevant medical and social history

III. OBJECTIVE EXAMINATION FINDINGS (CLINICIAN'S OBSERVATIONS)
- Vital signs and general appearance
- Physical examination findings:
  * Range of motion (ROM) measurements with specific degrees
  * Manual muscle testing grades (0-5)
  * Palpation findings and tender points
  * Special tests and orthopedic assessments
  * Neurological examination findings
- Functional capacity assessments
- Observation of movement patterns and gait

IV. TREATMENT PROVIDED (SESSION-SPECIFIC)
- Specific techniques and modalities used:
  * Manual therapy techniques
  * Therapeutic exercises prescribed
  * Modalities applied (heat, ice, electrical stimulation, etc.)
  * Acupuncture points or chiropractic adjustments
- Treatment parameters: duration, intensity, frequency
- Patient response to treatment
- Any adverse reactions or complications

V. ASSESSMENT & CLINICAL IMPRESSION
- Clinical assessment and diagnosis
- Progress since last visit
- Changes in functional status
- Barriers to recovery
- Prognosis and expected outcomes

VI. TREATMENT PLAN & GOALS
- Short-term and long-term goals
- Specific plan for next visit
- Home exercise program details
- Frequency and duration of continued care
- Referrals or consultations needed

VII. WORK STATUS & FUNCTIONAL CAPACITY
- Current work restrictions
- Functional limitations
- Ability to perform job duties
- Expected return to work timeline

VIII. OUTCOME MEASURES & PROGRESS TRACKING
- Standardized outcome measures (ODI, NDI, DASH, etc.)
- Pain scale ratings (0-10)
- Functional improvement metrics
- Patient satisfaction measures

⚠️ FINAL REMINDER:
- If information is NOT in the note, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate clinical information
- PAIN SCALES: Extract exact numbers (e.g., "6/10") not descriptions
- ROM MEASUREMENTS: Extract exact degrees, not ranges
- It is BETTER to have empty fields than incorrect clinical information

Now analyze this COMPLETE {doc_type} and extract ALL relevant clinical information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical clinical details:

{{
  "note_identity": {{
    "note_type": "{doc_type}",
    "visit_date": "",
    "encounter_date": "",
    "visit_type": "",
    "encounter_duration": "",
    "clinic_facility": "",
    "provider_credentials": ""
  }},
  
  "patient_information": {{
    "patient_name": "",
    "patient_dob": "",
    "patient_age": "",
    "time_since_injury": "",
    "visit_number": "",
    "treatment_frequency": ""
  }},
  
  "providers": {{
    "treating_provider": {{
      "name": "",
      "credentials": "",
      "specialty": "",
      "role": "Treating Provider"
    }},
    "assistant_provider": {{
      "name": "",
      "credentials": "",
      "role": ""
    }},
    "supervising_physician": {{
      "name": "",
      "credentials": ""
    }}
  }},
  
  "subjective_findings": {{
    "chief_complaint": "",
    "history_present_illness": "",
    "pain_characteristics": {{
      "location": "",
      "intensity": "",
      "quality": "",
      "radiation": "",
      "aggravating_factors": "",
      "relieving_factors": ""
    }},
    "functional_limitations": [],
    "patient_goals": "",
    "compliance_with_treatment": ""
  }},
  
  "objective_findings": {{
    "vital_signs": {{
      "blood_pressure": "",
      "heart_rate": "",
      "respiratory_rate": "",
      "temperature": ""
    }},
    "range_of_motion": [],
    "manual_muscle_testing": [],
    "palpation_findings": [],
    "special_tests": [],
    "functional_assessments": [],
    "neurological_findings": [],
    "observation_findings": ""
  }},
  
  "treatment_provided": {{
    "treatment_techniques": [],
    "therapeutic_exercises": [],
    "modalities_used": [],
    "manual_therapy": [],
    "treatment_parameters": {{
      "duration": "",
      "intensity": "",
      "patient_response": ""
    }},
    "adverse_reactions": ""
  }},
  
  "clinical_assessment": {{
    "assessment": "",
    "progress_since_last_visit": "",
    "changes_in_status": "",
    "clinical_impression": "",
    "prognosis": "",
    "barriers_to_recovery": []
  }},
  
  "treatment_plan": {{
    "short_term_goals": [],
    "long_term_goals": [],
    "plan_of_care": "",
    "home_exercise_program": [],
    "frequency_duration": "",
    "next_appointment": "",
    "referrals_needed": []
  }},
  
  "work_status": {{
    "current_work_status": "",
    "work_restrictions": [],
    "functional_capacity": "",
    "return_to_work_plan": ""
  }},
  
  "outcome_measures": {{
    "pain_scale": "",
    "functional_scores": [],
    "progress_metrics": [],
    "patient_satisfaction": ""
  }},
  
  "critical_clinical_findings": []
}}

⚠️ CRITICAL CLINICAL REMINDERS:
1. For "range_of_motion": Extract EXACT measurements with degrees
   - Include body part, motion, and specific degrees
   - Example: "Shoulder flexion: 120 degrees" not "limited shoulder flexion"

2. For "pain_scale": Extract EXACT numerical values (0-10 scale)
   - Example: "6/10" not "moderate pain"
   - Include location if specified

3. For "treatment_techniques": Extract SPECIFIC techniques used
   - Include parameters if specified (e.g., "US 1.5 W/cm² for 8 minutes")
   - Do not include techniques mentioned as options or for future use

4. For "work_restrictions": Extract EXACT limitations stated
   - Include weight limits, positional restrictions, duration
   - Example: "no lifting >10 lbs" not "light duty"

5. For "critical_clinical_findings": Include clinically significant changes
   - Worsening symptoms or functional decline
   - New neurological findings
   - Significant progress or setbacks
   - Compliance issues affecting treatment
""")

        # Build context guidance summary
        context_guidance_text = f"""
CLINICAL NOTE TYPE: {doc_type}
FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- Objective Findings: {critical_locations.get('objective_location', 'Search entire document')}
- Treatment Provided: {critical_locations.get('treatment_location', 'Search entire document')}
- Assessment/Plan: {critical_locations.get('assessment_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context clinical note extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "context_guidance": context_guidance_text
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context clinical note extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char clinical note")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context clinical note extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Clinical note exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large notes")
            
            return self._get_fallback_result(doc_type, fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted clinical note data.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted clinical data...")
        
        sections = []
        
        # Section 1: CLINICAL ENCOUNTER OVERVIEW
        sections.append("📋 CLINICAL ENCOUNTER OVERVIEW")
        sections.append("-" * 50)
        
        note_identity = raw_data.get("note_identity", {})
        overview_lines = [
            f"Note Type: {note_identity.get('note_type', doc_type)}",
            f"Visit Date: {note_identity.get('visit_date', fallback_date)}",
            f"Visit Type: {note_identity.get('visit_type', 'Not specified')}",
            f"Duration: {note_identity.get('encounter_duration', 'Not specified')}",
            f"Facility: {note_identity.get('clinic_facility', 'Not specified')}"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PROVIDER INFORMATION
        sections.append("\n👨‍⚕️ PROVIDER INFORMATION")
        sections.append("-" * 50)
        
        providers = raw_data.get("providers", {})
        provider_lines = []
        
        treating_provider = providers.get("treating_provider", {})
        if treating_provider.get("name"):
            provider_lines.append(f"Treating Provider: {treating_provider['name']}")
            if treating_provider.get("credentials"):
                provider_lines.append(f"  Credentials: {treating_provider['credentials']}")
            if treating_provider.get("specialty"):
                provider_lines.append(f"  Specialty: {treating_provider['specialty']}")
        
        sections.append("\n".join(provider_lines) if provider_lines else "No provider information extracted")
        
        # Section 3: SUBJECTIVE FINDINGS
        sections.append("\n🗣️ SUBJECTIVE FINDINGS")
        sections.append("-" * 50)
        
        subjective = raw_data.get("subjective_findings", {})
        subjective_lines = []
        
        if subjective.get("chief_complaint"):
            subjective_lines.append(f"Chief Complaint: {subjective['chief_complaint']}")
        
        # Pain characteristics
        pain_chars = subjective.get("pain_characteristics", {})
        pain_info = []
        if pain_chars.get("location"):
            pain_info.append(f"Location: {pain_chars['location']}")
        if pain_chars.get("intensity"):
            pain_info.append(f"Intensity: {pain_chars['intensity']}")
        if pain_chars.get("quality"):
            pain_info.append(f"Quality: {pain_chars['quality']}")
        
        if pain_info:
            subjective_lines.append(f"Pain: {', '.join(pain_info)}")
        
        # Functional limitations
        functional_limitations = subjective.get("functional_limitations", [])
        if functional_limitations:
            subjective_lines.append("\nFunctional Limitations:")
            for limitation in functional_limitations[:5]:
                if isinstance(limitation, dict):
                    desc = limitation.get("limitation", "")
                    if desc:
                        subjective_lines.append(f"  • {desc}")
                elif limitation:
                    subjective_lines.append(f"  • {limitation}")
        
        sections.append("\n".join(subjective_lines) if subjective_lines else "No subjective findings extracted")
        
        # Section 4: OBJECTIVE EXAMINATION
        sections.append("\n🔍 OBJECTIVE EXAMINATION")
        sections.append("-" * 50)
        
        objective = raw_data.get("objective_findings", {})
        objective_lines = []
        
        # Range of motion
        rom_measurements = objective.get("range_of_motion", [])
        if rom_measurements:
            objective_lines.append("Range of Motion:")
            for rom in rom_measurements[:5]:
                if isinstance(rom, dict):
                    body_part = rom.get("body_part", "")
                    motion = rom.get("motion", "")
                    measurement = rom.get("measurement", "")
                    if body_part and motion and measurement:
                        objective_lines.append(f"  • {body_part} {motion}: {measurement}")
                elif rom:
                    objective_lines.append(f"  • {rom}")
        
        # Manual muscle testing
        mmt = objective.get("manual_muscle_testing", [])
        if mmt:
            objective_lines.append("\nManual Muscle Testing:")
            for muscle in mmt[:3]:
                if isinstance(muscle, dict):
                    muscle_name = muscle.get("muscle", "")
                    grade = muscle.get("grade", "")
                    if muscle_name and grade:
                        objective_lines.append(f"  • {muscle_name}: {grade}/5")
                elif muscle:
                    objective_lines.append(f"  • {muscle}")
        
        # Special tests
        special_tests = objective.get("special_tests", [])
        if special_tests:
            objective_lines.append("\nSpecial Tests:")
            for test in special_tests[:3]:
                if isinstance(test, dict):
                    test_name = test.get("test", "")
                    result = test.get("result", "")
                    if test_name:
                        if result:
                            objective_lines.append(f"  • {test_name}: {result}")
                        else:
                            objective_lines.append(f"  • {test_name}")
                elif test:
                    objective_lines.append(f"  • {test}")
        
        sections.append("\n".join(objective_lines) if objective_lines else "No objective findings extracted")
        
        # Section 5: TREATMENT PROVIDED
        sections.append("\n💆 TREATMENT PROVIDED")
        sections.append("-" * 50)
        
        treatment = raw_data.get("treatment_provided", {})
        treatment_lines = []
        
        # Treatment techniques
        techniques = treatment.get("treatment_techniques", [])
        if techniques:
            treatment_lines.append("Treatment Techniques:")
            for technique in techniques[:5]:
                if isinstance(technique, dict):
                    tech_name = technique.get("technique", "")
                    if tech_name:
                        treatment_lines.append(f"  • {tech_name}")
                elif technique:
                    treatment_lines.append(f"  • {technique}")
        
        # Therapeutic exercises
        exercises = treatment.get("therapeutic_exercises", [])
        if exercises:
            treatment_lines.append("\nTherapeutic Exercises:")
            for exercise in exercises[:5]:
                if isinstance(exercise, dict):
                    ex_name = exercise.get("exercise", "")
                    if ex_name:
                        treatment_lines.append(f"  • {ex_name}")
                elif exercise:
                    treatment_lines.append(f"  • {exercise}")
        
        # Modalities
        modalities = treatment.get("modalities_used", [])
        if modalities:
            treatment_lines.append("\nModalities Used:")
            for modality in modalities[:3]:
                if isinstance(modality, dict):
                    mod_name = modality.get("modality", "")
                    if mod_name:
                        treatment_lines.append(f"  • {mod_name}")
                elif modality:
                    treatment_lines.append(f"  • {modality}")
        
        sections.append("\n".join(treatment_lines) if treatment_lines else "No treatment details extracted")
        
        # Section 6: CLINICAL ASSESSMENT
        sections.append("\n🏥 CLINICAL ASSESSMENT")
        sections.append("-" * 50)
        
        assessment = raw_data.get("clinical_assessment", {})
        assessment_lines = []
        
        if assessment.get("assessment"):
            assessment_lines.append(f"Assessment: {assessment['assessment']}")
        
        if assessment.get("progress_since_last_visit"):
            assessment_lines.append(f"Progress: {assessment['progress_since_last_visit']}")
        
        if assessment.get("clinical_impression"):
            assessment_lines.append(f"Clinical Impression: {assessment['clinical_impression']}")
        
        if assessment.get("prognosis"):
            assessment_lines.append(f"Prognosis: {assessment['prognosis']}")
        
        sections.append("\n".join(assessment_lines) if assessment_lines else "No clinical assessment extracted")
        
        # Section 7: TREATMENT PLAN
        sections.append("\n🎯 TREATMENT PLAN")
        sections.append("-" * 50)
        
        treatment_plan = raw_data.get("treatment_plan", {})
        plan_lines = []
        
        # Short-term goals
        short_term_goals = treatment_plan.get("short_term_goals", [])
        if short_term_goals:
            plan_lines.append("Short-term Goals:")
            for goal in short_term_goals[:3]:
                if isinstance(goal, dict):
                    goal_desc = goal.get("goal", "")
                    if goal_desc:
                        plan_lines.append(f"  • {goal_desc}")
                elif goal:
                    plan_lines.append(f"  • {goal}")
        
        # Home exercise program
        hep = treatment_plan.get("home_exercise_program", [])
        if hep:
            plan_lines.append("\nHome Exercise Program:")
            for exercise in hep[:3]:
                if isinstance(exercise, dict):
                    ex_desc = exercise.get("exercise", "")
                    if ex_desc:
                        plan_lines.append(f"  • {ex_desc}")
                elif exercise:
                    plan_lines.append(f"  • {exercise}")
        
        if treatment_plan.get("frequency_duration"):
            plan_lines.append(f"\nFrequency/Duration: {treatment_plan['frequency_duration']}")
        
        if treatment_plan.get("next_appointment"):
            plan_lines.append(f"Next Appointment: {treatment_plan['next_appointment']}")
        
        sections.append("\n".join(plan_lines) if plan_lines else "No treatment plan extracted")
        
        # Section 8: WORK STATUS
        sections.append("\n💼 WORK STATUS")
        sections.append("-" * 50)
        
        work_status = raw_data.get("work_status", {})
        work_lines = []
        
        if work_status.get("current_work_status"):
            work_lines.append(f"Current Status: {work_status['current_work_status']}")
        
        work_restrictions = work_status.get("work_restrictions", [])
        if work_restrictions:
            work_lines.append("\nWork Restrictions:")
            for restriction in work_restrictions[:5]:
                if isinstance(restriction, dict):
                    desc = restriction.get("restriction", "")
                    if desc:
                        work_lines.append(f"  • {desc}")
                elif restriction:
                    work_lines.append(f"  • {restriction}")
        
        if work_status.get("functional_capacity"):
            work_lines.append(f"\nFunctional Capacity: {work_status['functional_capacity']}")
        
        sections.append("\n".join(work_lines) if work_lines else "No work status information extracted")
        
        # Section 9: OUTCOME MEASURES
        sections.append("\n📊 OUTCOME MEASURES")
        sections.append("-" * 50)
        
        outcomes = raw_data.get("outcome_measures", {})
        outcome_lines = []
        
        if outcomes.get("pain_scale"):
            outcome_lines.append(f"Pain Scale: {outcomes['pain_scale']}")
        
        functional_scores = outcomes.get("functional_scores", [])
        if functional_scores:
            outcome_lines.append("\nFunctional Scores:")
            for score in functional_scores[:3]:
                if isinstance(score, dict):
                    score_name = score.get("measure", "")
                    score_value = score.get("score", "")
                    if score_name and score_value:
                        outcome_lines.append(f"  • {score_name}: {score_value}")
                elif score:
                    outcome_lines.append(f"  • {score}")
        
        sections.append("\n".join(outcome_lines) if outcome_lines else "No outcome measures extracted")
        
        # Section 10: CRITICAL CLINICAL FINDINGS
        sections.append("\n🚨 CRITICAL CLINICAL FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_clinical_findings", [])
        if critical_findings:
            for finding in critical_findings[:8]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("finding", "")
                    finding_priority = finding.get("priority", "")
                    if finding_desc:
                        if finding_priority:
                            sections.append(f"• [{finding_priority}] {finding_desc}")
                        else:
                            sections.append(f"• {finding_desc}")
                elif finding:
                    sections.append(f"• {finding}")
        else:
            # Check for critical findings in other sections
            critical_items = []
            
            # Check for worsening pain
            if outcomes.get("pain_scale"):
                pain_match = re.search(r'(\d+)/10', outcomes['pain_scale'])
                if pain_match and int(pain_match.group(1)) >= 7:
                    critical_items.append(f"High pain level: {outcomes['pain_scale']}")
            
            # Check for significant functional decline
            if assessment.get("progress_since_last_visit"):
                if any(term in assessment['progress_since_last_visit'].lower() for term in ['worse', 'declined', 'deteriorated', 'regressed']):
                    critical_items.append("Functional decline noted")
            
            if critical_items:
                for item in critical_items:
                    sections.append(f"• {item}")
            else:
                sections.append("No critical clinical findings identified")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Clinical note long summary built: {len(long_summary)} characters")
        
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
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word clinical note summary.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word clinical structured summary...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a clinical documentation specialist.

    TASK:
    Create a concise, factual clinical summary using ONLY information explicitly present in the long summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:
    [Report Title] | [Author/Physician or The person who signed the report] | [Date] | [Body parts] | [Diagnosis] | [Key Objective Findings] | [Medication] | [Treatments Provided] | [Clinical Assessment] | [Plan / Next Steps] | [MMI Status] | [Key Action Items] | [Work Status] | [Recommendation] | [Critical Finding] | Urgent Next Steps

    FORMAT & RULES:
- MUST be **30–60 words**.
- MUST be **ONE LINE**, pipe-delimited, no line breaks.
- NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
- NEVER fabricate: no invented dates, meds, restrictions, exam findings, or recommendations.
- NO narrative sentences. Use short factual fragments ONLY.
- Use the shortest, clearest key names:
  • Title = Report title  
  • Author = MD/DO/PA/NP or signer  
  • Date = Visit or exam date  
  • Work Status = current status (if given)  
  • Restrictions = physical restrictions (if given)  
  • Meds = medications explicitly listed  (if given)
  • Physical Exam = objective exam findings only (if given)
  • Treatment Progress = progress or response  (if given)
  • Auth Requests = items requested for authorization  (if given)
  • Follow-up = next appointment or instruction  (if given)
  • Critical Finding = one most clinically important finding (if given)
CONTENT PRIORITY (only if provided in the long summary):
1. Report Title  
2. Author  
3. Visit Date  
4. Diagnosis / body parts  
5. Work status & restrictions  
6. Medications  
7. Physical examination details  
8. Treatment progress  
9. Authorization requests  
10. Follow-up plan  
11. Critical finding

ABSOLUTELY FORBIDDEN:
- assumptions, interpretations, invented medications, or inferred diagnoses
- narrative writing
- placeholder text or “Not provided”
- duplicate pipes or empty pipe fields (e.g., "||")

Your final output MUST be between 30–60 words and follow the exact pipe-delimited style. 
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    CLINICAL LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured clinical summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })
            summary = response.content.strip()

            # Normalize whitespace
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Apply pipe cleaning function
            summary = self._clean_pipes_from_summary(summary)

            # Validate word count
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Clinical summary out of range ({wc} words). Attempting auto-fix...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior summary contained {wc} words. Rewrite it to be between 30 and 60 words. "
                        "DO NOT add fabricated details. Preserve all factual elements. Maintain pipe-delimited format."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                summary = self._clean_pipes_from_summary(summary)  # Clean pipes again after auto-fix

            logger.info(f"✅ Clinical summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Clinical summary generation failed: {e}")
            return "Summary unavailable due to processing error."


    def _create_clinical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback clinical summary directly from long summary"""
        
        # Extract key clinical information using regex patterns
        patterns = {
            'provider': r'Treating Provider:\s*([^\n]+)',
            'pain': r'Pain Scale:\s*([^\n]+)',
            'complaint': r'Chief Complaint:\s*([^\n]+)',
            'rom': r'Range of Motion:(.*?)(?:\n\n|\n[A-Z]|$)',
            'treatment': r'Treatment Techniques:(.*?)(?:\n\n|\n[A-Z]|$)',
            'assessment': r'Assessment:\s*([^\n]+)',
            'plan': r'Frequency/Duration:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} visit")
        
        # Add provider context
        if 'provider' in extracted:
            parts.append(f"by {extracted['provider']}")
        
        # Add chief complaint and pain
        if 'complaint' in extracted:
            complaint = extracted['complaint'][:60] + "..." if len(extracted['complaint']) > 60 else extracted['complaint']
            parts.append(f"for {complaint}")
        
        if 'pain' in extracted:
            parts.append(f"Pain: {extracted['pain']}")
        
        # Add key findings
        if 'rom' in extracted:
            first_rom = extracted['rom'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Findings: {first_rom}")
        
        # Add treatment
        if 'treatment' in extracted:
            first_treatment = extracted['treatment'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Treatment: {first_treatment}")
        
        # Add assessment and plan
        if 'assessment' in extracted:
            assessment = extracted['assessment'][:60] + "..." if len(extracted['assessment']) > 60 else extracted['assessment']
            parts.append(f"Assessment: {assessment}")
        
        if 'plan' in extracted:
            parts.append(f"Plan: {extracted['plan']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with continued clinical management", "following treatment protocols", "with ongoing progress monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used clinical fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, doc_type: str, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for clinical notes"""
        return {
            "note_identity": {
                "note_type": doc_type,
                "visit_date": fallback_date,
                "encounter_date": "",
                "visit_type": "",
                "encounter_duration": "",
                "clinic_facility": "",
                "provider_credentials": ""
            },
            "patient_information": {
                "patient_name": "",
                "patient_dob": "",
                "patient_age": "",
                "time_since_injury": "",
                "visit_number": "",
                "treatment_frequency": ""
            },
            "providers": {
                "treating_provider": {
                    "name": "",
                    "credentials": "",
                    "specialty": "",
                    "role": "Treating Provider"
                },
                "assistant_provider": {
                    "name": "",
                    "credentials": "",
                    "role": ""
                },
                "supervising_physician": {
                    "name": "",
                    "credentials": ""
                }
            },
            "subjective_findings": {
                "chief_complaint": "",
                "history_present_illness": "",
                "pain_characteristics": {
                    "location": "",
                    "intensity": "",
                    "quality": "",
                    "radiation": "",
                    "aggravating_factors": "",
                    "relieving_factors": ""
                },
                "functional_limitations": [],
                "patient_goals": "",
                "compliance_with_treatment": ""
            },
            "objective_findings": {
                "vital_signs": {
                    "blood_pressure": "",
                    "heart_rate": "",
                    "respiratory_rate": "",
                    "temperature": ""
                },
                "range_of_motion": [],
                "manual_muscle_testing": [],
                "palpation_findings": [],
                "special_tests": [],
                "functional_assessments": [],
                "neurological_findings": [],
                "observation_findings": ""
            },
            "treatment_provided": {
                "treatment_techniques": [],
                "therapeutic_exercises": [],
                "modalities_used": [],
                "manual_therapy": [],
                "treatment_parameters": {
                    "duration": "",
                    "intensity": "",
                    "patient_response": ""
                },
                "adverse_reactions": ""
            },
            "clinical_assessment": {
                "assessment": "",
                "progress_since_last_visit": "",
                "changes_in_status": "",
                "clinical_impression": "",
                "prognosis": "",
                "barriers_to_recovery": []
            },
            "treatment_plan": {
                "short_term_goals": [],
                "long_term_goals": [],
                "plan_of_care": "",
                "home_exercise_program": [],
                "frequency_duration": "",
                "next_appointment": "",
                "referrals_needed": []
            },
            "work_status": {
                "current_work_status": "",
                "work_restrictions": [],
                "functional_capacity": "",
                "return_to_work_plan": ""
            },
            "outcome_measures": {
                "pain_scale": "",
                "functional_scores": [],
                "progress_metrics": [],
                "patient_satisfaction": ""
            },
            "critical_clinical_findings": []
        }