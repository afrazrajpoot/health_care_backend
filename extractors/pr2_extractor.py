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
        
        # Build context-aware system prompt for PR-2
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical documentation specialist analyzing a COMPLETE PR-2 Progress Report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE PR-2 document at once, allowing you to:
- Track clinical progress across the entire treatment timeline
- Understand treatment response and patient compliance
- Identify patterns in symptom progression and functional improvement
- Connect subjective complaints with objective findings over time
- Provide comprehensive progress assessment without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate information
   - DO NOT fill in "typical" or "common" progress patterns
   - DO NOT use medical knowledge to "complete" incomplete information
   
   Examples:
   ✅ CORRECT: If document says "Pain improved from 7/10 to 4/10", extract: "subjective_pain_improvement": "7/10 to 4/10"
   ❌ WRONG: If document says "Pain improved", DO NOT extract: "subjective_pain_improvement": "7/10 to 4/10" (specific scores not stated)
   ✅ CORRECT: Extract: "subjective_pain_improvement": "improved" (only what's stated)

2. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY medications explicitly listed in current medication sections
   - Include dosage changes ONLY if explicitly stated
   - DO NOT extract:
     * Medications mentioned as discontinued
     * Medications mentioned in past history only
     * Medications recommended for future use
   
   Examples:
   ✅ CORRECT: Document states "Continuing Gabapentin 300mg TID, increased Meloxicam to 15mg daily"
   Extract: {{"current_medications": [{{"name": "Gabapentin", "dose": "300mg TID"}}, {{"name": "Meloxicam", "dose": "15mg daily"}}]}}
   
   ❌ WRONG: Document states "Previously took Oxycodone"
   DO NOT extract Oxycodone in current_medications

3. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
   Examples:
   ✅ CORRECT: If no functional improvement mentioned, return: "functional_improvement": ""
   ❌ WRONG: Return: "functional_improvement": "Not assessed" (use empty string instead)

4. **EXACT CLINICAL STATUS DESCRIPTORS**
   - Use EXACT wording from document for clinical status
   - DO NOT upgrade "stable" to "improved" or downgrade "worsened" to "stable"
   - Capture nuanced descriptions: "slightly improved", "markedly worsened", "essentially unchanged"
   
   Examples:
   ✅ CORRECT: Document says "Condition essentially unchanged"
   Extract: "clinical_status": "essentially unchanged"
   
   ❌ WRONG: Document says "Condition essentially unchanged"
   DO NOT extract: "clinical_status": "stable" (this changes the meaning)

5. **NO CLINICAL PROGNOSTICATION**
   - DO NOT predict future progress based on current status
   - DO NOT assume treatment effectiveness
   - DO NOT infer patient compliance unless explicitly stated
   
   Examples:
   ❌ WRONG: Document mentions "attending PT regularly"
   DO NOT assume: "good compliance with treatment" (this is inference)
   ✅ CORRECT: Extract: "pt_attendance": "regular" (if stated)

6. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
   Before returning your extraction, verify:
   □ Every medication change has a direct quote in the document
   □ Every status change is explicitly stated (not inferred from symptoms)
   □ Every work status modification is directly from restrictions section
   □ No fields are filled with "expected" or "typical" progress patterns
   □ Empty fields are truly empty (not "Not mentioned" or "Unknown")

PR-2 SPECIFIC EXTRACTION FOCUS - 5 CRITICAL PROGRESS CATEGORIES:

I. CLINICAL PROGRESS ASSESSMENT
- Subjective improvement: Patient-reported symptom changes
- Objective improvement: Examination findings compared to previous
- Functional progress: ADL and mobility changes
- Treatment response: How patient is responding to current interventions

II. CURRENT TREATMENT STATUS
- Medication adherence and adjustments
- Therapy attendance and participation
- Injection/procedure responses
- Compliance with home exercise program

III. WORK STATUS EVOLUTION
- Changes in work capacity since last visit
- Specific restriction modifications
- Return-to-work progression timeline
- Functional capacity evaluation results

IV. TREATMENT PLAN ADJUSTMENTS
- Medication changes: dose adjustments, additions, discontinuations
- Therapy modifications: frequency, duration, focus changes
- Procedure recommendations: new injections, surgical considerations
- Diagnostic updates: new imaging, test results

V. FOLLOW-UP PLANNING
- Next appointment timing and rationale
- Milestones for next evaluation
- Criteria for treatment success/failure
- Contingency planning for lack of progress

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate progress
- MEDICATIONS: Only extract current medications with explicit changes
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE PR-2 Progress Report and extract ALL relevant progress information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE PR-2 PROGRESS REPORT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all progress tracking details:

{{
  "report_metadata": {{
    "report_type": "PR-2 Progress Report",
    "report_date": "",
    "evaluation_date": "",
    "time_since_last_visit": "",
    "next_scheduled_appointment": ""
  }},
  
  "clinical_progress": {{
    "treating_physician": {{
      "name": "{primary_physician}",
      "specialty": "",
      "facility": ""
    }},
    "body_part_assessed": "",
    "subjective_improvement": {{
      "pain_level_changes": "",
      "symptom_progression": "",
      "patient_reported_function": "",
      "compliance_with_treatment": ""
    }},
    "objective_findings": {{
      "examination_changes": "",
      "rom_improvement": "",
      "strength_progress": "",
      "functional_testing": ""
    }},
    "overall_clinical_status": ""
  }},
  
  "current_treatment_status": {{
    "medication_adherence": "",
    "therapy_attendance": "",
    "injection_procedure_response": "",
    "home_exercise_compliance": "",
    "treatment_barriers": ""
  }},
  
  "medication_management": {{
    "current_medications": [
      {{
        "name": "Gabapentin",
        "dose": "300mg TID",
        "change": "unchanged",
        "effectiveness": "moderate pain relief"
      }},
      {{
        "name": "Meloxicam",
        "dose": "15mg daily",
        "change": "increased from 7.5mg",
        "effectiveness": "improved anti-inflammatory effect"
      }}
    ],
    "recent_medication_changes": [
      {{
        "medication": "Meloxicam",
        "change": "dose increased",
        "date": "current visit",
        "reason": "inadequate inflammation control"
      }}
    ],
    "discontinued_medications": [
      {{
        "name": "Cyclobenzaprine",
        "date_discontinued": "2 weeks ago",
        "reason": "sedation side effects"
      }}
    ]
  }},
  
  "therapy_progress": {{
    "current_therapy_regimen": {{
      "type": "Physical Therapy",
      "frequency": "2x/week",
      "focus_areas": ["L knee strengthening", "gait training"],
      "duration_completed": "4 weeks of 6 week program"
    }},
    "therapy_response": {{
      "strength_improvement": "quad strength improved from 3/5 to 4/5",
      "mobility_gains": "gait pattern improved, less antalgic",
      "pain_response": "pain with therapy decreased from 6/10 to 3/10",
      "functional_improvement": "able to ascend stairs with rail vs. unable previously"
    }},
    "therapy_modifications": {{
      "frequency_change": "",
      "exercise_additions": "added balance exercises",
      "exercise_removals": "discontinued painful squats",
      "intensity_adjustment": "increased resistance on knee extension"
    }}
  }},
  
  "work_status_evolution": {{
    "current_work_status": "Modified Duty",
    "work_restrictions": [
      "No lifting >15 lbs (increased from 10 lbs)",
      "No prolonged standing >30 minutes",
      "Limited stair climbing to 2 flights per day"
    ],
    "restriction_changes": {{
      "lifting_improvement": "increased from 10 lbs to 15 lbs",
      "standing_tolerance": "improved from 15 to 30 minutes",
      "stair_climbing": "now able 2 flights vs. unable previously"
    }},
    "return_to_work_progression": {{
      "current_phase": "modified duty",
      "next_phase": "full duty with temporary restrictions",
      "estimated_timeline": "4-6 weeks",
      "criteria_for_advancement": "pain <3/10, full ROM, normal gait"
    }}
  }},
  
  "diagnostic_updates": {{
    "recent_studies": [
      {{
        "test": "MRI L knee",
        "date": "2 weeks ago",
        "findings": "moderate joint effusion, meniscal tear unchanged",
        "impact_on_treatment": "confirmed meniscal pathology, supports current conservative approach"
      }}
    ],
    "pending_studies": [
      {{
        "test": "Follow-up x-ray",
        "scheduled_date": "next visit",
        "purpose": "assess arthritic progression"
      }}
    ]
  }},
  
  "treatment_plan_adjustments": {{
    "medication_changes": [
      {{
        "change": "Increase Gabapentin to 600mg TID",
        "rationale": "inadequate neuropathic pain control",
        "contingency": "if sedated, reduce to 300mg TID"
      }}
    ],
    "therapy_modifications": [
      {{
        "modification": "Advance to aquatic therapy",
        "rationale": "reduce weight-bearing stress",
        "frequency": "1x/week for 4 weeks"
      }}
    ],
    "procedure_recommendations": [
      {{
        "procedure": "L knee corticosteroid injection",
        "rationale": "persistent effusion and inflammation",
        "timing": "next 2 weeks if no improvement",
        "contingency": "if ineffective, consider ortho consult"
      }}
    ]
  }},
  
  "follow_up_planning": {{
    "next_appointment": {{
      "date": "4 weeks",
      "purpose": "Re-evaluate progress with new medication dose",
      "milestones_expected": "Pain <4/10, independent stair climbing, full ROM"
    }},
    "contingency_plans": [
      {{
        "scenario": "No improvement with increased Gabapentin",
        "action": "Consider switching to Pregabalin",
        "timing": "next appointment"
      }},
      {{
        "scenario": "Worsened pain or function",
        "action": "Move appointment sooner, consider injection",
        "timing": "contact office immediately"
      }}
    ],
    "long_term_goals": [
      "Return to full duty work in 8-12 weeks",
      "Independent ADLs without assistive device",
      "Pain management without opioid medications"
    ]
  }},
  
  "critical_progress_indicators": [
    {{
      "indicator": "Pain reduction from 7/10 to 4/10",
      "significance": "good medication response",
      "action": "continue current pharmacologic approach"
    }},
    {{
      "indicator": "Lifting capacity increased 10 lbs to 15 lbs",
      "significance": "functional improvement",
      "action": "continue strengthening program"
    }},
    {{
      "indicator": "Persistent knee effusion on MRI",
      "significance": "ongoing inflammation",
      "action": "consider corticosteroid injection"
    }},
    {{
      "indicator": "Improved gait pattern",
      "significance": "therapy effectiveness",
      "action": "advance gait training exercises"
    }}
  ],
  
  "overall_assessment": {{
    "progress_rating": "good",
    "treatment_effectiveness": "moderately effective",
    "patient_cooperation": "good compliance",
    "prognosis": "continued gradual improvement expected",
    "major_concerns": "persistent effusion may require intervention"
  }}
}}
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
        Build comprehensive narrative summary for PR-2 progress tracking.
        
        PR-2 style: "Clinical Status: [progress]. Treatment: [current interventions]. 
        Work Status: [restrictions]. Plan: [next steps]."
        """
        
        # Extract all progress data
        report_metadata = data.get("report_metadata", {})
        clinical_progress = data.get("clinical_progress", {})
        current_treatment = data.get("current_treatment_status", {})
        work_status = data.get("work_status_evolution", {})
        treatment_plan = data.get("treatment_plan_adjustments", {})
        follow_up = data.get("follow_up_planning", {})
        
        # Helper function for safe string conversion
        def safe_str(value, default=""):
            if not value:
                return default
            if isinstance(value, list):
                return ", ".join([str(x) for x in value if x])
            return str(value)
        
        # Build narrative sections
        narrative_parts = []
        
        # Section 1: CLINICAL STATUS & PROGRESS
        clinical_text = self._build_clinical_progress_narrative(clinical_progress, current_treatment)
        if clinical_text:
            narrative_parts.append(f"**Clinical Status:** {clinical_text}")
        
        # Section 2: TREATMENT RESPONSE & ADJUSTMENTS
        treatment_text = self._build_treatment_narrative(treatment_plan, data.get("medication_management", {}))
        if treatment_text:
            narrative_parts.append(f"**Treatment:** {treatment_text}")
        
        # Section 3: WORK STATUS EVOLUTION
        work_text = self._build_work_status_narrative(work_status)
        if work_text:
            narrative_parts.append(f"**Work Status:** {work_text}")
        
        # Section 4: FOLLOW-UP PLAN
        plan_text = self._build_followup_plan_narrative(follow_up)
        if plan_text:
            narrative_parts.append(f"**Plan:** {plan_text}")
        
        # Section 5: PHYSICIAN & DATE CONTEXT
        treating_physician = clinical_progress.get("treating_physician", {})
        physician_name = safe_str(treating_physician.get("name", ""))
        report_date = report_metadata.get("report_date", fallback_date)
        
        if physician_name:
            context_line = f"PR-2 by {physician_name} on {report_date}"
            narrative_parts.insert(0, context_line)
        
        # Join with proper formatting
        full_narrative = "\n\n".join(narrative_parts)
        
        logger.info(f"📝 PR-2 Narrative summary generated: {len(full_narrative)} characters")
        return full_narrative

    def _build_clinical_progress_narrative(self, clinical_progress: Dict, current_treatment: Dict) -> str:
        """Build clinical progress narrative section"""
        progress_parts = []
        
        # Overall status
        overall_status = clinical_progress.get("overall_clinical_status", "")
        if overall_status:
            progress_parts.append(overall_status)
        
        # Subjective improvements
        subjective = clinical_progress.get("subjective_improvement", {})
        pain_changes = subjective.get("pain_level_changes", "")
        symptom_progression = subjective.get("symptom_progression", "")
        
        if pain_changes:
            progress_parts.append(f"Pain: {pain_changes}")
        elif symptom_progression:
            progress_parts.append(symptom_progression)
        
        # Objective findings
        objective = clinical_progress.get("objective_findings", {})
        rom_improvement = objective.get("rom_improvement", "")
        strength_progress = objective.get("strength_progress", "")
        
        if rom_improvement:
            progress_parts.append(f"ROM: {rom_improvement}")
        if strength_progress:
            progress_parts.append(f"Strength: {strength_progress}")
        
        # Treatment compliance
        compliance = current_treatment.get("medication_adherence", "") or current_treatment.get("therapy_attendance", "")
        if compliance:
            progress_parts.append(f"Compliance: {compliance}")
        
        return ", ".join(progress_parts) if progress_parts else "Progress not specified"

    def _build_treatment_narrative(self, treatment_plan: Dict, medication_mgmt: Dict) -> str:
        """Build treatment response and adjustments narrative"""
        treatment_items = []
        
        # Medication changes
        recent_changes = medication_mgmt.get("recent_medication_changes", [])
        if recent_changes and isinstance(recent_changes, list):
            for change in recent_changes[:2]:  # Limit to 2 most recent
                if isinstance(change, dict):
                    med = change.get("medication", "")
                    change_desc = change.get("change", "")
                    if med and change_desc:
                        treatment_items.append(f"{med} {change_desc}")
        
        # Therapy modifications
        therapy_mods = treatment_plan.get("therapy_modifications", [])
        if therapy_mods and isinstance(therapy_mods, list):
            for mod in therapy_mods[:2]:
                if isinstance(mod, dict):
                    modification = mod.get("modification", "")
                    if modification:
                        treatment_items.append(modification)
        
        # Procedure recommendations
        procedures = treatment_plan.get("procedure_recommendations", [])
        if procedures and isinstance(procedures, list):
            for proc in procedures[:2]:
                if isinstance(proc, dict):
                    procedure = proc.get("procedure", "")
                    if procedure:
                        treatment_items.append(procedure)
        
        return "; ".join(treatment_items) if treatment_items else "Continuing current treatment"

    def _build_work_status_narrative(self, work_status: Dict) -> str:
        """Build work status evolution narrative"""
        parts = []
        
        # Current status
        current = work_status.get("current_work_status", "")
        if current:
            parts.append(current)
        
        # Restriction changes
        restriction_changes = work_status.get("restriction_changes", {})
        if restriction_changes:
            for change_type, change_desc in restriction_changes.items():
                if change_desc and isinstance(change_desc, str):
                    parts.append(change_desc)
        
        # Specific restrictions
        restrictions = work_status.get("work_restrictions", [])
        if restrictions and isinstance(restrictions, list):
            # Take most significant 2 restrictions
            significant_restrictions = []
            for restriction in restrictions:
                if isinstance(restriction, str) and restriction:
                    significant_restrictions.append(restriction)
                elif isinstance(restriction, dict) and restriction.get("restriction"):
                    significant_restrictions.append(restriction.get("restriction"))
            
            if significant_restrictions:
                parts.extend(significant_restrictions[:2])
        
        return ", ".join(parts) if parts else "Work status unchanged"

    def _build_followup_plan_narrative(self, follow_up: Dict) -> str:
        """Build follow-up planning narrative"""
        plan_items = []
        
        # Next appointment
        next_appt = follow_up.get("next_appointment", {})
        if next_appt:
            timing = next_appt.get("date", "")
            purpose = next_appt.get("purpose", "")
            if timing:
                plan_items.append(f"Follow-up in {timing}")
            elif purpose:
                plan_items.append(purpose)
        
        # Contingency plans
        contingencies = follow_up.get("contingency_plans", [])
        if contingencies and isinstance(contingencies, list):
            for contingency in contingencies[:1]:  # Just the primary contingency
                if isinstance(contingency, dict):
                    scenario = contingency.get("scenario", "")
                    action = contingency.get("action", "")
                    if scenario and action:
                        plan_items.append(f"If {scenario.lower()}: {action}")
        
        # Long-term goals
        goals = follow_up.get("long_term_goals", [])
        if goals and isinstance(goals, list):
            primary_goal = goals[0] if goals else ""
            if primary_goal and isinstance(primary_goal, str):
                plan_items.append(f"Goal: {primary_goal}")
            elif primary_goal and isinstance(primary_goal, dict) and primary_goal.get("goal"):
                plan_items.append(f"Goal: {primary_goal.get('goal')}")
        
        return "; ".join(plan_items) if plan_items else "Routine follow-up planned"

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for PR-2"""
        return {
            "report_metadata": {
                "report_type": "PR-2 Progress Report",
                "report_date": fallback_date,
                "evaluation_date": "",
                "time_since_last_visit": "",
                "next_scheduled_appointment": ""
            },
            "clinical_progress": {
                "treating_physician": {
                    "name": "",
                    "specialty": "",
                    "facility": ""
                },
                "body_part_assessed": "",
                "subjective_improvement": {
                    "pain_level_changes": "",
                    "symptom_progression": "",
                    "patient_reported_function": "",
                    "compliance_with_treatment": ""
                },
                "objective_findings": {
                    "examination_changes": "",
                    "rom_improvement": "",
                    "strength_progress": "",
                    "functional_testing": ""
                },
                "overall_clinical_status": ""
            },
            "current_treatment_status": {
                "medication_adherence": "",
                "therapy_attendance": "",
                "injection_procedure_response": "",
                "home_exercise_compliance": "",
                "treatment_barriers": ""
            },
            "work_status_evolution": {
                "current_work_status": "",
                "work_restrictions": [],
                "restriction_changes": {},
                "return_to_work_progression": {}
            },
            "treatment_plan_adjustments": {
                "medication_changes": [],
                "therapy_modifications": [],
                "procedure_recommendations": []
            },
            "follow_up_planning": {
                "next_appointment": {},
                "contingency_plans": [],
                "long_term_goals": []
            }
        }