"""
Specialist Consult Enhanced Extractor - Full Context with Context-Awareness
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
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction using DocumentContextAnalyzer guidance
    - Chain-of-thought reasoning for specialist recommendations
    - Optimized for consultation report specific patterns and clinical decision-making
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex patterns for consultation specific content
        self.consult_patterns = {
            'specialty': re.compile(r'\b(ortho|neuro|pain|psych|pm&r|cardio|pulm|rheum|surgery)\b', re.IGNORECASE),
            'recommendation_type': re.compile(r'\b(injection|surgery|therapy|medication|imaging|referral|follow-up)\b', re.IGNORECASE),
            'work_status': re.compile(r'\b(ttd|modified duty|full duty|light duty|sedentary|restrictions)\b', re.IGNORECASE),
            'urgency': re.compile(r'\b(urgent|emergent|routine|elective|asap)\b', re.IGNORECASE)
        }
        
        logger.info("✅ ConsultExtractorChained initialized (Full Context + Context-Aware)")

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
        Extract Consult data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Specialist Consultation)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("👨‍⚕️ STARTING CONSULT EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Log context guidance if available
        if context_analysis:
            primary_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            clinical_context = context_analysis.get("clinical_context", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Consulting Physician: {primary_physician.get('name', 'Unknown')}")
            logger.info(f"   Confidence: {primary_physician.get('confidence', 'Unknown')}")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Clinical Context: {clinical_context.get('referral_reason', 'Unknown')}")
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
            doc_type=doc_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )
        
        # Stage 2: Override consultant if context identified one with high confidence
        if context_analysis:
            context_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            if context_physician.get("name") and context_physician.get("confidence") in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified consultant: {context_physician.get('name')}")
                raw_result["consulting_physician_name"] = context_physician.get("name")
        
        # Stage 3: Fallback to DoctorDetector if no consultant identified
        if not raw_result.get("consulting_physician_name"):
            logger.info("🔍 No consultant from context/extraction, using DoctorDetector...")
            consultant_name = self._detect_consultant(text, page_zones)
            raw_result["consulting_physician_name"] = consultant_name
        
        # Stage 4: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 5: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info("=" * 80)
        logger.info("✅ CONSULT EXTRACTION COMPLETE (FULL CONTEXT)")
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
        Optimized for specialist consultation specific patterns and clinical decision-making.
        """
        logger.info("🔍 Processing ENTIRE consultation report in single context window with guidance...")
        
        # Extract guidance from context analysis
        primary_physician = ""
        focus_sections = []
        clinical_context = {}
        physician_reasoning = ""
        ambiguities = []
        
        if context_analysis:
            phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            primary_physician = phys_analysis.get("name", "")
            physician_reasoning = phys_analysis.get("reasoning", "")
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            clinical_context = context_analysis.get("clinical_context", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt for Consultations
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical documentation specialist analyzing a COMPLETE Specialist Consultation Report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE consultation report at once, allowing you to:
- Understand the complete referral context and clinical questions
- Correlate subjective complaints with objective examination findings
- Assess the specialist's diagnostic reasoning and clinical decision-making
- Identify treatment recommendations and their clinical rationale
- Provide comprehensive specialist assessment without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a finding/recommendation is NOT explicitly mentioned in the report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate clinical decisions
   - DO NOT fill in "typical" or "common" specialist recommendations
   - DO NOT use medical knowledge to "complete" incomplete information
   
   Examples:
   ✅ CORRECT: If report says "Recommend MRI lumbar spine", extract: "imaging_recommendations": ["MRI lumbar spine"]
   ❌ WRONG: If report says "Continue current medications", DO NOT extract specific medication names (not stated)
   ✅ CORRECT: Extract: "medication_recommendations": "Continue current medications" (exact wording)

2. **RECOMMENDATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY recommendations explicitly stated in assessment/plan sections
   - Include timing/scheduling ONLY if explicitly stated
   - DO NOT extract:
     * Treatments mentioned as "considered but not recommended"
     * Historical treatments unless explicitly recommended for continuation
     * Standard care that is not explicitly mentioned
   
   Examples:
   ✅ CORRECT: Report states "Recommend physical therapy 2x/week for 6 weeks"
   Extract: {{"therapy_recommendations": ["Physical therapy 2x/week for 6 weeks"]}}
   
   ❌ WRONG: Report states "Patient may benefit from therapy"
   DO NOT extract specific therapy recommendations (not specified)

3. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
   Examples:
   ✅ CORRECT: If no work restrictions mentioned, return: "work_restrictions": []
   ❌ WRONG: Return: "work_restrictions": "Not specified" (use empty list instead)

4. **EXACT CLINICAL TERMINOLOGY**
   - Use EXACT wording from report for diagnoses and recommendations
   - DO NOT upgrade "mild" to "moderate" or downgrade "severe" to "moderate"
   - Capture nuanced clinical language: "likely", "probable", "consistent with", "rule out"
   
   Examples:
   ✅ CORRECT: Report says "Clinical findings consistent with rotator cuff tendinopathy"
   Extract: "diagnostic_impression": "Clinical findings consistent with rotator cuff tendinopathy"
   
   ❌ WRONG: Report says "Clinical findings consistent with rotator cuff tendinopathy"
   DO NOT extract: "diagnostic_impression": "Rotator cuff tendinopathy" (changes meaning)

5. **NO CLINICAL DECISION-MAKING BEYOND REPORT**
   - DO NOT predict treatment outcomes
   - DO NOT suggest additional evaluations beyond what's recommended
   - DO NOT infer specialist preferences or practice patterns
   
   Examples:
   ❌ WRONG: Report mentions "corticosteroid injection"
   DO NOT assume: "will provide 3 months of pain relief" (outcome not stated)
   ✅ CORRECT: Extract: "procedure_recommendations": ["corticosteroid injection"]

6. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
   Before returning your extraction, verify:
   □ Every recommendation has a direct quote in the report
   □ Every diagnosis is explicitly stated (not inferred from symptoms)
   □ Every work restriction is directly from restrictions section
   □ No fields are filled with "expected" or "typical" specialist advice
   □ Empty fields are truly empty (not "Not mentioned" or "Unknown")

CONSULTATION SPECIFIC EXTRACTION FOCUS - 6 CRITICAL SPECIALIST CATEGORIES:

I. CONSULTATION CONTEXT & HISTORY
- Referral source and clinical questions
- Presenting symptoms and duration
- Previous treatments and responses
- Relevant medical and surgical history

II. SPECIALIST EXAMINATION FINDINGS
- Physical examination: ROM, strength, special tests
- Neurological assessment: sensation, reflexes, coordination
- Functional assessment: gait, posture, functional limitations
- Diagnostic test review: imaging, labs, other studies

III. DIAGNOSTIC IMPRESSION & ASSESSMENT
- Primary diagnosis and differential diagnoses
- Clinical correlation and diagnostic certainty
- Causation assessment (if work-related)
- Prognosis and expected clinical course

IV. TREATMENT RECOMMENDATIONS (Most Critical)
- Medication management: new prescriptions, adjustments, discontinuations
- Therapeutic interventions: PT, OT, other therapies
- Interventional procedures: injections, nerve blocks, other procedures
- Surgical recommendations: specific procedures, timing, indications

V. DIAGNOSTIC & CONSULTATION RECOMMENDATIONS
- Additional imaging studies with clinical rationale
- Laboratory or other diagnostic testing
- Additional specialist referrals
- Further evaluation or workup needs

VI. WORK STATUS & FUNCTIONAL CAPACITY
- Current work capacity and restrictions
- Specific functional limitations
- Duration of restrictions
- Return-to-work progression plan

⚠️ FINAL REMINDER:
- If information is NOT in the report, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate clinical decisions
- RECOMMENDATIONS: Only extract what is explicitly recommended
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE specialist consultation report and extract ALL relevant clinical information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE SPECIALIST CONSULTATION REPORT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all specialist assessment details:

{{
  "consultation_context": {{
    "report_type": "Specialist Consultation",
    "consultation_date": "",
    "referral_source": "",
    "referring_physician": "",
    "clinical_questions": "",
    "consultation_type": "initial/follow-up/pre-op/post-op"
  }},
  
  "clinical_presentation": {{
    "chief_complaint": "",
    "symptom_duration": "",
    "presenting_symptoms": [
      "Right shoulder pain",
      "Limited overhead reaching",
      "Night pain"
    ],
    "previous_treatments": [
      "Physical therapy - minimal improvement",
      "NSAIDs - partial relief",
      "Previous injection - good but temporary relief"
    ],
    "relevant_history": [
      "Work-related injury 6 months ago",
      "No prior shoulder problems",
      "Non-smoker, no diabetes"
    ]
  }},
  
  "consulting_physician": {{
    "name": "{primary_physician}",
    "specialty": "",
    "credentials": "",
    "facility": ""
  }},
  
  "examination_findings": {{
    "physical_exam": {{
      "inspection": "mild atrophy supraspinatus fossa",
      "palpation": "tenderness greater tuberosity",
      "range_of_motion": "forward flexion 120° (NL 180°), abduction 100° (NL 180°)",
      "strength": "supraspinatus 4/5, infraspinatus 5/5",
      "special_tests": "positive Neer, positive Hawkins, positive empty can"
    }},
    "neurological_exam": {{
      "sensation": "intact to light touch C5-T1",
      "reflexes": "biceps 2+, brachioradialis 2+",
      "motor": "deltoid 5/5, biceps 5/5, triceps 5/5"
    }},
    "functional_assessment": {{
      "gait": "normal",
      "posture": "forward shoulder posture",
      "functional_limitations": "unable to lift overhead, difficulty with reaching behind back"
    }}
  }},
  
  "diagnostic_impression": {{
    "primary_diagnosis": "Rotator cuff tendinopathy with impingement",
    "differential_diagnosis": [
      "Partial thickness rotator cuff tear",
      "Acromioclavicular joint arthritis",
      "Biceps tendinopathy"
    ],
    "diagnostic_certainty": "high",
    "clinical_correlation": "consistent with overhead work activities",
    "causation_assessment": "work-related",
    "prognosis": "good with appropriate treatment"
  }},
  
  "diagnostic_recommendations": {{
    "imaging_studies": [
      {{
        "study": "MRI right shoulder without contrast",
        "rationale": "evaluate for rotator cuff tear given refractory symptoms",
        "urgency": "routine"
      }}
    ],
    "laboratory_tests": [
      {{
        "test": "Complete blood count",
        "rationale": "pre-procedure baseline",
        "urgency": "routine"
      }}
    ],
    "additional_workup": [
      "Consider diagnostic ultrasound if MRI equivocal"
    ]
  }},
  
  "treatment_recommendations": {{
    "medication_management": [
      {{
        "medication": "Naproxen 500mg twice daily",
        "duration": "2 weeks",
        "rationale": "anti-inflammatory for tendinopathy",
        "new_prescription": true
      }},
      {{
        "medication": "Cyclobenzaprine 10mg at bedtime",
        "duration": "as needed",
        "rationale": "muscle relaxation for night pain",
        "new_prescription": true
      }}
    ],
    "therapy_recommendations": [
      {{
        "therapy_type": "Physical therapy",
        "frequency": "2 times per week",
        "duration": "6 weeks",
        "focus": "rotator cuff strengthening, scapular stabilization",
        "goals": "improve overhead function, reduce pain"
      }}
    ],
    "interventional_procedures": [
      {{
        "procedure": "Subacromial corticosteroid injection",
        "location": "right shoulder",
        "rationale": "reduce inflammation, facilitate therapy participation",
        "timing": "within 2 weeks",
        "urgency": "semi-urgent"
      }}
    ],
    "surgical_recommendations": [
      {{
        "procedure": "Arthroscopic subacromial decompression",
        "indication": "if conservative measures fail after 3 months",
        "urgency": "elective",
        "contingency": "dependent on MRI findings and treatment response"
      }}
    ]
  }},
  
  "additional_consultations": [
    {{
      "specialty": "Pain Management",
      "rationale": "if injection provides incomplete relief",
      "urgency": "routine"
    }},
    {{
      "specialty": "Work Capacity Evaluation",
      "rationale": "formal functional assessment for work restrictions",
      "urgency": "routine"
    }}
  ],
  
  "work_status_recommendations": {{
    "current_work_status": "Modified Duty",
    "work_restrictions": [
      "No overhead work",
      "No lifting greater than 10 pounds",
      "No repetitive reaching above shoulder level",
      "Limited pushing/pulling with right arm"
    ],
    "restriction_duration": "6 weeks",
    "reevaluation_timing": "after completion of injection and 4 weeks of therapy",
    "return_to_work_progression": [
      "Current: modified duty with above restrictions",
      "Next: light duty if 50% improvement in symptoms",
      "Goal: full duty after complete rehabilitation"
    ]
  }},
  
  "follow_up_planning": {{
    "next_appointment": {{
      "timing": "6 weeks",
      "purpose": "Re-evaluate treatment response, review MRI results",
      "contingencies": [
        "Sooner if worsening symptoms",
        "Phone update after MRI completed"
      ]
    }},
    "treatment_milestones": [
      "50% pain reduction after injection",
      "Improved overhead ROM after 4 weeks therapy",
      "Strength normalization after 8 weeks therapy"
    ],
    "discharge_criteria": [
      "Pain-free full ROM",
      "Normal strength",
      "Return to full work duties"
    ]
  }},
  
  "critical_recommendations": [
    {{
      "recommendation": "Subacromial corticosteroid injection within 2 weeks",
      "significance": "high",
      "rationale": "break pain cycle to enable effective therapy",
      "action": "schedule with interventional radiology/pain management"
    }},
    {{
      "recommendation": "MRI right shoulder to rule out rotator cuff tear",
      "significance": "high",
      "rationale": "guide surgical decision-making if conservative treatment fails",
      "action": "order MRI within 4 weeks"
    }},
    {{
      "recommendation": "Strict adherence to work restrictions for 6 weeks",
      "significance": "medium",
      "rationale": "prevent symptom exacerbation during healing phase",
      "action": "communicate restrictions to employer"
    }}
  ]
}}
""")

        # Build context guidance summary
        context_guidance_text = f"""
CONSULTING PHYSICIAN: {primary_physician or 'Not identified in context'}
REASONING: {physician_reasoning or 'See document for identification'}

FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'Assessment, Plan, Recommendations'}

CLINICAL CONTEXT:
- Referral Reason: {clinical_context.get('referral_reason', 'Not specified')}
- Clinical History: {clinical_context.get('clinical_history', 'Not provided')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context consultation extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_guidance_text,
                "primary_physician": primary_physician or "Extract from document",
                "physician_reasoning": physician_reasoning or "Use signature and documentation sections",
                "focus_sections": ', '.join(focus_sections) if focus_sections else "Standard consultation sections",
                "clinical_context": str(clinical_context),
                "ambiguities": str(ambiguities)
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context consultation extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted consultation data from complete {len(text):,} char document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context consultation extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(fallback_date)

    def _detect_consultant(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Fallback: Detect consultant using DoctorDetector"""
        logger.info("🔍 Fallback: Running DoctorDetector for consultant...")
        
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

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result from extracted consultation data"""
        logger.info("🔨 Building initial consultation extraction result...")
        
        # Extract core consultation information
        consultation_context = raw_data.get("consultation_context", {})
        consulting_physician = raw_data.get("consulting_physician", {})
        diagnostic_impression = raw_data.get("diagnostic_impression", {})
        
        # Build comprehensive consultation summary
        summary_line = self._build_consultation_narrative_summary(raw_data, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=consultation_context.get("consultation_date", fallback_date),
            summary_line=summary_line,
            examiner_name=raw_data.get("consulting_physician_name", ""),
            specialty=consulting_physician.get("specialty", ""),
            body_parts=self._extract_body_parts_from_consultation(raw_data),
            raw_data=raw_data,
        )
        
        logger.info(f"✅ Initial consultation result built (Consultant: {result.examiner_name})")
        return result

    def _extract_body_parts_from_consultation(self, data: Dict) -> List[str]:
        """Extract body parts from consultation data"""
        body_parts = []
        
        # From clinical presentation
        clinical_presentation = data.get("clinical_presentation", {})
        chief_complaint = clinical_presentation.get("chief_complaint", "")
        if chief_complaint:
            # Extract body parts from common patterns
            if any(part in chief_complaint.lower() for part in ['shoulder', 'arm', 'upper extremity']):
                body_parts.append('Upper Extremity')
            if any(part in chief_complaint.lower() for part in ['back', 'spine', 'lumbar']):
                body_parts.append('Spine')
            if any(part in chief_complaint.lower() for part in ['knee', 'leg', 'lower extremity']):
                body_parts.append('Lower Extremity')
        
        # From diagnostic impression
        diagnostic_impression = data.get("diagnostic_impression", {})
        primary_diagnosis = diagnostic_impression.get("primary_diagnosis", "")
        if primary_diagnosis and not body_parts:
            if any(part in primary_diagnosis.lower() for part in ['shoulder', 'rotator cuff']):
                body_parts.append('Upper Extremity')
            if any(part in primary_diagnosis.lower() for part in ['spine', 'disc', 'radiculopathy']):
                body_parts.append('Spine')
            if any(part in primary_diagnosis.lower() for part in ['knee', 'meniscus', 'acl']):
                body_parts.append('Lower Extremity')
        
        return body_parts if body_parts else []

    def _build_consultation_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive narrative summary for consultation reports.
        
        Consultation style: "Specialty: [type]. Assessment: [diagnosis]. Recommendations: [key actions]. Work Status: [restrictions]."
        """
        
        # Extract all consultation data
        consultation_context = data.get("consultation_context", {})
        consulting_physician = data.get("consulting_physician", {})
        diagnostic_impression = data.get("diagnostic_impression", {})
        treatment_recommendations = data.get("treatment_recommendations", {})
        work_status_recommendations = data.get("work_status_recommendations", {})
        critical_recommendations = data.get("critical_recommendations", [])
        
        # Build narrative sections
        narrative_parts = []
        
        # Section 1: SPECIALTY & ASSESSMENT
        assessment_text = self._build_assessment_narrative(consulting_physician, diagnostic_impression)
        if assessment_text:
            narrative_parts.append(f"**Assessment:** {assessment_text}")
        
        # Section 2: KEY RECOMMENDATIONS
        recommendations_text = self._build_recommendations_narrative(treatment_recommendations, critical_recommendations)
        if recommendations_text:
            narrative_parts.append(f"**Recommendations:** {recommendations_text}")
        
        # Section 3: WORK STATUS & FOLLOW-UP
        work_followup_text = self._build_work_followup_narrative(work_status_recommendations, consultation_context)
        if work_followup_text:
            narrative_parts.append(f"**Work Status:** {work_followup_text}")
        
        # Section 4: CONSULTANT & DATE CONTEXT
        consultant_name = self._safe_str(consulting_physician.get("name", ""))
        consult_date = consultation_context.get("consultation_date", fallback_date)
        
        if consultant_name:
            context_line = f"Consultation by {consultant_name} on {consult_date}"
            narrative_parts.insert(0, context_line)
        
        # Join with proper formatting
        full_narrative = "\n\n".join(narrative_parts)
        
        logger.info(f"📝 Consultation narrative summary generated: {len(full_narrative)} characters")
        return full_narrative

    def _safe_str(self, value, default="") -> str:
        """Safely convert any value to string, handling lists and nested structures."""
        if not value:
            return default
        
        if isinstance(value, list):
            # Flatten list and convert all items to strings
            string_items = []
            for item in value:
                if isinstance(item, (dict, list)):
                    # For complex structures, use JSON representation
                    try:
                        import json
                        string_items.append(json.dumps(item, default=str))
                    except:
                        string_items.append(str(item))
                elif item:
                    string_items.append(str(item))
            return ", ".join(string_items) if string_items else default
        
        if isinstance(value, dict):
            # For dictionaries, use JSON representation
            try:
                import json
                return json.dumps(value, default=str)
            except:
                return str(value)
        
        return str(value) if value else default

    def _build_assessment_narrative(self, consulting_physician: Dict, diagnostic_impression: Dict) -> str:
        """Build assessment and diagnosis narrative section"""
        assessment_parts = []
        
        # Specialty - safely convert to string
        specialty = self._safe_str(consulting_physician.get("specialty", ""))
        if specialty and specialty.strip():
            assessment_parts.append(specialty.strip())
        
        # Primary diagnosis - safely convert to string
        primary_dx = self._safe_str(diagnostic_impression.get("primary_diagnosis", ""))
        if primary_dx and primary_dx.strip():
            assessment_parts.append(primary_dx.strip())
        
        # Clinical correlation - safely convert to string
        correlation = self._safe_str(diagnostic_impression.get("clinical_correlation", ""))
        if correlation and correlation.strip():
            assessment_parts.append(correlation.strip())
        
        # Only join if we have valid string parts
        valid_parts = [part for part in assessment_parts if part and isinstance(part, str)]
        return " - ".join(valid_parts) if valid_parts else "Assessment not specified"

    def _build_recommendations_narrative(self, treatment_recommendations: Dict, critical_recommendations: list) -> str:
        """Build key recommendations narrative"""
        recommendation_items = []
        
        # Critical recommendations (highest priority)
        if critical_recommendations and isinstance(critical_recommendations, list):
            for critical in critical_recommendations[:2]:  # Top 2 critical recommendations
                if isinstance(critical, dict):
                    recommendation = self._safe_str(critical.get("recommendation", ""))
                    if recommendation and recommendation.strip():
                        recommendation_items.append(recommendation.strip())
        
        # Interventional procedures
        procedures = treatment_recommendations.get("interventional_procedures", [])
        if procedures and isinstance(procedures, list):
            for proc in procedures[:2]:
                if isinstance(proc, dict):
                    procedure = self._safe_str(proc.get("procedure", ""))
                    if procedure and procedure.strip():
                        recommendation_items.append(procedure.strip())
        
        # Therapy recommendations
        therapy = treatment_recommendations.get("therapy_recommendations", [])
        if therapy and isinstance(therapy, list):
            for tx in therapy[:1]:
                if isinstance(tx, dict):
                    therapy_type = self._safe_str(tx.get("therapy_type", ""))
                    if therapy_type and therapy_type.strip():
                        recommendation_items.append(therapy_type.strip())
        
        # Medication recommendations
        medications = treatment_recommendations.get("medication_management", [])
        if medications and isinstance(medications, list):
            for med in medications[:1]:
                if isinstance(med, dict):
                    medication = self._safe_str(med.get("medication", ""))
                    if medication and medication.strip():
                        recommendation_items.append(medication.strip())
        
        # Only return if we have valid recommendations
        valid_items = [item for item in recommendation_items if item and isinstance(item, str)]
        return "; ".join(valid_items) if valid_items else "Continue current management"

    def _build_work_followup_narrative(self, work_status_recommendations: Dict, consultation_context: Dict) -> str:
        """Build work status and follow-up planning narrative"""
        work_parts = []
        
        # Current work status
        current_status = self._safe_str(work_status_recommendations.get("current_work_status", ""))
        if current_status and current_status.strip():
            work_parts.append(current_status.strip())
        
        # Work restrictions
        restrictions = work_status_recommendations.get("work_restrictions", [])
        if restrictions and isinstance(restrictions, list):
            # Take most significant 2 restrictions
            significant_restrictions = []
            for restriction in restrictions:
                restriction_str = self._safe_str(restriction)
                if restriction_str and restriction_str.strip():
                    significant_restrictions.append(restriction_str.strip())
            
            if significant_restrictions:
                work_parts.extend(significant_restrictions[:2])
        
        # Follow-up timing
        follow_up = self._safe_str(consultation_context.get("consultation_type", ""))
        if "follow-up" in follow_up.lower():
            work_parts.append("scheduled follow-up")
        
        # Only return if we have valid work parts
        valid_parts = [part for part in work_parts if part and isinstance(part, str)]
        return ", ".join(valid_parts) if valid_parts else "Work status not specified"

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for consultations"""
        return {
            "consultation_context": {
                "report_type": "Specialist Consultation",
                "consultation_date": fallback_date,
                "referral_source": "",
                "referring_physician": "",
                "clinical_questions": "",
                "consultation_type": ""
            },
            "clinical_presentation": {
                "chief_complaint": "",
                "symptom_duration": "",
                "presenting_symptoms": [],
                "previous_treatments": [],
                "relevant_history": []
            },
            "consulting_physician": {
                "name": "",
                "specialty": "",
                "credentials": "",
                "facility": ""
            },
            "examination_findings": {
                "physical_exam": {
                    "inspection": "",
                    "palpation": "",
                    "range_of_motion": "",
                    "strength": "",
                    "special_tests": ""
                },
                "neurological_exam": {
                    "sensation": "",
                    "reflexes": "",
                    "motor": ""
                },
                "functional_assessment": {
                    "gait": "",
                    "posture": "",
                    "functional_limitations": ""
                }
            },
            "diagnostic_impression": {
                "primary_diagnosis": "",
                "differential_diagnosis": [],
                "diagnostic_certainty": "",
                "clinical_correlation": "",
                "causation_assessment": "",
                "prognosis": ""
            },
            "diagnostic_recommendations": {
                "imaging_studies": [],
                "laboratory_tests": [],
                "additional_workup": []
            },
            "treatment_recommendations": {
                "medication_management": [],
                "therapy_recommendations": [],
                "interventional_procedures": [],
                "surgical_recommendations": []
            },
            "additional_consultations": [],
            "work_status_recommendations": {
                "current_work_status": "",
                "work_restrictions": [],
                "restriction_duration": "",
                "reevaluation_timing": "",
                "return_to_work_progression": []
            },
            "follow_up_planning": {
                "next_appointment": {},
                "treatment_milestones": [],
                "discharge_criteria": []
            },
            "critical_recommendations": []
        }