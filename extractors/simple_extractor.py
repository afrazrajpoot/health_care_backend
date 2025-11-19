"""
Enhanced Simple Extractor with FULL Context-Awareness and Comprehensive Coverage
Handles all document types with dynamic, intelligent extraction
Version: 2.0 - Production Ready
"""

import re
import logging
import time
import json
from typing import Dict, Optional, List, Tuple, Union
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.doctor_detector import DoctorDetector
from extractors.prompt_manager import PromptManager

logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Production-grade universal extractor with comprehensive context-awareness.
    
    Features:
    - Dynamic extraction adapting to ANY medical document type
    - Context-guided intelligent field detection
    - Anti-hallucination safeguards with provenance tracking
    - Surgical event detection
    - Medication contradiction checking
    - Hierarchical summary generation
    - Comprehensive verification system
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.doctor_detector = DoctorDetector(llm)
        self.prompt_manager = PromptManager()
        logger.info("✅ SimpleExtractor v2.0 initialized with FULL CONTEXT-AWARENESS")
    
    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict] = None,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Universal context-aware extraction for ALL document types.
        
        Args:
            text: Full document text
            doc_type: Document type
            fallback_date: Fallback date
            context_analysis: Context from DocumentContextAnalyzer
            page_zones: Page-based text zones
            raw_text: Original flat text
        
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info(f"🚀 UNIVERSAL EXTRACTION v2.0: {doc_type}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # STEP 1: Extract context guidance
            context_guidance = self._extract_context_guidance(context_analysis)
            
            # STEP 2: Calculate document complexity
            complexity_score = self._calculate_document_complexity(text, doc_type)
            logger.info(f"📊 Document complexity: {complexity_score}/10")
            
            # STEP 3: Validate inputs
            if not text or not text.strip():
                raise ValueError("Empty document text provided")
            
            # STEP 4: Comprehensive extraction with context
            raw_data = self._extract_with_universal_framework(
                text=text,
                doc_type=doc_type,
                fallback_date=fallback_date,
                context_guidance=context_guidance,
                complexity_score=complexity_score
            )
            
            # STEP 5: Override physician if context detected one
            if context_guidance["primary_physician"] and context_guidance["physician_confidence"] in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified physician: {context_guidance['primary_physician']}")
                if isinstance(raw_data, dict) and "document_intelligence" in raw_data:
                    raw_data["document_intelligence"]["author"]["name"] = context_guidance["primary_physician"]
            
            # STEP 6: Run comprehensive verification
            verification = self._comprehensive_verification(text, raw_data, doc_type)
            raw_data["metadata"]["verification"] = verification
            
            if verification["recommended_action"] != "AUTO_ACCEPT":
                logger.warning(f"⚠️ Verification flagged {len(verification['issues'])} issues — human review required")
                raw_data["metadata"]["requires_human_review"] = True
            else:
                logger.info("✅ Verification passed: AUTO_ACCEPT")
            
            # STEP 7: Build hierarchical long summary
            long_summary = self._build_hierarchical_summary(
                raw_data=raw_data,
                doc_type=doc_type,
                verification=verification
            )
            
            # STEP 8: Generate short summary
            short_summary = self._generate_short_summary_from_long(long_summary, doc_type)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Extraction completed in {elapsed_time:.2f}s")
            logger.info(f"📊 Results: {len(long_summary)} chars long, {len(short_summary.split())} words short")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "raw_data": raw_data  # Include for debugging/auditing
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_type}: {str(e)}", exc_info=True)
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _calculate_document_complexity(self, text: str, doc_type: str) -> int:
        """
        Calculate document complexity score (1-10) to adjust extraction depth.
        
        Factors:
        - Document length
        - Technical terminology density
        - Number of sections
        - Presence of tables/structured data
        - Document type inherent complexity
        """
        score = 5  # Base score
        
        # Length factor
        word_count = len(text.split())
        if word_count > 5000:
            score += 2
        elif word_count > 2000:
            score += 1
        
        # Technical density
        technical_terms = [
            'diagnosis', 'prognosis', 'treatment', 'surgery', 'procedure',
            'imaging', 'mri', 'ct', 'emg', 'physical exam', 'medication',
            'restriction', 'mmi', 'impairment', 'causation', 'apportionment'
        ]
        tech_count = sum(1 for term in technical_terms if term in text.lower())
        if tech_count > 10:
            score += 1
        
        # Document type complexity
        complex_types = ["QME", "AME", "IME", "SURGERY_REPORT", "DISCHARGE", "FCE"]
        if doc_type in complex_types:
            score += 2
        
        return min(10, max(1, score))
    
    def _extract_context_guidance(self, context_analysis: Optional[Dict]) -> Dict:
        """Extract and structure context guidance."""
        if not context_analysis:
            logger.warning("⚠️ No context analysis provided")
            return {
                "primary_physician": "",
                "physician_confidence": "",
                "physician_reasoning": "",
                "focus_sections": [],
                "critical_locations": {},
                "ambiguities": []
            }
        
        phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
        
        guidance = {
            "primary_physician": phys_analysis.get("name", ""),
            "physician_confidence": phys_analysis.get("confidence", ""),
            "physician_reasoning": phys_analysis.get("reasoning", ""),
            "focus_sections": context_analysis.get("extraction_guidance", {}).get("focus_on_sections", []),
            "critical_locations": context_analysis.get("critical_findings_map", {}),
            "ambiguities": context_analysis.get("ambiguities_detected", [])
        }
        
        logger.info(f"🎯 Context Guidance: Physician={guidance['primary_physician']}, "
                   f"Sections={len(guidance['focus_sections'])}, "
                   f"Ambiguities={len(guidance['ambiguities'])}")
        
        return guidance
    
    def _extract_with_universal_framework(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_guidance: Dict,
        complexity_score: int
    ) -> Dict:
        """
        Universal extraction framework that adapts to ANY medical document.
        Uses escaped JSON in prompts to avoid LangChain parsing errors.
        """
        logger.info("🔍 Extracting with UNIVERSAL FRAMEWORK...")
        
        context_guidance_text = self._build_context_guidance_text(context_guidance)
        
        # FIXED: Double braces {{ }} for JSON in template
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical document extraction system for Workers' Compensation and general medicine.

DOCUMENT TYPE: {doc_type}
COMPLEXITY SCORE: {complexity_score}/10

CONTEXT GUIDANCE:
{context_guidance_text}

═══════════════════════════════════════════════════════════════════
EXTRACTION MISSION
═══════════════════════════════════════════════════════════════════

Create a COMPREHENSIVE, ACTIONABLE extraction that allows physicians to make 
informed decisions WITHOUT reading the full document.

Extract information across ALL relevant domains with provenance tracking.

═══════════════════════════════════════════════════════════════════
UNIVERSAL EXTRACTION DOMAINS
═══════════════════════════════════════════════════════════════════

DOMAIN 1: DOCUMENT INTELLIGENCE
• Document type (actual, not assumed)
• Report date, examination date, all relevant dates
• Author: Name, credentials, specialty, role
• Document purpose and legal context
• Completeness status

DOMAIN 2: PATIENT & INCIDENT CONTEXT
• Patient: Name, DOB, Age, Gender, MRN
• Injury details: Date, mechanism, body parts, setting
• Claim information: Claim #, adjuster, employer
• Chief complaint (patient's words)
• Symptom trajectory: improving/worsening/stable

DOMAIN 3: CLINICAL FINDINGS (HIGHEST PRIORITY)
A. Diagnoses:
   • Primary diagnoses with ICD-10 codes
   • Secondary/comorbid conditions
   • All affected body parts and anatomical structures
   • Severity indicators
   • Causation opinions (industrial vs non-industrial)
   • Differential diagnoses

B. Current Clinical Status:
   • Pain: VAS scores, locations, character, factors
   • Functional limitations (specific activities)
   • Range of motion (with measurements)
   • Strength testing results
   • Neurological findings (sensory, motor, reflexes)
   • Gait and mobility
   • Mental status

C. Objective Findings:
   • Vital signs
   • Physical exam by body system
   • Positive clinical tests (names and results)
   • Significant negative findings
   • Observable signs (swelling, atrophy, scars)

DOMAIN 4: DIAGNOSTIC WORKUP
• Imaging (MRI, CT, X-ray, US): Date, findings, impression
• Laboratory: Test name, result, reference range
• Electrodiagnostic (EMG/NCS): Nerves tested, findings
• Functional testing (FCE, ROM)
• Specialty consultations

DOMAIN 5: TREATMENT COMPREHENSIVE
A. Medications (CRITICAL - Be precise):
   Current: Name, dose, frequency, route, prescriber, efficacy
   Discontinued: Name, reason, date
   Recommended: Proposals for future use

B. Procedures & Surgeries:
   Past: Procedure, date, surgeon, outcome
   Planned: Recommendations, necessity rationale

C. Conservative Care:
   • Physical therapy: Frequency, modalities, progress
   • Injections: Type, location, relief, duration
   • Chiropractic, acupuncture, other therapies

DOMAIN 6: WORK STATUS & RESTRICTIONS (WORKERS' COMP CRITICAL)
• Current work status: Off/modified/full duty
• Specific restrictions:
  - Lift limits (weight, frequency)
  - Positional (bending, twisting, overhead)
  - Duration (standing, sitting, walking)
  - Environmental restrictions
  - Start date and expected duration
• Temporary/Permanent Disability status
• MMI/P&S status:
  - Date, impairment rating, apportionment
  - Future medical care needs
• Vocational implications

DOMAIN 7: MEDICAL NECESSITY & AUTHORIZATION
• Request details
• Determination: Approved/Denied/Modified
• Medical necessity rationale
• Guidelines cited (ODG, ACOEM, etc.)
• Approved parameters (frequency, duration)
• Alternatives considered

DOMAIN 8: PROGNOSIS & FUTURE CARE
• Short-term prognosis (3-6 months)
• Long-term prognosis
• Recovery timeline
• Future care needs by priority
• Follow-up plan and re-evaluation criteria

DOMAIN 9: MEDICAL-LEGAL (IME/QME/AME)
• Causation analysis: Industrial %, apportionment
• Consistency analysis
• Records reviewed
• Specific questions answered

DOMAIN 10: CRITICAL FINDINGS & RED FLAGS
• URGENT findings requiring immediate action
• Safety concerns
• Medication concerns (interactions, opioids, ADRs)
• Compliance issues
• Inconsistencies
• Need for specialist referral

═══════════════════════════════════════════════════════════════════
ANTI-HALLUCINATION RULES (ABSOLUTE)
═══════════════════════════════════════════════════════════════════

1. ONLY extract explicitly stated information
   ❌ NO inference, assumption, or typical values
   ✅ Use exact quotes for ambiguous items
   ✅ Mark uncertain items with [UNCLEAR]

2. For EVERY extracted finding include:
   • source_snippet: Exact text (max 200 chars)
   • confidence: HIGH/MEDIUM/LOW
   • page_reference: If identifiable

3. Medications (STRICTEST):
   ✅ Current = ONLY currently taking
   ✅ Dose ONLY if explicitly stated
   ❌ NO discontinued meds in "current"
   ❌ NO recommendations in "current"

4. If missing/unclear:
   ✅ Return empty string/list
   ✅ Add to missing_critical_fields[]
   ✅ Note in ambiguities[] if contradictory

5. Surgical procedures:
   ✅ Extract ALL mentions of surgery (past/planned)
   ✅ Keywords: arthroplasty, fusion, repair, replacement, ectomy, otomy, plasty

═══════════════════════════════════════════════════════════════════
OUTPUT JSON SCHEMA (Note: Double braces in actual template)
═══════════════════════════════════════════════════════════════════

Return this exact structure:

{{{{
  "document_intelligence": {{{{
    "detected_type": "",
    "detected_subtype": "",
    "report_date": "",
    "examination_date": "",
    "all_dates_mentioned": [],
    "author": {{{{
      "name": "",
      "credentials": "",
      "specialty": "",
      "role": "",
      "source_snippet": ""
    }}}},
    "document_purpose": "",
    "legal_context": "",
    "completeness_status": ""
  }}}},
  
  "patient_context": {{{{
    "name": "",
    "dob": "",
    "age": "",
    "gender": "",
    "mrn": "",
    "injury_details": {{{{
      "date_of_injury": "",
      "mechanism": "",
      "body_parts_injured": [],
      "setting": "",
      "source_snippet": ""
    }}}},
    "claim_info": {{{{
      "claim_number": "",
      "adjuster": "",
      "employer": ""
    }}}},
    "chief_complaint": "",
    "symptom_trajectory": ""
  }}}},
  
  "clinical_findings": {{{{
    "diagnoses": [
      {{{{
        "diagnosis": "",
        "icd10": "",
        "body_part": "",
        "laterality": "",
        "severity": "",
        "causation": "",
        "is_primary": true,
        "confidence": "",
        "source_snippet": ""
      }}}}
    ],
    "current_status": {{{{
      "pain": {{{{
        "locations": [],
        "severity_scales": [],
        "character": "",
        "exacerbating_factors": [],
        "relieving_factors": []
      }}}},
      "functional_limitations": [],
      "range_of_motion": [],
      "strength": [],
      "neurological": {{{{
        "sensory": [],
        "motor": [],
        "reflexes": []
      }}}},
      "gait_mobility": {{{{}}
    }}}},
    "objective_findings": {{{{
      "vitals": {{}},
      "physical_exam_by_system": {{}},
      "positive_tests": [],
      "significant_negatives": [],
      "observable_signs": []
    }}}}
  }}}},
  
  "diagnostics": {{{{
    "imaging": [
      {{{{
        "type": "",
        "date": "",
        "body_part": "",
        "facility": "",
        "key_findings": [],
        "impression": "",
        "recommendations": "",
        "source_snippet": ""
      }}}}
    ],
    "laboratory": [],
    "electrodiagnostic": [],
    "functional_testing": [],
    "consultations": []
  }}}},
  
  "treatment": {{{{
    "medications": {{{{
      "current": [
        {{{{
          "name": "",
          "dose": "",
          "frequency": "",
          "route": "",
          "prescriber": "",
          "date_started": "",
          "efficacy": "",
          "side_effects": "",
          "compliance": "",
          "source_snippet": ""
        }}}}
      ],
      "discontinued": [],
      "recommended": []
    }}}},
    "procedures": {{{{
      "completed": [
        {{{{
          "procedure": "",
          "date": "",
          "surgeon": "",
          "facility": "",
          "outcome": "",
          "complications": "",
          "source_snippet": ""
        }}}}
      ],
      "planned": []
    }}}},
    "conservative_care": {{{{
      "physical_therapy": {{{{
        "frequency": "",
        "duration": "",
        "modalities": [],
        "progress": "",
        "source_snippet": ""
      }}}},
      "injections": [],
      "chiropractic": {{}},
      "other_modalities": []
    }}}}
  }}}},
  
  "work_status": {{{{
    "current_status": "",
    "restrictions": [
      {{{{
        "category": "",
        "specification": "",
        "duration": "",
        "source_snippet": ""
      }}}}
    ],
    "temporary_disability": {{{{
      "status": "",
      "percentage": "",
      "dates": ""
    }}}},
    "permanent_disability": {{}},
    "mmi_status": {{{{
      "is_mmi": "",
      "date_reached": "",
      "impairment_rating": "",
      "apportionment": "",
      "justification": "",
      "future_medical_care": "",
      "source_snippet": ""
    }}}},
    "vocational_impact": {{{{
      "return_to_usual_occupation": "",
      "need_for_retraining": ""
    }}}}
  }}}},
  
  "utilization_review": {{{{
    "request_description": "",
    "determination": "",
    "medical_necessity_rationale": "",
    "guidelines_cited": [],
    "evidence_reviewed": [],
    "approved_parameters": {{}},
    "alternatives_considered": [],
    "denial_reasons": []
  }}}},
  
  "prognosis_future": {{{{
    "short_term_prognosis": "",
    "long_term_prognosis": "",
    "recovery_timeline": "",
    "factors_affecting_prognosis": [],
    "future_care_needs": [
      {{{{
        "category": "",
        "description": "",
        "priority": "",
        "timeframe": "",
        "source_snippet": ""
      }}}}
    ],
    "follow_up_plan": "",
    "re_evaluation_criteria": []
  }}}},
  
  "medical_legal": {{{{
    "causation_analysis": {{{{
      "industrial_percentage": "",
      "non_industrial_factors": [],
      "substantial_factors": [],
      "reasoning": "",
      "source_snippet": ""
    }}}},
    "consistency_analysis": {{}},
    "records_reviewed": [],
    "specific_questions": []
  }}}},
  
  "critical_findings": [
    {{{{
      "finding": "",
      "urgency": "CRITICAL|HIGH|MODERATE|LOW",
      "action_required": "",
      "timeframe": "",
      "source_snippet": ""
    }}}}
  ],
  
  "metadata": {{{{
    "extraction_confidence": "HIGH|MEDIUM|LOW",
    "extraction_date": "",
    "missing_critical_fields": [],
    "ambiguities": [
      {{{{
        "field": "",
        "issue": "",
        "possible_interpretations": []
      }}}}
    ],
    "requires_human_review": false,
    "review_reasons": []
  }}}}
}}}}

Extract comprehensively and accurately. Physician decisions depend on this data.
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TEXT:
{document_text}

Perform comprehensive extraction following ALL guidelines.
Priority: Completeness and accuracy over speed.
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            context_length = self._get_adaptive_context_length(complexity_score)
            document_text = text[:context_length]
            
            chain = chat_prompt | self.llm | self.parser
            extracted = chain.invoke({
                "document_text": document_text,
                "doc_type": doc_type,
                "complexity_score": complexity_score,
                "context_guidance_text": context_guidance_text
            })
            
            # Ensure proper structure
            if not isinstance(extracted, dict):
                logger.error("Extraction returned non-dict, using fallback")
                return self._create_fallback_data(fallback_date, doc_type)
            
            # Ensure date fallback
            if "document_intelligence" not in extracted:
                extracted["document_intelligence"] = {}
            if not extracted["document_intelligence"].get("report_date"):
                extracted["document_intelligence"]["report_date"] = fallback_date
            
            # Ensure metadata
            if "metadata" not in extracted:
                extracted["metadata"] = {}
            
            logger.info(f"✅ Universal extraction complete - {len(str(extracted))} chars")
            return extracted
            
        except Exception as e:
            logger.error(f"❌ Universal extraction failed: {e}", exc_info=True)
            return self._create_fallback_data(fallback_date, doc_type)
    
    def _comprehensive_verification(
        self,
        original_text: str,
        extracted_json: Dict,
        doc_type: str
    ) -> Dict:
        """
        Comprehensive verification with surgical detection and contradiction checking.
        """
        logger.info("🔎 Running comprehensive verification...")
        
        system_v = SystemMessagePromptTemplate.from_template("""
You are a verification assistant for medical extraction quality assurance.

Your tasks:
1. Validate presence of REQUIRED fields for this document type
2. Detect contradictions between different sections
3. Verify ALL surgical procedures mentioned in text are captured
4. Check medication logic (no discontinued meds in "current")
5. Verify work restrictions match MMI status
6. Identify missing critical information

Return EXACT JSON:
{{{{
  "ok": true/false,
  "recommended_action": "AUTO_ACCEPT" | "FLAG_FOR_HUMAN_REVIEW",
  "missing_fields": [],
  "issues": [
    {{{{
      "type": "MISSING|CONTRADICTION|SURGICAL_MISMATCH|MEDICATION_ERROR|LOGIC_ERROR",
      "field": "",
      "explanation": "",
      "source_snippet": "",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW"
    }}}}
  ],
  "contradictions": [],
  "surgical_events_found": [],
  "medication_concerns": []
}}}}

Document type: {doc_type}

Required fields by type:
- ALL: report_date, author, patient_name
- QME/IME/AME: diagnoses, causation, mmi_status, work_restrictions
- RFA/UR: request_description, determination, medical_necessity
- PR2/PR4: current_status, work_status, treatment_plan
- IMAGING: findings, impression
- SURGERY: procedure, surgeon, outcome

Surgical keywords to scan for: arthroplasty, fusion, repair, replacement, 
reconstruction, laminectomy, discectomy, meniscectomy, rotator cuff, ACL, 
total knee, total hip, carpal tunnel, spinal fusion
""")
        
        user_v = HumanMessagePromptTemplate.from_template("""
ORIGINAL TEXT:
{original_text}

EXTRACTED JSON:
{extracted_json}

Perform comprehensive verification and return structured JSON.
""")
        
        chat_v = ChatPromptTemplate.from_messages([system_v, user_v])
        
        try:
            chain = chat_v | self.llm | self.parser
            verification = chain.invoke({
                "original_text": original_text[:15000],  # Limit for verification
                "extracted_json": json.dumps(extracted_json, indent=2),
                "doc_type": doc_type
            })
            
            if not isinstance(verification, dict):
                logger.warning("Verification returned non-dict")
                return self._create_fallback_verification()
            
            # Normalize
            verification.setdefault("ok", False)
            verification.setdefault("recommended_action", "FLAG_FOR_HUMAN_REVIEW")
            verification.setdefault("missing_fields", [])
            verification.setdefault("issues", [])
            verification.setdefault("contradictions", [])
            
            logger.info(f"Verification: {verification['recommended_action']}, "
                       f"{len(verification['issues'])} issues")
            
            return verification
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return self._create_fallback_verification()
    
    def _create_fallback_verification(self) -> Dict:
        """Fallback verification result."""
        return {
            "ok": False,
            "recommended_action": "FLAG_FOR_HUMAN_REVIEW",
            "missing_fields": [],
            "issues": [{
                "type": "VERIFICATION_ERROR",
                "explanation": "Verification system failed - human review required",
                "severity": "CRITICAL"
            }]
        }
    
    def _build_hierarchical_summary(
        self,
        raw_data: Dict,
        doc_type: str,
        verification: Dict
    ) -> str:
        """
        Build hierarchical, physician-oriented long summary.
        
        Structure:
        1. Executive Summary
        2. Critical Action Items
        3. Diagnoses & Severity
        4. Current Clinical Status
        5. Treatment History & Current Plan
        6. Work Status & Restrictions
        7. Prognosis & Future Needs
        8. Verification Flags
        """
        logger.info("🔨 Building hierarchical long summary...")
        
        sections = []
        
        # === HEADER ===
        doc_intel = raw_data.get("document_intelligence", {})
        patient = raw_data.get("patient_context", {})
        
        header = [
            f"📄 {doc_type.upper()} COMPREHENSIVE SUMMARY",
            "=" * 70,
            f"Report Date: {doc_intel.get('report_date', 'Not specified')}",
            f"Author: {doc_intel.get('author', {}).get('name', 'Not specified')}",
            f"Patient: {patient.get('name', 'Not specified')}",
            f"DOB: {patient.get('dob', 'Not specified')}",
            ""
        ]
        sections.append("\n".join(header))
        
        # === 1. EXECUTIVE SUMMARY ===
        exec_lines = ["1. EXECUTIVE SUMMARY", "-" * 70]
        
        # Primary diagnoses
        diagnoses = raw_data.get("clinical_findings", {}).get("diagnoses", [])
        primary_dx = [d for d in diagnoses if d.get("is_primary")]
        if primary_dx:
            exec_lines.append(f"Primary Diagnosis: {primary_dx[0].get('diagnosis', 'Not specified')}")
        
        # Work status
        work_status = raw_data.get("work_status", {})
        if work_status.get("current_status"):
            exec_lines.append(f"Work Status: {work_status.get('current_status')}")
        
        # MMI
        mmi = work_status.get("mmi_status", {})
        if mmi.get("is_mmi"):
            exec_lines.append(f"MMI Status: {mmi.get('is_mmi')}")
        
        sections.append("\n".join(exec_lines))
        
        # === 2. CRITICAL ACTION ITEMS ===
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            crit_lines = ["\n2. CRITICAL ACTION ITEMS", "-" * 70]
            for finding in critical_findings[:10]:  # Top 10
                urgency = finding.get("urgency", "UNKNOWN")
                text = finding.get("finding", "")
                action = finding.get("action_required", "")
                crit_lines.append(f"• [{urgency}] {text}")
                if action:
                    crit_lines.append(f"  Action: {action}")
            sections.append("\n".join(crit_lines))
        
        # === 3. DIAGNOSES & SEVERITY ===
        if diagnoses:
            dx_lines = ["\n3. DIAGNOSES & CLINICAL FINDINGS", "-" * 70]
            for dx in diagnoses:
                dx_text = dx.get("diagnosis", "")
                icd = dx.get("icd10", "")
                body_part = dx.get("body_part", "")
                severity = dx.get("severity", "")
                
                line = f"• {dx_text}"
                if icd:
                    line += f" [{icd}]"
                if body_part:
                    line += f" - {body_part}"
                if severity:
                    line += f" ({severity})"
                dx_lines.append(line)
                
                src = dx.get("source_snippet", "")
                if src:
                    dx_lines.append(f"  Source: \"{src[:150]}...\"")
            
            sections.append("\n".join(dx_lines))
        
        # === 4. CURRENT CLINICAL STATUS ===
        current_status = raw_data.get("clinical_findings", {}).get("current_status", {})
        if current_status:
            status_lines = ["\n4. CURRENT CLINICAL STATUS", "-" * 70]
            
            # Pain
            pain = current_status.get("pain", {})
            if pain.get("locations"):
                status_lines.append(f"Pain: {', '.join(pain.get('locations', []))}")
                if pain.get("severity_scales"):
                    status_lines.append(f"  Severity: {', '.join(map(str, pain.get('severity_scales', [])))}")
            
            # Functional limitations
            func_lim = current_status.get("functional_limitations", [])
            if func_lim:
                status_lines.append(f"Functional Limitations: {', '.join(func_lim[:5])}")
            
            # ROM
            rom = current_status.get("range_of_motion", [])
            if rom:
                status_lines.append(f"Range of Motion: {', '.join(map(str, rom[:5]))}")
            
            # Neurological
            neuro = current_status.get("neurological", {})
            if neuro:
                if neuro.get("sensory"):
                    status_lines.append(f"Sensory: {', '.join(neuro.get('sensory', []))}")
                if neuro.get("motor"):
                    status_lines.append(f"Motor: {', '.join(neuro.get('motor', []))}")
            
            sections.append("\n".join(status_lines))
        
        # === 5. DIAGNOSTIC FINDINGS ===
        diagnostics = raw_data.get("diagnostics", {})
        imaging = diagnostics.get("imaging", [])
        if imaging:
            img_lines = ["\n5. DIAGNOSTIC IMAGING & TESTS", "-" * 70]
            for img in imaging[:10]:  # Top 10
                img_type = img.get("type", "")
                date = img.get("date", "")
                body_part = img.get("body_part", "")
                impression = img.get("impression", "")
                
                img_lines.append(f"• {img_type} - {body_part} ({date})")
                if impression:
                    img_lines.append(f"  Impression: {impression[:200]}")
                
                findings = img.get("key_findings", [])
                if findings:
                    img_lines.append(f"  Findings: {', '.join(findings[:3])}")
            
            sections.append("\n".join(img_lines))
        
        # === 6. TREATMENT HISTORY & CURRENT PLAN ===
        treatment = raw_data.get("treatment", {})
        treat_lines = ["\n6. TREATMENT HISTORY & CURRENT PLAN", "-" * 70]
        
        # Current Medications
        current_meds = treatment.get("medications", {}).get("current", [])
        if current_meds:
            treat_lines.append("Current Medications:")
            for med in current_meds[:10]:
                name = med.get("name", "")
                dose = med.get("dose", "")
                freq = med.get("frequency", "")
                med_line = f"  • {name}"
                if dose:
                    med_line += f" {dose}"
                if freq:
                    med_line += f" {freq}"
                treat_lines.append(med_line)
        
        # Procedures completed
        procedures = treatment.get("procedures", {}).get("completed", [])
        if procedures:
            treat_lines.append("\nCompleted Procedures:")
            for proc in procedures[:5]:
                proc_name = proc.get("procedure", "")
                date = proc.get("date", "")
                outcome = proc.get("outcome", "")
                treat_lines.append(f"  • {proc_name} ({date})")
                if outcome:
                    treat_lines.append(f"    Outcome: {outcome}")
        
        # Procedures planned
        planned = treatment.get("procedures", {}).get("planned", [])
        if planned:
            treat_lines.append("\nPlanned Procedures:")
            for proc in planned[:5]:
                proc_name = proc.get("procedure", "")
                necessity = proc.get("necessity_rationale", "")
                treat_lines.append(f"  • {proc_name}")
                if necessity:
                    treat_lines.append(f"    Rationale: {necessity[:150]}")
        
        # Conservative care
        pt = treatment.get("conservative_care", {}).get("physical_therapy", {})
        if pt:
            treat_lines.append(f"\nPhysical Therapy: {pt.get('frequency', '')} - {pt.get('progress', '')}")
        
        if treat_lines:
            sections.append("\n".join(treat_lines))
        
        # === 7. WORK STATUS & RESTRICTIONS ===
        if work_status:
            work_lines = ["\n7. WORK STATUS & RESTRICTIONS", "-" * 70]
            
            status = work_status.get("current_status", "")
            if status:
                work_lines.append(f"Current Status: {status}")
            
            restrictions = work_status.get("restrictions", [])
            if restrictions:
                work_lines.append("\nRestrictions:")
                for restr in restrictions[:10]:
                    category = restr.get("category", "")
                    spec = restr.get("specification", "")
                    duration = restr.get("duration", "")
                    work_lines.append(f"  • {category}: {spec}")
                    if duration:
                        work_lines.append(f"    Duration: {duration}")
            
            # MMI details
            if mmi:
                work_lines.append(f"\nMMI Status: {mmi.get('is_mmi', '')}")
                if mmi.get("date_reached"):
                    work_lines.append(f"  Date Reached: {mmi.get('date_reached')}")
                if mmi.get("impairment_rating"):
                    work_lines.append(f"  Impairment Rating: {mmi.get('impairment_rating')}")
                if mmi.get("apportionment"):
                    work_lines.append(f"  Apportionment: {mmi.get('apportionment')}")
            
            # Disability
            temp_dis = work_status.get("temporary_disability", {})
            if temp_dis.get("status"):
                work_lines.append(f"\nTemporary Disability: {temp_dis.get('status')}")
            
            sections.append("\n".join(work_lines))
        
        # === 8. UTILIZATION REVIEW / AUTHORIZATION ===
        ur = raw_data.get("utilization_review", {})
        if ur and ur.get("determination"):
            ur_lines = ["\n8. UTILIZATION REVIEW / AUTHORIZATION", "-" * 70]
            
            ur_lines.append(f"Request: {ur.get('request_description', 'Not specified')}")
            ur_lines.append(f"Determination: {ur.get('determination', '')}")
            
            rationale = ur.get("medical_necessity_rationale", "")
            if rationale:
                ur_lines.append(f"\nRationale: {rationale[:300]}")
            
            guidelines = ur.get("guidelines_cited", [])
            if guidelines:
                ur_lines.append(f"\nGuidelines Cited: {', '.join(guidelines)}")
            
            approved = ur.get("approved_parameters", {})
            if approved:
                ur_lines.append(f"\nApproved Parameters: {json.dumps(approved, indent=2)}")
            
            sections.append("\n".join(ur_lines))
        
        # === 9. PROGNOSIS & FUTURE CARE ===
        prognosis = raw_data.get("prognosis_future", {})
        if prognosis:
            prog_lines = ["\n9. PROGNOSIS & FUTURE CARE NEEDS", "-" * 70]
            
            if prognosis.get("short_term_prognosis"):
                prog_lines.append(f"Short-term: {prognosis.get('short_term_prognosis')}")
            
            if prognosis.get("long_term_prognosis"):
                prog_lines.append(f"Long-term: {prognosis.get('long_term_prognosis')}")
            
            if prognosis.get("recovery_timeline"):
                prog_lines.append(f"Timeline: {prognosis.get('recovery_timeline')}")
            
            future_needs = prognosis.get("future_care_needs", [])
            if future_needs:
                prog_lines.append("\nFuture Care Needs (Prioritized):")
                for need in future_needs[:10]:
                    priority = need.get("priority", "")
                    desc = need.get("description", "")
                    timeframe = need.get("timeframe", "")
                    prog_lines.append(f"  • [{priority}] {desc}")
                    if timeframe:
                        prog_lines.append(f"    Timeframe: {timeframe}")
            
            follow_up = prognosis.get("follow_up_plan", "")
            if follow_up:
                prog_lines.append(f"\nFollow-up: {follow_up}")
            
            sections.append("\n".join(prog_lines))
        
        # === 10. MEDICAL-LEGAL OPINIONS ===
        med_legal = raw_data.get("medical_legal", {})
        causation = med_legal.get("causation_analysis", {})
        if causation and causation.get("industrial_percentage"):
            legal_lines = ["\n10. MEDICAL-LEGAL OPINIONS", "-" * 70]
            
            legal_lines.append(f"Industrial Causation: {causation.get('industrial_percentage', '')}")
            
            non_ind = causation.get("non_industrial_factors", [])
            if non_ind:
                legal_lines.append(f"Non-Industrial Factors: {', '.join(non_ind)}")
            
            reasoning = causation.get("reasoning", "")
            if reasoning:
                legal_lines.append(f"\nReasoning: {reasoning[:300]}")
            
            sections.append("\n".join(legal_lines))
        
        # === 11. VERIFICATION & QA ===
        if verification and verification.get("issues"):
            verif_lines = ["\n11. VERIFICATION & QUALITY ASSURANCE", "-" * 70]
            verif_lines.append(f"Recommended Action: {verification.get('recommended_action', 'UNKNOWN')}")
            
            issues = verification.get("issues", [])
            if issues:
                verif_lines.append(f"\nIssues Detected ({len(issues)}):")
                for issue in issues[:10]:
                    issue_type = issue.get("type", "")
                    severity = issue.get("severity", "")
                    explanation = issue.get("explanation", "")
                    verif_lines.append(f"  • [{severity}] {issue_type}: {explanation}")
                    
                    src = issue.get("source_snippet", "")
                    if src:
                        verif_lines.append(f"    Source: \"{src[:100]}...\"")
            
            missing = verification.get("missing_fields", [])
            if missing:
                verif_lines.append(f"\nMissing Fields: {', '.join(missing)}")
            
            sections.append("\n".join(verif_lines))
        
        # === FOOTER ===
        metadata = raw_data.get("metadata", {})
        footer = [
            "\n" + "=" * 70,
            f"Extraction Confidence: {metadata.get('extraction_confidence', 'UNKNOWN')}",
            f"Requires Human Review: {metadata.get('requires_human_review', False)}",
            "=" * 70
        ]
        sections.append("\n".join(footer))
        
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Hierarchical summary built ({len(long_summary)} chars, {len(long_summary.split())} words)")
        
        return long_summary
    
    def _generate_short_summary_from_long(self, long_summary: str, doc_type: str) -> str:
        """
        Generate concise pipe-delimited short summary (30-60 words).
        Format: [Type] | [Author] | [Date] | [Body Parts] | [Findings] | [Meds] | [Plan]
        """
        logger.info("🎯 Generating structured short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You create CONCISE pipe-delimited medical summaries.

OUTPUT FORMAT:
[Type] | [Author] | [Date] | [Body Parts] | [Key Findings] | [Medications] | [Plan]

RULES:
- 30-60 words total
- Extract ONLY from provided summary
- NO assumptions or additions
- Use abbreviations: L/R, Bilat, Dx, Rx, PT, f/u, MMI, WC
- Omit segments if not in summary
- Critical info only

EXAMPLES:
"QME Report | Dr. Smith, MD | 10/28/25 | L shoulder | Superior labrum lesion, tendinosis, severe AC arthropathy, adhesive capsulitis | None listed | F/u PRN, conservative care"

"UR Decision | Dr. Jones | 09/15/25 | Cervical spine | MRI approved for radiculopathy evaluation, consistent with ODG | Gabapentin 300mg TID | PT 2x/wk x6wks"

"Surgery Report | Dr. Lee | 08/20/24 | R knee | Arthroscopic meniscectomy completed, medial meniscus tear | Post-op pain meds | F/u 2wks, PT start wk3"

Now create from this summary:
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG SUMMARY:
{long_summary}

Create pipe-delimited short summary (30-60 words):
""")
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary[:5000]})  # Limit input
            short_summary = response.content.strip()
            
            # Clean formatting
            short_summary = re.sub(r'\s+', ' ', short_summary).strip()
            short_summary = re.sub(r'\s*\|\s*', ' | ', short_summary)
            
            # Remove quotes if present
            short_summary = short_summary.strip('"').strip("'")
            
            word_count = len(short_summary.split())
            logger.info(f"✅ Short summary: {word_count} words")
            
            # Fallback if too long
            if word_count > 80:
                words = short_summary.split()
                short_summary = ' '.join(words[:60]) + "..."
            
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return self._create_fallback_short_summary(long_summary, doc_type)
    
    def _create_fallback_short_summary(self, long_summary: str, doc_type: str) -> str:
        """Create fallback short summary by extracting key sentences."""
        # Try to extract first meaningful content
        lines = long_summary.split('\n')
        meaningful_lines = [l.strip() for l in lines if l.strip() and not l.startswith('=') and not l.startswith('-')]
        
        # Find lines with actual content
        content_lines = []
        for line in meaningful_lines[5:]:  # Skip header
            if any(keyword in line.lower() for keyword in ['diagnosis', 'finding', 'status', 'treatment', 'medication']):
                content_lines.append(line)
            if len(content_lines) >= 3:
                break
        
        if content_lines:
            summary = ' | '.join(content_lines)
            words = summary.split()
            if len(words) > 60:
                summary = ' '.join(words[:60])
            return summary
        
        return f"{doc_type} | Comprehensive evaluation completed | Manual review recommended"
    
    def _build_context_guidance_text(self, context_guidance: Dict) -> str:
        """Build formatted context guidance text for LLM."""
        if isinstance(context_guidance, str):
            return context_guidance
        
        if not context_guidance:
            return "No specific guidance - extract all available information comprehensively."
        
        lines = []
        
        primary = context_guidance.get("primary_physician", "")
        confidence = context_guidance.get("physician_confidence", "")
        reasoning = context_guidance.get("physician_reasoning", "")
        
        if primary:
            lines.append(f"PRIMARY PHYSICIAN: {primary}")
            if confidence:
                lines.append(f"  Confidence: {confidence}")
            if reasoning:
                short_reason = reasoning[:400] if len(reasoning) > 400 else reasoning
                lines.append(f"  Reasoning: {short_reason}")
        
        focus_sections = context_guidance.get("focus_sections", [])
        if focus_sections:
            lines.append(f"\nPRIORITY SECTIONS ({len(focus_sections)}):")
            for section in focus_sections[:15]:
                lines.append(f"  • {section}")
        
        critical_locations = context_guidance.get("critical_locations", {})
        if critical_locations:
            lines.append("\nCRITICAL DATA LOCATIONS:")
            for key, location in list(critical_locations.items())[:15]:
                lines.append(f"  • {key}: {location}")
        
        ambiguities = context_guidance.get("ambiguities", [])
        if ambiguities:
            lines.append(f"\n⚠️ KNOWN AMBIGUITIES ({len(ambiguities)}):")
            for amb in ambiguities[:5]:
                amb_type = amb.get("type", "unknown")
                amb_desc = amb.get("description", "")[:200]
                lines.append(f"  • {amb_type}: {amb_desc}")
        
        return "\n".join(lines) if lines else "No specific guidance - extract comprehensively."
    
    def _get_adaptive_context_length(self, complexity_score: int) -> int:
        """Get adaptive context length based on complexity."""
        base_length = 16000
        
        if complexity_score >= 8:
            return 24000
        elif complexity_score >= 6:
            return 20000
        else:
            return base_length
    
    def _get_context_length(self, doc_type: str) -> int:
        """Legacy method for backward compatibility."""
        complex_docs = ["QME", "AME", "IME", "SURGERY_REPORT", "DISCHARGE", "FCE"]
        return 20000 if doc_type in complex_docs else 16000
    
    def _clean_extracted_data(self, data: Dict, fallback_date: str) -> Dict:
        """Clean and validate extracted data."""
        if not isinstance(data, dict):
            return {"document_intelligence": {"report_date": fallback_date}}
        
        if "document_intelligence" not in data:
            data["document_intelligence"] = {}
        if not data["document_intelligence"].get("report_date"):
            data["document_intelligence"]["report_date"] = fallback_date
        
        return data
    
    def _detect_physician(self, text: str, page_zones: Optional[Dict]) -> str:
        """Detect physician using DoctorDetector."""
        try:
            result = self.doctor_detector.detect_doctor(text=text, page_zones=page_zones)
            return result.get("doctor_name", "").strip()
        except Exception as e:
            logger.warning(f"Physician detection failed: {e}")
            return ""
    
    def _create_fallback_data(self, fallback_date: str, doc_type: str) -> Dict:
        """Create fallback data structure when extraction fails."""
        return {
            "document_intelligence": {
                "report_date": fallback_date,
                "detected_type": doc_type,
                "author": {"name": "Not detected"}
            },
            "patient_context": {},
            "clinical_findings": {
                "diagnoses": []
            },
            "critical_findings": [{
                "finding": "Extraction incomplete - comprehensive manual review required",
                "urgency": "HIGH",
                "action_required": "Human analyst must review original document"
            }],
            "metadata": {
                "extraction_confidence": "LOW",
                "requires_human_review": True,
                "review_reasons": ["Extraction engine failed or incomplete data"]
            }
        }
    
    def _create_error_response(self, doc_type: str, error_msg: str, fallback_date: str) -> Dict:
        """Create error response."""
        return {
            "long_summary": f"❌ {doc_type} EXTRACTION FAILED\n\nError: {error_msg}\n\nACTION REQUIRED: Complete manual review of original document necessary.",
            "short_summary": f"{doc_type} | {fallback_date} | EXTRACTION FAILED - Manual review required",
            "raw_data": self._create_fallback_data(fallback_date, doc_type)
        }