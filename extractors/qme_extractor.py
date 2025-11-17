"""
QME/AME/IME Enhanced Extractor - Full Context with Context-Awareness
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class QMEExtractorChained:
    """
    Enhanced QME extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Uses DocumentContextAnalyzer guidance = Context-aware extraction
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Dual summary generation: Long detailed + Short concise
    - UPDATED: Returns ONLY long and short summaries; all other fields removed
    - IMPROVEMENTS: Enhanced long summary for better narrative flow and readability;
                  Short summary refined for precision, consistency, and medical-legal focus
    """

    def __init__(self, llm: AzureChatOpenAI, summary_llm: Optional[AzureChatOpenAI] = None):
        self.llm = llm
        self.summary_llm = summary_llm or llm  # Use main LLM if summary LLM not provided
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        
        logger.info("✅ QMEExtractorChained initialized (Full Context + Context-Aware + Dual Summary Only + Improved Summaries)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,  # NEW: from DocumentContextAnalyzer
        raw_text: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Extract QME data with FULL CONTEXT and contextual awareness.
        
        UPDATED: Returns ONLY {"summary": {"long": "...", "short": "..."}} - all other fields removed.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer (CRITICAL)
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING QME EXTRACTION (FULL CONTEXT + DUAL SUMMARY ONLY)")
        logger.info("=" * 80)
        
        # Log context guidance if available
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

        # Log medicine/medications extraction
        meds = None
        if 'current_medications' in raw_result:
            meds = raw_result['current_medications']
        elif 'medications' in raw_result:
            meds = raw_result['medications']
        if meds:
            logger.info(f"✅ Extracted medications: {meds}")
        else:
            logger.warning("⚠️ No 'current_medications' or 'medications' found in QME extraction result.")

        # Stage 2: Override physician if context identified one with high confidence
        if context_analysis:
            context_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            if context_physician.get("name") and context_physician.get("confidence") in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified physician: {context_physician.get('name')}")
                raw_result["qme_physician_name"] = context_physician.get("name")

        # Stage 3: Fallback to DoctorDetector if no physician identified
        if not raw_result.get("qme_physician_name"):
            logger.info("🔍 No physician from context/extraction, using DoctorDetector...")
            examiner_name = self._detect_examiner(text, page_zones)
            raw_result["qme_physician_name"] = examiner_name

        # Stage 4: Build DUAL SUMMARY ONLY - all other fields discarded
        dual_summary = self._build_dual_summary_only(raw_result, doc_type, fallback_date)

        logger.info("=" * 80)
        logger.info("✅ QME EXTRACTION COMPLETE (DUAL SUMMARY ONLY)")
        logger.info("=" * 80)

        return dual_summary

    def _build_dual_summary_only(self, raw_data: Dict, doc_type: str, fallback_date: str) -> Dict[str, Dict[str, str]]:
        """Build ONLY long and short summaries; discard all other fields"""
        logger.info("🔨 Building dual summary only...")
        
        # Build LONG summary from structured data (convert dict to string)
        long_summary = self._build_comprehensive_narrative_summary(raw_data, doc_type, fallback_date)
        
        # Generate SHORT summary using the structured data (NOT the long summary text)
        short_summary = self._generate_short_summary_from_structured_data(raw_data)
        
        # Return ONLY the dual summary structure
        dual_summary = {
            "summary": {
                "long": long_summary,
                "short": short_summary
            }
        }
        
        logger.info(f"✅ Dual summary generated:")
        logger.info(f"   Long: {len(long_summary)} chars")
        logger.info(f"   Short: {len(short_summary)} chars")
        
        return dual_summary

    def _generate_short_summary_from_structured_data(self, raw_data: Dict) -> str:
        """
        Generate a concise short summary using the structured data from extraction.
        This uses the actual extracted fields rather than the long summary text.
        IMPROVEMENTS: 
        - More precise prioritization: MMI/WPI first, then restrictions, recommendations, meds.
        - Consistent abbreviations and telegraphic style.
        - Ensure body parts are integrated if relevant (e.g., "lumbar").
        - Enforce stricter length (100-200 chars) with fallback validation.
        """
        logger.info("🎯 Generating short summary from structured data...")
        
        # Extract key elements directly from structured data
        key_elements = self._extract_key_elements_for_short_summary(raw_data)
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal specialist creating ULTRA-CONCISE summaries of QME/AME/IME reports.

CRITICAL REQUIREMENTS:
- MAXIMUM 1-2 sentences, 100-200 characters total (count and trim if needed)
- Focus ONLY on the MOST critical medical-legal findings in this order: MMI/WPI → Work restrictions → Key recommendations → Critical meds/body parts
- Use telegraphic style: Bullet-like facts, no fluff, active voice
- Abbreviations: MMI (Max Medical Improvement), P&S (Permanent & Stationary), WPI (Whole Person Impairment), PT (Physical Therapy), DOI (Date of Injury)
- Integrate body parts naturally (e.g., "Lumbar strain: Not at MMI")

FORMAT RULES:
- Start with MMI/WPI if available (e.g., "At MMI, 12% lumbar WPI")
- Then restrictions (e.g., "No lifting >20lbs, light duty")
- Then 1 key rec (e.g., "Rec: PT x6 sessions")
- End with critical meds if any (e.g., "; On gabapentin")
- Use semicolons (;) or periods (.) to separate facts

EXAMPLES:
✅ "Not at MMI (lumbar); WPI deferred. Restrictions: no lift >25lbs, no bend. Rec: PT core strength; Ibuprofen PRN."
✅ "MMI reached DOI-related shoulder; 8% WPI. Off work 4wks. Rec: MRI if persistent."
✅ "P&S pending surgery (knee); No restrictions. On oxycodone, gabapentin."

Now create an ULTRA-CONCISE summary from these structured medical-legal findings:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
MEDICAL-LEGAL FINDINGS FOR SHORT SUMMARY:

MMI STATUS: {mmi_status}
WPI: {wpi_status} ({body_parts})
WORK RESTRICTIONS: {work_restrictions}
KEY RECOMMENDATIONS: {key_recommendations}
CRITICAL MEDICATIONS: {critical_medications}

Create the most concise possible medical-legal summary (1-2 sentences max, 100-200 chars):
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            chain = chat_prompt | self.summary_llm
            response = chain.invoke({
                "mmi_status": key_elements["mmi_status"],
                "wpi_status": key_elements["wpi"],
                "body_parts": key_elements["body_parts"],
                "work_restrictions": key_elements["work_restrictions"],
                "key_recommendations": ", ".join(key_elements["key_recommendations"][:2]),  # Limit to top 2
                "critical_medications": ", ".join(key_elements["critical_medications"][:2])
            })
            
            short_summary = response.content.strip()
            end_time = time.time()
            
            # Validate and clean up the short summary
            short_summary = self._clean_short_summary(short_summary)
            
            # Additional improvement: Ensure char count and add fallback if too vague
            if len(short_summary) < 50 or "review" in short_summary.lower():
                short_summary = self._create_fallback_short_summary_from_data(key_elements)
            
            logger.info(f"⚡ Short summary generated in {end_time - start_time:.2f}s: '{short_summary}' (Len: {len(short_summary)})")
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            # Fallback: create basic short summary from structured data
            return self._create_fallback_short_summary_from_data(key_elements)

    def _extract_key_elements_for_short_summary(self, raw_data: Dict) -> Dict:
        """Extract the most critical elements for short summary generation from structured data"""
        key_elements = {
            "mmi_status": "",
            "work_restrictions": "",
            "key_recommendations": [],
            "critical_medications": [],
            "wpi": "",
            "body_parts": ""
        }
        
        # Extract MMI status
        medical_legal = raw_data.get("medical_legal_conclusions", {})
        if medical_legal:
            mmi_data = medical_legal.get("mmi_status", {})
            if isinstance(mmi_data, dict):
                status = mmi_data.get("status", "")
                reason = mmi_data.get("reason", "")
                if status:
                    key_elements["mmi_status"] = status
                if reason and "deferred" in reason.lower():
                    key_elements["mmi_status"] += f" ({reason.split(',')[0]})"  # Add brief reason
            else:
                key_elements["mmi_status"] = str(mmi_data) if mmi_data else ""
        
        # Extract WPI
        if medical_legal:
            wpi_data = medical_legal.get("wpi_impairment", {})
            if isinstance(wpi_data, dict):
                total = wpi_data.get("total_wpi", "")
                if total:
                    key_elements["wpi"] = f"{total}%"
                if not total and wpi_data.get("breakdown"):
                    breakdown = wpi_data.get("breakdown", [])
                    if breakdown and isinstance(breakdown, list) and len(breakdown) > 0:
                        first = breakdown[0]
                        if isinstance(first, dict):
                            key_elements["wpi"] = f"{first.get('percentage', '')}%"
        
        # Extract work restrictions (more concise joining)
        work_status = raw_data.get("work_status", {})
        if work_status:
            restrictions = work_status.get("work_restrictions", [])
            if restrictions and isinstance(restrictions, list):
                key_elements["work_restrictions"] = " | ".join([str(r).strip() for r in restrictions[:4]])  # Use | for tighter separation
        
        # Extract body parts from diagnosis (improved mapping)
        diagnosis = raw_data.get("diagnosis", {})
        if diagnosis:
            primary_diagnoses = diagnosis.get("primary_diagnoses", [])
            if primary_diagnoses and isinstance(primary_diagnoses, list):
                body_parts = set()  # Use set to avoid duplicates
                for dx in primary_diagnoses:
                    if isinstance(dx, dict):
                        dx_name = str(dx.get("name", "")).lower()
                        mappings = {
                            "lumbar": "lumbar", "l-spine": "lumbar", "low back": "lumbar",
                            "cervical": "cervical", "neck": "cervical",
                            "shoulder": "shoulder", "rotator cuff": "shoulder",
                            "knee": "knee", "acl": "knee",
                            "back": "back", "spine": "spine"
                        }
                        for key, part in mappings.items():
                            if key in dx_name:
                                body_parts.add(part)
                                break
                if body_parts:
                    key_elements["body_parts"] = ", ".join(list(body_parts))
        
        # Extract key recommendations (prioritize surgical/procedural first)
        recommendations = raw_data.get("recommendations", {})
        if recommendations:
            # Surgical/future needs (highest priority)
            future_surg = recommendations.get("future_surgical_needs", [])
            if future_surg:
                key_elements["key_recommendations"].extend([
                    f"{s.get('procedure', '')} ({s.get('body_part', '')})"
                    for s in future_surg[:1] if isinstance(s, dict) and s.get('procedure')
                ])
            
            # Procedures
            procedures = recommendations.get("interventional_procedures", [])
            if procedures:
                key_elements["key_recommendations"].extend([
                    f"{p.get('procedure', '')}"
                    for p in procedures[:1] if isinstance(p, dict) and p.get('procedure')
                ])
            
            # Therapy
            therapy = recommendations.get("therapy", [])
            if therapy:
                key_elements["key_recommendations"].extend([
                    f"{t.get('type', '')} x{t.get('frequency', '')}"
                    for t in therapy[:2] if isinstance(t, dict) and t.get('type')
                ])
            
            # Diagnostics
            diagnostics = recommendations.get("diagnostic_tests", [])
            if diagnostics:
                key_elements["key_recommendations"].extend([
                    f"{d.get('test', '')}"
                    for d in diagnostics[:1] if isinstance(d, dict) and d.get('test')
                ])
        
        # Extract critical medications (focus on high-impact)
        medications = raw_data.get("medications", {})
        if medications:
            current_meds = medications.get("current_medications", [])
            critical_keywords = ['oxycodone', 'hydrocodone', 'morphine', 'fentanyl', 'tramadol', 'opioid', 'gabapentin', 'pregabalin', 'lyrica', 'amitriptyline', 'duloxetine']
            
            for med in current_meds:
                if isinstance(med, dict):
                    med_name = str(med.get("name", "")).lower()
                    if any(kw in med_name for kw in critical_keywords):
                        key_elements["critical_medications"].append(str(med.get("name", "")))
        
        return key_elements

    def _clean_short_summary(self, summary: str) -> str:
        """Clean and validate the short summary"""
        # Remove excessive whitespace and quotes
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = summary.replace('"', '').replace("'", "")
        
        # Enforce tighter length (100-200 chars)
        if len(summary) > 200:
            # Split on periods/semicolons and take first parts
            parts = re.split(r'[.;]+', summary)
            summary = '. '.join(parts[:3]) + '.'
            summary = summary[:200]
        elif len(summary) < 100:
            summary += " Review full report."  # Pad if too short, but only if not already complete
        
        # Ensure it starts with MMI/WPI if possible (post-process check)
        if "mmi" not in summary.lower() and "wpi" not in summary.lower() and "p&s" not in summary.lower():
            summary = "MMI status: " + summary
        
        return summary

    def _create_fallback_short_summary_from_data(self, key_elements: Dict) -> str:
        """Create a fallback short summary directly from structured data"""
        logger.info("🔄 Using fallback short summary generation from structured data")
        
        parts = []
        
        # MMI/WPI first
        mmi = key_elements["mmi_status"] or key_elements["wpi"]
        if mmi:
            parts.append(str(mmi))
        
        # Restrictions
        if key_elements["work_restrictions"]:
            parts.append(key_elements["work_restrictions"][:100])  # Truncate if long
        
        # Top rec
        if key_elements["key_recommendations"]:
            parts.append(key_elements["key_recommendations"][0])
        
        # Meds
        if key_elements["critical_medications"]:
            parts.append("Meds: " + ", ".join(key_elements["critical_medications"][:2]))
        
        # Body parts if no other content
        if key_elements["body_parts"] and len(parts) < 2:
            parts.insert(0, key_elements["body_parts"])
        
        fallback_summary = ". ".join(parts)
        
        if not fallback_summary.strip():
            fallback_summary = "QME/AME/IME evaluation: Key findings in full report."
        
        return fallback_summary[:200]

    def _build_comprehensive_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive narrative summary from structured data.
        IMPROVEMENTS:
        - More narrative flow: Connect sections with transitions (e.g., "Based on the diagnosis...").
        - Cleaner formatting: Remove raw dict keys like "name:", integrate into sentences.
        - Prioritize medical-legal sections: Bold or emphasize MMI/WPI in text.
        - Limit verbosity: Cap lists at 3-4 items, use "et al." for more.
        - Add causal links: E.g., "Pain exacerbated by work duties leads to restrictions."
        """
        logger.info("📝 Building comprehensive narrative summary from structured data...")
        
        # Helper function to safely convert values to string (improved for narrative)
        def safe_str(value, default="", max_items=4):
            if not value:
                return default
            if isinstance(value, list):
                flat = []
                for item in value[:max_items]:
                    if isinstance(item, dict):
                        # Narrative integration: e.g., "Acute lumbar strain (ICD-10: S39.012A, resolved)"
                        name = item.get("name", "")
                        icd = item.get("icd_10", "")
                        status = item.get("status", "")
                        dose = item.get("dose", "")  # For meds
                        purpose = item.get("purpose", "")  # For meds/therapy
                        if name:
                            parts = [name]
                            if icd:
                                parts.append(f"ICD-10: {icd}")
                            if status:
                                parts.append(f"({status})")
                            if dose:
                                parts.append(f"dose: {dose}")
                            if purpose:
                                parts.append(f"for {purpose}")
                            flat.append(", ".join(parts))
                    elif item:
                        flat.append(str(item))
                return "; ".join(flat) if flat else default
            elif isinstance(value, dict):
                # For nested like ROM: "Flexion: 40° (limited by pain)"
                return ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in value.items() if v])
            return str(value)
        
        summary_parts = []
        
        # 1. PATIENT & CASE INFORMATION (Narrative intro)
        patient_info = data.get("patient_information", {})
        report_meta = data.get("report_metadata", {})
        physicians = data.get("physicians", {})
        
        intro = f"This {doc_type} report for patient {safe_str(patient_info.get('patient_name'))}, age {safe_str(patient_info.get('patient_age'))}, evaluates injuries from DOI {safe_str(patient_info.get('date_of_injury'))} (Claim: {safe_str(patient_info.get('claim_number'))}) while employed as {safe_str(patient_info.get('employer'))}. Evaluated by {safe_str(physicians.get('qme_physician', {}).get('name'))}, {safe_str(physicians.get('qme_physician', {}).get('specialty'))}, dated {safe_str(report_meta.get('report_date', fallback_date))}."
        summary_parts.append("**Patient & Case Overview:**\n" + intro)
        
        # 2. DIAGNOSIS (Narrative: "The primary diagnoses include...")
        diagnosis = data.get("diagnosis", {})
        dx_narrative = "The primary diagnoses include: " + safe_str(diagnosis.get("primary_diagnoses", []))
        if diagnosis.get("secondary_diagnoses"):
            dx_narrative += f". Secondary: {safe_str(diagnosis.get('secondary_diagnoses', []))}"
        summary_parts.append("**Diagnosis:**\n" + dx_narrative)
        
        # 3. CLINICAL STATUS (Connect to symptoms)
        clinical = data.get("clinical_status", {})
        clinical_narrative = f"Chief complaint: {safe_str(clinical.get('chief_complaint'))}. Current pain: {safe_str(clinical.get('pain_scores', {}).get('current'))}/10."
        if clinical.get("objective_findings"):
            obj = safe_str(clinical.get("objective_findings", {}))
            if obj:
                clinical_narrative += f" Objective: {obj}."
        summary_parts.append("**Clinical Status:**\n" + clinical_narrative)
        
        # 4. MEDICATIONS (List cleanly)
        medications = data.get("medications", {})
        meds_narrative = "Current medications: " + safe_str(medications.get("current_medications", []), max_items=5)
        summary_parts.append("**Medications:**\n" + meds_narrative)
        
        # 5. MEDICAL-LEGAL CONCLUSIONS (Emphasize with **)
        medical_legal = data.get("medical_legal_conclusions", {})
        legal_narrative = f"**MMI Status:** {safe_str(medical_legal.get('mmi_status', {}).get('status'))}; Reason: {safe_str(medical_legal.get('mmi_status', {}).get('reason'))}. **WPI:** {safe_str(medical_legal.get('wpi_impairment', {}).get('total_wpi'))} (breakdown: {safe_str(medical_legal.get('wpi_impairment', {}).get('breakdown', []))})."
        summary_parts.append("**Medical-Legal Conclusions:**\n" + legal_narrative)
        
        # 6. WORK STATUS & RESTRICTIONS (Link to clinical)
        work_status = data.get("work_status", {})
        work_narrative = f"Current work status: {safe_str(work_status.get('current_status'))}. Restrictions: {safe_str(work_status.get('work_restrictions', []))}. Prognosis for RTW: {safe_str(work_status.get('prognosis_for_return_to_work'))}."
        summary_parts.append("**Work Status:**\n" + work_narrative)
        
        # 7. RECOMMENDATIONS (Action-oriented)
        recommendations = data.get("recommendations", {})
        rec_narrative = "Recommendations include: " + safe_str(recommendations.get("therapy", [])) + "; " + safe_str(recommendations.get("interventional_procedures", [])) + "; Diagnostics: " + safe_str(recommendations.get("diagnostic_tests", [])) + "."
        summary_parts.append("**Recommendations:**\n" + rec_narrative)
        
        # Join with transitions
        full_summary = " ".join([part for part in summary_parts if part.strip()])
        
        # Truncate if too long, preserving legal sections
        if len(full_summary) > 3500:  # Tighter limit for readability
            # Prioritize: Keep legal, work, recs; trim others
            priority_parts = [p for p in summary_parts if any(kw in p.lower() for kw in ['medical-legal', 'work', 'recommendations'])]
            non_priority = [p for p in summary_parts if p not in priority_parts]
            full_summary = " ".join(non_priority[:2] + priority_parts)  # Intro + clinical + priorities
            full_summary = full_summary[:3500] + "... (Full details in report)"
        
        logger.info(f"📝 Comprehensive summary generated: {len(full_summary)} characters")
        return full_summary

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        This mimics Gemini's approach of processing the entire document at once.
        """
        logger.info("🔍 Processing ENTIRE document in single context window with guidance...")
        
        # Extract guidance from context analysis
        primary_physician = ""
        focus_sections = []
        critical_locations = {}
        physician_reasoning = ""
        ambiguities = []
        
        if context_analysis:
            phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            primary_physician = phys_analysis.get("name", "")
            physician_reasoning = phys_analysis.get("reasoning", "")
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE QME/AME/IME report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing you to:
- Understand the complete case narrative from start to finish
- Connect findings across all sections (history → examination → conclusions)
- Identify relationships between symptoms, diagnoses, and recommendations
- Provide comprehensive, context-aware extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate information
   - DO NOT fill in "typical" or "common" values
   - DO NOT use medical knowledge to "complete" incomplete information
   
   Examples:
   ✅ CORRECT: If document says "Patient takes Gabapentin 300mg TID", extract: {{"name": "Gabapentin", "dose": "300mg TID"}}
   ❌ WRONG: If document says "Patient takes Gabapentin", DO NOT extract: {{"name": "Gabapentin", "dose": "300mg TID"}} (dose not stated)
   ✅ CORRECT: Extract: {{"name": "Gabapentin", "dose": ""}} (dose field empty)

2. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY medications explicitly listed in the "Current Medications" or "Medications" section
   - Include dosage ONLY if explicitly stated
   - DO NOT extract:
     * Medications mentioned as discontinued
     * Medications mentioned in past medical history
     * Medications recommended for future use (put those in future_medications)
     * Medications you "think" the patient should be taking
   
   Examples:
   ✅ CORRECT: Document states "Current Medications: Gabapentin 300mg TID, Meloxicam 15mg daily"
   Extract: {{"current_medications": [{{"name": "Gabapentin", "dose": "300mg TID"}}, {{"name": "Meloxicam", "dose": "15mg daily"}}]}}
   
   ❌ WRONG: Document states "Patient previously took Oxycodone but discontinued 6 months ago"
   DO NOT extract Oxycodone in current_medications
   
   ❌ WRONG: Document states "Consider adding Amitriptyline for sleep"
   DO NOT extract Amitriptyline in current_medications (put in future_medications)

3. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
   Examples:
   ✅ CORRECT: If no pain score mentioned, return: "pain_score_current": ""
   ❌ WRONG: Return: "pain_score_current": "Not mentioned" (use empty string instead)

4. **EXACT QUOTES FOR CRITICAL FIELDS**
   - For MMI status, WPI, Work Restrictions: use EXACT wording from document
   - DO NOT paraphrase or interpret
   - If exact value not found, return empty
   
   Examples:
   ✅ CORRECT: Document says "Patient has reached MMI as of 10/15/2024"
   Extract: "mmi_status": {{"status": "Patient has reached MMI as of 10/15/2024"}}
   
   ❌ WRONG: Document says "Patient improving with treatment"
   DO NOT extract: "mmi_status": {{"status": "Not at MMI"}} (this is inference, not stated)

5. **NO CLINICAL ASSUMPTIONS**
   - DO NOT assume typical dosages, frequencies, or durations
   - DO NOT assume standard procedures or treatments
   - DO NOT assume body parts if not explicitly stated
   
   Examples:
   ❌ WRONG: Document mentions "knee injection"
   DO NOT assume: "corticosteroid injection" (steroid type not stated)
   ✅ CORRECT: Extract: "knee injection" (exact wording)

6. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
   Before returning your extraction, verify:
   □ Every medication has a direct quote in the document
   □ Every diagnosis is explicitly stated (not inferred from symptoms)
   □ Every recommendation is directly from "Recommendations" or "Plan" section
   □ No fields are filled with "typical" or "standard" values
   □ Empty fields are truly empty (not "Not mentioned" or "Unknown")

EXTRACTION FOCUS - 6 CRITICAL MEDICAL-LEGAL CATEGORIES:

I. CORE IDENTITY
- Patient name, age, DOB
- Date of Injury (DOI) - often in history section
- Report date - check header and conclusion
- QME Physician: **USE THE PRIMARY PHYSICIAN IDENTIFIED IN CONTEXT GUIDANCE ABOVE**
  * This is the REPORT AUTHOR, not treating physicians mentioned in history
  * Reasoning: {physician_reasoning}

II. DIAGNOSIS
- Primary diagnosis(es) - synthesize from examination findings AND conclusion
- ICD-10 codes if mentioned anywhere in document
- Affected body part(s) - consolidate all mentions throughout document

III. CLINICAL STATUS
- Past surgeries - scan entire history section for surgical history
- Current chief complaint - patient's own words from subjective section
- Pain score (current/max on 0-10 scale) - look in subjective complaints
- Objective findings:
  * ROM limitations - from physical examination section
  * Gait abnormalities - from observation/ambulation section
  * Positive tests - from clinical tests section (e.g., Hawkins, Neer, McMurray)
  * Effusion/swelling - from inspection/palpation findings

IV. MEDICATIONS ⚠️ CRITICAL - ZERO ASSUMPTIONS
- Current medications - from medication list or current medications section
- **ONLY extract medications EXPLICITLY listed as "current" or "taking"**
- **DO NOT extract discontinued, past, or recommended future medications**
- Categorize into: narcotics/opioids, nerve pain meds, anti-inflammatories, other
- Include dosages ONLY if explicitly stated (e.g., "Gabapentin 300mg TID")
- If dosage not stated, leave dose field empty
- Focus on CURRENT medications, not historical discontinued meds

Example extraction:
Document states: "Current Medications: 1. Gabapentin 300mg three times daily, 2. Meloxicam 15mg once daily, 3. Tramadol 50mg as needed for pain. Patient discontinued Oxycodone 3 months ago."

✅ CORRECT extraction:
{{
  "medications": {{
    "current_medications": [
      {{"name": "Gabapentin", "dose": "300mg TID", "purpose": "nerve pain"}},
      {{"name": "Meloxicam", "dose": "15mg daily", "purpose": "anti-inflammatory"}},
      {{"name": "Tramadol", "dose": "50mg PRN", "purpose": "pain"}}
    ]
  }}
}}

❌ WRONG - DO NOT include:
- Oxycodone (discontinued)
- Any medications not explicitly listed

V. MEDICAL-LEGAL CONCLUSIONS (MOST CRITICAL - HIGHEST PRIORITY)
**FOCUS ON THESE SECTIONS:** {focus_sections}
**CRITICAL LOCATIONS:** {critical_locations}

- MMI/P&S Status:
  * Look for explicit statement (e.g., "Patient has reached MMI as of [date]")
  * If MMI deferred, extract SPECIFIC REASON (e.g., "pending MRI results", "awaiting surgical outcome")
  * Location hint: {mmi_location}

- WPI (Whole Person Impairment):
  * Look for percentage WITH body part (e.g., "15% WPI to left shoulder")
  * Include method used (e.g., "per AMA Guides 5th Edition")
  * If WPI deferred, extract SPECIFIC REASON
  * Location hint: {wpi_location}


VI. ACTIONABLE RECOMMENDATIONS (SECOND HIGHEST PRIORITY)
**These are critical for immediate clinical action:**

- Future treatment: Be SPECIFIC
  * Surgeries: Include procedure name and body part (e.g., "total knee arthroplasty")
  * Injections: Include type and location (e.g., "ESI C5-6", "corticosteroid injection R shoulder")
  * Therapy: Include type and frequency (e.g., "PT 2x/week for 6 weeks")
  * Diagnostics: Include test type and body part (e.g., "MRI L-spine without contrast")

- Work restrictions: Extract EXACT functional limitations
  * Be specific: "no lifting >10 lbs" not "modified duty"
  * Include positional restrictions: "no overhead reaching", "no kneeling/squatting"
  * Include duration if stated: "restrictions for 8 weeks"
  * Location hint: {work_restrictions_location}

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- MEDICATIONS: Only extract what is explicitly listed as current
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE QME report and extract ALL relevant information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE QME/AME/IME DOCUMENT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical details:

{{
  "patient_information": {{
    "patient_name": "",
    "patient_age": "",
    "patient_dob": "",
    "date_of_injury": "",
    "claim_number": "",
    "employer": ""
  }},
  
  "report_metadata": {{
    "report_title": "",
    "report_date": "",
    "evaluation_date": "",
    "report_type": "QME/AME/IME"
  }},
  
  "physicians": {{
    "qme_physician": {{
      "name": "{primary_physician}",
      "specialty": "",
      "credentials": "",
      "role": "Evaluating Physician/QME/AME"
    }},
    "treating_physicians": [],
    "consulting_physicians": [],
    "referring_source": {{
      "name": "",
      "type": ""
    }}
  }},
  
  "diagnosis": {{
    "primary_diagnoses": [],
    "secondary_diagnoses": [],
    "historical_conditions": []
  }},
  
  "clinical_status": {{
    "chief_complaint": "",
    "pain_scores": {{
      "current": "",
      "maximum": "",
      "location": ""
    }},
    "functional_limitations": [],
    "past_surgeries": [],
    "objective_findings": {{
      "rom_limitations": "",
      "gait": "",
      "positive_tests": "",
      "other_findings": ""
    }}
  }},
  
  "medications": {{
    "current_medications": [],
    "future_medications": []
  }},
  
  "treatment_history": {{
    "past_treatments": [],
    "current_treatments": []
  }},
  
  "medical_legal_conclusions": {{
    "mmi_status": {{
      "status": "",
      "reason": "",
      "reasoning": ""
    }},
    "wpi_impairment": {{
      "total_wpi": "",
      "breakdown": [],
      "reasoning": ""
    }},
  }},
  
  "work_status": {{
    "current_status": "",
    "work_restrictions": [],
    "prognosis_for_return_to_work": ""
  }},
  
  "recommendations": {{
    "diagnostic_tests": [],
    "interventional_procedures": [],
    "specialist_referrals": [],
    "therapy": [],
    "future_surgical_needs": []
  }},
  
  "critical_findings": []
}}

⚠️ CRITICAL REMINDERS:
1. For "work_restrictions": Extract EXACT wording from document
   - If document says "no lifting", extract: "no lifting" (NOT "no lifting >10 lbs")
   - If document says "no standing", extract: "no standing" (NOT "no prolonged standing >15 min")
   - DO NOT add weight limits, time limits, or specifics not stated

2. For "current_medications": Extract ONLY from "Current Medications" section
   - Include dosage ONLY if explicitly stated
   - DO NOT extract discontinued medications
   - DO NOT extract recommended future medications (use future_medications for those)

3. For "critical_findings": Include MAIN actionable points only (max 5-8 items)
   - Focus on: MMI status, required procedures, required QMEs, important diagnostic tests
   - DO NOT include minor details or routine follow-ups
""")

        # Build context guidance summary
        context_guidance_text = f"""
PRIMARY PHYSICIAN (Report Author): {primary_physician or 'Not identified in context'}
REASONING: {physician_reasoning or 'See document for identification'}

FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- MMI Status: {critical_locations.get('mmi_location', 'Search entire document')}
- WPI Percentage: {critical_locations.get('wpi_location', 'Search entire document')}
- Work Restrictions: {critical_locations.get('work_restrictions_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text,
                "context_guidance": context_guidance_text,
                "primary_physician": primary_physician or "Extract from document",
                "physician_reasoning": physician_reasoning or "Use credentials and signature section",
                "focus_sections": ', '.join(focus_sections) if focus_sections else "All sections",
                "critical_locations": str(critical_locations),
                "mmi_location": critical_locations.get('mmi_location', 'Search document'),
                "wpi_location": critical_locations.get('wpi_location', 'Search document'),
                "work_restrictions_location": critical_locations.get('work_restrictions_location', 'Search document'),
                "ambiguities": str(ambiguities)
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(fallback_date)

    def _detect_examiner(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Fallback: Detect QME/AME examiner using DoctorDetector"""
        logger.info("🔍 Fallback: Running DoctorDetector for QME physician...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ QME Physician detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid QME physician found")
            return ""

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure"""
        return {
            "patient_information": {
                "patient_name": "",
                "patient_age": "",
                "patient_dob": "",
                "date_of_injury": "",
                "claim_number": "",
                "employer": ""
            },
            "report_metadata": {
                "report_date": fallback_date,
                "report_title": "",
                "evaluation_date": ""
            },
            "physicians": {
                "qme_physician": {
                    "name": "",
                    "specialty": ""
                }
            },
            "diagnosis": {
                "primary_diagnoses": []
            },
            "clinical_status": {
                "chief_complaint": "",
                "pain_scores": {
                    "current": ""
                }
            },
            "medications": {
                "current_medications": []
            },
            "medical_legal_conclusions": {
                "mmi_status": {
                    "status": ""
                },
                "wpi_impairment": {
                    "total_wpi": ""
                }
            },
            "work_status": {
                "current_status": "",
                "work_restrictions": []
            },
            "recommendations": {
                "therapy": []
            }
        }   