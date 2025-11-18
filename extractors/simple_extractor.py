"""
Enhanced Simple Extractor with FULL Context-Awareness (like QME Extractor)
Handles all document types with context-guided extraction
"""

import re
import logging
import time
import json
from typing import Dict, Optional, List, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.doctor_detector import DoctorDetector
from extractors.prompt_manager import PromptManager

logger = logging.getLogger("document_ai")

class SimpleExtractor:
    """
    Enhanced extractor with FULL CONTEXT-AWARENESS for all document types.
    
    Features:
    - Context-guided extraction (like QME extractor)
    - Type-specific prompts via PromptManager
    - Anti-hallucination rules
    - Critical findings focus
    - Comprehensive extraction without missing details
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.doctor_detector = DoctorDetector(llm)
        self.prompt_manager = PromptManager()
        logger.info("✅ SimpleExtractor initialized with CONTEXT-AWARENESS")
    
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
        Context-aware extraction for ALL document types.
        
        Args:
            text: Full document text
            doc_type: Document type
            fallback_date: Fallback date
            context_analysis: CRITICAL - Context from DocumentContextAnalyzer
            page_zones: Page-based text zones
            raw_text: Original flat text
        
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info(f"🚀 CONTEXT-AWARE EXTRACTION: {doc_type}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # STEP 1: Extract and log context guidance
            context_guidance = self._extract_context_guidance(context_analysis)
            
            # STEP 2: Validate inputs
            if not text or not text.strip():
                raise ValueError("Empty document text provided")
            
            # STEP 3: Extract with FULL CONTEXT-AWARENESS
            raw_data = self._extract_with_context_guidance(
                text=text,
                doc_type=doc_type,
                fallback_date=fallback_date,
                context_guidance=context_guidance
            )
            
            # STEP 4: Override physician if context detected one with high confidence
            if context_guidance["primary_physician"] and context_guidance["physician_confidence"] in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified physician: {context_guidance['primary_physician']}")
                raw_data["physician_name"] = context_guidance["primary_physician"]
            else:
                # Fallback to DoctorDetector
                physician_name = self._detect_physician(text, page_zones)
                if physician_name:
                    raw_data["physician_name"] = physician_name
            
            # STEP 5: Build comprehensive long summary
            long_summary = self._build_comprehensive_long_summary(
                raw_data=raw_data,
                doc_type=doc_type,
                fallback_date=fallback_date,
                context_guidance=context_guidance
            )
            
            # STEP 6: Generate short summary
            short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Context-aware extraction completed in {elapsed_time:.2f}s")
            logger.info(f"📊 Results: {len(long_summary)} chars long, {len(short_summary.split())} words short")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_type}: {str(e)}")
            return self._create_error_response(doc_type, str(e), fallback_date)
    
    def _extract_context_guidance(self, context_analysis: Optional[Dict]) -> Dict:
        """
        Extract and structure context guidance (like QME extractor).
        
        Returns structured guidance dict with:
        - primary_physician
        - physician_confidence
        - physician_reasoning
        - focus_sections
        - critical_locations
        - ambiguities
        """
        if not context_analysis:
            logger.warning("⚠️ No context analysis provided - proceeding without guidance")
            return {
                "primary_physician": "",
                "physician_confidence": "",
                "physician_reasoning": "",
                "focus_sections": [],
                "critical_locations": {},
                "ambiguities": []
            }
        
        # Extract physician analysis
        phys_analysis = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
        primary_physician = phys_analysis.get("name", "")
        physician_confidence = phys_analysis.get("confidence", "")
        physician_reasoning = phys_analysis.get("reasoning", "")
        
        # Extract extraction guidance
        focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
        
        # Extract critical findings map
        critical_locations = context_analysis.get("critical_findings_map", {})
        
        # Extract ambiguities
        ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Log context guidance
        logger.info(f"🎯 Context Guidance Received:")
        logger.info(f"   Primary Physician: {primary_physician or 'Unknown'}")
        logger.info(f"   Confidence: {physician_confidence or 'Unknown'}")
        logger.info(f"   Focus Sections: {focus_sections}")
        logger.info(f"   Critical Locations: {list(critical_locations.keys())}")
        logger.info(f"   Ambiguities: {len(ambiguities)} detected")
        
        return {
            "primary_physician": primary_physician,
            "physician_confidence": physician_confidence,
            "physician_reasoning": physician_reasoning,
            "focus_sections": focus_sections,
            "critical_locations": critical_locations,
            "ambiguities": ambiguities
        }
    
    def _extract_with_context_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_guidance: Dict
    ) -> Dict:
        """
        Extract with FULL CONTEXT-AWARENESS (like QME extractor).
        
        Uses context guidance to:
        - Direct LLM to critical sections
        - Provide location hints for key data
        - Alert LLM to ambiguities
        - Prevent hallucinations
        """
        logger.info("🔍 Extracting with CONTEXT GUIDANCE...")
        
        # Build context guidance text for LLM
        context_guidance_text = self._build_context_guidance_text(context_guidance)
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical document specialist with CONTEXTUAL GUIDANCE for precise extraction.

DOCUMENT TYPE: {doc_type}

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. EXTRACT ONLY EXPLICITLY STATED INFORMATION
   - If a field/value is NOT in the document, return EMPTY string/list
   - DO NOT infer, assume, or extrapolate
   - DO NOT fill in typical or common values
   - Empty fields are ACCEPTABLE and PREFERRED over guessing

2. MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS
   - Extract ONLY medications explicitly listed as "current" or "taking"
   - Include dosage ONLY if explicitly stated
   - DO NOT extract: discontinued meds, past meds, future recommendations
   - If dosage not stated, leave dose field empty

3. DIAGNOSIS & FINDINGS - EXACT EXTRACTION
   - Extract diagnoses exactly as stated
   - DO NOT infer diagnoses from symptoms
   - DO NOT add medical interpretation
   - List all body parts explicitly mentioned

4. CRITICAL FINDINGS FOCUS
   {focus_sections_text}
   
   CRITICAL DATA LOCATIONS:
   {critical_locations_text}

5. KNOWN AMBIGUITIES:
   {ambiguities_text}
   - Be extra careful in these areas
   - Use exact quotes when ambiguous

EXTRACTION CATEGORIES (Extract ALL that apply):

I. CORE IDENTITY
   - Document date, report date
   - Patient name, age, DOI (if applicable)
   - Author/physician (use context guidance above)

II. DIAGNOSIS & CLINICAL FINDINGS
   - Primary diagnoses (exact wording)
   - Secondary/comorbid conditions
   - Affected body parts (explicit only)
   - Symptoms with severity (if stated)

III. CLINICAL ASSESSMENT
   - Vital signs (if present)
   - Physical exam findings (objective only)
   - Test results (lab, imaging, etc.)
   - Pain scores (if documented)

IV. MEDICATIONS
   - Current medications (with dosages if stated)
   - Future medication recommendations
   - Medication changes/adjustments

V. TREATMENTS & PROCEDURES
   - Past treatments/procedures
   - Current treatment plan
   - Future treatment recommendations

VI. ASSESSMENTS & DECISIONS (Critical for decision documents)
   - Medical necessity determination
   - Authorization status (approved/denied)
   - Appeal decisions
   - Utilization review findings

VII. WORK STATUS (If applicable)
   - Work restrictions (exact wording)
   - RTW status
   - Disability ratings (if mentioned)
   - MMI/P&S status (if mentioned)

VIII. RECOMMENDATIONS & PLAN
   - Diagnostic tests recommended
   - Specialist referrals
   - Therapy recommendations
   - Follow-up plans

IX. CRITICAL FINDINGS & ACTION ITEMS
   - Urgent/important findings
   - Required follow-ups
   - Time-sensitive actions

STRICT RULES:
- NEVER add information not in the document
- NEVER assume typical values
- If information is not present, return EMPTY
- Use exact quotes for critical data
- Verify every extraction against the document

Now extract from this {doc_type} document:
""")
        
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE DOCUMENT TEXT:

{document_text}

Extract into COMPREHENSIVE structured JSON (following all anti-hallucination rules):

{{
    "document_metadata": {{
        "document_date": "",
        "report_date": "",
        "document_type": "{doc_type}",
        "author_physician": ""
    }},
    "patient_information": {{
        "patient_name": "",
        "patient_age": "",
        "date_of_injury": ""
    }},
    "diagnosis": {{
        "primary_diagnoses": [],
        "secondary_diagnoses": [],
        "affected_body_parts": []
    }},
    "clinical_findings": {{
        "chief_complaint": "",
        "vital_signs": {{}},
        "physical_exam": {{}},
        "test_results": {{}},
        "pain_assessment": {{}}
    }},
    "medications": {{
        "current_medications": [],
        "future_medications": []
    }},
    "treatments": {{
        "past_treatments": [],
        "current_treatment_plan": [],
        "recommended_treatments": []
    }},
    "assessments_decisions": {{
        "medical_necessity": "",
        "authorization_status": "",
        "decision_rationale": "",
        "appeal_status": ""
    }},
    "work_status": {{
        "work_restrictions": [],
        "rtw_status": "",
        "disability_rating": "",
        "mmi_status": ""
    }},
    "recommendations": {{
        "diagnostic_tests": [],
        "specialist_referrals": [],
        "therapy_recommendations": [],
        "follow_up_plan": ""
    }},
    "critical_findings": []
}}

REMEMBER: Empty fields are better than guessed information!
""")
        
        # Build focus sections text
        focus_sections_text = "Focus on these sections:\n" + "\n".join([
            f"   - {section}" for section in context_guidance["focus_sections"]
        ]) if context_guidance["focus_sections"] else "   - All sections equally important"
        
        # Build critical locations text
        critical_locations = context_guidance["critical_locations"]
        critical_locations_text = "\n".join([
            f"   - {key}: {value}" for key, value in critical_locations.items()
        ]) if critical_locations else "   - Search entire document"
        
        # Build ambiguities text
        ambiguities = context_guidance["ambiguities"]
        ambiguities_text = f"{len(ambiguities)} detected:\n" + "\n".join([
            f"   - {amb.get('type')}: {amb.get('description')}" for amb in ambiguities
        ]) if ambiguities else "None detected"
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            # Use appropriate context length
            context_length = self._get_context_length(doc_type)
            context_text = text[:context_length]
            
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "document_text": context_text,
                "doc_type": doc_type,
                "context_guidance": context_guidance_text,
                "focus_sections_text": focus_sections_text,
                "critical_locations_text": critical_locations_text,
                "ambiguities_text": ambiguities_text
            })
            
            # Clean and validate
            result = self._clean_extracted_data(result, fallback_date)
            
            logger.info(f"✅ Context-aware extraction complete - {len(result)} fields extracted")
            return result
            
        except Exception as e:
            logger.error(f"❌ Context-aware extraction failed: {e}")
            return self._create_fallback_data(fallback_date, doc_type)
    
    def _build_context_guidance_text(self, context_guidance: Dict) -> str:
        """Build formatted context guidance text for LLM"""
        lines = []
        
        if context_guidance["primary_physician"]:
            lines.append(f"PRIMARY PHYSICIAN (Report Author): {context_guidance['primary_physician']}")
            lines.append(f"  Confidence: {context_guidance['physician_confidence']}")
            lines.append(f"  Reasoning: {context_guidance['physician_reasoning']}")
        
        if context_guidance["focus_sections"]:
            lines.append(f"\nFOCUS ON THESE SECTIONS: {', '.join(context_guidance['focus_sections'])}")
        
        if context_guidance["critical_locations"]:
            lines.append("\nCRITICAL DATA LOCATIONS:")
            for key, location in context_guidance["critical_locations"].items():
                lines.append(f"  - {key}: {location}")
        
        if context_guidance["ambiguities"]:
            lines.append(f"\n⚠️ KNOWN AMBIGUITIES ({len(context_guidance['ambiguities'])} detected):")
            for amb in context_guidance["ambiguities"][:3]:  # Limit to 3
                lines.append(f"  - {amb.get('type')}: {amb.get('description')}")
        
        return "\n".join(lines) if lines else "No specific guidance provided - extract all available information."
    
    def _build_comprehensive_long_summary(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str,
        context_guidance: Dict
    ) -> str:
        """
        Build comprehensive long summary from extracted data.
        Includes ALL critical information without losing details.
        """
        logger.info("🔨 Building comprehensive long summary...")
        
        sections = []
        
        # SECTION 1: DOCUMENT HEADER
        sections.append(f"📄 {doc_type} COMPREHENSIVE REPORT")
        sections.append("=" * 60)
        
        doc_metadata = raw_data.get("document_metadata", {})
        patient_info = raw_data.get("patient_information", {})
        
        header_lines = [
            f"Document Type: {doc_metadata.get('document_type', doc_type)}",
            f"Report Date: {doc_metadata.get('report_date', fallback_date)}",
            f"Author/Physician: {raw_data.get('physician_name', doc_metadata.get('author_physician', 'Not specified'))}"
        ]
        
        if patient_info.get("patient_name"):
            header_lines.append(f"Patient: {patient_info['patient_name']}")
        if patient_info.get("date_of_injury"):
            header_lines.append(f"Date of Injury: {patient_info['date_of_injury']}")
        
        sections.append("\n".join(header_lines))
        
        # SECTION 2: DIAGNOSIS & CLINICAL FINDINGS
        diagnosis = raw_data.get("diagnosis", {})
        if diagnosis.get("primary_diagnoses") or diagnosis.get("affected_body_parts"):
            sections.append("\nDIAGNOSIS")
            sections.append("-" * 40)
            
            if diagnosis.get("primary_diagnoses"):
                sections.append("Primary Diagnoses:")
                for dx in diagnosis["primary_diagnoses"][:10]:
                    sections.append(f"  • {dx}")
            
            if diagnosis.get("secondary_diagnoses"):
                sections.append("Secondary/Comorbid Conditions:")
                for dx in diagnosis["secondary_diagnoses"][:5]:
                    sections.append(f"  • {dx}")
            
            if diagnosis.get("affected_body_parts"):
                sections.append(f"Affected Body Parts: {', '.join(diagnosis['affected_body_parts'])}")
        
        # SECTION 3: CLINICAL ASSESSMENT
        clinical = raw_data.get("clinical_findings", {})
        if any(clinical.values()):
            sections.append("\nCLINICAL ASSESSMENT")
            sections.append("-" * 40)
            
            if clinical.get("chief_complaint"):
                sections.append(f"Chief Complaint: {clinical['chief_complaint']}")
            
            if clinical.get("vital_signs"):
                sections.append("Vital Signs:")
                for key, value in clinical["vital_signs"].items():
                    sections.append(f"  • {key}: {value}")
            
            if clinical.get("physical_exam"):
                sections.append("Physical Examination:")
                for key, value in clinical["physical_exam"].items():
                    sections.append(f"  • {key}: {value}")
            
            if clinical.get("pain_assessment"):
                sections.append(f"Pain Assessment: {clinical['pain_assessment']}")
        
        # SECTION 4: MEDICATIONS
        medications = raw_data.get("medications", {})
        if medications.get("current_medications") or medications.get("future_medications"):
            sections.append("\nMEDICATIONS")
            sections.append("-" * 40)
            
            if medications.get("current_medications"):
                sections.append("Current Medications:")
                for med in medications["current_medications"][:15]:
                    if isinstance(med, dict):
                        med_str = f"{med.get('name', '')}"
                        if med.get('dose'):
                            med_str += f" - {med['dose']}"
                        if med.get('purpose'):
                            med_str += f" ({med['purpose']})"
                        sections.append(f"  • {med_str}")
                    else:
                        sections.append(f"  • {med}")
            
            if medications.get("future_medications"):
                sections.append("Future Medication Recommendations:")
                for med in medications["future_medications"][:5]:
                    sections.append(f"  • {med}")
        
        # SECTION 5: ASSESSMENTS & DECISIONS (Critical for decision documents)
        assessments = raw_data.get("assessments_decisions", {})
        if any(assessments.values()):
            sections.append("\nASSESSMENTS & DECISIONS")
            sections.append("-" * 40)
            
            if assessments.get("authorization_status"):
                sections.append(f"Authorization Status: {assessments['authorization_status']}")
            
            if assessments.get("medical_necessity"):
                sections.append(f"Medical Necessity: {assessments['medical_necessity']}")
            
            if assessments.get("decision_rationale"):
                sections.append(f"Rationale: {assessments['decision_rationale']}")
            
            if assessments.get("appeal_status"):
                sections.append(f"Appeal Status: {assessments['appeal_status']}")
        
        # SECTION 6: WORK STATUS
        work_status = raw_data.get("work_status", {})
        if any(work_status.values()):
            sections.append("\nWORK STATUS & RESTRICTIONS")
            sections.append("-" * 40)
            
            if work_status.get("work_restrictions"):
                sections.append("Work Restrictions:")
                for restriction in work_status["work_restrictions"][:10]:
                    sections.append(f"  • {restriction}")
            
            if work_status.get("rtw_status"):
                sections.append(f"Return to Work Status: {work_status['rtw_status']}")
            
            if work_status.get("mmi_status"):
                sections.append(f"MMI Status: {work_status['mmi_status']}")
            
            if work_status.get("disability_rating"):
                sections.append(f"Disability Rating: {work_status['disability_rating']}")
        
        # SECTION 7: RECOMMENDATIONS
        recommendations = raw_data.get("recommendations", {})
        if any(recommendations.values()):
            sections.append("\nRECOMMENDATIONS & PLAN")
            sections.append("-" * 40)
            
            if recommendations.get("diagnostic_tests"):
                sections.append("Diagnostic Tests Recommended:")
                for test in recommendations["diagnostic_tests"][:8]:
                    sections.append(f"  • {test}")
            
            if recommendations.get("therapy_recommendations"):
                sections.append("Therapy Recommendations:")
                for therapy in recommendations["therapy_recommendations"][:5]:
                    sections.append(f"  • {therapy}")
            
            if recommendations.get("specialist_referrals"):
                sections.append("Specialist Referrals:")
                for referral in recommendations["specialist_referrals"][:5]:
                    sections.append(f"  • {referral}")
            
            if recommendations.get("follow_up_plan"):
                sections.append(f"Follow-up Plan: {recommendations['follow_up_plan']}")
        
        # SECTION 8: CRITICAL FINDINGS
        critical_findings = raw_data.get("critical_findings", [])
        if critical_findings:
            sections.append("\nCRITICAL FINDINGS & ACTION ITEMS")
            sections.append("-" * 40)
            for finding in critical_findings[:10]:
                sections.append(f"  • {finding}")
        
        # SECTION 9: TREATMENTS (if present)
        treatments = raw_data.get("treatments", {})
        if any(treatments.values()):
            sections.append("\nTREATMENT PLAN")
            sections.append("-" * 40)
            
            if treatments.get("current_treatment_plan"):
                sections.append("Current Treatment:")
                for tx in treatments["current_treatment_plan"][:5]:
                    sections.append(f"  • {tx}")
            
            if treatments.get("recommended_treatments"):
                sections.append("Recommended Treatments:")
                for tx in treatments["recommended_treatments"][:5]:
                    sections.append(f"  • {tx}")
        
        long_summary = "\n\n".join(sections)
        word_count = len(long_summary.split())
        
        logger.info(f"✅ Comprehensive long summary built: {word_count} words, {len(sections)} sections")
        
        return long_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate concise short summary (30-60 words) from comprehensive long summary.
        Pipe-delimited format: [Type] | [Author] | [Date] | [Body Parts] | [Findings] | [Meds] | [Plan]
        """
        logger.info("🎯 Generating structured short summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are creating CONCISE pipe-delimited summaries.

OUTPUT FORMAT (pipe-delimited):
[Type] | [Author] | [Date] | [Body Parts] | [Key Findings] | [Medications] | [Plan/Recommendations]

RULES:
- 30-60 words total
- Extract ONLY explicit info from long summary
- NO assumptions
- Use abbreviations: L/R, Bilat, Dx, Rx, PT, f/u
- Omit segments if not in summary

EXAMPLES:
✅ "UR Decision | Dr. Smith | 09/15/25 | L knee | MRI approved for ACL evaluation | Meloxicam 15mg BID | PT 2x/wk x6wks"
✅ "Surgery Report | Dr. Jones | 08/20/24 | R shoulder | Arthroscopic rotator cuff repair completed | Post-op pain meds | f/u 2wks, PT start wk 3"

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
            response = chain.invoke({"long_summary": long_summary})
            short_summary = response.content.strip()
            
            # Clean
            short_summary = re.sub(r'\s+', ' ', short_summary).strip()
            short_summary = re.sub(r'\s*\|\s*', ' | ', short_summary)
            
            word_count = len(short_summary.split())
            logger.info(f"✅ Short summary: {word_count} words")
            
            return short_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary failed: {e}")
            return self._create_fallback_short_summary(long_summary, doc_type)
    
    def _create_fallback_short_summary(self, long_summary: str, doc_type: str) -> str:
        """Fallback short summary from long summary"""
        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]+', long_summary)
        key_sentence = next((s.strip() for s in sentences if len(s.strip()) > 50), "")
        
        if key_sentence:
            words = key_sentence.split()
            return ' '.join(words[:50])
        
        return f"{doc_type} report - comprehensive evaluation completed."
    
    # ... rest of the helper methods (keep existing ones)
    
    def _get_context_length(self, doc_type: str) -> int:
        """Determine context length"""
        complex_docs = ["QME", "AME", "IME", "SURGERY_REPORT", "DISCHARGE"]
        return 20000 if doc_type in complex_docs else 16000
    
    def _clean_extracted_data(self, data: Dict, fallback_date: str) -> Dict:
        """Clean extracted data"""
        if not isinstance(data, dict):
            return {"document_metadata": {"report_date": fallback_date}}
        
        # Ensure date
        if "document_metadata" not in data:
            data["document_metadata"] = {}
        if not data["document_metadata"].get("report_date"):
            data["document_metadata"]["report_date"] = fallback_date
        
        return data
    
    def _detect_physician(self, text: str, page_zones: Optional[Dict]) -> str:
        """Detect physician"""
        try:
            result = self.doctor_detector.detect_doctor(text=text, page_zones=page_zones)
            return result.get("doctor_name", "").strip()
        except:
            return ""
    
    def _create_fallback_data(self, fallback_date: str, doc_type: str) -> Dict:
        """Fallback data"""
        return {
            "document_metadata": {"report_date": fallback_date, "document_type": doc_type},
            "diagnosis": {"primary_diagnoses": [], "affected_body_parts": []},
            "critical_findings": ["Extraction incomplete - manual review recommended"]
        }
    
    def _create_error_response(self, doc_type: str, error_msg: str, fallback_date: str) -> Dict:
        """Error response"""
        return {
            "long_summary": f"{doc_type} extraction failed: {error_msg}. Manual review required.",
            "short_summary": f"{doc_type} | {fallback_date} | Extraction error"
        }
