"""
Imaging Reports Enhanced Extractor - Full Context with Context-Awareness
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


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction using DocumentContextAnalyzer guidance
    - Chain-of-thought reasoning for radiological findings
    - Optimized for imaging report specific patterns and clinical significance
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex patterns for imaging specific content
        self.imaging_patterns = {
            'modality': re.compile(r'\b(MRI|CT|X-RAY|XRAY|ULTRASOUND|US|MAMMOGRAM|PET|SPECT|DEXA)\b', re.IGNORECASE),
            'body_part': re.compile(r'\b(shoulder|knee|spine|wrist|hip|ankle|elbow|hand|foot|brain|chest|abdomen|pelvis)\b', re.IGNORECASE),
            'contrast': re.compile(r'\b(with|without)\s+contrast\b', re.IGNORECASE),
            'finding_severity': re.compile(r'\b(mild|moderate|severe|minimal|marked|advanced)\b', re.IGNORECASE)
        }
        
        logger.info("✅ ImagingExtractorChained initialized (Full Context + Context-Aware)")

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
        Extract Imaging data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (MRI, CT, X-ray, Ultrasound, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
        """
        logger.info("=" * 80)
        logger.info("📊 STARTING IMAGING EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Log context guidance if available
        if context_analysis:
            primary_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            clinical_context = context_analysis.get("clinical_context", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Primary Physician: {primary_physician.get('name', 'Unknown')}")
            logger.info(f"   Confidence: {primary_physician.get('confidence', 'Unknown')}")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Clinical Context: {clinical_context.get('indication', 'Unknown')}")
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
        
        # Stage 2: Override radiologist if context identified one with high confidence
        if context_analysis:
            context_physician = context_analysis.get("physician_analysis", {}).get("primary_physician", {})
            if context_physician.get("name") and context_physician.get("confidence") in ["high", "medium"]:
                logger.info(f"🎯 Using context-identified radiologist: {context_physician.get('name')}")
                raw_result["radiologist_name"] = context_physician.get("name")
        
        # Stage 3: Fallback to DoctorDetector if no radiologist identified
        if not raw_result.get("radiologist_name"):
            logger.info("🔍 No radiologist from context/extraction, using DoctorDetector...")
            radiologist_name = self._detect_radiologist(text, page_zones)
            raw_result["radiologist_name"] = radiologist_name
        
        # Stage 4: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 5: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info("=" * 80)
        logger.info("✅ IMAGING EXTRACTION COMPLETE (FULL CONTEXT)")
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
        Optimized for imaging report specific patterns and clinical significance.
        """
        logger.info("🔍 Processing ENTIRE imaging report in single context window with guidance...")
        
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
        
        # Build context-aware system prompt for Imaging
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert radiology documentation specialist analyzing a COMPLETE imaging report with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE imaging report at once, allowing you to:
- Understand the complete clinical context and indication
- Correlate findings across different sequences and views
- Assess the clinical significance of all abnormalities
- Identify incidental findings and their relevance
- Provide comprehensive radiological assessment without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a finding/value is NOT explicitly mentioned in the report, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate findings
   - DO NOT fill in "typical" or "common" imaging findings
   - DO NOT use radiological knowledge to "complete" incomplete information
   
   Examples:
   ✅ CORRECT: If report says "3cm mass in right upper lobe", extract: "mass_size": "3cm"
   ❌ WRONG: If report says "mass in right upper lobe", DO NOT extract: "mass_size": "3cm" (size not stated)
   ✅ CORRECT: Extract: "mass_size": "" (size field empty)

2. **FINDINGS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY findings explicitly described in "FINDINGS" or "IMPRESSION" sections
   - Include measurements ONLY if explicitly stated
   - DO NOT extract:
     * Findings mentioned as "rule out" or "possible"
     * Historical comparisons unless explicitly compared
     * Normal variants that are not explicitly described as findings
   
   Examples:
   ✅ CORRECT: Report states "1.5cm cystic lesion in right kidney, likely simple cyst"
   Extract: {{"primary_finding": "1.5cm cystic lesion in right kidney", "characterization": "likely simple cyst"}}
   
   ❌ WRONG: Report states "no evidence of renal mass"
   DO NOT extract renal mass as a finding

3. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
   Examples:
   ✅ CORRECT: If no contrast mentioned, return: "contrast_used": ""
   ❌ WRONG: Return: "contrast_used": "Not mentioned" (use empty string instead)

4. **EXACT RADIOLOGICAL TERMINOLOGY**
   - Use EXACT wording from report for findings and impressions
   - DO NOT upgrade "mild" to "moderate" or downgrade "severe" to "moderate"
   - Capture nuanced descriptions: "subtle", "questionable", "probable", "likely"
   
   Examples:
   ✅ CORRECT: Report says "subtle increased signal on T2"
   Extract: "finding_description": "subtle increased signal on T2"
   
   ❌ WRONG: Report says "subtle increased signal on T2"
   DO NOT extract: "finding_description": "increased signal on T2" (removes "subtle")

5. **NO CLINICAL INTERPRETATION BEYOND REPORT**
   - DO NOT predict clinical significance beyond what's stated
   - DO NOT suggest follow-up studies unless explicitly recommended
   - DO NOT infer treatment implications
   
   Examples:
   ❌ WRONG: Report mentions "compression fracture"
   DO NOT assume: "osteoporotic fracture" (etiology not stated)
   ✅ CORRECT: Extract: "compression fracture" (exact wording)

6. **VERIFICATION CHECKLIST BEFORE SUBMISSION**
   Before returning your extraction, verify:
   □ Every finding has a direct quote in the report
   □ Every measurement is explicitly stated (not estimated)
   □ Every "normal" finding is explicitly described as normal
   □ No fields are filled with "typical" or "expected" findings
   □ Empty fields are truly empty (not "Not mentioned" or "Unknown")

IMAGING SPECIFIC EXTRACTION FOCUS - 6 CRITICAL RADIOLOGICAL CATEGORIES:

I. STUDY IDENTIFICATION & TECHNIQUE
- Modality and specific protocol (e.g., "MRI brain without contrast")
- Body part and laterality (be specific: "right shoulder", "lumbar spine")
- Contrast usage and type if mentioned
- Technical adequacy and limitations

II. CLINICAL CONTEXT
- Indication for study from clinical history
- Specific clinical questions to be answered
- Relevant patient history affecting interpretation

III. MAJOR FINDINGS (Most Critical)
- Primary abnormality or most significant finding
- Size, location, and characterization of lesions
- Acute vs chronic findings differentiation
- Comparison to previous studies if explicitly stated

IV. INCIDENTAL FINDINGS
- Clinically relevant incidental discoveries
- Characterization and recommended follow-up if stated
- Significance assessment based on report language

V. IMPRESSION & CONCLUSIONS
- Summary of clinically significant findings
- Differential diagnosis if provided
- Recommendations for follow-up if explicitly stated
- Overall study interpretation

VI. COMPARISON & PROGRESSION
- Explicit comparison to prior studies with dates
- Interval change descriptions
- Stability or progression assessment

⚠️ FINAL REMINDER:
- If information is NOT in the report, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate findings
- FINDINGS: Only extract what is explicitly described
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE imaging report and extract ALL relevant radiological information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE IMAGING REPORT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all radiological details:

{{
  "study_identification": {{
    "report_type": "{doc_type}",
    "study_date": "",
    "accession_number": "",
    "facility": "",
    "radiologist": {{
      "name": "{primary_physician}",
      "credentials": "",
      "specialty": "Radiology"
    }}
  }},
  
  "imaging_technique": {{
    "modality": "{doc_type}",
    "body_part": "",
    "laterality": "",
    "contrast_used": "",
    "contrast_type": "",
    "technical_quality": "",
    "protocol_details": ""
  }},
  
  "clinical_context": {{
    "indication": "",
    "clinical_history": "",
    "specific_questions": "",
    "relevant_history": ""
  }},
  
  "major_findings": {{
    "primary_finding": {{
      "description": "",
      "location": "",
      "size": "",
      "characteristics": "",
      "significance": ""
    }},
    "secondary_findings": [
      {{
        "description": "Mild degenerative changes",
        "location": "L4-L5",
        "significance": "chronic"
      }},
      {{
        "description": "Small joint effusion",
        "location": "right knee",
        "significance": "acute"
      }}
    ],
    "normal_findings": [
      "No acute fracture",
      "No dislocation",
      "Normal alignment"
    ]
  }},
  
  "detailed_analysis": {{
    "bone_structures": {{
      "fractures": "",
      "degenerative_changes": "",
      "alignment": "",
      "bone_marrow": ""
    }},
    "joint_structures": {{
      "cartilage": "",
      "menisci": "",
      "ligaments": "",
      "tendons": "",
      "effusion": ""
    }},
    "soft_tissues": {{
      "masses": "",
      "edema": "",
      "atrophy": "",
      "calcifications": ""
    }},
    "neurological_structures": {{
      "spinal_cord": "",
      "nerve_roots": "",
      "neural_foramina": ""
    }}
  }},
  
  "incidental_findings": [
    {{
      "description": "Small renal cyst",
      "location": "right kidney",
      "size": "1.2cm",
      "characterization": "benign appearing",
      "recommendation": "no follow-up needed"
    }},
    {{
      "description": "Hepatic steatosis",
      "location": "liver",
      "severity": "mild",
      "recommendation": "clinical correlation"
    }}
  ],
  
  "comparison_studies": [
    {{
      "study_type": "MRI",
      "body_part": "right shoulder",
      "date": "01/15/2024",
      "interval_change": "no significant change",
      "stability": "stable"
    }}
  ],
  
  "impression_conclusions": {{
    "overall_impression": "",
    "primary_diagnosis": "",
    "differential_diagnosis": [
      "Rotator cuff tear",
      "Tendinosis",
      "Partial thickness tear"
    ],
    "clinical_correlation": "",
    "recommendations": [
      "Follow-up MRI in 6 months if symptomatic",
      "Orthopedic consultation recommended"
    ],
    "urgency": "routine"
  }},
  
  "critical_findings": [
    {{
      "finding": "Full-thickness rotator cuff tear",
      "significance": "high",
      "action": "Orthopedic surgery consultation",
      "urgency": "semi-urgent"
    }},
    {{
      "finding": "Acute fracture",
      "significance": "high", 
      "action": "Immediate orthopedic evaluation",
      "urgency": "urgent"
    }},
    {{
      "finding": "Mass with suspicious features",
      "significance": "high",
      "action": "Biopsy or further characterization",
      "urgency": "semi-urgent"
    }}
  ],
  
  "quality_assessment": {{
    "study_adequacy": "diagnostic",
    "limitations": "patient motion artifact",
    "technical_notes": "limited due to body habitus"
  }}
}}
""")

        # Build context guidance summary
        context_guidance_text = f"""
PRIMARY RADIOLOGIST: {primary_physician or 'Not identified in context'}
REASONING: {physician_reasoning or 'See document for identification'}

FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'Findings, Impression, Technique'}

CLINICAL CONTEXT:
- Indication: {clinical_context.get('indication', 'Not specified')}
- Clinical History: {clinical_context.get('clinical_history', 'Not provided')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context imaging extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "full_document_text": text,
                "doc_type": doc_type,
                "context_guidance": context_guidance_text,
                "primary_physician": primary_physician or "Extract from document",
                "physician_reasoning": physician_reasoning or "Use signature and interpretation sections",
                "focus_sections": ', '.join(focus_sections) if focus_sections else "Standard radiology sections",
                "clinical_context": str(clinical_context),
                "ambiguities": str(ambiguities)
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context imaging extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted imaging data from complete {len(text):,} char document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context imaging extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(fallback_date, doc_type)

    def _detect_radiologist(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Fallback: Detect radiologist using DoctorDetector"""
        logger.info("🔍 Fallback: Running DoctorDetector for radiologist...")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(f"✅ Radiologist detected: {detection_result['doctor_name']}")
            return detection_result["doctor_name"]
        else:
            logger.warning("⚠️ No valid radiologist found")
            return ""

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result from extracted imaging data"""
        logger.info("🔨 Building initial imaging extraction result...")
        
        # Extract core imaging information
        study_identification = raw_data.get("study_identification", {})
        imaging_technique = raw_data.get("imaging_technique", {})
        impression_conclusions = raw_data.get("impression_conclusions", {})
        
        # Build comprehensive imaging summary
        summary_line = self._build_imaging_narrative_summary(raw_data, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=study_identification.get("study_date", fallback_date),
            summary_line=summary_line,
            examiner_name=raw_data.get("radiologist_name", ""),
            specialty="Radiology",
            body_parts=[imaging_technique.get("body_part", "")] if imaging_technique.get("body_part") else [],
            raw_data=raw_data,
        )
        
        logger.info(f"✅ Initial imaging result built (Radiologist: {result.examiner_name})")
        return result

    def _build_imaging_narrative_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive narrative summary for imaging reports.
        
        Imaging style: "Modality: [type]. Findings: [key abnormalities]. Impression: [clinical significance]."
        """
        
        # Extract all imaging data
        study_identification = data.get("study_identification", {})
        imaging_technique = data.get("imaging_technique", {})
        major_findings = data.get("major_findings", {})
        impression_conclusions = data.get("impression_conclusions", {})
        critical_findings = data.get("critical_findings", [])
        
        # Helper function for safe string conversion
        def safe_str(value, default=""):
            if not value:
                return default
            if isinstance(value, list):
                return ", ".join([str(x) for x in value if x])
            return str(value)
        
        # Build narrative sections
        narrative_parts = []
        
        # Section 1: MODALITY & TECHNIQUE
        technique_text = self._build_technique_narrative(imaging_technique, doc_type)
        if technique_text:
            narrative_parts.append(f"**Study:** {technique_text}")
        
        # Section 2: KEY FINDINGS
        findings_text = self._build_findings_narrative(major_findings, critical_findings)
        if findings_text:
            narrative_parts.append(f"**Findings:** {findings_text}")
        
        # Section 3: IMPRESSION & SIGNIFICANCE
        impression_text = self._build_impression_narrative(impression_conclusions)
        if impression_text:
            narrative_parts.append(f"**Impression:** {impression_text}")
        
        # Section 4: RADIOLOGIST & DATE CONTEXT
        radiologist_info = study_identification.get("radiologist", {})
        radiologist_name = safe_str(radiologist_info.get("name", ""))
        study_date = study_identification.get("study_date", fallback_date)
        
        if radiologist_name:
            context_line = f"Interpreted by {radiologist_name} on {study_date}"
            narrative_parts.insert(0, context_line)
        
        # Join with proper formatting
        full_narrative = "\n\n".join(narrative_parts)
        
        logger.info(f"📝 Imaging narrative summary generated: {len(full_narrative)} characters")
        return full_narrative

    def _build_technique_narrative(self, imaging_technique: Dict, doc_type: str) -> str:
        """Build imaging technique narrative section"""
        technique_parts = []
        
        # Modality and body part
        modality = imaging_technique.get("modality", doc_type)
        body_part = imaging_technique.get("body_part", "")
        laterality = imaging_technique.get("laterality", "")
        
        if modality and body_part:
            technique_str = f"{modality}"
            if body_part:
                technique_str += f" {body_part}"
            if laterality:
                technique_str += f" ({laterality})"
            technique_parts.append(technique_str)
        
        # Contrast information
        contrast = imaging_technique.get("contrast_used", "")
        if contrast:
            technique_parts.append(contrast)
        
        # Technical quality
        quality = imaging_technique.get("technical_quality", "")
        if quality and quality != "diagnostic":
            technique_parts.append(f"Quality: {quality}")
        
        return " - ".join(technique_parts) if technique_parts else f"{doc_type} study"

    def _build_findings_narrative(self, major_findings: Dict, critical_findings: list) -> str:
        """Build key findings narrative"""
        findings_items = []
        
        # Primary finding (highest priority)
        primary = major_findings.get("primary_finding", {})
        if isinstance(primary, dict):
            primary_desc = primary.get("description", "")
            if primary_desc:
                findings_items.append(primary_desc)
        
        # Critical findings (high urgency)
        if critical_findings and isinstance(critical_findings, list):
            for critical in critical_findings[:2]:  # Top 2 critical findings
                if isinstance(critical, dict):
                    finding = critical.get("finding", "")
                    if finding:
                        findings_items.append(finding)
        
        # Secondary findings
        secondary = major_findings.get("secondary_findings", [])
        if secondary and isinstance(secondary, list):
            for finding in secondary[:2]:  # Top 2 secondary findings
                if isinstance(finding, dict):
                    finding_desc = finding.get("description", "")
                    if finding_desc and finding_desc not in findings_items:
                        findings_items.append(finding_desc)
        
        return "; ".join(findings_items) if findings_items else "No significant abnormalities"

    def _build_impression_narrative(self, impression_conclusions: Dict) -> str:
        """Build impression and clinical significance narrative"""
        impression_parts = []
        
        # Overall impression
        overall = impression_conclusions.get("overall_impression", "")
        if overall:
            impression_parts.append(overall)
        
        # Primary diagnosis
        diagnosis = impression_conclusions.get("primary_diagnosis", "")
        if diagnosis and diagnosis != overall:
            impression_parts.append(diagnosis)
        
        # Recommendations
        recommendations = impression_conclusions.get("recommendations", [])
        if recommendations and isinstance(recommendations, list):
            primary_rec = recommendations[0] if recommendations else ""
            if primary_rec and isinstance(primary_rec, str):
                impression_parts.append(f"Recommend: {primary_rec}")
            elif primary_rec and isinstance(primary_rec, dict) and primary_rec.get("recommendation"):
                impression_parts.append(f"Recommend: {primary_rec.get('recommendation')}")
        
        # Urgency
        urgency = impression_conclusions.get("urgency", "")
        if urgency and urgency != "routine":
            impression_parts.append(f"Urgency: {urgency}")
        
        return "; ".join(impression_parts) if impression_parts else "Routine findings"

    def _get_fallback_result(self, fallback_date: str, doc_type: str) -> Dict:
        """Return minimal fallback result structure for imaging"""
        return {
            "study_identification": {
                "report_type": doc_type,
                "study_date": fallback_date,
                "accession_number": "",
                "facility": "",
                "radiologist": {
                    "name": "",
                    "credentials": "",
                    "specialty": "Radiology"
                }
            },
            "imaging_technique": {
                "modality": doc_type,
                "body_part": "",
                "laterality": "",
                "contrast_used": "",
                "contrast_type": "",
                "technical_quality": "",
                "protocol_details": ""
            },
            "clinical_context": {
                "indication": "",
                "clinical_history": "",
                "specific_questions": "",
                "relevant_history": ""
            },
            "major_findings": {
                "primary_finding": {
                    "description": "",
                    "location": "",
                    "size": "",
                    "characteristics": "",
                    "significance": ""
                },
                "secondary_findings": [],
                "normal_findings": []
            },
            "detailed_analysis": {
                "bone_structures": {
                    "fractures": "",
                    "degenerative_changes": "",
                    "alignment": "",
                    "bone_marrow": ""
                },
                "joint_structures": {
                    "cartilage": "",
                    "menisci": "",
                    "ligaments": "",
                    "tendons": "",
                    "effusion": ""
                },
                "soft_tissues": {
                    "masses": "",
                    "edema": "",
                    "atrophy": "",
                    "calcifications": ""
                },
                "neurological_structures": {
                    "spinal_cord": "",
                    "nerve_roots": "",
                    "neural_foramina": ""
                }
            },
            "incidental_findings": [],
            "comparison_studies": [],
            "impression_conclusions": {
                "overall_impression": "",
                "primary_diagnosis": "",
                "differential_diagnosis": [],
                "clinical_correlation": "",
                "recommendations": [],
                "urgency": "routine"
            },
            "critical_findings": [],
            "quality_assessment": {
                "study_adequacy": "",
                "limitations": "",
                "technical_notes": ""
            }
        }