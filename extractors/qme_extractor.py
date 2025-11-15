"""
QME/AME/IME Enhanced Extractor - Medical-Legal Focus (Parallel Processing)
Optimized for 6 critical categories with thread-based parallel chunk processing
"""
import logging
import re
import json
import time
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class QMEExtractorChained:
    """
    Enhanced QME extractor with parallel processing for 6 medical-legal categories
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        
        # Pre-compile regex
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        
        # Optimized chunking settings (4000 chars with 200 overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Optimal for QME reports
            chunk_overlap=200,  # Good overlap for context
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],  # Section-aware
            length_function=len,
        )
        
        logger.info("✅ QMEExtractorChained initialized with parallel processing")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """Extract QME data with parallel chunk processing"""
        logger.info("=" * 80)
        logger.info("🏥 STARTING QME MEDICAL-LEGAL EXTRACTION (PARALLEL)")
        logger.info("=" * 80)
        
        # Stage 1: Extract clinical data using parallel chunked processing
        raw_result = self._extract_medical_legal_data_parallel(text, doc_type, fallback_date)
        
        # Stage 2: Detect examiner via DoctorDetector
        examiner_name = self._detect_examiner(text, page_zones)
        raw_result["qme_physician_name"] = examiner_name
        
        # Stage 3: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info("=" * 80)
        logger.info("✅ QME MEDICAL-LEGAL EXTRACTION COMPLETE")
        logger.info("=" * 80)
        
        return final_result

    def _extract_medical_legal_data_parallel(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        Stage 1: Extract with PARALLEL chunk processing for speedup
        """
        logger.info(f"🔍 Stage 1: Splitting document (length: {len(text)} chars)")
        chunks = self.splitter.split_text(text)
        logger.info(f"📦 Created {len(chunks)} chunks for PARALLEL processing")
        
        # Build prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a senior QME/IME/AME Medical-Legal Extraction Engine.
Your job is to extract HIGH-ACCURACY structured data from long medical-legal reports
(QME, AME, IME, PR-2, PR-4, PTP notes, orthopedic reports, pain management reports).

🎯 PRIMARY GOAL:
Return a COMPLETE, NO-MISSING-FIELDS structured extraction for ALL medically and legally
relevant information contained in the report.

YOU MUST EXTRACT THE FOLLOWING CATEGORIES WITH MAXIMUM COMPLETENESS:

===================================================================
I. CORE CASE IDENTITY (REQUIRED)
- Applicant/Case Name
- Patient name (even if repeated in multiple sections)
- Patient DOB and/or age
- Date of Injury (DOI)
- Dates: evaluation date, exam date, report date
- Evaluating physician name, credentials, and specialty/subspecialty
- Referring attorney or insurer (if mentioned)

===================================================================
II. DIAGNOSES (FULL LIST)
Extract ALL diagnoses exactly as written, including:
- Musculoskeletal diagnoses
- Psych diagnoses
- Internal medicine and rheumatology findings
- Chronic conditions
- Differential diagnoses
- Laterality (left/right/bilateral)
- ICD-10 codes if included

The diagnoses list MUST be complete, not abbreviated.

===================================================================
III. SURGICAL & TREATMENT HISTORY
Extract ALL:
- Past surgeries (arthroscopy, TKA, THA, meniscectomy, RFA, injections)
- Dates of surgeries
- Prior conservative care (PT, acupuncture, chiropractic)
- Relevant imaging and findings

===================================================================
IV. EXAM FINDINGS & CLINICAL STATUS
Extract EXACT values and descriptions:
- Pain score (current and highest)
- Gait abnormalities
- ROM limits (degrees, flexion, extension)
- Positive special tests (McMurray, Lachman, FABER, Hawkins, SLRT, etc.)
- Neurologic deficits
- Swelling, effusion, tenderness
- Muscle weakness
- Imaging summaries if included

===================================================================
V. MEDICATIONS (COMPLETE LIST)
Categorize medications into:
- Narcotics/opioids
- Neuropathic pain meds (Gabapentin, Duloxetine, Amitriptyline, etc.)
- Anti-inflammatories (NSAIDs, Meloxicam, Ibuprofen)
- Other long-term meds (statins, antihistamines, inhalers, dermatologic meds)
Include dosages WHEN explicitly provided.

===================================================================
VI. MEDICAL-LEGAL OPINIONS (CRITICAL)
Extract EXACT legal conclusions:
- MMI / P&S Status (Yes, No, or Deferred)
- If deferred → extract the REASON
- WPI % (whole person impairment)
- If deferred → extract the REASON
- Apportionment explanation
- Industrial vs non-industrial percentages
- Causation summary (industrial vs degenerative)

===================================================================
VII. FUTURE TREATMENT & RECOMMENDATIONS
Extract ALL future care recommendations:
- Surgeries
- Injections (steroid, PRP, genicular nerve blocks, RFA)
- Diagnostic testing (MRI, CT, sleep study)
- Specialist referrals (Rheumatology QME, Psych QME, Neuro QME)
- Medication changes
- Follow-up QME timelines

===================================================================
VIII. WORK STATUS & RESTRICTIONS
Extract:
- TTD / TPD / Full Duty / Modified Duty
- Specific functional restrictions (e.g., no kneeling, lifting <10 lbs)
- RTW status and reasoning
- Anticipated RTW timeline if stated

===================================================================
EXTRACTION RULES:
- DO NOT INVENT OR FILL IN MISSING INFORMATION.
- Only extract text explicitly found in the document.
- Use empty strings "" or empty lists [] for missing values.
- Preserve original medical wording EXACTLY.
- If something appears in multiple sections, include it only once.
- If data is uncertain, include the exact phrasing used (e.g., “Osteoarthritis vs labral injury”).

===================================================================
OUTPUT:
Return ONLY the JSON structure required by the user prompt.
Do NOT include commentary, summaries, or additional text.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
Extract ALL medically and legally relevant data from the following QME/AME/IME text.

You MUST NOT miss ANY information.

TEXT:
{text}

Return JSON using this exact structure and field names:

{{
  "category_1_core_identity": {{
    "applicant_or_case_name": "",
    "patient_name": "",
    "patient_age": "",
    "patient_dob": "",
    "date_of_injury": "",
    "evaluation_date": "",
    "report_date": "",
    "qme_physician_name": "",
    "qme_specialty": ""
  }},
  "category_2_diagnosis": {{
    "primary_diagnoses": [],
    "icd_codes": [],
    "affected_body_parts": []
  }},
  "category_3_clinical_status": {{
    "past_surgeries": [],
    "current_chief_complaint": "",
    "pain_score_current": "",
    "pain_score_max": "",
    "objective_findings": {{
      "rom_limitations": "",
      "gait_abnormalities": "",
      "positive_tests": "",
      "effusion_swelling": "",
      "neurologic_findings": "",
      "other_objective": ""
    }}
  }},
  "category_4_medications": {{
    "narcotics_opioids": [],
    "nerve_pain_meds": [],
    "anti_inflammatories": [],
    "other_long_term_meds": []
  }},
  "category_5_medical_legal_conclusions": {{
    "mmi_status": "",
    "mmi_deferred_reason": "",
    "wpi_percentage": "",
    "wpi_deferred_reason": "",
    "apportionment_industrial": "",
    "apportionment_nonindustrial": "",
    "causation_summary": ""
  }},
  "category_6_actionable_recommendations": {{
    "future_surgery": "",
    "future_injections": "",
    "future_therapy": "",
    "future_diagnostics": "",
    "specialist_referrals": [],
    "work_status": "",
    "work_restrictions_specific": [],
    "follow_up_timeline": ""
  }}
}}
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            # Parallel processing with ThreadPoolExecutor
            partial_results = []
            
            logger.info(f"🚀 Processing {len(chunks)} chunks in PARALLEL...")
            start_time = time.time()  # Use time.time() instead of asyncio
            
            # Determine optimal number of workers (max 5 to avoid rate limits)
            max_workers = min(len(chunks), 5)
            logger.info(f"⚙️ Using {max_workers} parallel workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(self._process_chunk, chat_prompt, chunk, i+1, len(chunks)): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        partial_results.append((chunk_idx, result))
                    except Exception as e:
                        logger.error(f"❌ Chunk {chunk_idx+1} processing failed: {e}")
                        partial_results.append((chunk_idx, None))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Sort by original chunk order and filter None results
            partial_results.sort(key=lambda x: x[0])
            successful_results = [r[1] for r in partial_results if r[1] is not None]
            
            logger.info(f"⚡ Parallel processing completed in {processing_time:.2f}s")
            logger.info(f"✅ Successfully processed {len(successful_results)}/{len(chunks)} chunks")
            
            if not successful_results:
                logger.error("❌ No chunks processed successfully!")
                return self._get_fallback_result(fallback_date)
            
            # Merge partial extractions
            merged_result = self._merge_medical_legal_extractions(successful_results, fallback_date)
            logger.info(f"✅ Chunked extraction completed: {len(successful_results)} chunks merged")
            
            # Log extracted categories
            self._log_extracted_categories(merged_result)
            
            return merged_result
            
        except Exception as e:
            logger.error(f"❌ Parallel extraction failed: {e}", exc_info=True)
            return self._get_fallback_result(fallback_date)

    def _process_chunk(self, chat_prompt, chunk: str, chunk_num: int, total_chunks: int) -> Dict:
        """Process a single chunk (called in parallel threads)"""
        try:
            logger.info(f"🔄 Processing chunk {chunk_num}/{total_chunks}")
            
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({"text": chunk})
            
            logger.debug(f"✅ Chunk {chunk_num}/{total_chunks} completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing chunk {chunk_num}/{total_chunks}: {e}")
            raise

    def _merge_medical_legal_extractions(self, partials: List[Dict], fallback_date: str) -> Dict:
        """Merge extractions from multiple chunks (FIXED: safe type handling)"""
        if not partials:
            return self._get_fallback_result(fallback_date)
        
        logger.info(f"🔄 Merging {len(partials)} partial extractions...")
        
        merged = self._get_fallback_result(fallback_date)
        
        try:
            # Merge Category 1: Core Identity (take most complete)
            for partial in partials:
                cat1 = partial.get("category_1_core_identity", {})
                if not isinstance(cat1, dict):
                    continue
                    
                for key, value in cat1.items():
                    if value and not merged["category_1_core_identity"].get(key):
                        # Safe string conversion
                        merged["category_1_core_identity"][key] = str(value).strip() if value else ""
            
            # Merge Category 2: Diagnosis (union of lists)
            for partial in partials:
                cat2 = partial.get("category_2_diagnosis", {})
                if not isinstance(cat2, dict):
                    continue
                    
                for field in ["primary_diagnoses", "icd_codes", "affected_body_parts"]:
                    values = cat2.get(field, [])
                    if isinstance(values, list):
                        merged["category_2_diagnosis"][field].extend([
                            str(v).strip() for v in values if v and str(v).strip()
                        ])
                    elif isinstance(values, str) and values.strip():
                        merged["category_2_diagnosis"][field].extend([
                            v.strip() for v in values.split(",") if v.strip()
                        ])
            
            # Deduplicate diagnosis lists
            for field in ["primary_diagnoses", "icd_codes", "affected_body_parts"]:
                seen = set()
                deduped = []
                for item in merged["category_2_diagnosis"][field]:
                    if item not in seen:
                        seen.add(item)
                        deduped.append(item)
                merged["category_2_diagnosis"][field] = deduped
            
            # Merge Category 3: Clinical Status
            for partial in partials:
                cat3 = partial.get("category_3_clinical_status", {})
                if not isinstance(cat3, dict):
                    continue
                
                # Surgeries (list)
                surgeries = cat3.get("past_surgeries", [])
                if isinstance(surgeries, list):
                    merged["category_3_clinical_status"]["past_surgeries"].extend([
                        str(s).strip() for s in surgeries if s and str(s).strip()
                    ])
                
                # String fields (take most complete)
                for field in ["current_chief_complaint", "pain_score_current", "pain_score_max"]:
                    value = cat3.get(field, "")
                    current_len = len(str(merged["category_3_clinical_status"].get(field, "")))
                    new_len = len(str(value).strip()) if value else 0
                    if new_len > current_len:
                        merged["category_3_clinical_status"][field] = str(value).strip()
                
                # Objective findings (CRITICAL FIX: check if dict before iterating)
                obj_findings = cat3.get("objective_findings", {})
                if isinstance(obj_findings, dict):
                    for key, value in obj_findings.items():
                        current_len = len(str(merged["category_3_clinical_status"]["objective_findings"].get(key, "")))
                        new_len = len(str(value).strip()) if value else 0
                        if new_len > current_len:
                            merged["category_3_clinical_status"]["objective_findings"][key] = str(value).strip()
            
            # Deduplicate surgeries
            seen = set()
            deduped_surgeries = []
            for surgery in merged["category_3_clinical_status"]["past_surgeries"]:
                if surgery not in seen:
                    seen.add(surgery)
                    deduped_surgeries.append(surgery)
            merged["category_3_clinical_status"]["past_surgeries"] = deduped_surgeries
            
            # Merge Category 4: Medications (union of lists)
            for partial in partials:
                cat4 = partial.get("category_4_medications", {})
                if not isinstance(cat4, dict):
                    continue
                    
                for field in ["narcotics_opioids", "nerve_pain_meds", "anti_inflammatories", "other_long_term_meds"]:
                    meds = cat4.get(field, [])
                    if isinstance(meds, list):
                        merged["category_4_medications"][field].extend([
                            str(m).strip() for m in meds if m and str(m).strip()
                        ])
                    elif isinstance(meds, str) and meds.strip():
                        merged["category_4_medications"][field].extend([
                            m.strip() for m in meds.split(",") if m.strip()
                        ])
            
            # Deduplicate medications
            for field in ["narcotics_opioids", "nerve_pain_meds", "anti_inflammatories", "other_long_term_meds"]:
                seen = set()
                deduped = []
                for med in merged["category_4_medications"][field]:
                    if med not in seen:
                        seen.add(med)
                        deduped.append(med)
                merged["category_4_medications"][field] = deduped
            
            # Merge Category 5: Medical-Legal Conclusions (take most complete)
            for partial in partials:
                cat5 = partial.get("category_5_medical_legal_conclusions", {})
                if not isinstance(cat5, dict):
                    continue
                    
                for key, value in cat5.items():
                    current_len = len(str(merged["category_5_medical_legal_conclusions"].get(key, "")))
                    new_len = len(str(value).strip()) if value else 0
                    if new_len > current_len:
                        merged["category_5_medical_legal_conclusions"][key] = str(value).strip()
            
            # Merge Category 6: Actionable Recommendations
            for partial in partials:
                cat6 = partial.get("category_6_actionable_recommendations", {})
                if not isinstance(cat6, dict):
                    continue
                
                # String fields (take most complete)
                for field in ["future_surgery", "future_injections", "future_therapy", "future_diagnostics", "work_status"]:
                    value = cat6.get(field, "")
                    current_len = len(str(merged["category_6_actionable_recommendations"].get(field, "")))
                    new_len = len(str(value).strip()) if value else 0
                    if new_len > current_len:
                        merged["category_6_actionable_recommendations"][field] = str(value).strip()
                
                # Work restrictions (list)
                restrictions = cat6.get("work_restrictions_specific", [])
                if isinstance(restrictions, list):
                    merged["category_6_actionable_recommendations"]["work_restrictions_specific"].extend([
                        str(r).strip() for r in restrictions if r and str(r).strip()
                    ])
                elif isinstance(restrictions, str) and restrictions.strip():
                    merged["category_6_actionable_recommendations"]["work_restrictions_specific"].extend([
                        r.strip() for r in restrictions.split(",") if r.strip()
                    ])
            
            # Deduplicate work restrictions
            seen = set()
            deduped_restrictions = []
            for restriction in merged["category_6_actionable_recommendations"]["work_restrictions_specific"]:
                if restriction not in seen:
                    seen.add(restriction)
                    deduped_restrictions.append(restriction)
            merged["category_6_actionable_recommendations"]["work_restrictions_specific"] = deduped_restrictions
            
            logger.info("✅ Merge completed successfully")
            return merged
            
        except Exception as e:
            logger.error(f"❌ Merge failed: {e}", exc_info=True)
            return self._get_fallback_result(fallback_date)
  
    def _log_extracted_categories(self, data: Dict):
        """Log extracted data organized by medical-legal categories"""
        logger.info("=" * 80)
        logger.info("📋 EXTRACTED MEDICAL-LEGAL DATA BY CATEGORY:")
        logger.info("=" * 80)
        
        # Category I: Core Identity
        cat1 = data.get("category_1_core_identity", {})
        logger.info("\n📌 CATEGORY I: CORE IDENTITY")
        logger.info(f"  Patient: {cat1.get('patient_name', 'Not found')} (Age: {cat1.get('patient_age', 'N/A')}, DOB: {cat1.get('patient_dob', 'N/A')})")
        logger.info(f"  Date of Injury: {cat1.get('date_of_injury', 'Not found')}")
        logger.info(f"  Report Date: {cat1.get('report_date', 'Not found')}")
        logger.info(f"  QME Physician: {cat1.get('qme_physician_name', 'Not found')} ({cat1.get('qme_specialty', 'N/A')})")
        
        # Category II: Diagnosis
        cat2 = data.get("category_2_diagnosis", {})
        logger.info("\n🩺 CATEGORY II: DIAGNOSIS")
        logger.info(f"  Primary Diagnoses: {', '.join(cat2.get('primary_diagnoses', [])) or 'Not found'}")
        logger.info(f"  ICD Codes: {', '.join(cat2.get('icd_codes', [])) or 'Not found'}")
        logger.info(f"  Affected Body Parts: {', '.join(cat2.get('affected_body_parts', [])) or 'Not found'}")
        
        # Category III: Clinical Status
        cat3 = data.get("category_3_clinical_status", {})
        logger.info("\n🏥 CATEGORY III: CLINICAL STATUS")
        logger.info(f"  Past Surgeries: {', '.join(cat3.get('past_surgeries', [])) or 'None'}")
        logger.info(f"  Chief Complaint: {cat3.get('current_chief_complaint', 'Not found')}")
        logger.info(f"  Pain Score: {cat3.get('pain_score_current', 'N/A')}/10 (Max: {cat3.get('pain_score_max', 'N/A')}/10)")
        obj = cat3.get("objective_findings", {})
        logger.info(f"  Objective Findings:")
        logger.info(f"    - ROM: {obj.get('rom_limitations', 'Normal')}")
        logger.info(f"    - Gait: {obj.get('gait_abnormalities', 'Normal')}")
        logger.info(f"    - Tests: {obj.get('positive_tests', 'None')}")
        
        # Category IV: Medications
        cat4 = data.get("category_4_medications", {})
        logger.info("\n💊 CATEGORY IV: MEDICATIONS")
        logger.info(f"  Narcotics/Opioids: {', '.join(cat4.get('narcotics_opioids', [])) or 'None'}")
        logger.info(f"  Nerve Pain Meds: {', '.join(cat4.get('nerve_pain_meds', [])) or 'None'}")
        logger.info(f"  Anti-inflammatories: {', '.join(cat4.get('anti_inflammatories', [])) or 'None'}")
        logger.info(f"  Other Long-term: {', '.join(cat4.get('other_long_term_meds', [])) or 'None'}")
        
        # Category V: Medical-Legal Conclusions (CRITICAL)
        cat5 = data.get("category_5_medical_legal_conclusions", {})
        logger.info("\n⚖️ CATEGORY V: MEDICAL-LEGAL CONCLUSIONS (CRITICAL)")
        logger.info(f"  MMI Status: {cat5.get('mmi_status', 'Not stated')}")
        if cat5.get('mmi_deferred_reason'):
            logger.info(f"    └─ Reason Deferred: {cat5.get('mmi_deferred_reason')}")
        logger.info(f"  WPI: {cat5.get('wpi_percentage', 'Not stated')}")
        if cat5.get('wpi_deferred_reason'):
            logger.info(f"    └─ Reason Deferred: {cat5.get('wpi_deferred_reason')}")
        logger.info(f"  Apportionment: {cat5.get('apportionment_industrial', 'N/A')}% Industrial / {cat5.get('apportionment_nonindustrial', 'N/A')}% Non-industrial")
        
        # Category VI: Actionable Recommendations (MOST IMPORTANT)
        cat6 = data.get("category_6_actionable_recommendations", {})
        logger.info("\n🎯 CATEGORY VI: ACTIONABLE RECOMMENDATIONS (CRITICAL FOR REVIEW)")
        logger.info(f"  Future Surgery: {cat6.get('future_surgery', 'None recommended')}")
        logger.info(f"  Future Injections: {cat6.get('future_injections', 'None recommended')}")
        logger.info(f"  Future Therapy: {cat6.get('future_therapy', 'None recommended')}")
        logger.info(f"  Future Diagnostics: {cat6.get('future_diagnostics', 'None recommended')}")
        logger.info(f"  Work Status: {cat6.get('work_status', 'Not specified')}")
        restrictions = cat6.get('work_restrictions_specific', [])
        if restrictions:
            logger.info(f"  Specific Work Restrictions:")
            for restriction in restrictions:
                logger.info(f"    - {restriction}")
        else:
            logger.info(f"  Specific Work Restrictions: None specified")
        
        logger.info("=" * 80)

    def _get_fallback_result(self, fallback_date: str) -> Dict:
        """Return minimal fallback result structure organized by categories"""
        return {
            "category_1_core_identity": {
                "patient_name": "",
                "patient_age": "",
                "patient_dob": "",
                "date_of_injury": "",
                "report_date": fallback_date,
                "qme_physician_name": "",
                "qme_specialty": ""
            },
            "category_2_diagnosis": {
                "primary_diagnoses": [],
                "icd_codes": [],
                "affected_body_parts": []
            },
            "category_3_clinical_status": {
                "past_surgeries": [],
                "current_chief_complaint": "",
                "pain_score_current": "",
                "pain_score_max": "",
                "objective_findings": {
                    "rom_limitations": "",
                    "gait_abnormalities": "",
                    "positive_tests": "",
                    "effusion_swelling": "",
                    "other_objective": ""
                }
            },
            "category_4_medications": {
                "narcotics_opioids": [],
                "nerve_pain_meds": [],
                "anti_inflammatories": [],
                "other_long_term_meds": []
            },
            "category_5_medical_legal_conclusions": {
                "mmi_status": "",
                "mmi_deferred_reason": "",
                "wpi_percentage": "",
                "wpi_deferred_reason": "",
                "apportionment_industrial": "",
                "apportionment_nonindustrial": ""
            },
            "category_6_actionable_recommendations": {
                "future_surgery": "",
                "future_injections": "",
                "future_therapy": "",
                "future_diagnostics": "",
                "work_status": "",
                "work_restrictions_specific": []
            }
        }

    def _detect_examiner(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """Stage 2: Detect QME/AME examiner using DoctorDetector"""
        logger.info("🔍 Stage 2: Running DoctorDetector for QME physician...")
        
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

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build initial result from categorized data"""
        logger.info("🔨 Stage 3: Building initial result from categorized data...")
        
        # Extract core identity
        cat1 = raw_data.get("category_1_core_identity", {})
        cat2 = raw_data.get("category_2_diagnosis", {})
        cat5 = raw_data.get("category_5_medical_legal_conclusions", {})
        cat6 = raw_data.get("category_6_actionable_recommendations", {})
        
        # Build summary
        summary_line = self._build_medical_legal_summary(raw_data, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cat1.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=raw_data.get("qme_physician_name", ""),
            specialty=cat1.get("qme_specialty", ""),
            body_parts=cat2.get("affected_body_parts", []),
            raw_data=raw_data,
        )
        
        logger.info(f"✅ Initial result built (Physician: {result.examiner_name})")
        return result

    def _build_medical_legal_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build concise summary focused on medical-legal key points"""
        cat1 = data.get("category_1_core_identity", {})
        cat2 = data.get("category_2_diagnosis", {})
        cat5 = data.get("category_5_medical_legal_conclusions", {})
        cat6 = data.get("category_6_actionable_recommendations", {})
        
        parts = []
        
        # Date and physician
        date = cat1.get("report_date", fallback_date)
        physician = data.get("qme_physician_name", "")
        specialty = cat1.get("qme_specialty", "")
        
        parts.append(f"{date}: {doc_type}")
        if physician:
            if specialty:
                parts.append(f"by {physician} ({self._abbreviate_specialty(specialty)})")
            else:
                parts.append(f"by {physician}")
        
        # Body parts and diagnoses
        body_parts = cat2.get("affected_body_parts", [])
        diagnoses = cat2.get("primary_diagnoses", [])
        
        if body_parts:
            parts.append(f"for {', '.join(body_parts[:2])}")
        
        if diagnoses:
            parts.append(f"= Dx: {', '.join(diagnoses[:2])}")
        
        # Medical-legal conclusions (CRITICAL)
        mmi = cat5.get("mmi_status", "")
        wpi = cat5.get("wpi_percentage", "")
        apport_ind = cat5.get("apportionment_industrial", "")
        
        conclusions = []
        if mmi:
            conclusions.append(mmi)
        if wpi:
            conclusions.append(f"WPI: {wpi}")
        if apport_ind:
            conclusions.append(f"Apportionment: {apport_ind}% industrial")
        
        if conclusions:
            parts.append(f"| {'; '.join(conclusions)}")
        
        # Actionable recommendations
        future_tx = []
        if cat6.get("future_surgery"):
            future_tx.append(cat6.get("future_surgery"))
        if cat6.get("future_injections"):
            future_tx.append(cat6.get("future_injections"))
        if cat6.get("future_therapy"):
            future_tx.append(cat6.get("future_therapy"))
        
        work_status = cat6.get("work_status", "")
        restrictions = cat6.get("work_restrictions_specific", [])
        
        if future_tx or work_status or restrictions:
            rec_parts = []
            if future_tx:
                rec_parts.append(f"Tx: {', '.join(future_tx[:2])}")
            if work_status:
                rec_parts.append(f"Work: {work_status}")
            elif restrictions:
                rec_parts.append(f"Work: {', '.join(restrictions[:2])}")
            
            parts.append(f"→ {'; '.join(rec_parts)}")
        
        summary = " ".join(parts)
        
        # Truncate if too long
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:70]) + "..."
        
        logger.info(f"📝 Summary generated: {len(summary.split())} words")
        return summary

    def _abbreviate_specialty(self, specialty: str) -> str:
        """Abbreviate medical specialties"""
        abbreviations = {
            "Orthopedic Surgery": "Ortho",
            "Orthopedics": "Ortho",
            "Neurology": "Neuro",
            "Pain Management": "Pain",
            "Psychiatry": "Psych",
            "Psychology": "Psych",
            "Physical Medicine & Rehabilitation": "PM&R",
            "Physical Medicine and Rehabilitation": "PM&R",
            "Internal Medicine": "IM",
            "Occupational Medicine": "Occ Med",
        }
        return abbreviations.get(specialty, specialty[:12])

    def _validate_physician_full_name(self, name: str) -> str:
        """Validate physician name has proper credentials"""
        if not name or name.lower() in ["not specified", "not found", "none", "n/a", ""]:
            return ""
        
        name_lower = name.lower()
        
        reject_terms = {
            "admin", "administrator", "case manager", "coordinator", "manager",
            "therapist", "technician", "assistant", "technologist"
        }
        
        if any(term in name_lower for term in reject_terms):
            return ""
        
        if not self.medical_credential_pattern.search(name_lower):
            return ""
        
        words = name.split()
        if len(words) < 2:
            return ""
        
        has_proper_structure = (
            (len(words) >= 3 and any(title in words[0].lower() for title in ["dr", "dr."])) or
            (len(words) >= 2 and any(title in words[-1].lower() for title in ["md", "do", "m.d.", "d.o."]))
        )
        
        if not has_proper_structure:
            if len(words) >= 2 and self.medical_credential_pattern.search(name_lower):
                return name
            return ""
        
        return name
