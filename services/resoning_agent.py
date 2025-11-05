"""
Enhanced Report Analyzer with LLM Chaining and Verification
Integrates with modular document_detector.py for accurate document type detection
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime, timedelta
import re, json
import logging

# Import our modular document detector
from utils.document_detector import DocumentTypeDetector
from models.data_models import DocumentType
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

from models.data_models import DocumentAnalysis, VerificationResult, BriefSummary

# ============================================================================
# ENHANCED REPORT ANALYZER WITH CHAINING & VERIFICATION
# ============================================================================

class EnhancedReportAnalyzer:
    """
    Enhanced analyzer with:
    1. Document type detection using modular DocumentTypeDetector
    2. LLM chaining for extraction
    3. Verification layer for accuracy
    4. Doctor name validation with title checking
    """
    
    def __init__(self):
        """Initialize with LLM and all components"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )
        
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)
        self.verification_parser = JsonOutputParser(pydantic_object=VerificationResult)
        
        # Initialize document type detector from modular architecture
        self.document_detector = DocumentTypeDetector(self.llm)
        
        logger.info("✅ EnhancedReportAnalyzer initialized with document detector")
    
    def detect_document_type(self, document_text: str) -> str:
        """
        Detect document type using modular DocumentTypeDetector.
        Returns standardized document type string.
        """
        try:
            logger.info("🔍 Detecting document type using modular detector...")
            doc_type_enum = self.document_detector.detect(document_text)
            doc_type = doc_type_enum.value
            logger.info(f"✅ Document type detected: {doc_type}")
            return doc_type
        except Exception as e:
            logger.error(f"❌ Document type detection failed: {str(e)}")
            return "medical_document"
    
    def validate_doctor_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate doctor name has proper title (Dr., MD, DO).
        Returns (is_valid, corrected_name)
        """
        if not name or name == "Not specified":
            return True, name
        
        # Check if has title
        has_title = any(title in name for title in ["Dr.", "Dr ", "MD", "DO", "M.D.", "D.O."])
        
        if has_title:
            return True, name
        else:
            # Check if looks like a name (2+ words with capitals)
            words = name.strip().split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if w):
                # Add Dr. prefix
                corrected = f"Dr. {name}"
                logger.warning(f"⚠️ Added 'Dr.' title to physician name: {name} → {corrected}")
                return False, corrected
            else:
                # Not a valid doctor name
                logger.warning(f"⚠️ Invalid doctor name format: {name}")
                return False, "Not specified"
    
    def create_enhanced_extraction_prompt(self, detected_doc_type: str) -> PromptTemplate:
        """
        Create extraction prompt with detected document type guidance.
        Enhanced with better doctor name extraction rules.
        """
        template = """
        You are a medical document analysis expert. Perform a SINGLE, DEEP, COMPREHENSIVE analysis of this medical document.
        
        DETECTED DOCUMENT TYPE: {detected_doc_type}
        (Use this as guidance for what to expect and prioritize in extraction)
        
        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        DEEP ANALYSIS INSTRUCTIONS (Perform ALL in one pass):
        
        1. DEEP STRUCTURE & PATTERN ANALYSIS:
           - Document type already detected as: {detected_doc_type}
           - Use this to guide extraction priorities (e.g., if QME/AME → focus on impairment/apportionment; if imaging → focus on findings)
           - Scan for sections: Patient Demographics, Subjective, Objective, Assessment, Plan, Findings, Impressions, Treatment, Follow-up
           - Identify workers comp: Look for "WRKCMP", "Work Comp", "State Comp", "industrial injury", claim references
           - Assess urgency: Pain scales (7+/10=urgent), symptoms (severe/acute), findings (fracture/infection=critical)
           - Levels: normal, elevated, urgent, critical
        
        2. PATIENT & CLAIM EXTRACTION (DEEP FOCUS ON CLAIM):
           - Patient Name: Full name from demographics (e.g., "Patient: John Doe" → "John Doe"). If not present, "Not specified"
           - DOB: Explicit "DOB:", "Date of Birth:", birth patterns. If not present, "Not specified"
           - CLAIM NUMBER (CRITICAL): Deeply scan for keywords like "CLAIM", "Claim #", "Claim Number", "CL#" exactly near a number
             * If keyword present + number follows/nearby (within 20 chars), extract as claim_number
             * If no keyword, use "Not specified" even if numbers exist
           - DOI: Injury/event dates in history/workers comp context. If not present, "Not specified"
        
        3. CLINICAL & FUNCTIONAL EXTRACTION:
           - Status: From urgency analysis. If not inferable, "Not specified"
           - Diagnosis: Primary condition + 2-3 key objective findings (5-10 words total). If not present, "Not specified"
           - Key Concern: Main issue in 2-3 words. If not present, "Not specified"
           
           - BODY PART ANALYSIS (MULTIPLE SUPPORT):
             * Primary Body Part: Extract main body part (e.g., "lumbar spine", "right knee"). If not present, "Not specified"
             * Multiple Body Parts: If document mentions multiple distinct body parts, analyze EACH separately in body_parts_analysis array
             * For each body part include: body_part, diagnosis, key_concern, clinical_summary (2-3 sentences), treatment_plan (2-3 sentences), extracted_recommendation, adls_affected, work_restrictions
           
           - ADLs Affected: Limited activities in 2-3 words. If not present, "Not specified"
           - Work Restrictions: Limitations in 2-3 words. If not present, "Not specified"
        
        4. EXTRACTED RECOMMENDATION KEYWORDS (STRICT):
           - STRICTLY keyword-based: Scan for 'recommend', 'recommended', 'recommendation', 'plan', 'plans', 'follow-up', 'therapy', 'PT', 'medication', 'surgery', 'consult', 'referral' in Plan/Assessment/Follow-up sections
           - ONLY if keywords found: Extract immediate key keywords/phrases (comma-separated, e.g., 'PT twice weekly, follow-up 4 weeks')
           - If NONE found, set to 'Not specified'
        
        5. EXTRACTED DECISION KEYWORDS (DIRECT EXTRACTION):
           - Scan for decisions in Assessment/Plan/Impressions: Extract key keywords/phrases (comma-separated medical terms)
           - Look for "Decision:", "Plan:", "Judgment:", "Proceed with", etc.
           - If none found, "Not specified"
        
        6. DOCTOR EXTRACTION (CRITICAL - STRICT VALIDATION):
           - CONSULTING DOCTOR:
             * MUST have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O."
             * Look in signatures, consultations, specialist mentions
             * Extract FULL NAME (first + last) with title
             * STRICT RULE: If name found without title, extract it but LLM MUST note this for verification
             * If no consultant found, "Not specified"
           
           - REFERRAL DOCTOR:
             * MUST have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O."
             * Look for "Referred to", "Referral to", "Referred by", "PCP", "Primary Care"
             * Extract FULL NAME (first + last) with title
             * STRICT RULE: If name found without title, extract it but LLM MUST note this for verification
             * If no referral found, "Not specified"
           
           - IMPORTANT: Do NOT extract patient names, signature names without context, or administrative names as doctors
        
        7. EXTRACTED UR DECISION KEYWORDS & DENIAL REASON:
           - Scan for 'Utilization Review', 'UR', 'prior authorization', 'PA', 'approved', 'denied', 'coverage decision'
           - ONLY if keywords found: Extract decision keywords (comma-separated)
           - If denial: Extract reason (1-2 sentences). Set ur_denial_reason
           - If NONE found, ur_decision='Not specified', ur_denial_reason=None
        
        8. AI OUTCOME KEYWORDS (GENERATED):
           - Based on analysis: Generate outcome prediction keywords (comma-separated)
           - Tie to evidence from extracted data
        
        9. TASK NEED ANALYSIS (CRITICAL):
           - Set is_task_needed=TRUE ONLY if:
             * PENDING actions, treatments NOT YET completed
             * FUTURE appointments needing scheduling
             * NEW recommendations requiring implementation
             * AUTHORIZATIONS needed
             * REFERRALS to process
           - Set FALSE if:
             * Everything COMPLETED/DONE
             * No pending actions
             * Purely historical/descriptive
        
        10. DEEP DATE ANALYSIS & REASONING:
           - Extract ALL dates: Convert to YYYY-MM-DD
           - Contexts: 50-100 chars around each
           - Reasoning: Classify as DOB/DOI/RD using document flow
           - Confidence scores (0.0-1.0)
           - If no dates: Empty lists, note in reasoning
        
        11. DOCUMENT OVERVIEW:
           - Document Type: Use detected type ({detected_doc_type}) as primary, refine if needed
           - Summary Points: 3-5 key points, each 2-3 words
        
        12. CONFIDENCE & VERIFICATION METADATA:
           - extraction_confidence: Assign overall confidence (0.0-1.0) based on:
             * 1.0: All critical fields extracted with high certainty
             * 0.8: Most fields extracted, some ambiguity
             * 0.6: Several fields uncertain or missing
             * 0.4: Many critical fields missing or uncertain
             * 0.2: Minimal extraction possible
           - verification_notes: List any uncertainties (e.g., "Doctor name extracted without title", "Claim number ambiguous")
        
        RULES FOR ACCURACY:
        - Extract from document text only; "Not specified" otherwise
        - Doctor names MUST have titles; note if missing for verification
        - Keywords: Comma-separated terms only, no sentences
        - Cross-reference data for consistency
        - Output valid JSON matching schema
        - Include extraction_confidence and verification_notes
        
        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date", "detected_doc_type"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
    
    def create_verification_prompt(self) -> PromptTemplate:
        """
        Create verification prompt to validate extraction quality.
        Second LLM call for quality assurance.
        """
        template = """
        You are a medical document extraction quality assurance expert. Verify the extracted data for accuracy and consistency.
        
        ORIGINAL DOCUMENT:
        {document_text}
        
        EXTRACTED DATA:
        {extracted_data}
        
        VERIFICATION CHECKLIST:
        
        1. DOCTOR NAME VALIDATION:
           - Check consulting_doctor and referral_doctor fields
           - CRITICAL: Verify each doctor name has proper title (Dr., MD, DO)
           - Flag if name lacks title or seems like patient/admin name
           - Score: High confidence if titled, Low if questionable
        
        2. DOCUMENT TYPE ACCURACY:
           - Verify document_type matches actual document content
           - Check if detected type ({detected_doc_type}) aligns with extracted data
           - Flag mismatches
        
        3. DATA CONSISTENCY:
           - Check dates are in YYYY-MM-DD format and logical (DOB < DOI < RD)
           - Verify diagnosis matches body_part context
           - Check if extracted_recommendation aligns with treatment_plan
        
        4. COMPLETENESS:
           - Flag if critical fields are "Not specified" when document likely contains info
           - Check if body_parts_analysis is populated for multi-body-part documents
        
        5. KEYWORD EXTRACTION QUALITY:
           - Verify extracted_recommendation contains actual keywords (not sentences)
           - Check extracted_decision is concise and medical-term based
           - Validate ur_decision follows strict keyword rules
        
        6. CONFIDENCE ASSESSMENT:
           - Evaluate overall extraction quality
           - Assign confidence score (0.0-1.0)
           - Determine if manual review needed
        
        VERIFICATION OUTPUT:
        {{
          "is_valid": boolean (true if passes validation, false if critical issues),
          "confidence_score": float (0.0-1.0, overall quality),
          "issues_found": [list of specific issues, e.g., "consulting_doctor lacks title: 'John Smith'"],
          "corrections_made": {{field: corrected_value}} (suggest corrections),
          "needs_review": boolean (true if manual review recommended)
        }}
        
        Analyze thoroughly and provide detailed verification.
        
        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "extracted_data", "detected_doc_type"],
            partial_variables={"format_instructions": self.verification_parser.get_format_instructions()},
        )
    
    def verify_extraction(self, document_text: str, extracted_analysis: DocumentAnalysis, detected_doc_type: str) -> VerificationResult:
        """
        Verify extraction using second LLM call.
        Validates doctor names, document type, data consistency.
        """
        try:
            logger.info("🔍 Starting extraction verification (second LLM call)...")
            
            verification_prompt = self.create_verification_prompt()
            chain = verification_prompt | self.llm | self.verification_parser
            
            result = chain.invoke({
                "document_text": document_text[:5000],  # Limit context
                "extracted_data": json.dumps(extracted_analysis.dict(), indent=2),
                "detected_doc_type": detected_doc_type
            })
            
            verification = VerificationResult(**result)
            
            logger.info(f"✅ Verification complete:")
            logger.info(f"   - Valid: {verification.is_valid}")
            logger.info(f"   - Confidence: {verification.confidence_score:.2f}")
            logger.info(f"   - Issues: {len(verification.issues_found)}")
            logger.info(f"   - Needs Review: {verification.needs_review}")
            
            if verification.issues_found:
                for issue in verification.issues_found:
                    logger.warning(f"   ⚠️ {issue}")
            
            return verification
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            # Return default verification result
            return VerificationResult(
                is_valid=False,
                confidence_score=0.5,
                issues_found=[f"Verification failed: {str(e)}"],
                corrections_made={},
                needs_review=True
            )
    
    def apply_corrections(self, analysis: DocumentAnalysis, verification: VerificationResult) -> DocumentAnalysis:
        """
        Apply corrections from verification to extracted analysis.
        Validates and corrects doctor names specifically.
        """
        try:
            # Apply suggested corrections
            for field, corrected_value in verification.corrections_made.items():
                if hasattr(analysis, field):
                    logger.info(f"🔧 Applying correction: {field} = {corrected_value}")
                    setattr(analysis, field, corrected_value)
            
            # Validate doctor names specifically
            is_valid_consulting, corrected_consulting = self.validate_doctor_name(analysis.consulting_doctor)
            if not is_valid_consulting:
                analysis.consulting_doctor = corrected_consulting
                verification.issues_found.append(f"Corrected consulting_doctor: {corrected_consulting}")
            
            is_valid_referral, corrected_referral = self.validate_doctor_name(analysis.referral_doctor)
            if not is_valid_referral:
                analysis.referral_doctor = corrected_referral
                verification.issues_found.append(f"Corrected referral_doctor: {corrected_referral}")
            
            # Update verification metadata
            analysis.extraction_confidence = verification.confidence_score
            analysis.verified = True
            analysis.verification_notes = verification.issues_found
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Correction application failed: {str(e)}")
            return analysis
    
    def extract_document_data_with_reasoning(self, document_text: str) -> DocumentAnalysis:
        """
        Enhanced extraction with 3-stage chain:
        Stage 1: Document type detection (modular detector)
        Stage 2: Comprehensive extraction (main LLM)
        Stage 3: Verification and correction (verification LLM)
        """
        try:
            logger.info("🚀 Starting 3-stage extraction chain...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # STAGE 1: Document type detection using modular detector
            logger.info("📄 Stage 1: Document type detection")
            detected_doc_type = self.detect_document_type(document_text)
            
            # STAGE 2: Comprehensive extraction
            logger.info("🔍 Stage 2: Comprehensive extraction")
            prompt = self.create_enhanced_extraction_prompt(detected_doc_type)
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "document_text": document_text,
                "current_date": current_date,
                "detected_doc_type": detected_doc_type
            })
            
            initial_analysis = DocumentAnalysis(**result)
            
            # Log initial extraction results
            logger.info(f"✅ Initial extraction completed:")
            logger.info(f"   - Patient: {initial_analysis.patient_name}")
            logger.info(f"   - Document Type: {initial_analysis.document_type}")
            logger.info(f"   - Consulting Doctor: {initial_analysis.consulting_doctor}")
            logger.info(f"   - Referral Doctor: {initial_analysis.referral_doctor}")
            logger.info(f"   - Extraction Confidence: {initial_analysis.extraction_confidence:.2f}")
            
            # STAGE 3: Verification and correction
            logger.info("✅ Stage 3: Verification and correction")
            verification = self.verify_extraction(document_text, initial_analysis, detected_doc_type)
            
            # Apply corrections
            final_analysis = self.apply_corrections(initial_analysis, verification)
            
            logger.info(f"🎉 3-stage extraction chain completed:")
            logger.info(f"   - Final Confidence: {final_analysis.extraction_confidence:.2f}")
            logger.info(f"   - Verified: {final_analysis.verified}")
            logger.info(f"   - Issues Found: {len(final_analysis.verification_notes)}")
            logger.info(f"   - Needs Review: {verification.needs_review}")
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ 3-stage extraction chain failed: {str(e)}")
            return self.create_fallback_analysis()
    
    def generate_brief_summary(self, document_text: str) -> str:
        """Generate brief summary (unchanged)"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            summary_prompt = PromptTemplate(
                template="""
                Generate a concise 1-2 sentence professional summary of this medical document.
                
                DOCUMENT TEXT:
                {document_text}
                
                CURRENT DATE: {current_date}
                
                Focus: Patient condition, key findings, recommendations. Use clinical language.
                
                {format_instructions}
                """,
                input_variables=["document_text", "current_date"],
                partial_variables={"format_instructions": self.brief_summary_parser.get_format_instructions()},
            )
            
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({"document_text": document_text, "current_date": current_date})
            
            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated summary: {brief_summary}")
            return brief_summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {str(e)}")
            return "Brief summary unavailable"
    
    def create_fallback_analysis(self) -> DocumentAnalysis:
        """Create fallback analysis when extraction fails (unchanged)"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=timestamp,
            doi=timestamp,
            status="Not specified",
            rd=timestamp,
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="Not specified",
            key_concern="Not specified",
            extracted_recommendation="Not specified",
            extracted_decision="Not specified",
            ur_decision="Not specified",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor="Not specified",
            referral_doctor="Not specified",
            ai_outcome="Insufficient data; full evaluation needed",
            document_type="medical_document",
            summary_points=["Not specified"],
            date_reasoning=None,
            is_task_needed=False,
            extraction_confidence=0.0,
            verified=False,
            verification_notes=["Fallback analysis - extraction failed"]
        )
    
    def get_date_reasoning(self, document_text: str) -> Dict[str, Any]:
        """Get date reasoning results (unchanged)"""
        try:
            full_analysis = self.extract_document_data_with_reasoning(document_text)
            return {
                "date_reasoning": full_analysis.date_reasoning.dict() if full_analysis.date_reasoning else {}
            }
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}}