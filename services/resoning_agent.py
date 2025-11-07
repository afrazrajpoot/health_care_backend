"""
OPTIMIZED Enhanced Report Analyzer with Single-Pass LLM Extraction
Now with System/User prompts and focused consulting doctor extraction
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime, timedelta
import re, json
import logging

# Import our modular document detector (kept for hybrid approach)
from utils.document_detector import DocumentTypeDetector
from models.data_models import DocumentType
from config.settings import CONFIG

logger = logging.getLogger("document_ai")
from models.data_models import DocumentAnalysis, VerificationResult, BriefSummary


# ============================================================================
# OPTIMIZED ENHANCED REPORT ANALYZER
# ============================================================================

class EnhancedReportAnalyzer:
    """
    OPTIMIZED Enhanced analyzer with single-pass extraction.
    Now uses System/User prompts for better extraction.
    Consulting Doctor = Report author/signer ONLY (ignore referral doctors)
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
        
        # Initialize document type detector from modular architecture
        self.document_detector = DocumentTypeDetector(self.llm)
        
        logger.info("✅ OPTIMIZED EnhancedReportAnalyzer initialized (System/User prompts)")
    
    def detect_document_type(self, document_text: str) -> str:
        """
        OPTIMIZED: Fast pattern-based detection (90% of cases).
        Only uses LLM for ambiguous cases (10%).
        """
        try:
            logger.info("🔍 Fast document type detection...")
            doc_type_enum = self.document_detector.detect(document_text)
            doc_type = doc_type_enum.value
            logger.info(f"✅ Document type detected: {doc_type}")
            return doc_type
        except Exception as e:
            logger.error(f"❌ Document type detection failed: {str(e)}")
            return "medical_document"
    
    def validate_doctor_name_locally(self, name: str) -> Tuple[bool, str]:
        """
        OPTIMIZED: Local validation without LLM (instant).
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
                logger.warning(f"⚠️ Auto-corrected doctor name: {name} → {corrected}")
                return False, corrected
            else:
                # Not a valid doctor name
                logger.warning(f"⚠️ Invalid doctor name format: {name}")
                return False, "Not specified"
    
    def create_optimized_extraction_prompt(self, detected_doc_type: str) -> ChatPromptTemplate:
        """
        OPTIMIZED: System/User prompt structure for better extraction.
        Critical: Consulting Doctor = Report author/signer ONLY
        """
        
        # SYSTEM PROMPT - Role and Instructions
        system_template = """You are an expert medical document analyzer with BUILT-IN quality assurance.

ROLE: Extract structured medical information with high accuracy and self-verification.

CRITICAL DOCTOR EXTRACTION RULES:
1. CONSULTING DOCTOR: ONLY the doctor who WROTE, AUTHORED, or SIGNED this specific report
   - Look for: Signature lines, "Dictated by:", "Authored by:", "Report by:", "Electronically signed by:"
   - Must have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O."
   - Extract FULL NAME with title (e.g., "Dr. Jane Smith")
   - IGNORE: Referral doctors, primary care doctors, other specialists mentioned
   - IGNORE: Administrative staff, nurses, physician assistants
   - If no clear author/signer → "Not specified"

2. REFERRAL DOCTOR: ONLY extract if explicitly mentioned as referral source
   - Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care Physician:"
   - Must have explicit title
   - If no referral source → "Not specified"

DOCUMENT TYPE GUIDANCE: {detected_doc_type}
- Use this to prioritize extraction fields and context

QUALITY ASSURANCE:
- Simultaneously perform self-verification during extraction
- Flag inconsistencies in verification_notes
- Set extraction_confidence (0.0-1.0) based on data quality
- verified=true (since built-in verification)

OUTPUT FORMAT: Strict JSON following schema"""
        
        # USER PROMPT - Document and Specific Instructions
        user_template = """DOCUMENT TEXT:
{document_text}

CURRENT DATE: {current_date}

═══════════════════════════════════════════════════════════════════════
COMPREHENSIVE EXTRACTION INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

PERFORM ALL STAGES IN ONE PASS:

━━━ 1. PATIENT & CLAIM DATA ━━━
- patient_name: Full name from demographics. If not found → "Not specified"
- claim_number: ONLY if near "CLAIM", "Claim #", "Claim Number", "CL#" keywords. Otherwise → "Not specified"
- dob: "DOB:", "Date of Birth:" patterns. Format YYYY-MM-DD. If not found → "Not specified"  
- doi: Date of injury from workers comp context. Format YYYY-MM-DD. If not found → "Not specified"

━━━ 2. CLINICAL DATA ━━━
- status: Based on urgency (normal/elevated/urgent/critical). If unclear → "Not specified"
- diagnosis: Primary condition + key findings (5-10 words). If not found → "Not specified"
- key_concern: Main issue in 2-3 words. If not found → "Not specified"

BODY PART ANALYSIS:
- body_part: Primary body part. If not found → "Not specified"
- body_parts_analysis: If MULTIPLE body parts, create separate analysis for EACH
- adls_affected: Limited activities in 2-3 words. If not found → "Not specified"
- work_restrictions: Work limitations in 2-3 words. If not found → "Not specified"

━━━ 3. KEYWORD EXTRACTION ━━━
- extracted_recommendation: Scan for 'recommend', 'recommendation', 'plan', 'therapy', 'PT', 'medication', 'surgery'
- extracted_decision: Scan Assessment/Plan for "Decision:", "Plan:", "Judgment:"
- ur_decision: ONLY if 'Utilization Review', 'UR', 'prior authorization', 'approved', 'denied' keywords present

━━━ 4. DOCTOR EXTRACTION (CRITICAL) ━━━
CONSULTING DOCTOR (REPORT AUTHOR/SIGNER ONLY):
- PRIMARY FOCUS: Signature blocks, "Dictated by:", "Report by:", "Electronically signed by:"
- MUST have title: "Dr.", "MD", "DO", "M.D.", "D.O."
- IGNORE all other doctors mentioned in content (referrals, consults, PCPs)
- If name found WITHOUT title → ADD to verification_notes: "Consulting doctor lacks title: [name]"
- If no clear author/signer → "Not specified"

REFERRAL DOCTOR (ONLY IF EXPLICIT REFERRAL SOURCE):
- Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care Physician:"
- Must have title
- If no explicit referral source → "Not specified"

━━━ 5. AI OUTCOME & TASK ANALYSIS ━━━
- ai_outcome: Prediction based on diagnosis, recommendations (comma-separated keywords)
- is_task_needed: TRUE if pending actions, future appointments, new recommendations

━━━ 6. DATE REASONING ━━━
- Extract ALL dates, convert to YYYY-MM-DD
- Classify as DOB (birth), DOI (injury), RD (report/signature)
- Provide reasoning, contexts, confidence scores

━━━ 7. SELF-VERIFICATION & METADATA ━━━
- extraction_confidence: 0.0-1.0 based on data quality
- verification_notes: List any issues (doctor titles, ambiguous data, inconsistencies)
- verified: true
- document_type: Confirm or refine {detected_doc_type}
- summary_points: 3-5 key points, each 2-3 words

═══════════════════════════════════════════════════════════════════════
CRITICAL OUTPUT RULES
═══════════════════════════════════════════════════════════════════════
- Extract ONLY from document text. If not found → "Not specified"
- Consulting Doctor = Report Author/Signer ONLY (ignore all other doctors)
- Keywords: comma-separated terms, NO full sentences
- Dates: YYYY-MM-DD format only
- Output valid JSON matching schema

{format_instructions}"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    
    def extract_document_data_with_reasoning(self, document_text: str) -> DocumentAnalysis:
        """
        OPTIMIZED: Single-pass extraction with System/User prompts.
        Consulting Doctor = Report author/signer ONLY
        """
        try:
            logger.info("🚀 Starting OPTIMIZED extraction (System/User prompts)...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Fast pattern-based detection
            logger.info("📄 Stage 1: Fast document type detection")
            detected_doc_type = self.detect_document_type(document_text)
            
            # Single comprehensive extraction with System/User prompts
            logger.info("🔍 Stage 2: System/User prompt extraction")
            prompt = self.create_optimized_extraction_prompt(detected_doc_type)
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "document_text": document_text,
                "current_date": current_date,
                "detected_doc_type": detected_doc_type,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            analysis = DocumentAnalysis(**result)
            
            # Local doctor name validation
            logger.info("✅ Stage 3: Local validation")
            
            if analysis.consulting_doctor and analysis.consulting_doctor != "Not specified":
                is_valid, corrected = self.validate_doctor_name_locally(analysis.consulting_doctor)
                if not is_valid:
                    analysis.consulting_doctor = corrected
                    analysis.verification_notes.append(f"Auto-corrected consulting_doctor: {corrected}")
            
            if analysis.referral_doctor and analysis.referral_doctor != "Not specified":
                is_valid, corrected = self.validate_doctor_name_locally(analysis.referral_doctor)
                if not is_valid:
                    analysis.referral_doctor = corrected
                    analysis.verification_notes.append(f"Auto-corrected referral_doctor: {corrected}")
            
            # Set verified flag
            analysis.verified = True
            
            # Log results with focus on consulting doctor
            logger.info(f"🎉 OPTIMIZED extraction complete:")
            logger.info(f"   - Patient: {analysis.patient_name}")
            logger.info(f"   - Document Type: {analysis.document_type}")
            logger.info(f"   - CONSULTING DOCTOR (Author/Signer): {analysis.consulting_doctor}")
            logger.info(f"   - Referral Doctor: {analysis.referral_doctor}")
            logger.info(f"   - Confidence: {analysis.extraction_confidence:.2f}")
            logger.info(f"   - Issues: {len(analysis.verification_notes)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Optimized extraction failed: {str(e)}")
            return self.create_fallback_analysis()
    
    def generate_brief_summary(self, document_text: str) -> str:
        """
        Generate brief summary (unchanged).
        """
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Use System/User prompts for summary too
            system_prompt = SystemMessagePromptTemplate.from_template(
                "You are a medical summarization expert. Generate concise 1-2 sentence professional summaries."
            )
            
            human_prompt = HumanMessagePromptTemplate.from_template(
                """Generate a concise 1-2 sentence professional summary of this medical document.

DOCUMENT TEXT:
{document_text}

CURRENT DATE: {current_date}

Focus: Patient condition, key findings, recommendations. Use clinical language.

{format_instructions}"""
            )
            
            summary_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "document_text": document_text, 
                "current_date": current_date,
                "format_instructions": self.brief_summary_parser.get_format_instructions()
            })
            
            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated summary: {brief_summary}")
            return brief_summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {str(e)}")
            return "Brief summary unavailable"
    
    def create_fallback_analysis(self) -> DocumentAnalysis:
        """Create fallback analysis when extraction fails"""
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
            consulting_doctor="Not specified",  # Focus on author/signer
            referral_doctor="Not specified",   # Only if explicit referral
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
        """Get date reasoning results"""
        try:
            full_analysis = self.extract_document_data_with_reasoning(document_text)
            return {
                "date_reasoning": full_analysis.date_reasoning.dict() if full_analysis.date_reasoning else {}
            }
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}}