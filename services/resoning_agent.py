
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime, timedelta
import re, json
import logging
from services.report_analyzer import ReportAnalyzer

from config.settings import CONFIG

logger = logging.getLogger("document_ai")

# Pydantic models (enhanced with extracted keywords for recommendations/decision/outcome)
class DateReasoning(BaseModel):
    """Structured reasoning about dates found in document"""
    extracted_dates: List[str] = Field(..., description="All dates found in the document in YYYY-MM-DD format")
    date_contexts: Dict[str, str] = Field(..., description="Context around each date found")
    reasoning: str = Field(..., description="Step-by-step reasoning for date assignments")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each date assignment (0.0-1.0)")
    predicted_assignments: Dict[str, str] = Field(..., description="Predicted date assignments")

class BodyPartAnalysis(BaseModel):
    """Analysis for a specific body part"""
    body_part: str = Field(..., description="Specific body part involved")
    diagnosis: str = Field(..., description="Diagnosis for this body part")
    key_concern: str = Field(..., description="Key concern for this body part")
    extracted_recommendation: str = Field(..., description="Recommendations for this body part")
    adls_affected: str = Field(..., description="ADLs affected for this body part")
    work_restrictions: str = Field(..., description="Work restrictions for this body part")

class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema (enhanced)"""
    patient_name: str = Field(..., description="Full name of the patient")
    claim_number: str = Field(..., description="Claim number. Use 'Not specified' if not found")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    doi: str = Field(..., description="Date of injury in YYYY-MM-DD format")
    status: str = Field(..., description="Current status: normal, urgent, critical, etc.")
    rd: str = Field(..., description="Report date in YYYY-MM-DD format")
    
    # Single body part (backward compatibility) or multiple body parts
    body_part: str = Field(..., description="Primary body part involved")
    body_parts_analysis: List[BodyPartAnalysis] = Field(default=[], description="Detailed analysis for each body part")
    
    diagnosis: str = Field(..., description="Primary diagnosis and key findings")
    key_concern: str = Field(..., description="Main clinical concern in 2-3 words")
    extracted_recommendation: str = Field(..., description="Extracted key recommendation keywords/phrases")
    extracted_decision: str = Field(..., description="Extracted key decision/judgment keywords/phrases")
    ur_decision: str = Field(..., description="Extracted UR decision keywords/phrases")
    ur_denial_reason: Optional[str] = Field(None, description="UR denial reason if applicable")
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
    consulting_doctor: str = Field(default="Not specified", description="Name of consultant doctor")
    ai_outcome: str = Field(..., description="AI-generated key outcome prediction keywords/phrases")
    document_type: str = Field(..., description="Type of document")
    summary_points: List[str] = Field(..., description="3-5 key points, each 2-3 words")
    date_reasoning: Optional[DateReasoning] = Field(None, description="Reasoning behind date assignments")

class BriefSummary(BaseModel):
    """Structured brief summary of the report"""
    brief_summary: str = Field(..., description="A concise 1-2 sentence summary of the entire report")

# State schema for LangGraph workflow (unchanged)
class ReasoningState(TypedDict, total=False):
    document_text: str
    document_type: str
    current_date: str
    regex_dates: list
    llm_date_analysis: dict
    extraction_complete: bool
    all_dates: dict
    date_clues: dict
    context_analysis_complete: bool
    date_reasoning: dict
    reasoning_complete: bool
    final_date_assignments: dict
    date_reasoning_complete: bool
    validated_date_assignments: dict

class EnhancedReportAnalyzer(ReportAnalyzer):
    """Enhanced analyzer with comprehensive document reasoning, extracted keywords for recommendations/decision/outcome, consulting doctor (optimized to single LLM call)"""
    
    def __init__(self):
        super().__init__()
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)
    
    def create_enhanced_extraction_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Perform a SINGLE, DEEP, COMPREHENSIVE analysis of this medical document. Analyze the entire text holistically for structure, patterns, clinical context, patient info, urgency, workers comp indicators, dates, consulting doctor, extracted recommendation/decision/outcome keywords. Extract ONLY from document text if present; otherwise use 'Not specified'. Do not make separate calls—reason step-by-step internally and output everything in one structured JSON.

        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        DEEP ANALYSIS INSTRUCTIONS (Perform ALL in one pass):
        
        1. DEEP STRUCTURE & PATTERN ANALYSIS:
           - Scan for sections: Patient Demographics, Subjective, Objective, Assessment, Plan, Findings, Impressions, Treatment, Follow-up.
           - Detect document type: progress_report (structured S/O/A/P), imaging_report (findings/impressions), procedure_note (injections/procedures), consultation_note (referrals), initial_evaluation (new patient), or medical_document.
           - Identify workers comp: Look for "WRKCMP", "Work Comp", "State Comp", "industrial injury", claim references.
           - Assess urgency: Pain scales (7+/10=urgent), symptoms (severe/acute), findings (fracture/infection= critical). Levels: normal, elevated, urgent, critical.
           - Functional impact: ADLs (sitting/standing/walking limits), work restrictions (light duty/no lifting).
        
        2. PATIENT & CLAIM EXTRACTION (DEEP FOCUS ON CLAIM - EXTRACT IF PRESENT):
           - Patient Name: Full name from demographics (e.g., "Patient: John Doe" → "John Doe"). If not present, "Not specified".
           - DOB: Explicit "DOB:", "Date of Birth:", birth patterns (e.g., "Born 01/01/1980"). If not present, "Not specified".
           - CLAIM NUMBER (CRITICAL RULE): Deeply scan for keywords like "CLAIM", "Claim #", "Claim Number", "CL#" exactly near a number (e.g., "Claim #12345" → "12345"). If keyword present + number follows/nearby (within 20 chars), extract as claim_number. If no keyword, use "Not specified" even if numbers exist. Prioritize this pattern over assumptions.
           - DOI: Injury/event dates in history/workers comp context. If not present, "Not specified".
        
        3. CLINICAL & FUNCTIONAL EXTRACTION (EXTRACT IF PRESENT):
           - Status: From urgency analysis (e.g., high pain → "urgent"). If not inferable, "Not specified".
           - Diagnosis: Primary condition + 2-3 key objective findings (5-10 words total). If not present, "Not specified".
           - Key Concern: Main issue in 2-3 words (e.g., "Chronic back pain"). If not present, "Not specified".
           
           - BODY PART ANALYSIS (MULTIPLE SUPPORT):
             * Primary Body Part: Extract the main body part involved (e.g., "lumbar spine", "right knee"). If not present, "Not specified".
             * Multiple Body Parts: If document mentions multiple distinct body parts (e.g., "right shoulder and left knee", "cervical and lumbar spine"), analyze EACH separately in body_parts_analysis array. For each body part include:
               - body_part: Specific body part name
               - diagnosis: Diagnosis specific to this body part
               - key_concern: Key concern for this body part  
               - extracted_recommendation: Recommendations specific to this body part
               - adls_affected: ADLs affected by this body part
               - work_restrictions: Work restrictions for this body part
           
           - ADLs Affected: Limited activities in 2-3 words (e.g., "Prolonged sitting"). If not present, "Not specified".
           - Work Restrictions: Limitations in 2-3 words (e.g., "No heavy lifting"). If not present, "Not specified".
        
        4. EXTRACTED RECOMMENDATION KEYWORDS (STRICT KEYWORD-BASED EXTRACTION IF PRESENT):
           - STRICTLY keyword-based: Scan the entire document text for presence of recommendation-related keywords such as 'recommend', 'recommended', 'recommendation', 'plan', 'plans', 'follow-up', 'follow up', 'therapy', 'PT', 'medication', 'surgery', 'consult', 'referral', or similar clinical action terms in Plan/Assessment/Follow-up sections.
           - ONLY if one or more of these keywords are found in the document text: Extract the immediate key keywords/phrases directly following or around them (comma-separated, e.g., 'PT twice weekly, follow-up 4 weeks, surgical consult'). No full sentences—just core terms. Keep concise.
           - If NONE of these keywords are present anywhere in the document text after a thorough scan, set to 'Not specified'. Do not infer or generate if keywords absent.
        
        5. EXTRACTED DECISION KEYWORDS (DIRECT EXTRACTION USING MEDICAL TERMS IF PRESENT):
           - Deeply scan for decisions or clinical judgments in Assessment/Plan/Impressions sections: Extract key keywords/phrases only using medical terms (comma-separated, e.g., "conservative management, no surgery, initiate meds"). Look for "Decision:", "Plan:", "Judgment:", "Proceed with", etc. No full sentences—just core medical terms. If multiple, list concisely. If none found after thorough scan, "Not specified".
        
        6. CONSULTING DOCTOR EXTRACTION (DEEP SCAN IF PRESENT):
           - Deeply scan entire document for consultant doctor name: Look in signatures (e.g., "Consulting MD: Dr. Jane Smith"), consultations (e.g., "Consulted with Dr. Robert Lee for orthopedics"), referrals (e.g., "Referred to consultant Dr. Emily Chen"), or explicit mentions of consultants/specialists. Extract full names (first + last) where possible. Prioritize 'consultant' or specialist roles. List unique name. If none, empty list.
        
        6.5. EXTRACTED UR DECISION KEYWORDS & DENIAL REASON (STRICT KEYWORD-BASED EXTRACTION IF PRESENT):
            - STRICTLY keyword-based: Deeply scan the entire document text for UR (Utilization Review) or prior authorization-related keywords such as 'Utilization Review', 'UR', 'prior authorization', 'prior auth', 'PA', 'approved', 'denied', 'coverage decision', 'authorization status', 'insurance review' in Plan/Assessment/Billing/Authorization sections.
            - ONLY if one or more of these keywords are found in the document text: Extract the immediate key keywords/phrases directly following or around them (comma-separated, e.g., 'approved 12 PT sessions, denied surgical intervention, PA required for MRI'). No full sentences—just core terms. Keep concise.
            - If the extraction includes denial-related terms ('denied', 'not approved', 'rejected', etc.), also extract the specific denial reason: Look for phrases explaining why (e.g., 'denied due to lack of medical necessity', 'not covered under policy'). Set ur_denial_reason to a concise summary of the reason (1-2 sentences or key phrases). If no clear reason or not a denial, set to None.
            - If NONE of these keywords are present anywhere in the document text after a thorough scan, set ur_decision to 'Not specified' and ur_denial_reason to None. Do not infer or generate if keywords absent.
        
        7. AI OUTCOME KEYWORDS (GENERATED BASED ON ANALYSIS):
           - Based on deep analysis of diagnosis, status, key_concern, extracted_recommendation, extracted_decision, ur_decision, ur_denial_reason, and consulting_doctor: Generate key outcome prediction keywords/phrases only (comma-separated, e.g., "full recovery 6 weeks, monitor pain, low risk"). Straightforward for doctor, no sentences—just core terms tied to evidence.
        
        8. DEEP DATE ANALYSIS & REASONING (Integrated - EXTRACT IF PRESENT):
           - Extract ALL dates: Convert MM/DD/YYYY etc. to YYYY-MM-DD. Ignore non-dates (ages, IDs). If no dates, empty lists and note in reasoning.
           - Contexts: 50-100 chars around each.
           - Reasoning: Step-by-step—classify as DOB (birth context), DOI (injury), RD (report/signature/end). Use document flow (early=DOB/DOI, late=RD). Handle relatives (e.g., "2 weeks ago" → subtract from current_date).
           - Confidence: High (labeled), Medium (contextual), Low (ambiguous). Scores 0.0-1.0.
           - Assignments: Predict DOB/DOI/RD based on norms; set rd to latest/most recent if unclear.
           - If no dates: Empty lists, note in reasoning.
        
        9. DOCUMENT OVERVIEW (EXTRACT IF PRESENT):
           - Document Type: From structure/patterns. If unclear, "medical_document".
           - Summary Points: 3-5 key points, each 2-3 words (e.g., "Lumbar strain diagnosed", "Pain reduced to 4/10"). If none, ["Not specified"].
        
        RULES FOR ACCURACY & PRECISION:
        - Extract EVERYTHING directly from document text if present; otherwise "Not specified" or empty. No assumptions or generations except for AI outcome.
        - Workers Comp: Claim only if keyword + number; prioritize injury/DOI context.
        - Holistic: Cross-reference (e.g., claim near DOI → workers comp; consulting_doctor with extracted keywords → inform AI outcome; ur_decision with extracted_recommendation → flag coverage issues; ur_denial_reason with ur_decision → highlight denial impacts).
        - Keywords: Extract/list as comma-separated terms only—strictly no full sentences or narratives.
        - Consultants: Focus on consultant/specialist roles; extract verifiable names from context; no assumptions.
        - AI Outcome Keywords: Always generate—concise terms only, straight to doctor, base on evidence from extracted data.
        - Recommendation Keywords: STRICTLY only if specified keywords present; else 'Not specified'.
        - UR Decision Keywords: STRICTLY only if specified UR keywords present; else 'Not specified'.
        - Denial Reason: ONLY if denial indicated in ur_decision; extract concisely from text; else None.
        - Body Parts: If multiple body parts mentioned, analyze each separately in body_parts_analysis array.
        - Output ONLY valid JSON matching the schema. Include date_reasoning and all fields fully.

        OUTPUT FORMAT:
        - body_part: Always include the primary body part
        - body_parts_analysis: Array of detailed analysis for each distinct body part mentioned
        - If only one body part: body_parts_analysis should contain one entry matching the primary body_part
        - If multiple body parts: body_parts_analysis should contain entries for each, and body_part should be the primary/most significant one

        {format_instructions}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
    
    def extract_document_data_with_reasoning(self, document_text: str) -> DocumentAnalysis:
        """Enhanced extraction with comprehensive document reasoning, extracted keyword recs/decision/outcome, consulting doctor (optimized to single LLM call)"""
        try:
            logger.info(f"🔍 Starting enhanced deep document analysis (single LLM call)...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # SINGLE comprehensive chain: Handles patterns, extraction, dates, doctors, extracted keywords in one prompt/output
            prompt = self.create_enhanced_extraction_prompt()
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "document_text": document_text,
                "current_date": current_date
            })
            
            # Validate and log
            logger.info(f"🔍 LLM extraction result keys: {list(result.keys())}")
            if 'date_reasoning' in result and result['date_reasoning']:
                logger.info(f"✅ Date reasoning integrated: {len(result['date_reasoning']['extracted_dates'])} dates found")
            if 'consulting_doctor' in result:
                logger.info(f"✅ Consulting doctor extracted: {result['consulting_doctor']}")
            if 'extracted_recommendation' in result:
                logger.info(f"✅ Extracted recommendation keywords: {result['extracted_recommendation'][:50]}...")
            if 'extracted_decision' in result:
                logger.info(f"✅ Extracted decision keywords: {result['extracted_decision'][:50]}...")
            if 'ur_decision' in result:
                logger.info(f"✅ Extracted UR decision keywords: {result['ur_decision'][:50]}...")
            if 'ur_denial_reason' in result and result['ur_denial_reason']:
                logger.info(f"✅ UR denial reason extracted: {result['ur_denial_reason'][:50]}...")
            if 'ai_outcome' in result:
                logger.info(f"✅ AI outcome keywords generated: {result['ai_outcome'][:50]}...")
            if 'body_parts_analysis' in result:
                logger.info(f"✅ Body parts analysis: {len(result['body_parts_analysis'])} body parts found")
            
            final_analysis = DocumentAnalysis(**result)
            
            logger.info(f"✅ Enhanced extraction completed (1 LLM call)")
            logger.info(f"📊 Key Results:")
            logger.info(f"   - Patient: {final_analysis.patient_name}")
            logger.info(f"   - Claim: {final_analysis.claim_number}")
            logger.info(f"   - Diagnosis: {final_analysis.diagnosis}")
            logger.info(f"   - Status: {final_analysis.status}")
            logger.info(f"   - Body Parts: {len(final_analysis.body_parts_analysis)} analyzed")
            logger.info(f"   - Extracted Recommendation: {final_analysis.extracted_recommendation}")
            logger.info(f"   - Extracted Decision: {final_analysis.extracted_decision}")
            logger.info(f"   - Extracted UR Decision: {final_analysis.ur_decision}")
            logger.info(f"   - UR Denial Reason: {final_analysis.ur_denial_reason}")
            logger.info(f"   - Consulting Doctor: {final_analysis.consulting_doctor}")
            logger.info(f"   - Body Part: {final_analysis.body_part}")
            logger.info(f"   - AI Outcome: {final_analysis.ai_outcome}")
            logger.info(f"   - Document Type: {final_analysis.document_type}")
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ Enhanced document analysis failed: {str(e)}")
            return self.create_fallback_analysis()
    
    def generate_brief_summary(self, document_text: str) -> str:
        """Generate brief summary with contextual understanding (single LLM call)"""
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
            consulting_doctor=None,
            ai_outcome="Insufficient data; full evaluation needed",
            document_type="medical_document",
            summary_points=["Not specified"],
            date_reasoning=None
        )
    
    def get_date_reasoning(self, document_text: str) -> Dict[str, Any]:
        """Get date reasoning results without full document extraction (calls full extraction for consistency)"""
        try:
            # For consistency, run full extraction and return date_reasoning part
            full_analysis = self.extract_document_data_with_reasoning(document_text)
            return {
                "date_reasoning": full_analysis.date_reasoning.dict() if full_analysis.date_reasoning else {}
            }
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}}