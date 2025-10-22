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

# Pydantic models (unchanged)
class DateReasoning(BaseModel):
    """Structured reasoning about dates found in document"""
    extracted_dates: List[str] = Field(..., description="All dates found in the document in YYYY-MM-DD format")
    date_contexts: Dict[str, str] = Field(..., description="Context around each date found")
    reasoning: str = Field(..., description="Step-by-step reasoning for date assignments")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each date assignment (0.0-1.0)")
    predicted_assignments: Dict[str, str] = Field(..., description="Predicted date assignments")

class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema"""
    patient_name: str = Field(..., description="Full name of the patient")
    claim_number: str = Field(..., description="Claim number. Use 'Not specified' if not found")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    doi: str = Field(..., description="Date of injury in YYYY-MM-DD format")
    status: str = Field(..., description="Current status: normal, urgent, critical, etc.")
    rd: str = Field(..., description="Report date in YYYY-MM-DD format")
    diagnosis: str = Field(..., description="Primary diagnosis and key findings (comma-separated if multiple, 5-10 words total)")
    key_concern: str = Field(..., description="Main clinical concern in 2-3 words")
    next_step: str = Field(..., description="Recommended next steps in 2-3 words")
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
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
    """Enhanced analyzer with comprehensive document reasoning (optimized to single LLM call)"""
    
    def __init__(self):
        super().__init__()
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)
    
    def create_enhanced_extraction_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Perform a SINGLE, DEEP, COMPREHENSIVE analysis of this medical document. Analyze the entire text holistically for structure, patterns, clinical context, patient info, urgency, workers comp indicators, dates, and extractions. Do not make separate calls—reason step-by-step internally and output everything in one structured JSON.

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
        
        2. PATIENT & CLAIM EXTRACTION (DEEP FOCUS ON CLAIM):
           - Patient Name: Full name from demographics (e.g., "Patient: John Doe" → "John Doe"). Avoid non-names.
           - DOB: Explicit "DOB:", "Date of Birth:", birth patterns (e.g., "Born 01/01/1980").
           - CLAIM NUMBER (CRITICAL RULE): Deeply scan for keywords like "CLAIM", "Claim #", "Claim Number", "CL#" exactly near a number (e.g., "Claim #12345" → "12345"). If keyword present + number follows/nearby (within 20 chars), extract as claim_number. If no keyword, use "Not specified" even if numbers exist. Prioritize this pattern over assumptions.
           - DOI: Injury/event dates in history/workers comp context.
        
        3. CLINICAL & FUNCTIONAL EXTRACTION:
           - Status: From urgency analysis (e.g., high pain → "urgent").
           - Diagnosis: Primary condition + 2-3 key objective findings (5-10 words total).
           - Key Concern: Main issue in 2-3 words (e.g., "Chronic back pain").
           - Next Steps: Recommendations in 2-3 words (e.g., "Follow-up in 2 weeks").
           - ADLs Affected: Limited activities in 2-3 words (e.g., "Prolonged sitting").
           - Work Restrictions: Limitations in 2-3 words (e.g., "No heavy lifting").
        
        4. DEEP DATE ANALYSIS & REASONING (Integrated):
           - Extract ALL dates: Convert MM/DD/YYYY etc. to YYYY-MM-DD. Ignore non-dates (ages, IDs).
           - Contexts: 50-100 chars around each.
           - Reasoning: Step-by-step—classify as DOB (birth context), DOI (injury), RD (report/signature/end). Use document flow (early=DOB/DOI, late=RD). Handle relatives (e.g., "2 weeks ago" → subtract from current_date).
           - Confidence: High (labeled), Medium (contextual), Low (ambiguous). Scores 0.0-1.0.
           - Assignments: Predict DOB/DOI/RD based on norms; set rd to latest/most recent if unclear.
           - If no dates: Empty lists, note in reasoning.
        
        5. DOCUMENT OVERVIEW:
           - Document Type: From structure/patterns.
           - Summary Points: 3-5 key points, each 2-3 words (e.g., "Lumbar strain diagnosed", "Pain reduced to 4/10").
        
        RULES FOR ACCURACY:
        - REAL medical docs: Chronological flow, standard patterns. Missing = "Not specified" after deep search.
        - Workers Comp: Claim only if keyword + number; prioritize injury/DOI context.
        - Holistic: Cross-reference (e.g., claim near DOI → workers comp).
        - Output ONLY valid JSON matching the schema. Include date_reasoning fully.

        {format_instructions}
        """
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
    
    def extract_document_data_with_reasoning(self, document_text: str) -> DocumentAnalysis:
        """Enhanced extraction with comprehensive document reasoning (optimized to single LLM call)"""
        try:
            logger.info(f"🔍 Starting enhanced deep document analysis (single LLM call)...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # SINGLE comprehensive chain: Handles patterns, extraction, dates in one prompt/output
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
            
            final_analysis = DocumentAnalysis(**result)
            
            logger.info(f"✅ Enhanced extraction completed (1 LLM call)")
            logger.info(f"📊 Key Results:")
            logger.info(f"   - Patient: {final_analysis.patient_name}")
            logger.info(f"   - Claim: {final_analysis.claim_number}")
            logger.info(f"   - Diagnosis: {final_analysis.diagnosis}")
            logger.info(f"   - Status: {final_analysis.status}")
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
            status="normal",
            rd=timestamp,
            diagnosis="Not specified",
            key_concern="Not specified",
            next_step="Not specified",
            adls_affected="Not specified",
            work_restrictions="Not specified",
            document_type="Medical Document",
            summary_points=["Processing completed", "Analysis unavailable"],
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