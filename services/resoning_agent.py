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

# Pydantic models
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

# State schema for LangGraph workflow
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

# Pydantic models for AI-driven pattern analysis
class DocumentTypeDetection(BaseModel):
    document_type: str = Field(..., description="Detected document type")

class SectionIdentification(BaseModel):
    sections_found: List[str] = Field(..., description="Identified sections")

class PatientPatternExtraction(BaseModel):
    patient_name_found: bool = Field(..., description="Whether patient name was found")
    dob_found: bool = Field(..., description="Whether DOB was found")
    claim_number_found: bool = Field(..., description="Whether claim number was found")
    potential_names: List[str] = Field(default=[], description="Potential patient names")
    potential_claims: List[str] = Field(default=[], description="Potential claim numbers")

class UrgencyDetection(BaseModel):
    urgency_level: str = Field(..., description="Detected urgency level")
    pain_indicators: List[str] = Field(default=[], description="Pain indicators")
    critical_findings: List[str] = Field(default=[], description="Critical findings")
    urgency_keywords: List[str] = Field(default=[], description="Urgency keywords")

class WorkersCompDetection(BaseModel):
    is_workers_comp: bool = Field(..., description="Whether it's workers comp")
    indicators: List[str] = Field(default=[], description="Indicators found")
    work_status: str = Field(..., description="Work status")

class DateExtractionAI(BaseModel):
    extracted_dates: List[str] = Field(..., description="Extracted dates in YYYY-MM-DD format")
    date_contexts: Dict[str, str] = Field(..., description="Context for each date")
    reasoning: str = Field(..., description="Step-by-step reasoning")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores")
    predicted_assignments: Dict[str, str] = Field(..., description="Predicted assignments")

class ComprehensiveDocumentReasoningSystem:
    """Enhanced reasoning system that understands medical report patterns and context"""
    
    def __init__(self, llm):
        self.llm = llm
        self.document_type_parser = JsonOutputParser(pydantic_object=DocumentTypeDetection)
        self.sections_parser = JsonOutputParser(pydantic_object=SectionIdentification)
        self.patient_patterns_parser = JsonOutputParser(pydantic_object=PatientPatternExtraction)
        self.urgency_parser = JsonOutputParser(pydantic_object=UrgencyDetection)
        self.workers_comp_parser = JsonOutputParser(pydantic_object=WorkersCompDetection)
        self.date_ai_parser = JsonOutputParser(pydantic_object=DateExtractionAI)
    
    def create_comprehensive_analysis_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Perform DEEP contextual analysis of this medical document to understand its STRUCTURE, PATTERNS, and CLINICAL CONTEXT.
        
        DOCUMENT TEXT:
        {document_text}
        
        CURRENT DATE: {current_date}
        
        CRITICAL ANALYSIS TASKS:
        
        1. DOCUMENT STRUCTURE ANALYSIS:
           - Identify sections: Patient Demographics, Clinical Findings, Diagnosis, Treatment, Follow-up
           - Look for headers, labels, and patterns in the document layout
           - Understand the flow: Patient info → Clinical assessment → Treatment → Recommendations
        
        2. PATIENT INFORMATION PATTERNS:
           - Patient Name: Look for "Patient:", demographics section, near DOB
           - DOB: Look for "DOB:", "Date of Birth:", birth date patterns
           - Claim Number: Look for "Claim #", "Claim Number:", 
        
        3. CLINICAL CONTEXT UNDERSTANDING:
           - Document Type: Progress Note, Imaging Report, Consultation, etc.
           - Status: Urgent/Critical based on symptoms, pain levels, clinical findings
           - Diagnosis: Primary condition and key findings from objective section
           - Key Concerns: Main symptoms or clinical issues mentioned
        
        4. TREATMENT & FOLLOW-UP PATTERNS:
           - Next Steps: Look for "Follow up", "Return to clinic", recommendations
           - ADLs Affected: Activities mentioned as limited or problematic
           - Work Restrictions: "Light duty", "No heavy lifting", work limitations
        
        5. WORKERS COMP CONTEXT (if applicable):
           - Look for "WRKCMP", "Work Comp", "State Comp", industrial injury context
           - Claim number: Must be near "claim" keyword or "claim number"
        
        SPECIFIC PATTERNS TO RECOGNIZE:
        
        - Progress Reports (PR-2): Have structured sections with patient info, subjective, objective, assessment
        - Imaging Reports: Focus on findings, impressions, technical details
        - Injection Notes: Document specific procedures, medications, follow-up timing
        - Workers Comp: Include claim numbers, work status, injury details
        
        EXTRACTION RULES:
        
        - Patient Name: Extract full name from demographics (e.g., "Jhon Doe")
        - Claim Number: Look near "Claim #", but must have a keyword "claim" or "claim number" nearby
        - Status: Infer from clinical urgency (pain 8/10 → urgent, routine findings → normal)
        - Diagnosis: Primary condition + 2-3 key findings from objective assessment
        - Key Concerns: 2-3 word summary of main clinical issues
        - Next Steps: 2-3 word summary of recommendations
        - ADLs: Activities mentioned as affected (sitting, standing, walking, etc.)
        - Work Restrictions: Specific limitations mentioned
        
        REAL-WORLD CONTEXT:
        - This is REAL medical documentation with REAL patients
        - Information follows standard medical documentation patterns
        - Missing information should be marked as "Not specified"
        
        {format_instructions}
        
        Return ONLY valid JSON with your comprehensive analysis.
        """
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
    
    def analyze_document_patterns(self, document_text: str) -> Dict[str, Any]:
        """Analyze document patterns and structure using AI-driven LLM calls for professional reasoning"""
        patterns = {}
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # AI-driven document type detection
            patterns["document_type"] = self._ai_detect_document_type(document_text, current_date)
            
            # AI-driven section identification
            patterns["sections_found"] = self._ai_identify_sections(document_text, current_date)
            
            # AI-driven patient patterns extraction
            patterns["patient_info_patterns"] = self._ai_extract_patient_patterns(document_text, current_date)
            
            # AI-driven clinical urgency detection
            patterns["clinical_urgency_indicators"] = self._ai_detect_urgency_indicators(document_text, current_date)
            
            # AI-driven workers comp detection
            patterns["workers_comp_indicators"] = self._ai_detect_workers_comp_patterns(document_text, current_date)
            
            logger.info(f"🔍 AI-driven pattern analysis completed: {json.dumps(patterns, indent=2)}")
            
        except Exception as e:
            logger.error(f"❌ AI pattern analysis failed: {str(e)}")
            # Fallback to basic static if AI fails
            patterns = self._fallback_pattern_analysis(document_text)
        
        return patterns
    
    def _ai_detect_document_type(self, text: str, current_date: str) -> str:
        """AI-driven document type detection with professional reasoning"""
        prompt = PromptTemplate(
            template="""
            Analyze this medical document and determine its type based on content, structure, and clinical patterns.
            
            DOCUMENT TEXT:
            {document_text}
            
            CURRENT DATE: {current_date}
            
            PROFESSIONAL REASONING:
            - Consider sections like subjective/objective/assessment/plan for progress notes
            - Look for imaging findings/impressions for radiology reports
            - Check for procedure details/medications for injection notes
            - Identify consultation/referral language for consult notes
            - Detect workers comp/claim references for comp-related docs
            
            Output ONLY the document type: progress_report, imaging_report, procedure_note, consultation_note, initial_evaluation, or medical_document.
            Provide brief reasoning in the response, but ensure JSON validity.
            
            {format_instructions}
            """,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.document_type_parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.document_type_parser
        result = chain.invoke({"document_text": text, "current_date": current_date})
        return result["document_type"]
    
    def _ai_identify_sections(self, text: str, current_date: str) -> List[str]:
        """AI-driven section identification with contextual understanding"""
        prompt = PromptTemplate(
            template="""
            Identify key sections in this medical document based on headers, labels, and structural flow.
            
            DOCUMENT TEXT:
            {document_text}
            
            CURRENT DATE: {current_date}
            
            REASONING:
            - Scan for common headers: Patient:, Subjective:, Objective:, Assessment:, Plan:
            - Infer sections from content transitions (e.g., demographics → history → findings)
            - Output unique sections: patient_demographics, diagnosis, treatment_plan, subjective, objective, etc.
            
            {format_instructions}
            """,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.sections_parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.sections_parser
        result = chain.invoke({"document_text": text, "current_date": current_date})
        return result["sections_found"]
    
    def _ai_extract_patient_patterns(self, text: str, current_date: str) -> Dict[str, Any]:
        """AI-driven patient information pattern extraction"""
        prompt = PromptTemplate(
            template="""
            Extract patient information patterns from this medical document with deep contextual analysis.
            
            DOCUMENT TEXT:
            {document_text}
            
            CURRENT DATE: {current_date}
            
            PROFESSIONAL EXTRACTION:
            - Patient Name: Infer full names from demographics, avoid non-names
            - DOB: Detect birth date mentions, formats like MM/DD/YYYY
            - Claim Number: Identify claim number must be near "claim" keyword or "claim number"
            - Flag presence and list potentials with reasoning for accuracy
            
            {format_instructions}
            """,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.patient_patterns_parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.patient_patterns_parser
        result = chain.invoke({"document_text": text, "current_date": current_date})
        return result
    
    def _ai_detect_urgency_indicators(self, text: str, current_date: str) -> Dict[str, Any]:
        """AI-driven clinical urgency detection"""
        prompt = PromptTemplate(
            template="""
            Assess clinical urgency in this medical document using evidence-based reasoning.
            
            DOCUMENT TEXT:
            {document_text}
            
            CURRENT DATE: {current_date}
            
            CLINICAL REASONING:
            - Evaluate pain scales (e.g., 7+/10 = urgent), symptoms (severe/acute), findings (fracture/infection)
            - Classify level: normal, elevated, urgent, critical
            - List supporting indicators for transparency
            
            {format_instructions}
            """,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.urgency_parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.urgency_parser
        result = chain.invoke({"document_text": text, "current_date": current_date})
        return result
    
    def _ai_detect_workers_comp_patterns(self, text: str, current_date: str) -> Dict[str, Any]:
        """AI-driven workers compensation pattern detection"""
        prompt = PromptTemplate(
            template="""
            Detect workers compensation context in this medical document.
            
            DOCUMENT TEXT:
            {document_text}
            
            CURRENT DATE: {current_date}
            
            OCCUPATIONAL REASONING:
            - Look for indicators: WRKCMP, work-related injury, claim number
            - Infer work status: light duty, off work, full duty from limitations
            - Flag as workers_comp if evidence supports
            
            {format_instructions}
            """,
            input_variables=["document_text", "current_date"],
            partial_variables={"format_instructions": self.workers_comp_parser.get_format_instructions()},
        )
        chain = prompt | self.llm | self.workers_comp_parser
        result = chain.invoke({"document_text": text, "current_date": current_date})
        return result
    
    def _fallback_pattern_analysis(self, text: str) -> Dict[str, Any]:
        """Static fallback for pattern analysis if AI fails"""
        logger.warning("⚠️ Using static fallback for pattern analysis")
        text_lower = text.lower()
        
        return {
            "document_type": next((t for t in ["progress_report", "imaging_report", "procedure_note", "consultation_note", "initial_evaluation", "medical_document"] if any(term in text_lower for term in self._get_keywords_for_type(t))), "medical_document"),
            "sections_found": [],  # Simplified
            "patient_info_patterns": {"patient_name_found": bool(re.search(r'Patient:\s*[A-Za-z\s]+', text, re.IGNORECASE)), "dob_found": bool(re.search(r'DOB:\s*\d', text, re.IGNORECASE)), "claim_number_found": bool(re.search(r'Claim\s*#?', text, re.IGNORECASE)), "potential_names": [], "potential_claims": []},
            "clinical_urgency_indicators": {"urgency_level": "normal", "pain_indicators": [], "critical_findings": [], "urgency_keywords": []},
            "workers_comp_indicators": {"is_workers_comp": any(term in text_lower for term in ["wrkcmp", "work comp"]), "indicators": [], "work_status": "unknown"}
        }
    
    def _get_keywords_for_type(self, doc_type: str) -> List[str]:
        """Helper for fallback keywords"""
        keywords = {
            "progress_report": ["progress report", "pr-2", "follow-up"],
            "imaging_report": ["mri", "ct", "x-ray", "imaging"],
            "procedure_note": ["injection", "inject"],
            "consultation_note": ["consultation", "referral"],
            "initial_evaluation": ["initial evaluation", "new patient"]
        }
        return keywords.get(doc_type, [])

class EnhancedReportAnalyzer(ReportAnalyzer):
    """Enhanced analyzer with comprehensive document reasoning"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_system = ComprehensiveDocumentReasoningSystem(self.llm)
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)
    
    def create_enhanced_extraction_prompt(self) -> PromptTemplate:
        template = """
        You are a medical document analysis expert. Perform COMPREHENSIVE analysis of this medical document using deep understanding of medical patterns and context.
        
        DOCUMENT TEXT:
        {document_text}
        
        PATTERN ANALYSIS RESULTS:
        {pattern_analysis}
        
        CURRENT DATE: {current_date}
        
        COMPREHENSIVE EXTRACTION INSTRUCTIONS:
        
        Based on the pattern analysis, extract the following with DEEP CONTEXTUAL UNDERSTANDING:
        
        1. PATIENT INFORMATION:
           - Patient Name: Use the most likely full name from demographics
           - Claim Number: Extract from claim number , must be near "claim" keyword or "claim number"
           - DOB: Extract date of birth from explicit labels
           - DOI: Look for injury dates in workers comp context
        
        2. CLINICAL ASSESSMENT:
           - Status: Use urgency indicators (normal, urgent, critical)
           - Diagnosis: Primary condition + key objective findings
           - Key Concern: 2-3 word summary of main clinical issue
           - Next Steps: Follow-up recommendations in 2-3 words
        
        3. FUNCTIONAL IMPACT:
           - ADLs Affected: Activities limited by condition (2-3 words)
           - Work Restrictions: Work limitations mentioned (2-3 words)
        
        4. DOCUMENT CONTEXT:
           - Document Type: Based on content and structure
           - Summary Points: 3-5 key clinical points (2-3 words each)
        
        CRITICAL: DO NOT include 'date_reasoning' field in your response. Only extract the fields specified above.
        
        CONTEXTUAL REASONING RULES:
        
        - Workers Comp Context: If WRKCMP/claim patterns found, prioritize claim number extraction
        - Urgency Assessment: Use pain levels, critical findings to determine status
        - Name Extraction: Prefer full names from patient demographics section
        - Diagnosis: Focus on objective findings and assessment section
        - Missing Information: Use "Not specified" only after thorough search
        
        REAL-WORLD MEDICAL UNDERSTANDING:
        - This is actual patient care documentation
        - Information follows standard medical documentation flows
        - Clinical context matters for accurate extraction
        
        {format_instructions}
        
        Return ONLY valid JSON with your comprehensive extraction. DO NOT include date_reasoning.
        """
        return PromptTemplate(
            template=template,
            input_variables=["document_text", "pattern_analysis", "current_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
    
    def extract_document_data_with_reasoning(self, document_text: str) -> DocumentAnalysis:
        """Enhanced extraction with comprehensive document reasoning"""
        try:
            logger.info(f"🔍 Starting comprehensive document reasoning...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Step 1: Analyze document patterns and structure (now fully AI-driven)
            pattern_analysis = self.reasoning_system.analyze_document_patterns(document_text)
            logger.info(f"🔍 Pattern analysis: {json.dumps(pattern_analysis, indent=2)}")
            
            # Step 2: Use comprehensive reasoning for extraction
            prompt = self.create_enhanced_extraction_prompt()
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "document_text": document_text,
                "pattern_analysis": json.dumps(pattern_analysis, indent=2),
                "current_date": current_date
            })
            
            # DEBUG: Check what's in the result
            logger.info(f"🔍 LLM extraction result keys: {list(result.keys())}")
            
            # FIXED: Remove date_reasoning from result if LLM included it
            if 'date_reasoning' in result:
                logger.warning("⚠️ LLM included 'date_reasoning' in output; removing to avoid conflict")
                del result['date_reasoning']
            
            # Step 3: Create date reasoning using AI-driven extraction
            date_reasoning = self._create_ai_date_reasoning(document_text, current_date)
            
            # Create final analysis - FIXED: Pass date_reasoning separately
            final_analysis = DocumentAnalysis(
                **result,  # All the fields from LLM except date_reasoning
                date_reasoning=date_reasoning  # Our AI-driven date reasoning
            )
            
            logger.info(f"✅ Comprehensive extraction completed")
            logger.info(f"📊 Extraction Results:")
            logger.info(f"   - Patient: {final_analysis.patient_name}")
            logger.info(f"   - Claim: {final_analysis.claim_number}")
            logger.info(f"   - Diagnosis: {final_analysis.diagnosis}")
            logger.info(f"   - Status: {final_analysis.status}")
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ Comprehensive document analysis failed: {str(e)}")
            return self.create_fallback_analysis()
    
    def _create_ai_date_reasoning(self, document_text: str, current_date: str) -> Optional[DateReasoning]:
        """AI-driven date reasoning with professional step-by-step analysis"""
        try:
            prompt = PromptTemplate(
                template="""
                You are a medical date analysis expert. Perform thorough, evidence-based reasoning to extract and contextualize all dates in this document.
                
                DOCUMENT TEXT:
                {document_text}
                
                CURRENT DATE: {current_date}
                
                PROFESSIONAL DATE EXTRACTION & REASONING:
                
                1. IDENTIFY ALL DATES:
                   - Scan for explicit dates in various formats (MM/DD/YYYY, YYYY-MM-DD, etc.)
                   - Convert all to YYYY-MM-DD standard
                   - Ignore non-date numbers (e.g., ages, claim IDs)
                
                2. CONTEXTUAL ANALYSIS:
                   - For each date, extract 50-100 characters of surrounding context
                   - Classify potential role: DOB (birth patterns), DOI (injury/event), RD (report/signature date)
                   - Consider document flow: earliest = DOI/DOB, latest = RD
                
                3. ASSIGNMENT REASONING:
                   - Step-by-step: Why this date fits DOB/DOI/RD? Evidence from context?
                   - Confidence: High (explicit label), Medium (inferred context), Low (ambiguous)
                   - Predicted assignments based on medical document norms
                
                4. EDGE CASES:
                   - Relative dates (e.g., "2 weeks ago") → Infer from current date
                   - No dates found → Empty lists, note in reasoning
                   - Ambiguities → Flag lowest confidence
                
                REAL-WORLD MEDICAL CONTEXT:
                - Dates follow chronological patient care timeline
                - Report dates often at end/signature
                - Birth/injury dates in demographics/history
                
                Output structured JSON with transparent, professional reasoning.
                
                {format_instructions}
                """,
                input_variables=["document_text", "current_date"],
                partial_variables={"format_instructions": self.reasoning_system.date_ai_parser.get_format_instructions()},
            )
            
            chain = prompt | self.llm | self.reasoning_system.date_ai_parser
            ai_result = chain.invoke({"document_text": document_text, "current_date": current_date})
            
            # Convert to DateReasoning for consistency
            date_reasoning = DateReasoning(**ai_result)
            
            logger.info(f"✅ AI-driven date reasoning completed: {len(ai_result['extracted_dates'])} dates found")
            return date_reasoning
            
        except Exception as e:
            logger.warning(f"❌ AI date reasoning failed: {e}. Falling back to pattern-based.")
            return self._create_date_reasoning_from_patterns({}, document_text)  # Legacy fallback
    
    def _create_date_reasoning_from_patterns(self, pattern_analysis: Dict[str, Any], document_text: str) -> Optional[DateReasoning]:
        """Legacy pattern-based date reasoning (fallback only)"""
        try:
            # Extract dates using simple patterns
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',
                r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'
            ]
            
            extracted_dates = []
            date_contexts = {}
            
            for pattern in date_patterns:
                matches = re.finditer(pattern, document_text)
                for match in matches:
                    date_str = match.group()
                    if date_str not in extracted_dates:
                        extracted_dates.append(date_str)
                        # Get context around the date
                        start = max(0, match.start() - 50)
                        end = min(len(document_text), match.end() + 50)
                        context = document_text[start:end]
                        date_contexts[date_str] = context
            
            # Only return DateReasoning if we actually found dates
            if extracted_dates:
                return DateReasoning(
                    extracted_dates=extracted_dates,
                    date_contexts=date_contexts,
                    reasoning=f"Pattern-based extraction (fallback). Document type: {pattern_analysis.get('document_type', 'unknown')}",
                    confidence_scores={},
                    predicted_assignments={}
                )
            else:
                logger.info("ℹ️ No dates found for date_reasoning")
                return None
                
        except Exception as e:
            logger.warning(f"Date reasoning from patterns failed: {e}")
            return None
    
    def generate_brief_summary(self, document_text: str) -> str:
        """Generate brief summary with contextual understanding"""
        try:
            # Use AI-driven pattern analysis to inform summary
            pattern_analysis = self.reasoning_system.analyze_document_patterns(document_text)
            
            summary_prompt = PromptTemplate(
                template="""
                Generate a concise 1-2 sentence summary of this medical document based on its content and context.
                
                DOCUMENT TEXT:
                {document_text}
                
                DOCUMENT CONTEXT:
                {document_context}
                
                Focus on: Patient condition, key findings, and main recommendations.
                Be professional and concise. Use clinical language appropriate for medical records.
                
                {format_instructions}
                """,
                input_variables=["document_text", "document_context"],
                partial_variables={"format_instructions": self.brief_summary_parser.get_format_instructions()},
            )
            
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "document_text": document_text,
                "document_context": f"Document type: {pattern_analysis.get('document_type', 'medical_document')}. Urgency: {pattern_analysis.get('clinical_urgency_indicators', {}).get('urgency_level', 'normal')}"
            })
            
            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated contextual summary: {brief_summary}")
            return brief_summary
            
        except Exception as e:
            logger.error(f"❌ Contextual summary generation failed: {str(e)}")
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
        """Get date reasoning results without full document extraction (AI-driven)"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Run AI-driven date reasoning directly
            logger.info("🔍 Starting AI-driven date reasoning...")
            
            date_reasoning = self._create_ai_date_reasoning(document_text, current_date)
            
            # Include pattern analysis for completeness
            pattern_analysis = self.reasoning_system.analyze_document_patterns(document_text)
            
            return {
                "date_reasoning": date_reasoning.dict() if date_reasoning else {},
                "pattern_analysis": pattern_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}, "pattern_analysis": {}}