from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime, timedelta
import re
import logging
import json
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
    claim_number: str = Field(..., description="Claim number or case ID. Use 'Not specified' if not found")
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

class ComprehensiveDocumentReasoningSystem:
    """Enhanced reasoning system that understands medical report patterns and context"""
    
    def __init__(self, llm):
        self.llm = llm
    
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
           - Claim Number: Look for "Claim #", "Case #", "Claim Number:", insurance references
        
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
           - Claim numbers often follow patterns like numbers/letters combinations
        
        SPECIFIC PATTERNS TO RECOGNIZE:
        
        - Progress Reports (PR-2): Have structured sections with patient info, subjective, objective, assessment
        - Imaging Reports: Focus on findings, impressions, technical details
        - Injection Notes: Document specific procedures, medications, follow-up timing
        - Workers Comp: Include claim numbers, work status, injury details
        
        EXTRACTION RULES:
        
        - Patient Name: Extract full name from demographics (e.g., "Cynthia L Williams")
        - Claim Number: Look near "Claim #", case numbers, insurance references
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
        """Analyze document patterns and structure before LLM extraction"""
        patterns = {
            "document_type": self._detect_document_type(document_text),
            "sections_found": self._identify_sections(document_text),
            "patient_info_patterns": self._extract_patient_patterns(document_text),
            "clinical_urgency_indicators": self._detect_urgency_indicators(document_text),
            "workers_comp_indicators": self._detect_workers_comp_patterns(document_text)
        }
        return patterns
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content patterns"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["progress report", "pr-2", "follow-up", "follow up"]):
            return "progress_report"
        elif any(term in text_lower for term in ["mri", "ct", "x-ray", "imaging", "scan findings"]):
            return "imaging_report"
        elif any(term in text_lower for term in ["injection", "inject", "viscosupplementation", "injection note"]):
            return "procedure_note"
        elif any(term in text_lower for term in ["consultation", "referral", "qme", "ame"]):
            return "consultation_note"
        elif any(term in text_lower for term in ["initial evaluation", "new patient"]):
            return "initial_evaluation"
        else:
            return "medical_document"
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify document sections based on patterns"""
        sections = []
        lines = text.split('\n')
        
        for line in lines:
            line_clean = line.strip()
            if any(section in line_clean.lower() for section in ["patient:", "name:", "dob:"]):
                sections.append("patient_demographics")
            elif any(section in line_clean.lower() for section in ["diagnosis:", "assessment:", "findings:"]):
                sections.append("diagnosis")
            elif any(section in line_clean.lower() for section in ["treatment:", "plan:", "recommendations:"]):
                sections.append("treatment_plan")
            elif any(section in line_clean.lower() for section in ["subjective:", "history:"]):
                sections.append("subjective")
            elif any(section in line_clean.lower() for section in ["objective:", "exam:"]):
                sections.append("objective")
        
        return list(set(sections))
    
    def _extract_patient_patterns(self, text: str) -> Dict[str, Any]:
        """Extract patient information patterns"""
        patterns = {
            "patient_name_found": False,
            "dob_found": False,
            "claim_number_found": False,
            "potential_names": [],
            "potential_claims": []
        }
        
        # Look for name patterns
        name_patterns = [
            r'Patient:\s*([A-Za-z\s,\.]+)',
            r'Name:\s*([A-Za-z\s,\.]+)',
            r'([A-Z][a-z]+ [A-Z] [A-Z][a-z]+)',  # First M Last pattern
            r'([A-Z][a-z]+, [A-Z][a-z]+)'  # Last, First pattern
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) > 5:  # Reasonable name length
                    patterns["potential_names"].append(name)
                    patterns["patient_name_found"] = True
        
        # Look for claim number patterns
        claim_patterns = [
            r'Claim\s*[#]?:\s*([A-Z0-9\-]+)',
            r'Case\s*[#]?:\s*([A-Z0-9\-]+)',
            r'Claim\s*Number:\s*([A-Z0-9\-]+)',
            r'([A-Z]+\d+)'  # Pattern like INT845298
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim = match.group(1).strip()
                if claim and len(claim) > 3:  # Reasonable claim number length
                    patterns["potential_claims"].append(claim)
                    patterns["claim_number_found"] = True
        
        # Look for DOB patterns
        dob_patterns = [
            r'DOB:\s*([0-9/\-]+)',
            r'Date of Birth:\s*([0-9/\-]+)',
            r'Birth Date:\s*([0-9/\-]+)'
        ]
        
        for pattern in dob_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns["dob_found"] = True
                break
        
        return patterns
    
    def _detect_urgency_indicators(self, text: str) -> Dict[str, Any]:
        """Detect clinical urgency indicators"""
        text_lower = text.lower()
        indicators = {
            "urgency_level": "normal",
            "pain_indicators": [],
            "critical_findings": [],
            "urgency_keywords": []
        }
        
        # Pain indicators
        pain_patterns = [
            r'pain\s*(\d+)/10',
            r'pain\s*scale\s*(\d+)',
            r'rated\s*(\d+)/10'
        ]
        
        for pattern in pain_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                pain_level = int(match.group(1))
                indicators["pain_indicators"].append(f"pain_{pain_level}/10")
                if pain_level >= 7:
                    indicators["urgency_level"] = "urgent"
                elif pain_level >= 5:
                    indicators["urgency_level"] = "elevated"
        
        # Critical findings
        critical_terms = ["fracture", "infection", "bleeding", "emergency", "urgent", "stat", "critical"]
        for term in critical_terms:
            if term in text_lower:
                indicators["critical_findings"].append(term)
                indicators["urgency_level"] = "critical"
                break
        
        # Urgency keywords
        urgency_keywords = ["severe", "acute", "worsening", "progressive", "uncontrolled"]
        for keyword in urgency_keywords:
            if keyword in text_lower:
                indicators["urgency_keywords"].append(keyword)
                if indicators["urgency_level"] == "normal":
                    indicators["urgency_level"] = "elevated"
        
        return indicators
    
    def _detect_workers_comp_patterns(self, text: str) -> Dict[str, Any]:
        """Detect workers compensation patterns"""
        text_lower = text.lower()
        patterns = {
            "is_workers_comp": False,
            "indicators": [],
            "work_status": "unknown"
        }
        
        workers_comp_indicators = [
            "wrkcmp", "work comp", "workers comp", "state comp", "industrial", 
            "work related", "occupational", "claim #", "case #"
        ]
        
        for indicator in workers_comp_indicators:
            if indicator in text_lower:
                patterns["indicators"].append(indicator)
                patterns["is_workers_comp"] = True
        
        # Work status patterns
        status_terms = {
            "light duty": ["light duty", "modified duty", "restricted duty"],
            "off work": ["off work", "no work", "unable to work"],
            "full duty": ["full duty", "return to work", "released to work"]
        }
        
        for status, terms in status_terms.items():
            for term in terms:
                if term in text_lower:
                    patterns["work_status"] = status
                    break
        
        return patterns

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
           - Claim Number: Extract from claim/case number patterns
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
            
            # Step 1: Analyze document patterns and structure
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
            
            # Step 3: Create date reasoning from pattern analysis
            date_reasoning = self._create_date_reasoning_from_patterns(pattern_analysis, document_text)
            
            # Create final analysis - FIXED: Pass date_reasoning separately
            final_analysis = DocumentAnalysis(
                **result,  # All the fields from LLM except date_reasoning
                date_reasoning=date_reasoning  # Our own date reasoning
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
    
    def _create_date_reasoning_from_patterns(self, pattern_analysis: Dict[str, Any], document_text: str) -> Optional[DateReasoning]:
        """Create date reasoning from pattern analysis"""
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
                    reasoning=f"Pattern-based extraction. Document type: {pattern_analysis.get('document_type', 'unknown')}",
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
            # Use pattern analysis to inform summary
            pattern_analysis = self.reasoning_system.analyze_document_patterns(document_text)
            
            summary_prompt = PromptTemplate(
                template="""
                Generate a concise 1-2 sentence summary of this medical document based on its content and context.
                
                DOCUMENT TEXT:
                {document_text}
                
                DOCUMENT CONTEXT:
                {document_context}
                
                Focus on: Patient condition, key findings, and main recommendations.
                Be professional and concise.
                
                {format_instructions}
                """,
                input_variables=["document_text", "document_context"],
                partial_variables={"format_instructions": self.brief_summary_parser.get_format_instructions()},
            )
            
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "document_text": document_text,
                "document_context": f"Document type: {pattern_analysis.get('document_type', 'medical_document')}"
            })
            
            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated contextual summary: {brief_summary}")
            return brief_summary
            
        except Exception as e:
            logger.error(f"❌ Contextual summary generation failed: {str(e)}")
            return "Brief summary unavailable"
    
    def create_fallback_analysis(self) -> DocumentAnalysis:
        """Create fallback analysis when extraction fails"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=datetime.now().strftime("%Y-%m-%d"),
            doi=datetime.now().strftime("%Y-%m-%d"),
            status="normal",
            rd=datetime.now().strftime("%Y-%m-%d"),
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
        """Get date reasoning results without full document extraction"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            document_type = self.detect_document_type_preview(document_text)
            
            # Run date reasoning using the separate agent
            logger.info("🔍 Starting date reasoning with LangGraph...")
            
            # For now, return basic date reasoning from patterns
            pattern_analysis = self.reasoning_system.analyze_document_patterns(document_text)
            date_reasoning = self._create_date_reasoning_from_patterns(pattern_analysis, document_text)
            
            return {
                "date_reasoning": date_reasoning.dict() if date_reasoning else {},
                "pattern_analysis": pattern_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}, "pattern_analysis": {}}