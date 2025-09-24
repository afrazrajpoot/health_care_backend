


import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from models.schemas import ComprehensiveAnalysis, PatientInfo, WorkStatusAlert, ErrorResponse
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

class ReportAnalyzer:
    """Enhanced service for comprehensive medical/legal report analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.1,  # Lower temperature for more consistent parsing
            timeout=120  # Longer timeout for complex analysis
        )
        
    def get_current_datetime(self) -> str:
        """Get current datetime string"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def detect_document_type_preview(self, document_text: str) -> str:
        """
        Quick document type detection for logging purposes
        
        Args:
            document_text: Document content to analyze
            
        Returns:
            Detected document type as string
        """
        text_lower = document_text.lower()
        
        # Medical imaging reports
        if any(term in text_lower for term in ['mri', 'ct scan', 'x-ray', 'ultrasound', 'mammography', 'radiolog']):
            return "Medical Imaging Report"
        
        # Lab/pathology reports  
        if any(term in text_lower for term in ['lab result', 'pathology', 'blood test', 'urinalysis', 'biopsy']):
            return "Laboratory/Pathology Report"
        
        # Progress reports
        if any(term in text_lower for term in ['progress report', 'pr-2', 'follow-up', 'treatment progress']):
            return "Progress Report"
        
        # IME reports
        if any(term in text_lower for term in ['independent medical examination', 'ime', 'medical evaluation']):
            return "Independent Medical Examination"
        
        # Authorization requests
        if any(term in text_lower for term in ['request for authorization', 'rfa', 'pre-authorization', 'treatment request']):
            return "Request for Authorization (RFA)"
        
        # Denial letters
        if any(term in text_lower for term in ['denied', 'denial', 'not authorized', 'coverage denied']):
            return "Denial/Coverage Decision"
        
        # Work status documents
        if any(term in text_lower for term in ['ttd', 'temporary total disability', 'work restriction', 'return to work']):
            return "Work Status Document"
        
        # Legal documents
        if any(term in text_lower for term in ['legal opinion', 'attorney', 'litigation', 'deposition']):
            return "Legal Document"
        
        # Generic medical report
        if any(term in text_lower for term in ['patient', 'diagnosis', 'treatment', 'medical', 'physician']):
            return "Medical Report"
        
        return "Unknown Document Type"
    
    def create_analysis_prompt(self) -> PromptTemplate:
        """Create the comprehensive analysis prompt template"""
        template = """
You are a medical/legal document analysis expert specializing in healthcare document classification and analysis. 

INPUT DOCUMENT TEXT:
{document_text}

CURRENT DATE/TIME: {current_datetime}

TASK:
Analyze this document and provide a structured response. First, identify the DOCUMENT TYPE based on content patterns, then extract key information.

DOCUMENT TYPE DETECTION:
Identify the document type based on these indicators:

**Medical Reports:**
- MRI Report/Imaging Report: Contains "MRI", "CT", "X-Ray", imaging findings, radiological terms
- Lab Report: Contains test results, normal ranges, lab values, pathology findings
- Progress Report/PR-2: Contains progress notes, treatment updates, work status changes
- Independent Medical Examination (IME): Contains "IME", independent evaluation, medical opinions
- Physician Report: Contains diagnosis, treatment plans, medical recommendations

**Insurance/Legal Documents:**
- Request for Authorization (RFA): Contains "RFA", treatment authorization requests, procedure codes
- Denial Report: Contains "denied", "rejection", coverage decisions, appeal information
- Claim Review: Contains claim numbers, coverage determinations, benefit analysis
- Treatment Authorization: Contains pre-authorization, approval/denial of treatments

**Work Status Documents:**
- Disability Evaluation: Contains TTD, permanent disability ratings, work restrictions
- Return to Work Assessment: Contains work capacity, job modifications, fitness for duty
- Functional Capacity Evaluation: Contains physical abilities, work limitations

**Other Professional Documents:**
- Legal Opinion: Contains legal analysis, case law references, recommendations
- Case Management Notes: Contains care coordination, treatment planning
- Utilization Review: Contains medical necessity determinations

ANALYSIS REQUIREMENTS:

1. DOCUMENT TYPE & TITLE:
   - Identify specific document type from the patterns above
   - Create a precise, professional title that includes the document type
   - Examples: "MRI Lumbar Spine Report", "IME Orthopedic Evaluation", "RFA Physical Therapy Request"

2. ENHANCED SUMMARY (4-6 bullet points):
   - **First bullet**: Document type and primary purpose
   - **Key findings**: Most important medical/legal conclusions
   - **Current status**: TTD status, work capacity, treatment stage
   - **Critical dates**: Important deadlines, follow-up dates
   - **Recommendations**: Next steps, treatment plans, work restrictions
   - **Alerts**: Any urgent issues requiring immediate attention

3. STRUCTURED DATA EXTRACTION:
   - patient_name: Extract full patient name (return null if not found)
   - patient_email: Extract email address (return null if not found) 
   - claim_no: Extract claim/case/file number (return null if not found)
   - report_title: Use the document type + specific content (e.g., "MRI Cervical Spine Report")
   - time_day: Extract report date or use current date if missing
   - status: Determine urgency level:
     * "urgent" - Critical findings, TTD >45 days, emergency conditions, overdue reviews
     * "normal" - Standard reports, routine updates, mild-moderate findings  
     * "low" - Administrative documents, informational reports, minor findings

4. WORK STATUS ALERTS:
   Generate alerts for urgent situations:
   - TTD exceeding 45 days without clear end date
   - Critical imaging findings (fractures, severe degeneration, tumors)
   - Treatment denials affecting patient care
   - Overdue medical evaluations or follow-ups
   - Contradictory medical opinions requiring resolution
   - Missing required documentation

Alert categories:
   - "Work Status Review": TTD duration, return-to-work issues
   - "Medical Urgency": Critical findings, deteriorating conditions
   - "Treatment Authorization": RFA denials, coverage issues  
   - "Follow-Up Required": Overdue appointments, missing documentation
   - "Legal Review": Conflicting opinions, complex cases

OUTPUT FORMAT:
{{
  "original_report": "Full extracted document text exactly as provided...",
  "report_json": {{
    "patient_name": "Full patient name or null",
    "patient_email": "email@domain.com or null", 
    "claim_no": "Claim/case number or null",
    "report_title": "Document Type: Specific Title (e.g., 'MRI Report: Lumbar Spine Assessment')",
    "time_day": "YYYY-MM-DDTHH:MM:SSZ format",
    "status": "urgent/normal/low"
  }},
  "summary": [
    "• Document Type: [Specific type] - [Primary purpose/scope]",
    "• Key Medical Findings: [Most important clinical information]", 
    "• Current Work Status: [TTD/Modified/Regular duty status with dates]",
    "• Treatment Recommendations: [Proposed treatments, restrictions, modifications]",
    "• Next Steps: [Follow-up appointments, additional testing, deadlines]",
    "• Priority Items: [Any urgent issues requiring immediate attention]"
  ],
  "work_status_alert": [
    {{
      "alert_type": "Specific alert category",
      "title": "Clear, actionable alert title", 
      "date": "Relevant date (YYYY-MM-DD)",
      "status": "urgent/normal/low priority"
    }}
  ]
}}

CRITICAL GUIDELINES:
- Document type identification is MANDATORY - never use generic titles
- Include medical/legal terminology appropriately based on document type
- Summary must start with document type identification
- Prioritize patient safety and compliance issues
- Extract precise information - do not invent details not in the document
- Use professional healthcare/legal terminology consistent with document type
- Ensure JSON is perfectly formatted and valid

Begin analysis now:
"""
        
        return PromptTemplate(
            input_variables=["document_text", "current_datetime"],
            template=template
        )
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate JSON response from LLM"""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove any markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            parsed_data = json.loads(response_text)
            
            logger.info("✅ Successfully parsed JSON response from LLM")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {str(e)}")
            logger.error(f"📄 Raw response: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON response from AI analysis: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error parsing response: {str(e)}")
            raise
    
    def validate_analysis_data(self, parsed_data: Dict[str, Any]) -> ComprehensiveAnalysis:
        """Validate and structure the parsed data"""
        try:
            # Extract patient info
            patient_data = parsed_data.get("report_json", {})
            patient_info = PatientInfo(**patient_data)
            
            # Extract alerts
            alerts_data = parsed_data.get("work_status_alert", [])
            alerts = [WorkStatusAlert(**alert) for alert in alerts_data]
            
            # Create comprehensive analysis
            analysis = ComprehensiveAnalysis(
                original_report=parsed_data.get("original_report", ""),
                report_json=patient_info,
                summary=parsed_data.get("summary", []),
                work_status_alert=alerts
            )
            
            logger.info("✅ Successfully validated analysis data")
            logger.info(f"📊 Summary points: {len(analysis.summary)}")
            logger.info(f"🚨 Alerts generated: {len(analysis.work_status_alert)}")
            
            return analysis
            
        except ValidationError as e:
            logger.error(f"❌ Data validation error: {str(e)}")
            raise ValueError(f"Invalid analysis data structure: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error validating analysis data: {str(e)}")
            raise
    
    def analyze_document(self, document_text: str) -> ComprehensiveAnalysis:
        """
        Perform comprehensive analysis of a document
        
        Args:
            document_text: Full text content of the document
            
        Returns:
            ComprehensiveAnalysis object with structured results
        """
        try:
            logger.info("🔍 Starting comprehensive document analysis...")
            logger.info(f"📄 Document length: {len(document_text)} characters")
            
            # Quick document type detection for logging
            detected_type = self.detect_document_type_preview(document_text)
            logger.info(f"📋 Detected document type: {detected_type}")
            
            # Check if document has meaningful content
            if not document_text.strip():
                raise ValueError("Document appears to be empty or contains no readable text")
            
            if len(document_text.strip()) < 50:
                raise ValueError("Document content appears too short to be meaningful")
            
            # Create prompt
            prompt = self.create_analysis_prompt()
            current_datetime = self.get_current_datetime()
            
            # Format prompt with document content
            formatted_prompt = prompt.format(
                document_text=document_text,
                current_datetime=current_datetime
            )
            
            logger.info(f"🚀 Sending {detected_type} analysis request to GPT-4o...")
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content
            
            logger.info(f"📨 Received response ({len(response_text)} characters)")
            
            # Parse JSON response
            parsed_data = self.parse_json_response(response_text)
            
            # Validate and structure data
            analysis = self.validate_analysis_data(parsed_data)
            
            # Enhanced logging with document type information
            patient_name = analysis.report_json.patient_name or "Unknown"
            report_title = analysis.report_json.report_title or f"{detected_type} (Untitled)"
            status = analysis.report_json.status
            
            logger.info(f"📄 Document Type: {detected_type}")
            logger.info(f"👤 Patient: {patient_name}")
            logger.info(f"📑 Report Title: {report_title}")
            logger.info(f"⚡ Status: {status}")
            logger.info(f"📝 Summary Points: {len(analysis.summary)}")
            
            if analysis.work_status_alert:
                logger.info(f"🚨 Alerts Generated: {len(analysis.work_status_alert)}")
                for alert in analysis.work_status_alert:
                    logger.info(f"   ⚠️ {alert.alert_type}: {alert.title} ({alert.status})")
            else:
                logger.info("✅ No alerts required for this document")
            
            logger.info("✅ Comprehensive analysis completed successfully")
            return analysis
            
        except ValueError as e:
            # Handle document content issues
            logger.error(f"❌ Document analysis failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in document analysis: {str(e)}")
            raise ValueError(f"Analysis failed due to: {str(e)}")
    
    def compare_summaries(self, previous_summary: List[str], current_summary: List[str]) -> str:
        """
        Use AI to compare previous and current summaries and generate last changes description.
        
        Args:
            previous_summary: List of strings from previous document summary
            current_summary: List of strings from current document summary
            
        Returns:
            String describing the key changes
        """
        try:
            logger.info("🔄 Starting AI summary comparison...")
            
            template = """
You are a medical report comparison expert. Compare these two summaries for the same patient:

PREVIOUS SUMMARY:
{previous_summary}

CURRENT SUMMARY:
{current_summary}

TASK:
- Identify key changes, improvements, deteriorations, new findings, or resolved issues
- Focus on medical status, work restrictions, treatments, diagnoses, and urgency levels
- Be concise: 3-5 bullet points describing main differences
- Start with "Key Changes:" 
- Use professional medical terminology

Output a single string with the comparison.
"""
            
            prompt = PromptTemplate(
                input_variables=["previous_summary", "current_summary"],
                template=template
            )
            
            formatted_prompt = prompt.format(
                previous_summary="\n".join(previous_summary),
                current_summary="\n".join(current_summary)
            )
            
            response = self.llm.invoke(formatted_prompt)
            changes = response.content.strip()
            
            logger.info(f"✅ Generated changes description ({len(changes)} characters)")
            return changes
            
        except Exception as e:
            logger.error(f"❌ Failed to compare summaries: {str(e)}")
            return f"Unable to generate changes due to error: {str(e)}"
    
    def handle_analysis_error(self, error: Exception, document_text: str = "") -> ErrorResponse:
        """
        Create structured error response for analysis failures
        
        Args:
            error: The exception that occurred
            document_text: Original document text for context
            
        Returns:
            ErrorResponse with helpful guidance
        """
        error_msg = str(error)
        guidance = []
        
        # Determine error type and provide specific guidance
        if "empty" in error_msg.lower() or "no readable text" in error_msg.lower():
            guidance = [
                "Ensure the document is not password-protected or corrupted.",
                "If this is a scanned document, try uploading a higher-quality scan.",
                "Verify the document contains actual text content, not just images."
            ]
        elif "too short" in error_msg.lower():
            guidance = [
                "The document appears to contain insufficient content for analysis.",
                "Ensure you uploaded the complete document, not a partial file.",
                "Check if the document is properly formatted and readable."
            ]
        elif "json" in error_msg.lower():
            guidance = [
                "The AI analysis service encountered a processing error.",
                "Try uploading the document again in a few moments.",
                "If the problem persists, contact technical support."
            ]
        else:
            guidance = [
                "Ensure the document is not password-protected.",
                "Upload a higher-quality scan if it is an image-based PDF.", 
                "If this is a handwritten document, provide a typed copy.",
                "Contact support if the issue persists."
            ]
        
        return ErrorResponse(
            error=True,
            message=f"The document could not be processed because {error_msg}.",
            guidance=guidance
        )