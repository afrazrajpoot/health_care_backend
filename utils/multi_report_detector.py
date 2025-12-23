# multi_report_detector.py

"""
Multi-Report Detector using Azure OpenAI o3-pro reasoning model.
Detects if a document contains multiple separate reports by analyzing
the Document AI Summarizer output with AI reasoning.
"""

import logging
import json
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config.settings import CONFIG

logger = logging.getLogger("multi_report_detector")


class MultiReportDetectionResult(BaseModel):
    """Result model for multi-report detection"""
    is_multiple: bool = Field(description="True if multiple separate reports are detected in the document")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")
    reason: str = Field(description="Clear explanation of why this determination was made")
    report_count_estimate: int = Field(description="Estimated number of separate reports in the document")
    reports_identified: list = Field(default_factory=list, description="List of identified report types/descriptions if multiple")


class MultiReportDetector:
    """
    Detects if a document contains multiple separate reports using Azure OpenAI o3-pro.
    Uses AI reasoning to understand context and make intelligent decisions.
    """
    
    def __init__(self):
        """Initialize the detector with Azure OpenAI o3-pro model"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_o3_model"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.1,
            timeout=120,
        )
        self.parser = JsonOutputParser(pydantic_object=MultiReportDetectionResult)
        logger.info("✅ MultiReportDetector initialized with Azure OpenAI o3-pro model")
    
    SYSTEM_PROMPT = """You are an expert medical document analyst. Your task is to determine if a document summary describes a SINGLE medical report or MULTIPLE separate reports combined into one file.

## YOUR TASK
Analyze the provided Document AI Summarizer output and determine:
1. Is this a single cohesive report OR multiple separate reports?
2. How confident are you in this assessment?
3. If multiple reports, approximately how many and what types?

## WHAT COUNTS AS MULTIPLE REPORTS
Multiple reports means the document contains DISTINCT, SEPARATE medical documents that were combined/scanned together, such as:
- Different report types (e.g., a Progress Report AND an Authorization Decision in one file)
- Reports from different dates that are clearly separate documents (not follow-up notes within one report)
- Reports for different patients (very strong indicator)
- Reports with different claim numbers
- Reports from different providers/facilities that aren't just referenced

## WHAT IS STILL A SINGLE REPORT
A single report may contain:
- Multiple dates (visit history, treatment timeline within one report)
- References to other reports or external documents
- Multiple sections (history, exam, assessment, plan)
- Multiple body parts or conditions discussed
- Multiple providers mentioned (referring physician, consulting specialist)
- Attachments or addendums that are part of the same report

## KEY INDICATORS TO LOOK FOR

**Strong indicators of MULTIPLE reports:**
- Explicit mentions like "Report 1", "Report 2", "First document", "Second document"
- Different patient names appearing as the primary subject
- Different claim numbers for the same patient
- Completely different report headers/types (e.g., "Progress Report" followed by "Utilization Review Decision")
- Clear document separators or restart of report formatting
- Drastically different dates with complete separate report structures

**Indicators of SINGLE report:**
- Continuous narrative flow
- Single patient throughout
- Single claim number
- Logical progression (history → exam → assessment → plan)
- References to previous visits or reports (not the reports themselves)
- Consistent formatting and structure

## IMPORTANT RULES
1. Be CONSERVATIVE - only flag as multiple if you're reasonably confident
2. A report discussing multiple visits/dates is usually STILL one report
3. A report referencing external documents is STILL one report
4. Focus on whether there are truly SEPARATE document structures, not just multiple topics
5. When in doubt, lean toward "single report" to avoid false positives

## OUTPUT FORMAT
Provide your analysis in the following JSON format:
{format_instructions}
"""

    USER_PROMPT = """Analyze the following Document AI Summarizer output and determine if it represents a single report or multiple reports combined into one document.

## DOCUMENT SUMMARY TO ANALYZE:
{summary_text}

## YOUR ANALYSIS:
Think through this carefully:
1. Is there a single coherent narrative/structure, or are there distinct separate documents?
2. Are there different patients, different claim numbers, or completely separate report types?
3. What evidence supports your conclusion?

Provide your determination in JSON format."""

    def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
        """
        Analyze Document AI Summarizer output to detect multiple reports using AI reasoning.
        
        Args:
            summary_text: The summarizer output text to analyze
            
        Returns:
            dict with:
                - is_multiple: bool - True if multiple reports detected
                - confidence: str - "high", "medium", "low"
                - reason: str - Explanation of the determination
                - report_count_estimate: int - Estimated number of reports
                - reports_identified: list - List of identified reports if multiple
        """
        # Handle empty or very short text
        if not summary_text or len(summary_text.strip()) < 50:
            logger.info("📄 Document too short for multi-report analysis")
            return {
                "is_multiple": False,
                "confidence": "high",
                "reason": "Document too short to contain multiple reports",
                "report_count_estimate": 1,
                "reports_identified": []
            }
        
        try:
            logger.info("🔍 Analyzing document for multiple reports using o3-pro reasoning...")
            logger.info(f"📝 Summary text length: {len(summary_text)} characters")
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT)
            ])
            
            # Create the chain
            chain = prompt | self.llm | self.parser
            
            # Truncate very long summaries to avoid token limits
            max_chars = 15000
            truncated_summary = summary_text[:max_chars] if len(summary_text) > max_chars else summary_text
            if len(summary_text) > max_chars:
                logger.info(f"⚠️ Summary truncated from {len(summary_text)} to {max_chars} chars for analysis")
            
            # Run the analysis
            result = chain.invoke({
                "summary_text": truncated_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Normalize result
            if isinstance(result, dict):
                detection_result = result
            else:
                try:
                    detection_result = result.dict()
                except Exception:
                    detection_result = {
                        "is_multiple": False,
                        "confidence": "low",
                        "reason": "Failed to parse AI response",
                        "report_count_estimate": 1,
                        "reports_identified": []
                    }
            
            # Log results
            logger.info("=" * 80)
            logger.info("📊 MULTI-REPORT DETECTION RESULTS (o3-pro):")
            logger.info("=" * 80)
            logger.info(f"   Is Multiple Reports: {detection_result.get('is_multiple', False)}")
            logger.info(f"   Confidence: {detection_result.get('confidence', 'unknown')}")
            logger.info(f"   Estimated Report Count: {detection_result.get('report_count_estimate', 1)}")
            logger.info(f"   Reason: {detection_result.get('reason', 'No reason provided')}")
            if detection_result.get('reports_identified'):
                logger.info(f"   Reports Identified: {detection_result.get('reports_identified')}")
            logger.info("=" * 80)
            
            return detection_result
            
        except Exception as e:
            logger.error(f"❌ Multi-report detection failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return safe default on error
            return {
                "is_multiple": False,
                "confidence": "low",
                "reason": f"Detection failed due to error: {str(e)}",
                "report_count_estimate": 1,
                "reports_identified": []
            }


# Singleton instance
_detector_instance = None

def get_multi_report_detector() -> MultiReportDetector:
    """Get singleton MultiReportDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        logger.info("🚀 Initializing Multi-Report Detector...")
        _detector_instance = MultiReportDetector()
    return _detector_instance


def detect_multiple_reports(summary_text: str) -> Dict[str, Any]:
    """
    Convenience function to detect multiple reports in a document summary.
    
    Args:
        summary_text: The Document AI Summarizer output to analyze
        
    Returns:
        Detection result dictionary
    """
    detector = get_multi_report_detector()
    return detector.detect_multiple_reports(summary_text)
