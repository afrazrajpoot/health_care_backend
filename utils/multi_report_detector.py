"""
Simplified Multi-Report Detector using Azure OpenAI.
Relies on LLM reasoning to accurately detect multiple reports from summary.
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
    is_multiple: bool = Field(description="True if multiple separate reports are detected")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")
    reasoning: str = Field(description="Step-by-step reasoning explaining the decision")
    report_count: int = Field(description="Number of separate reports identified")
    report_types: list = Field(default_factory=list, description="Types of reports found (e.g., QME, PR2, AME)")


class MultiReportDetector:
    """Detects if a document contains multiple separate reports using LLM reasoning"""
    
    def __init__(self):
        """Initialize with Azure OpenAI model"""
        deployment_name = CONFIG.get("azure_openai_o3_model") or CONFIG.get("azure_openai_deployment")
        
        if not deployment_name:
            raise ValueError("No Azure OpenAI deployment configured")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=deployment_name,
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0,
            timeout=120,
        )
        self.parser = JsonOutputParser(pydantic_object=MultiReportDetectionResult)
        logger.info(f"✅ MultiReportDetector initialized with deployment: {deployment_name}")

    SYSTEM_PROMPT = """You are an expert at analyzing medical document summaries to determine if they contain multiple separate reports or just a single report.

## YOUR TASK
Analyze the provided summary and determine if it describes:
- **SINGLE REPORT**: One document that may reference or mention other reports
- **MULTIPLE REPORTS**: Two or more complete separate reports combined into one document

## KEY INDICATORS OF MULTIPLE REPORTS

**Strong indicators of MULTIPLE separate reports:**
1. **Multiple Report Headers/Titles**: Summary explicitly mentions distinct report sections like "The document contains a QME report followed by a PR-2 report"
2. **Multiple Evaluation Dates**: Different dates for different evaluations (not just follow-up dates within same report)
3. **Multiple Evaluating Physicians**: Different doctors conducting separate evaluations (not just referencing another doctor's work)
4. **Multiple Patient Info Sections**: Summary indicates separate patient information blocks for different reports
5. **Sequential Report Structure**: Summary describes document structure like "First section is X report, second section is Y report"

**Weak indicators (do NOT indicate multiple reports alone):**
1. **References to Other Reports**: "Reviewed the previous PR-2" or "Based on Dr. Smith's QME" - these are just CITATIONS
2. **Records Reviewed Section**: Listing of documents reviewed as part of the evaluation
3. **Medical History**: References to prior treatments, evaluations, or medical records
4. **Follow-up Dates**: Multiple dates within the same evaluation report

## DECISION PROCESS

**Step 1: Identify Report Type Mentions**
List all medical report types mentioned (QME, PR-2, PR-4, AME, IME, DFR, etc.)

**Step 2: Determine Context**
For each mention, is it:
- A: An actual separate report section in this document?
- B: A reference/citation to another document?
- C: Part of the medical history/records reviewed?

**Step 3: Count Actual Reports**
Count only the "A" items (actual separate report sections)

**Step 4: Make Decision**
- If count = 1: is_multiple = false
- If count ≥ 2: is_multiple = true

## EXAMPLES

**EXAMPLE 1 - SINGLE REPORT (references others)**
Summary: "This is a QME report evaluating the patient on 03/15/2024. The doctor reviewed the patient's PR-2 from 01/10/2024 and the previous AME by Dr. Smith dated 12/05/2023. Based on these records and current examination..."

Analysis:
- Report types mentioned: QME, PR-2, AME
- Context: QME is the main report (A), PR-2 is referenced (B), AME is referenced (B)
- Actual reports: 1 (only the QME)
- Decision: is_multiple = FALSE
- Reasoning: "Single QME report that references two other reports (PR-2 and AME). The PR-2 and AME are cited as reviewed documents, not included as separate reports."

**EXAMPLE 2 - MULTIPLE REPORTS**
Summary: "This document contains two separate reports. The first section is a Qualified Medical Evaluation (QME) dated 02/15/2024 conducted by Dr. Johnson with its own patient information and examination findings. The second section is a Progress Report PR-2 dated 03/20/2024 conducted by Dr. Williams with a separate patient information section and treatment recommendations."

Analysis:
- Report types mentioned: QME, PR-2
- Context: QME is actual report section (A), PR-2 is actual report section (A)
- Actual reports: 2 (both QME and PR-2)
- Decision: is_multiple = TRUE
- Reasoning: "Document contains two distinct report sections: a QME from 02/15/2024 by Dr. Johnson and a PR-2 from 03/20/2024 by Dr. Williams. Each has its own structure, date, and physician."

**EXAMPLE 3 - SINGLE REPORT (with history)**
Summary: "PR-2 Progress Report dated 04/10/2024. Patient was initially evaluated via QME on 01/15/2024. Current examination shows improvement. Records reviewed include: previous QME report, PR-2 from 02/20/2024, imaging studies."

Analysis:
- Report types mentioned: PR-2, QME
- Context: PR-2 is main report (A), QME is historical reference (C), previous PR-2 is reviewed record (B)
- Actual reports: 1 (only current PR-2)
- Decision: is_multiple = FALSE
- Reasoning: "Single PR-2 report that references prior evaluations as part of patient history and records reviewed. No separate report sections present."

## OUTPUT FORMAT
Respond ONLY with valid JSON matching this structure:
{format_instructions}

Be decisive and clear in your reasoning. Focus on whether the document actually CONTAINS multiple reports, not whether it MENTIONS multiple reports."""

    USER_PROMPT = """Analyze this document summary and determine if it contains multiple separate reports:

## DOCUMENT SUMMARY:
{summary_text}

## YOUR ANALYSIS:
Provide your response in the JSON format specified."""

    def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
        """
        Analyze document summary to detect multiple reports using LLM reasoning.
        
        Args:
            summary_text: The summarizer output to analyze
            
        Returns:
            dict with detection results
        """
        if not summary_text or len(summary_text.strip()) < 50:
            logger.info("📄 Document too short for analysis")
            return {
                "is_multiple": False,
                "confidence": "high",
                "reasoning": "Document too short to contain multiple reports",
                "report_count": 1,
                "report_types": []
            }
        
        try:
            logger.info("🔍 Analyzing document for multiple reports...")
            logger.info(f"📝 Summary length: {len(summary_text)} characters")
            
            # Create prompt and chain
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT)
            ])
            
            chain = prompt | self.llm | self.parser
            
            # Truncate if needed
            max_chars = 15000
            truncated = summary_text[:max_chars]
            if len(summary_text) > max_chars:
                logger.info(f"⚠️ Truncated from {len(summary_text)} to {max_chars} chars")
            
            # Run analysis
            result = chain.invoke({
                "summary_text": truncated,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Normalize result
            if isinstance(result, dict):
                detection_result = result
            else:
                detection_result = result.dict()
            
            # Log results
            logger.info("=" * 80)
            logger.info("📊 MULTI-REPORT DETECTION RESULTS:")
            logger.info("=" * 80)
            logger.info(f"   Multiple Reports: {detection_result.get('is_multiple', False)}")
            logger.info(f"   Confidence: {detection_result.get('confidence', 'unknown')}")
            logger.info(f"   Report Count: {detection_result.get('report_count', 1)}")
            logger.info(f"   Report Types: {detection_result.get('report_types', [])}")
            logger.info(f"   Reasoning: {detection_result.get('reasoning', 'No reasoning provided')}")
            logger.info("=" * 80)
            
            return detection_result
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "is_multiple": False,
                "confidence": "low",
                "reasoning": f"Detection failed: {str(e)}",
                "report_count": 1,
                "report_types": []
            }


# Singleton instance
_detector_instance = None
_initialization_failed = False

def get_multi_report_detector() -> MultiReportDetector:
    """Get singleton MultiReportDetector instance"""
    global _detector_instance, _initialization_failed
    
    if _detector_instance is None and not _initialization_failed:
        try:
            logger.info("🚀 Initializing Multi-Report Detector...")
            _detector_instance = MultiReportDetector()
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            _initialization_failed = True
            
            class DummyDetector:
                def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
                    logger.warning("⚠️ Detector not available - returning default")
                    return {
                        "is_multiple": False,
                        "confidence": "low",
                        "reasoning": "Detector initialization failed",
                        "report_count": 1,
                        "report_types": []
                    }
            _detector_instance = DummyDetector()
    
    return _detector_instance


def detect_multiple_reports(summary_text: str) -> Dict[str, Any]:
    """Convenience function to detect multiple reports"""
    detector = get_multi_report_detector()
    return detector.detect_multiple_reports(summary_text)