# multi_report_detector.py

"""
Multi-Report Detector using Azure OpenAI o3-pro reasoning model.
Detects if a document contains multiple separate reports by analyzing
the Document AI Summarizer output with AI reasoning.
"""

import logging
import json
import re
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
    Detects if a document contains multiple separate reports using Azure OpenAI.
    Uses AI reasoning to understand context and make intelligent decisions.
    Falls back to regular deployment if o3-pro model is not available.
    """
    
    def __init__(self):
        """Initialize the detector with Azure OpenAI model (o3-pro preferred, falls back to regular deployment)"""
        # Try to use o3-pro model first, fallback to regular deployment if not available
        o3_model = CONFIG.get("azure_openai_o3_model")
        regular_deployment = CONFIG.get("azure_openai_deployment")
        
        # Determine which deployment to use
        deployment_name = o3_model if o3_model else regular_deployment
        model_type = "o3-pro" if o3_model else "regular deployment"
        
        if not deployment_name:
            raise ValueError("No Azure OpenAI deployment configured. Please set AZURE_OPENAI_O3_MODEL_NAME or AZURE_OPENAI_DEPLOYMENT_NAME")
        
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                deployment_name=deployment_name,
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.1,
                timeout=120,
            )
            self.parser = JsonOutputParser(pydantic_object=MultiReportDetectionResult)
            self.deployment_name = deployment_name
            self.model_type = model_type
            logger.info(f"✅ MultiReportDetector initialized with Azure OpenAI {model_type} (deployment: {deployment_name})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MultiReportDetector: {str(e)}")
            # If o3-pro fails and we have a regular deployment, try that
            if o3_model and regular_deployment and deployment_name == o3_model:
                logger.info(f"🔄 Falling back to regular deployment: {regular_deployment}")
                try:
                    self.llm = AzureChatOpenAI(
                        azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                        api_key=CONFIG.get("azure_openai_api_key"),
                        deployment_name=regular_deployment,
                        api_version=CONFIG.get("azure_openai_api_version"),
                        temperature=0.1,
                        timeout=120,
                    )
                    self.parser = JsonOutputParser(pydantic_object=MultiReportDetectionResult)
                    self.deployment_name = regular_deployment
                    self.model_type = "regular deployment (fallback)"
                    logger.info(f"✅ MultiReportDetector initialized with fallback deployment: {regular_deployment}")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback deployment also failed: {str(fallback_error)}")
                    raise
            else:
                raise
    
    SYSTEM_PROMPT = """You are an expert medical document analyst. Your PRIMARY task is to identify ALL REPORT TITLES that appear at the TOP OF EACH REPORT SECTION throughout the document.

## YOUR PRIMARY TASK
1. **SCAN THE ENTIRE DOCUMENT FOR ANY REPORT TITLES**:
   - Look for ANY report titles/headers that appear at the BEGINNING OF EACH REPORT SECTION
   - Report titles appear at the TOP of each report section, not just at the very top of the document
   - Scan through the entire document looking for ANY report section headers/titles
   - Report titles can be ANY type of medical document/report, including but NOT LIMITED TO:
     * "QUALIFIED MEDICAL EVALUATION" or "QME"
     * "PROGRESS REPORT - PR2" or "PR-2" or "PR2"
     * "PROGRESS REPORT - PR4" or "PR-4" or "PR4"
     * "DOCTOR'S FIRST REPORT" or "DFR"
     * "AGREED MEDICAL EVALUATION" or "AME"
     * "INDEPENDENT MEDICAL EVALUATION" or "IME"
     * "CONSULTATION REPORT" or "CONSULT"
     * "AUTHORIZATION REQUEST" or "TREATMENT AUTHORIZATION"
     * "DISCHARGE SUMMARY" or "DISCHARGE REPORT"
     * "OPERATIVE REPORT" or "SURGERY REPORT"
     * "RADIOLOGY REPORT" or "IMAGING REPORT"
     * "PATHOLOGY REPORT" or "LAB REPORT"
     * "PHYSICAL THERAPY REPORT" or "PT REPORT"
     * "OCCUPATIONAL THERAPY REPORT" or "OT REPORT"
     * "PSYCHOLOGICAL EVALUATION" or "PSYCH REPORT"
     * "FUNCTIONAL CAPACITY EVALUATION" or "FCE"
     * "WORKERS' COMPENSATION REPORT"
     * "MEDICAL RECORDS REVIEW"
     * "DEPOSITION SUMMARY"
     * "EXPERT WITNESS REPORT"
     * ANY OTHER TYPE OF MEDICAL REPORT OR DOCUMENT TITLE
   - **IMPORTANT**: Do NOT limit yourself to only known report types. Identify ANY distinct report title/header that appears at the top of a report section, regardless of whether it's a common type or not.
   
2. **IDENTIFY EACH REPORT SECTION BY ITS TITLE**:
   - Each report section starts with its own title/header
   - Look for titles that appear after page breaks, section breaks, or document separators
   - A new report title indicates a NEW REPORT SECTION
   - Report titles are typically:
     * In ALL CAPS or Title Case
     * Appear at the beginning of a line or section
     * Are followed by report content (patient info, dates, etc.)
     * May include words like: "REPORT", "EVALUATION", "SUMMARY", "RECORD", "ASSESSMENT", "NOTE", "LETTER", etc.
   - Example: If you see "QUALIFIED MEDICAL EVALUATION" at one point, then later see "PROGRESS REPORT - PR2", these are TWO DIFFERENT REPORTS
   - Example: If you see "CONSULTATION REPORT" and later see "RADIOLOGY REPORT", these are TWO DIFFERENT REPORTS
   - Example: If you see "DISCHARGE SUMMARY" and later see "OPERATIVE REPORT", these are TWO DIFFERENT REPORTS
   
3. **COUNT DISTINCT REPORT TITLES FOUND**:
   - List ALL distinct report titles you find throughout the document, regardless of type
   - Do NOT ignore report titles just because they're not in your predefined list
   - If you find 2 or more DISTINCT report titles (of ANY type) = MULTIPLE REPORTS = INVALID DOCUMENT
   - If you find only 1 report title = SINGLE REPORT = VALID DOCUMENT
   
4. **Determine confidence and report types found**

## WHAT COUNTS AS MULTIPLE REPORTS (INVALID DOCUMENT)
**CRITICAL RULE: If the document contains MULTIPLE DISTINCT REPORT TITLES, it is INVALID and must be flagged as multiple reports.**

Multiple reports means the document contains DISTINCT, SEPARATE medical documents that were combined/scanned together, identified by:

1. **MULTIPLE REPORT TITLES/HEADERS** - This is the PRIMARY indicator:
   - Document contains ANY 2+ distinct report titles = MULTIPLE REPORTS = INVALID
   - Examples (but NOT limited to these):
     * "QUALIFIED MEDICAL EVALUATION" title AND "PROGRESS REPORT - PR2" title = MULTIPLE REPORTS = INVALID
     * "QME" header AND "PR2" header = MULTIPLE REPORTS = INVALID
     * "CONSULTATION REPORT" AND "RADIOLOGY REPORT" = MULTIPLE REPORTS = INVALID
     * "DISCHARGE SUMMARY" AND "OPERATIVE REPORT" = MULTIPLE REPORTS = INVALID
     * "PROGRESS REPORT - PR2" title AND "PROGRESS REPORT - PR4" title = MULTIPLE REPORTS = INVALID
     * "QME" title AND "PR4" title = MULTIPLE REPORTS = INVALID
     * "AME" title AND "PR2" title = MULTIPLE REPORTS = INVALID
     * "IME" title AND any Progress Report title = MULTIPLE REPORTS = INVALID
     * "AUTHORIZATION REQUEST" AND "PROGRESS REPORT" = MULTIPLE REPORTS = INVALID
     * ANY combination of 2+ distinct report titles (of ANY type) = MULTIPLE REPORTS = INVALID

2. **Different report types for the same patient** - Still counts as multiple reports:
   - ANY combination of distinct report types in one file = MULTIPLE REPORTS
   - Examples include but are NOT limited to:
     * QME (Qualified Medical Evaluation) AND PR2 (Progress Report) in one file
     * QME AND PR4 in one file
     * PR2 AND PR4 in one file
     * Consultation Report AND Radiology Report in one file
     * Discharge Summary AND Operative Report in one file
     * Progress Report AND Authorization Request in one file
     * QME/AME/IME AND any Progress Report (PR2, PR4) in one file
     * QME/AME/IME AND Consultation report in one file
     * Progress Report AND Authorization Decision in one file
     * ANY combination of distinct report types (regardless of whether they're common or uncommon)

3. Other indicators (secondary):
   - Reports from different dates that are clearly separate documents
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

**PRIMARY INDICATOR - MULTIPLE REPORT TITLES (INVALID):**
- **Multiple distinct report titles/headers** appearing at the TOP OF DIFFERENT REPORT SECTIONS throughout the document:
  - Scan the ENTIRE document from beginning to end
  - Look for ANY report titles that appear at the START of each report section (not just at the very top of the document)
  - Report titles can appear anywhere in the document - at the beginning, middle, or after page breaks
  - **DO NOT limit yourself to only known report types** - identify ANY distinct report title, regardless of type
  - Examples of multiple reports (but NOT limited to these):
    - "QUALIFIED MEDICAL EVALUATION" title at one section AND "PROGRESS REPORT - PR2" title at another section = INVALID
    - "CONSULTATION REPORT" at one section AND "RADIOLOGY REPORT" at another section = INVALID
    - "DISCHARGE SUMMARY" at one section AND "OPERATIVE REPORT" at another section = INVALID
    - "QME" header at one section AND "PR2" header at another section = INVALID
    - "QUALIFIED MEDICAL EVALUATION" at beginning AND "PROGRESS REPORT - PR4" later in document = INVALID
    - "PROGRESS REPORT - PR2" at one section AND "PROGRESS REPORT - PR4" at another section = INVALID
    - "AUTHORIZATION REQUEST" at one section AND "PROGRESS REPORT" at another section = INVALID
    - ANY 2+ distinct report titles (of ANY type) appearing at different sections = INVALID

**Secondary indicators of MULTIPLE reports:**
- Explicit mentions like "Report 1", "Report 2", "First document", "Second document"
- Different patient names appearing as the primary subject
- Different claim numbers for the same patient
- Completely different report headers/types appearing sequentially
- Clear document separators, page breaks, or restart of report formatting with new headers
- Drastically different dates with complete separate report structures

**Indicators of SINGLE report:**
- Continuous narrative flow
- Single patient throughout
- Single claim number
- Logical progression (history → exam → assessment → plan)
- References to previous visits or reports (not the reports themselves)
- Consistent formatting and structure

## CRITICAL RULES
1. **PRIMARY FOCUS: Check for MULTIPLE REPORT TITLES** - Scan the ENTIRE document and if you find 2+ distinct report titles appearing at different sections, flag as MULTIPLE REPORTS (INVALID)
2. **Report Title Detection - SCAN ENTIRE DOCUMENT FOR ANY REPORT TYPE**:
   - **IMPORTANT**: Do NOT just look at the very top of the document
   - **IMPORTANT**: Do NOT limit yourself to only known/common report types
   - Scan through the ENTIRE document from beginning to end
   - Look for ANY report titles that appear at the TOP OF EACH REPORT SECTION
   - Report titles can appear:
     * At the beginning of the document
     * After page breaks or section separators
     * In the middle of the document (indicating a new report section starts)
     * At any point where a new report section begins
   - Report titles typically:
     * Are in ALL CAPS or Title Case
     * Include words like: "REPORT", "EVALUATION", "SUMMARY", "RECORD", "ASSESSMENT", "NOTE", "LETTER", "CONSULTATION", "AUTHORIZATION", etc.
     * Appear as headers or major section titles
   - Common report titles include: "QUALIFIED MEDICAL EVALUATION", "QME", "PROGRESS REPORT - PR2", "PR-2", "PROGRESS REPORT - PR4", "PR-4", "DOCTOR'S FIRST REPORT", "DFR", "CONSULTATION REPORT", "AME", "IME", "RADIOLOGY REPORT", "DISCHARGE SUMMARY", "OPERATIVE REPORT", etc.
   - **BUT ALSO identify ANY other report titles you find, even if they're not in the common list**
   - If document has ANY 2+ distinct report titles at different sections = MULTIPLE REPORTS = INVALID
3. **Key distinction**: 
   - Multiple sections within ONE report type (e.g., QME with history/exam/assessment sections) = SINGLE REPORT = VALID
   - Multiple DIFFERENT report titles (e.g., "QME" title AND "PR2" title) = MULTIPLE REPORTS = INVALID
4. A report discussing multiple visits/dates is usually STILL one report (unless they have different titles)
5. A report referencing external documents is STILL one report (unless the other report's title is present)
6. **Examples of MULTIPLE REPORTS (INVALID)** - ANY combination of 2+ distinct report titles:
   - Document with "QUALIFIED MEDICAL EVALUATION" title AND "PROGRESS REPORT - PR2" title = MULTIPLE REPORTS = INVALID
   - Document with "CONSULTATION REPORT" title AND "RADIOLOGY REPORT" title = MULTIPLE REPORTS = INVALID
   - Document with "DISCHARGE SUMMARY" title AND "OPERATIVE REPORT" title = MULTIPLE REPORTS = INVALID
   - Document with "QME" header AND "PR2" header = MULTIPLE REPORTS = INVALID
   - Document with "PROGRESS REPORT - PR2" title AND "PROGRESS REPORT - PR4" title = MULTIPLE REPORTS = INVALID
   - Document with "AUTHORIZATION REQUEST" title AND "PROGRESS REPORT" title = MULTIPLE REPORTS = INVALID
   - Document with ANY 2+ distinct report titles (regardless of type) = MULTIPLE REPORTS = INVALID
7. **Examples of SINGLE REPORT (VALID)**:
   - A document with only ONE report title (even with multiple sections) = SINGLE REPORT = VALID
   - A document with only "QUALIFIED MEDICAL EVALUATION" title (even with multiple sections) = SINGLE REPORT = VALID
   - A document with only "PROGRESS REPORT - PR2" title (even with multiple dates/visits) = SINGLE REPORT = VALID
   - A document with only "CONSULTATION REPORT" title = SINGLE REPORT = VALID
   - A report that mentions another report type but doesn't include its title = SINGLE REPORT = VALID

## OUTPUT FORMAT
Provide your analysis in the following JSON format:
{format_instructions}
"""

    USER_PROMPT = """Analyze the following Document AI Summarizer output and identify ALL REPORT TITLES that appear at the TOP OF EACH REPORT SECTION throughout the entire document.

## DOCUMENT SUMMARY TO ANALYZE:
{summary_text}

## YOUR ANALYSIS - STEP BY STEP:

**STEP 1: SCAN ENTIRE DOCUMENT FOR ANY REPORT SECTION TITLES**
Carefully scan through the ENTIRE document from beginning to end, looking for ANY report titles/headers that appear at the TOP OF EACH REPORT SECTION.

Report titles appear:
- At the beginning of each report section (not just the very top of the document)
- After page breaks or section separators
- As major section headings that indicate a new report is starting
- Report titles are typically in ALL CAPS or Title Case
- Report titles often include words like: "REPORT", "EVALUATION", "SUMMARY", "RECORD", "ASSESSMENT", "NOTE", "LETTER", "CONSULTATION", "AUTHORIZATION", etc.

Examples of report titles to look for (but DO NOT limit yourself to only these):
  * "QUALIFIED MEDICAL EVALUATION" or "QME" (appears at top of QME report section)
  * "PROGRESS REPORT - PR2" or "PR-2" or "PR2" (appears at top of PR2 report section)
  * "PROGRESS REPORT - PR4" or "PR-4" or "PR4" (appears at top of PR4 report section)
  * "DOCTOR'S FIRST REPORT" or "DFR" (appears at top of DFR report section)
  * "AGREED MEDICAL EVALUATION" or "AME" (appears at top of AME report section)
  * "INDEPENDENT MEDICAL EVALUATION" or "IME" (appears at top of IME report section)
  * "CONSULTATION REPORT" or "CONSULT" (appears at top of consultation report section)
  * "RADIOLOGY REPORT" or "IMAGING REPORT" (appears at top of radiology report section)
  * "DISCHARGE SUMMARY" or "DISCHARGE REPORT" (appears at top of discharge report section)
  * "OPERATIVE REPORT" or "SURGERY REPORT" (appears at top of operative report section)
  * "AUTHORIZATION REQUEST" or "TREATMENT AUTHORIZATION" (appears at top of authorization section)
  * "PHYSICAL THERAPY REPORT" or "PT REPORT" (appears at top of PT report section)
  * "FUNCTIONAL CAPACITY EVALUATION" or "FCE" (appears at top of FCE report section)
  * ANY OTHER TYPE OF MEDICAL REPORT OR DOCUMENT TITLE

**CRITICAL**: 
- Look for titles that appear at the START of each report section, not just references to report types in the text.
- Do NOT limit yourself to only known/common report types. Identify ANY distinct report title you find.
- If you see ANY title that looks like it's the header of a new report section, include it in your analysis.

**STEP 2: LIST ALL DISTINCT REPORT TITLES FOUND**
Go through the document and list EVERY distinct report title you find that appears at the top of a report section, regardless of whether it's a common type or not:
- Title 1: [e.g., "QUALIFIED MEDICAL EVALUATION" or "CONSULTATION REPORT" or "RADIOLOGY REPORT" or ANY other report title]
- Title 2: [e.g., "PROGRESS REPORT - PR2" or "DISCHARGE SUMMARY" or ANY other report title]
- Title 3: [if any more...]

**IMPORTANT**: Include ALL report titles you find, even if they're not in the common list. Do NOT skip titles just because they're unfamiliar.

**STEP 3: COUNT AND DETERMINE**
- How many DISTINCT report titles did you find? [Count]
- If you found 2 or more distinct titles (of ANY type) = MULTIPLE REPORTS = INVALID DOCUMENT
- If you found only 1 title = SINGLE REPORT = VALID DOCUMENT

**STEP 4: PROVIDE EVIDENCE**
List the exact report titles you found and where they appear in the document (e.g., "Found 'QUALIFIED MEDICAL EVALUATION' at the beginning, then found 'PROGRESS REPORT - PR2' later in the document" or "Found 'CONSULTATION REPORT' at line X, then found 'RADIOLOGY REPORT' at line Y").

## CRITICAL RULE: If you find 2+ distinct report titles (of ANY type) that appear at the top of different report sections, this document is INVALID and must be flagged as multiple reports, even if they're for the same patient. Do NOT limit yourself to only known report types - identify ANY distinct report titles you find.

Provide your determination in JSON format with:
- is_multiple: true if 2+ distinct report section titles found (of ANY type), false if only 1 title
- reports_identified: List of ALL report section titles you found (exact titles as they appear, including any uncommon types)
- reason: Explain which report titles you found (including their types), where they appear, and why this is single/multiple"""

    def _detect_report_titles_pattern(self, text: str) -> Dict[str, Any]:
        """
        Pattern-based detection to identify multiple report titles/headers.
        Scans the ENTIRE document for report titles that appear at the top of each report section.
        Returns detection result if multiple distinct titles are found.
        """
        text_upper = text.upper()
        text_lines = text.split('\n')
        
        # Report title patterns - looking for titles that appear at the top of report sections
        report_titles_found = []
        report_title_positions = {}  # Track where each title appears
        
        # Scan ENTIRE document, not just first 20 lines
        # Look for report titles that appear at the start of lines (indicating new report sections)
        full_text_upper = text_upper
        
        # QME title patterns - look for titles at start of lines or after page breaks
        qme_patterns = [
            r'(?:^|\n)\s*QUALIFIED\s+MEDICAL\s+EVALUATION\s*(?:\(QME\)|REPORT|FOR|:)?',
            r'(?:^|\n)\s*QME\s+(?:REPORT|EVALUATION|EXAM)',
            r'(?:^|\n)\s*QUALIFIED\s+MEDICAL\s+EVALUATION\s+(?:REPORT|FOR)',
        ]
        for pattern in qme_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "QME" not in report_titles_found:
                    report_titles_found.append("QME")
                    # Get line number
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["QME"] = line_num
                break
        
        # AME title patterns
        ame_patterns = [
            r'(?:^|\n)\s*AGREED\s+MEDICAL\s+EVALUATION\s*(?:\(AME\)|REPORT|FOR|:)?',
            r'(?:^|\n)\s*AME\s+(?:REPORT|EVALUATION|EXAM)',
            r'(?:^|\n)\s*AGREED\s+MEDICAL\s+EVALUATION\s+(?:REPORT|FOR)',
        ]
        for pattern in ame_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "AME" not in report_titles_found:
                    report_titles_found.append("AME")
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["AME"] = line_num
                break
        
        # IME title patterns
        ime_patterns = [
            r'(?:^|\n)\s*INDEPENDENT\s+MEDICAL\s+EVALUATION\s*(?:\(IME\)|REPORT|FOR|:)?',
            r'(?:^|\n)\s*IME\s+(?:REPORT|EVALUATION|EXAM)',
            r'(?:^|\n)\s*INDEPENDENT\s+MEDICAL\s+EVALUATION\s+(?:REPORT|FOR)',
        ]
        for pattern in ime_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "IME" not in report_titles_found:
                    report_titles_found.append("IME")
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["IME"] = line_num
                break
        
        # PR2 title patterns - look for "PROGRESS REPORT" with "PR2" or "PR-2" at start of lines
        pr2_patterns = [
            r'(?:^|\n)\s*PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*2',
            r'(?:^|\n)\s*PR\s*[-]?\s*2\s+(?:PRIMARY\s+TREATING\s+PHYSICIAN\'?S\s+)?(?:PROGRESS\s+REPORT|REPORT)',
            r'(?:^|\n)\s*PR\s*[-]?\s*2\s*[:]?\s*PRIMARY\s+TREATING\s+PHYSICIAN',
            r'(?:^|\n)\s*PROGRESS\s+REPORT.*?PR\s*[-]?\s*2',
        ]
        for pattern in pr2_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "PR2" not in report_titles_found:
                    report_titles_found.append("PR2")
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["PR2"] = line_num
                break
        
        # PR4 title patterns - look for titles at start of lines
        pr4_patterns = [
            r'(?:^|\n)\s*PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*4',
            r'(?:^|\n)\s*PR\s*[-]?\s*4\s+(?:PROGRESS\s+REPORT|REPORT)',
            r'(?:^|\n)\s*PERMANENT\s+AND\s+STATIONARY',
            r'(?:^|\n)\s*PERMANENT\s+STATIONARY',
            r'(?:^|\n)\s*PROGRESS\s+REPORT.*?PR\s*[-]?\s*4',
        ]
        for pattern in pr4_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "PR4" not in report_titles_found:
                    report_titles_found.append("PR4")
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["PR4"] = line_num
                break
        
        # DFR title patterns - look for titles at start of lines
        dfr_patterns = [
            r'(?:^|\n)\s*DOCTOR[\'S]?\s+FIRST\s+REPORT',
            r'(?:^|\n)\s*DFR\s+(?:REPORT|FORM)',
            r'(?:^|\n)\s*DOCTOR[\'S]?\s+FIRST\s+REPORT\s+(?:OF|FOR)',
        ]
        for pattern in dfr_patterns:
            matches = re.finditer(pattern, full_text_upper, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if "DFR" not in report_titles_found:
                    report_titles_found.append("DFR")
                    line_num = text_upper[:match.start()].count('\n') + 1
                    report_title_positions["DFR"] = line_num
                break
        
        # Remove duplicates while preserving order
        unique_titles = []
        for title in report_titles_found:
            if title not in unique_titles:
                unique_titles.append(title)
        
        # If we find 2+ distinct report titles, it's multiple reports
        if len(unique_titles) >= 2:
            # Build detailed reason with positions
            position_info = []
            for title in unique_titles:
                if title in report_title_positions:
                    position_info.append(f"{title} (found at line {report_title_positions[title]})")
                else:
                    position_info.append(title)
            
            return {
                "is_multiple": True,
                "confidence": "high",
                "reason": f"Pattern detection found multiple distinct report section titles: {', '.join(position_info)}. These titles appear at the top of different report sections throughout the document.",
                "report_count_estimate": len(unique_titles),
                "reports_identified": unique_titles
            }
        
        return None  # No clear pattern match

    def _detect_report_types_pattern(self, text: str) -> Dict[str, Any]:
        """
        Legacy pattern-based detection (kept for backward compatibility).
        Now calls the more specific title detection method.
        """
        return self._detect_report_titles_pattern(text)

    def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
        """
        Analyze Document AI Summarizer output to detect multiple reports using AI reasoning.
        Uses pattern-based detection as a first pass, then LLM analysis.
        
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
        
        # First, try pattern-based detection to find multiple report titles
        pattern_result = self._detect_report_titles_pattern(summary_text)
        if pattern_result:
            logger.info(f"🔍 Pattern detection found multiple reports: {pattern_result.get('reports_identified')}")
            logger.info(f"   Reason: {pattern_result.get('reason')}")
            # If pattern detection finds multiple distinct report types with high confidence, return immediately
            # This catches QME+PR2, QME+PR4, etc. combinations reliably
            if pattern_result.get('confidence') == 'high' and len(pattern_result.get('reports_identified', [])) >= 2:
                logger.info("✅ Pattern detection confirms multiple reports - returning result immediately")
                logger.info("=" * 80)
                logger.info("📊 MULTI-REPORT DETECTION RESULTS (Pattern-based):")
                logger.info("=" * 80)
                logger.info(f"   Is Multiple Reports: {pattern_result.get('is_multiple', False)}")
                logger.info(f"   Confidence: {pattern_result.get('confidence', 'unknown')}")
                logger.info(f"   Estimated Report Count: {pattern_result.get('report_count_estimate', 2)}")
                logger.info(f"   Reason: {pattern_result.get('reason', 'No reason provided')}")
                logger.info(f"   Reports Identified: {pattern_result.get('reports_identified', [])}")
                logger.info("=" * 80)
                return pattern_result
        
        try:
            logger.info(f"🔍 Analyzing document for multiple reports using {self.model_type}...")
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
            logger.info(f"📊 MULTI-REPORT DETECTION RESULTS ({self.model_type}):")
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
            error_msg = str(e)
            logger.error(f"❌ Multi-report detection failed: {error_msg}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Check if it's a 404 error (deployment not found)
            if "404" in error_msg or "Resource not found" in error_msg:
                logger.error(f"⚠️ Deployment '{self.deployment_name}' not found. Please check your Azure OpenAI configuration.")
                logger.error(f"   Available config: o3_model={CONFIG.get('azure_openai_o3_model')}, regular_deployment={CONFIG.get('azure_openai_deployment')}")
            
            # Return safe default on error - don't fail the document processing
            return {
                "is_multiple": False,
                "confidence": "low",
                "reason": f"Detection failed due to error: {error_msg}",
                "report_count_estimate": 1,
                "reports_identified": []
            }


# Singleton instance
_detector_instance = None
_initialization_failed = False

def get_multi_report_detector() -> MultiReportDetector:
    """Get singleton MultiReportDetector instance with error handling"""
    global _detector_instance, _initialization_failed
    
    if _detector_instance is None and not _initialization_failed:
        try:
            logger.info("🚀 Initializing Multi-Report Detector...")
            _detector_instance = MultiReportDetector()
        except Exception as e:
            logger.error(f"❌ Failed to initialize MultiReportDetector: {str(e)}")
            _initialization_failed = True
            # Return a dummy instance that will return safe defaults
            class DummyDetector:
                def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
                    logger.warning("⚠️ MultiReportDetector not available - returning safe default")
                    return {
                        "is_multiple": False,
                        "confidence": "low",
                        "reason": "MultiReportDetector initialization failed - detection disabled",
                        "report_count_estimate": 1,
                        "reports_identified": []
                    }
            _detector_instance = DummyDetector()
    
    if _detector_instance is None:
        # Last resort fallback
        class DummyDetector:
            def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
                return {
                    "is_multiple": False,
                    "confidence": "low",
                    "reason": "MultiReportDetector not available",
                    "report_count_estimate": 1,
                    "reports_identified": []
                }
        return DummyDetector()
    
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
