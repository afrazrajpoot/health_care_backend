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
    
    SYSTEM_PROMPT = """You are an expert medical document analyst. Your task is to determine if a document contains MULTIPLE SEPARATE REPORTS or is a SINGLE REPORT that may reference other documents.

## CRITICAL DISTINCTION: ACTUAL REPORTS vs. REFERENCES

**A SINGLE report often REFERENCES other reports without containing them:**
- "I reviewed the patient's PR2 dated 01/15/2024" - This is a REFERENCE, not a separate report
- "Based on the QME by Dr. Smith" - This is a REFERENCE
- "The PR-2 report indicated..." - This is a REFERENCE
- "See attached AME dated..." - This is a REFERENCE
- "Records reviewed: PR2, QME, imaging reports" - These are REFERENCES

**MULTIPLE reports means ACTUAL SEPARATE DOCUMENTS combined together:**
- Each report has its own distinct HEADER/TITLE at the TOP of a section
- Each report has its own patient information section
- Each report has its own date and provider signature
- There is a clear document boundary/separation between reports

## HOW TO IDENTIFY ACTUAL REPORT HEADERS vs. REFERENCES

**Actual Report Headers (indicate separate reports):**
1. Appear at the TOP of a document section, not in flowing text
2. Are standalone titles on their own line or prominently displayed
3. Are followed by report metadata (patient name, DOB, date of evaluation, etc.)
4. Mark the BEGINNING of a complete report structure
5. Example: A line that just says "QUALIFIED MEDICAL EVALUATION" followed by patient info

**References (do NOT indicate separate reports):**
1. Appear WITHIN the narrative text of a report
2. Are preceded by words like: "reviewed", "referenced", "attached", "per the", "based on", "see", "copy of"
3. Are followed by words like: "dated", "by Dr.", "indicates", "shows", "revealed"
4. Are mentioned in a "Records Reviewed" or "Documents Reviewed" section
5. Example: "I reviewed the PR-2 dated 03/15/2024 which showed..."

## EXAMPLES

**SINGLE REPORT (even with multiple report type mentions):**
- A QME report that states "I reviewed the patient's PR2 reports from January and March"
- A report with a "Records Reviewed" section listing "QME, PR2, PR4, imaging studies"
- A consultation that references "the prior AME by Dr. Johnson dated 06/01/2024"
- A report that mentions "as documented in the DFR from the initial injury"

**MULTIPLE REPORTS (separate documents combined):**
- Document starts with "QUALIFIED MEDICAL EVALUATION" header, then later has a completely new "PROGRESS REPORT - PR2" header with its own patient info section
- Two distinct report sections, each with their own title, date, and signature
- Clear document boundaries where one report ends and another begins

## YOUR ANALYSIS STEPS

1. **Identify the PRIMARY report type** - What is the main report in this document?
2. **Look for OTHER report type mentions** - Are they HEADERS or REFERENCES?
3. **For each mention, determine:**
   - Is it at the TOP of a section (header) or WITHIN text (reference)?
   - Is it preceded/followed by reference indicators?
   - Does it have its own patient info, date, and structure?
4. **Conclusion:** Only flag as multiple reports if you find 2+ ACTUAL report headers

## OUTPUT FORMAT
{format_instructions}
"""

    USER_PROMPT = """Analyze the following document summary and determine if it contains MULTIPLE SEPARATE REPORTS or is a SINGLE REPORT that may reference other documents.

## DOCUMENT SUMMARY:
{summary_text}

## YOUR ANALYSIS:

**Step 1: Identify the PRIMARY report**
What is the main report type in this document?

**Step 2: Find all report type mentions**
List every mention of report types (QME, PR2, PR4, AME, IME, DFR, etc.)

**Step 3: Classify each mention**
For each mention, determine:
- Is this an ACTUAL REPORT HEADER (standalone title at top of section)?
- Or is this a REFERENCE (mentioned within text, preceded by "reviewed", "see", "dated", etc.)?

**Step 4: Look for reference indicators**
Check if mentions are preceded/followed by:
- "reviewed", "referenced", "see", "per the", "based on", "copy of", "attached"
- "dated", "by Dr.", "indicates", "shows", "was reviewed"
- "Records Reviewed:", "Documents Reviewed:"

**Step 5: Determine if truly multiple reports**
- If only ONE actual report header exists (others are just references) = SINGLE REPORT
- If TWO OR MORE actual report headers exist with their own sections = MULTIPLE REPORTS

## CRITICAL RULE
A single report that REFERENCES other reports is still a SINGLE REPORT. Only flag as multiple if there are clearly SEPARATE report sections with their own headers and structure.

Provide your response in JSON format."""

    def _is_reference_context(self, text: str, match_start: int, match_end: int) -> bool:
        """
        Check if a report title match appears in a reference/citation context rather than as an actual report header.
        Returns True if this is likely a reference to another report, not an actual report section.
        """
        # Get surrounding context (200 chars before and after)
        context_start = max(0, match_start - 200)
        context_end = min(len(text), match_end + 200)
        
        before_context = text[context_start:match_start].upper()
        after_context = text[match_end:context_end].upper()
        
        # Get the line containing the match
        line_start = text.rfind('\n', 0, match_start) + 1
        line_end = text.find('\n', match_end)
        if line_end == -1:
            line_end = len(text)
        full_line = text[line_start:line_end].strip().upper()
        
        # Reference indicator words that suggest this is a mention, not a report header
        reference_indicators_before = [
            'REVIEWED', 'REVIEW OF', 'REVIEWING', 'SEE', 'REFER TO', 'REFERENCED',
            'ATTACHED', 'ENCLOSED', 'PER THE', 'PER MY', 'IN THE', 'IN MY',
            'FROM THE', 'FROM MY', 'BASED ON', 'ACCORDING TO', 'AS NOTED IN',
            'AS STATED IN', 'AS DOCUMENTED IN', 'AS PER', 'PREVIOUS', 'PRIOR',
            'RECEIVED', 'OBTAINED', 'REVIEWED THE', 'I REVIEWED', 'WE REVIEWED',
            'UPON REVIEW', 'AFTER REVIEW', 'RECORDS REVIEWED', 'DOCUMENTS REVIEWED',
            'SUBMITTED', 'PROVIDED', 'FORWARDED', 'INCLUDED', 'COPY OF',
            'DATED', 'OF THE', 'THE PATIENT\'S', 'PATIENT\'S', 'HIS', 'HER',
            'THEIR', 'THIS PATIENT\'S', 'MR.', 'MS.', 'MRS.', 'DR.',
        ]
        
        reference_indicators_after = [
            'DATED', 'FROM', 'BY DR', 'BY DOCTOR', 'PERFORMED BY', 'CONDUCTED BY',
            'WAS REVIEWED', 'WERE REVIEWED', 'INDICATES', 'INDICATED', 'SHOWS',
            'SHOWED', 'REVEALED', 'DEMONSTRATES', 'DOCUMENTED', 'NOTED',
            'STATES', 'STATED', 'REPORTS', 'REPORTED', 'CONCLUDED', 'FINDINGS',
        ]
        
        # Check for reference indicators before the match
        for indicator in reference_indicators_before:
            if indicator in before_context:
                logger.debug(f"   Found reference indicator '{indicator}' before match")
                return True
        
        # Check for reference indicators after the match
        for indicator in reference_indicators_after:
            if indicator in after_context[:100]:  # Only check first 100 chars after
                logger.debug(f"   Found reference indicator '{indicator}' after match")
                return True
        
        # Check if the line contains typical reference patterns
        reference_line_patterns = [
            r'REVIEWED.*(?:QME|PR-?2|PR-?4|AME|IME|DFR)',
            r'(?:QME|PR-?2|PR-?4|AME|IME|DFR).*DATED',
            r'(?:QME|PR-?2|PR-?4|AME|IME|DFR).*(?:BY DR|BY DOCTOR)',
            r'(?:SEE|REFER TO|ATTACHED).*(?:QME|PR-?2|PR-?4|AME|IME|DFR)',
            r'(?:COPY OF|THE).*(?:QME|PR-?2|PR-?4|AME|IME|DFR)',
            r'(?:PREVIOUS|PRIOR).*(?:QME|PR-?2|PR-?4|AME|IME|DFR)',
        ]
        
        for pattern in reference_line_patterns:
            if re.search(pattern, full_line, re.IGNORECASE):
                logger.debug(f"   Line matches reference pattern: {pattern}")
                return True
        
        return False

    def _is_standalone_title(self, text: str, match_start: int, match_end: int) -> bool:
        """
        Check if a report title match appears as a standalone title/header.
        A standalone title typically:
        - Appears on its own line or nearly so
        - Is at the beginning of a document section
        - Is followed by report metadata (patient name, date, etc.)
        """
        # Get the line containing the match
        line_start = text.rfind('\n', 0, match_start) + 1
        line_end = text.find('\n', match_end)
        if line_end == -1:
            line_end = len(text)
        
        full_line = text[line_start:line_end].strip()
        matched_text = text[match_start:match_end].strip()
        
        # Check if the match takes up most of the line (indicating a standalone title)
        # Allow for some extra characters like colons, report numbers, etc.
        line_length = len(full_line)
        match_length = len(matched_text)
        
        if line_length > 0 and match_length / line_length >= 0.5:
            # The match is at least 50% of the line, likely a standalone title
            return True
        
        # Check if the line is short (typical for headers)
        if line_length < 80:
            # Short line, check if it looks like a header
            header_indicators = [
                r'^[\s]*(?:QUALIFIED\s+MEDICAL\s+EVALUATION|QME|AME|IME|PR-?2|PR-?4|DFR)',
                r'^[\s]*(?:PROGRESS\s+REPORT)',
                r'^[\s]*(?:DOCTOR\'?S?\s+FIRST\s+REPORT)',
                r'^[\s]*(?:AGREED\s+MEDICAL\s+EVALUATION)',
                r'^[\s]*(?:INDEPENDENT\s+MEDICAL\s+EVALUATION)',
            ]
            for pattern in header_indicators:
                if re.match(pattern, full_line, re.IGNORECASE):
                    return True
        
        return False

    def _detect_report_titles_pattern(self, text: str) -> Dict[str, Any]:
        """
        Context-aware pattern-based detection to identify multiple report titles/headers.
        Distinguishes between actual report section headers vs. references to other reports.
        Only flags as multiple reports if there are clearly separate report sections.
        """
        text_upper = text.upper()
        
        # Report titles found as actual headers (not references)
        actual_report_titles = []
        report_title_positions = {}
        
        # Referenced reports (mentioned but not actual separate documents)
        referenced_reports = []
        
        logger.debug("🔍 Starting context-aware report title detection...")
        
        # Define report type patterns with their identifiers
        report_patterns = {
            "QME": [
                r'QUALIFIED\s+MEDICAL\s+EVALUATION\s*(?:\(QME\)|REPORT|FOR|:)?',
                r'QME\s+(?:REPORT|EVALUATION|EXAM)',
            ],
            "AME": [
                r'AGREED\s+MEDICAL\s+EVALUATION\s*(?:\(AME\)|REPORT|FOR|:)?',
                r'AME\s+(?:REPORT|EVALUATION|EXAM)',
            ],
            "IME": [
                r'INDEPENDENT\s+MEDICAL\s+EVALUATION\s*(?:\(IME\)|REPORT|FOR|:)?',
                r'IME\s+(?:REPORT|EVALUATION|EXAM)',
            ],
            "PR2": [
                r'PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*2',
                r'PR\s*[-]?\s*2\s+(?:PRIMARY\s+TREATING\s+PHYSICIAN\'?S\s+)?(?:PROGRESS\s+REPORT|REPORT)',
                r'PR\s*[-]?\s*2\s*[:]?\s*PRIMARY\s+TREATING\s+PHYSICIAN',
            ],
            "PR4": [
                r'PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*4',
                r'PR\s*[-]?\s*4\s+(?:PROGRESS\s+REPORT|REPORT)',
                r'PERMANENT\s+AND\s+STATIONARY\s+REPORT',
            ],
            "DFR": [
                r'DOCTOR\'?S?\s+FIRST\s+REPORT(?:\s+OF\s+OCCUPATIONAL)?',
                r'DFR\s+(?:REPORT|FORM)',
            ],
        }
        
        for report_type, patterns in report_patterns.items():
            for pattern in patterns:
                # Look for matches at the start of lines (potential headers)
                full_pattern = r'(?:^|\n)\s*' + pattern
                matches = list(re.finditer(full_pattern, text, re.MULTILINE | re.IGNORECASE))
                
                for match in matches:
                    match_start = match.start()
                    match_end = match.end()
                    line_num = text[:match_start].count('\n') + 1
                    
                    logger.debug(f"   Found potential {report_type} at line {line_num}")
                    
                    # Check if this is a reference context or an actual report header
                    if self._is_reference_context(text, match_start, match_end):
                        logger.debug(f"   -> Classified as REFERENCE (not actual report)")
                        if report_type not in referenced_reports:
                            referenced_reports.append(report_type)
                    elif self._is_standalone_title(text, match_start, match_end):
                        logger.debug(f"   -> Classified as ACTUAL REPORT HEADER")
                        if report_type not in actual_report_titles:
                            actual_report_titles.append(report_type)
                            report_title_positions[report_type] = line_num
                    else:
                        # Ambiguous - check if it appears early in the document (likely main report)
                        # or later (likely reference)
                        if line_num <= 30 and report_type not in actual_report_titles:
                            # Early in document, more likely to be the main report title
                            logger.debug(f"   -> Early in document, classified as POTENTIAL REPORT HEADER")
                            actual_report_titles.append(report_type)
                            report_title_positions[report_type] = line_num
                        else:
                            logger.debug(f"   -> Ambiguous, classified as REFERENCE")
                            if report_type not in referenced_reports:
                                referenced_reports.append(report_type)
                    
                    break  # Only process first match for each pattern
        
        logger.debug(f"   Actual report headers found: {actual_report_titles}")
        logger.debug(f"   Referenced reports (not counted): {referenced_reports}")
        
        # Only flag as multiple reports if we have 2+ ACTUAL report headers
        if len(actual_report_titles) >= 2:
            position_info = []
            for title in actual_report_titles:
                if title in report_title_positions:
                    position_info.append(f"{title} (header at line {report_title_positions[title]})")
                else:
                    position_info.append(title)
            
            return {
                "is_multiple": True,
                "confidence": "high",
                "reason": f"Found {len(actual_report_titles)} distinct report section headers: {', '.join(position_info)}. These appear to be separate reports combined in one document.",
                "report_count_estimate": len(actual_report_titles),
                "reports_identified": actual_report_titles
            }
        
        # If only references were found (no actual headers), it's a single report
        if len(actual_report_titles) <= 1 and len(referenced_reports) > 0:
            logger.debug(f"   Single report that references other reports: {referenced_reports}")
        
        return None  # No clear pattern match for multiple reports

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
