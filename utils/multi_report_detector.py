"""
Robust Multi-Report Detector with multiple detection strategies.
Uses ensemble approach for consistent and reliable detection.
"""

import logging
import json
import re
from typing import Dict, Any, List, Tuple
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
    report_types: list = Field(default_factory=list, description="Types of reports found")
    detection_method: str = Field(default="hybrid", description="Method used for detection")


class MultiReportDetector:
    """Detects if a document contains multiple separate reports using multiple strategies"""
    
    # Known report type patterns
    REPORT_TYPE_PATTERNS = {
        'QME': r'\b(?:QME|Qualified Medical Evaluation|Q\.M\.E\.)\b',
        'PQME': r'\b(?:PQME|Panel Qualified Medical Evaluation|P\.Q\.M\.E\.)\b',
        'AME': r'\b(?:AME|Agreed Medical Evaluation|A\.M\.E\.)\b',
        'PR-2': r'\b(?:PR-2|PR2|Progress Report|P\.R\.\s*2)\b',
        'PR-3': r'\b(?:PR-3|PR3|P\.R\.\s*3)\b',
        'PR-4': r'\b(?:PR-4|PR4|P\.R\.\s*4)\b',
        'IME': r'\b(?:IME|Independent Medical Evaluation|I\.M\.E\.)\b',
        'DFR': r'\b(?:DFR|Doctor First Report|D\.F\.R\.)\b',
    }
    
    # Phrases that indicate separate report sections
    SEPARATION_INDICATORS = [
        r'document contains (?:two|three|multiple|several) (?:separate )?reports?',
        r'first (?:section|part|report) is .{0,50}second (?:section|part|report) is',
        r'followed by (?:a |an )?(?:separate |another |second )?(?:QME|PR-2|PR-3|PR-4|AME|IME)',
        r'combined with (?:a |an )?(?:QME|PR-2|PR-3|PR-4|AME|IME)',
        r'includes both (?:a |an )?(?:QME|PR-2|PR-3|PR-4|AME|IME).{0,50}and (?:a |an )?(?:QME|PR-2|PR-3|PR-4|AME|IME)',
        r'separate patient information (?:sections?|blocks?)',
        r'two distinct reports?',
        r'multiple report types?',
    ]
    
    # Phrases that indicate single report (references/reviews)
    SINGLE_REPORT_INDICATORS = [
        r'reviewed? (?:the )?(?:previous|prior|earlier)',
        r'referenced? (?:the )?(?:previous|prior|earlier)',
        r'based on (?:the )?(?:previous|prior|earlier)',
        r'according to (?:the )?(?:previous|prior|earlier)',
        r'discussed? (?:in |by )?(?:Dr\.|Doctor)',
        r'mentioned? in the',
        r'records? reviewed?',
        r'medical history',
        r'treatment history',
    ]
    
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

    def _pattern_based_detection(self, text: str) -> Tuple[bool, float, str, List[str]]:
        """
        Pattern-based detection using regex and heuristics.
        
        Returns:
            (is_multiple, confidence_score, reasoning, report_types)
        """
        text_lower = text.lower()
        
        # Step 1: Find all report types mentioned
        found_types = []
        for report_type, pattern in self.REPORT_TYPE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_types.append(report_type)
        
        # Step 2: Check for separation indicators
        separation_count = 0
        separation_matches = []
        for pattern in self.SEPARATION_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                separation_count += len(matches)
                separation_matches.extend(matches)
        
        # Step 3: Check for single report indicators
        single_report_count = 0
        for pattern in self.SINGLE_REPORT_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                single_report_count += len(matches)
        
        # Step 4: Special handling for QME/AME/PQME
        is_qme_type = any(rt in found_types for rt in ['QME', 'PQME', 'AME'])
        
        # Decision logic
        if separation_count >= 2:
            # Strong evidence of multiple reports
            confidence = 0.9 if separation_count >= 3 else 0.75
            reasoning = f"Pattern detection found {separation_count} strong indicators of separate reports: {separation_matches[:2]}"
            return True, confidence, reasoning, found_types
        
        elif separation_count == 1 and len(found_types) >= 2:
            # Moderate evidence: one separator + multiple types
            if is_qme_type and single_report_count > separation_count:
                # QME likely reviewing other reports
                confidence = 0.8
                reasoning = f"QME/AME report likely reviewing other evaluations (found {single_report_count} reference indicators)"
                return False, confidence, reasoning, found_types
            else:
                confidence = 0.7
                reasoning = f"Found separation indicator and {len(found_types)} different report types"
                return True, confidence, reasoning, found_types
        
        elif len(found_types) >= 3 and single_report_count > len(found_types):
            # Multiple types mentioned but many references - likely single report
            confidence = 0.75
            reasoning = f"Found {len(found_types)} report types but {single_report_count} reference indicators suggest single report reviewing others"
            return False, confidence, reasoning, found_types
        
        else:
            # Default to single report
            confidence = 0.6
            reasoning = f"Pattern analysis suggests single report (types found: {found_types})"
            return False, confidence, reasoning, found_types

    SIMPLIFIED_SYSTEM_PROMPT = """You are a medical document analyzer. Your task is to determine if a document summary describes ONE report or MULTIPLE SEPARATE reports.

## KEY RULES:

1. **MULTIPLE REPORTS means**: The document contains 2+ complete, separate reports of DIFFERENT types (e.g., QME + PR-2)

2. **SINGLE REPORT means**: One report that may reference or review other reports

3. **QME/AME/PQME** that discuss multiple doctors' evaluations = SINGLE REPORT (it's reviewing others)

4. **Strong indicators of MULTIPLE**:
   - "Document contains a QME report AND a PR-2 report"
   - "First section is [Type A], second section is [Type B]"
   - "Two separate patient information sections"

5. **NOT indicators of multiple**:
   - "Reviewed the previous PR-2"
   - "Dr. Smith's QME was considered"
   - "Based on earlier evaluations"

## YOUR TASK:
1. Identify report types mentioned
2. Determine if they are ACTUAL separate reports or just REFERENCES
3. Count ACTUAL separate reports only
4. Make a clear decision

Respond with valid JSON only:
{format_instructions}"""

    USER_PROMPT = """Analyze this summary:

{summary_text}

Provide your analysis as JSON."""

    def _llm_based_detection(self, text: str) -> Dict[str, Any]:
        """
        LLM-based detection with simplified prompt.
        
        Returns:
            dict with detection results
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SIMPLIFIED_SYSTEM_PROMPT),
                ("human", self.USER_PROMPT)
            ])
            
            chain = prompt | self.llm | self.parser
            
            # Truncate if needed
            max_chars = 12000
            truncated = text[:max_chars]
            
            result = chain.invoke({
                "summary_text": truncated,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            if isinstance(result, dict):
                return result
            return result.dict()
            
        except Exception as e:
            logger.error(f"❌ LLM detection failed: {str(e)}")
            return None

    def _ensemble_decision(
        self,
        pattern_result: Tuple[bool, float, str, List[str]],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine pattern-based and LLM-based results for final decision.
        """
        pattern_is_multiple, pattern_confidence, pattern_reasoning, pattern_types = pattern_result
        
        # If LLM failed, use pattern-based only
        if llm_result is None:
            confidence_map = {0.9: "high", 0.8: "high", 0.75: "high", 0.7: "medium", 0.6: "medium"}
            return {
                "is_multiple": pattern_is_multiple,
                "confidence": confidence_map.get(pattern_confidence, "low"),
                "reasoning": f"[Pattern-based only] {pattern_reasoning}",
                "report_count": len(pattern_types) if pattern_is_multiple else 1,
                "report_types": pattern_types,
                "detection_method": "pattern_only"
            }
        
        # Both methods available - use voting with confidence weighting
        llm_is_multiple = llm_result.get("is_multiple", False)
        llm_confidence_str = llm_result.get("confidence", "medium")
        
        # Convert confidence strings to scores
        confidence_scores = {"high": 0.9, "medium": 0.7, "low": 0.5}
        llm_confidence = confidence_scores.get(llm_confidence_str, 0.7)
        
        # Agreement check
        if pattern_is_multiple == llm_is_multiple:
            # Both agree - high confidence
            final_confidence = max(pattern_confidence, llm_confidence)
            confidence_str = "high" if final_confidence >= 0.8 else "medium"
            
            combined_reasoning = f"Both methods agree. Pattern: {pattern_reasoning}. LLM: {llm_result.get('reasoning', '')}"
            
            return {
                "is_multiple": pattern_is_multiple,
                "confidence": confidence_str,
                "reasoning": combined_reasoning,
                "report_count": llm_result.get("report_count", len(pattern_types) if pattern_is_multiple else 1),
                "report_types": llm_result.get("report_types", pattern_types),
                "detection_method": "ensemble_agreement"
            }
        else:
            # Disagreement - use higher confidence method
            if pattern_confidence > llm_confidence:
                confidence_str = "medium"  # Reduce confidence due to disagreement
                reasoning = f"Methods disagree (using pattern-based). Pattern: {pattern_reasoning}. LLM: {llm_result.get('reasoning', '')}"
                
                return {
                    "is_multiple": pattern_is_multiple,
                    "confidence": confidence_str,
                    "reasoning": reasoning,
                    "report_count": len(pattern_types) if pattern_is_multiple else 1,
                    "report_types": pattern_types,
                    "detection_method": "ensemble_pattern_priority"
                }
            else:
                confidence_str = "medium"  # Reduce confidence due to disagreement
                reasoning = f"Methods disagree (using LLM). Pattern: {pattern_reasoning}. LLM: {llm_result.get('reasoning', '')}"
                
                return {
                    "is_multiple": llm_is_multiple,
                    "confidence": confidence_str,
                    "reasoning": reasoning,
                    "report_count": llm_result.get("report_count", 1),
                    "report_types": llm_result.get("report_types", pattern_types),
                    "detection_method": "ensemble_llm_priority"
                }

    def detect_multiple_reports(self, summary_text: str) -> Dict[str, Any]:
        """
        Analyze document summary using ensemble approach.
        
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
                "report_types": [],
                "detection_method": "length_check"
            }
        
        try:
            logger.info("🔍 Starting multi-strategy detection...")
            logger.info(f"📝 Summary length: {len(summary_text)} characters")
            
            # Strategy 1: Pattern-based detection
            logger.info("🔎 Running pattern-based detection...")
            pattern_result = self._pattern_based_detection(summary_text)
            pattern_is_multiple, pattern_conf, pattern_reason, pattern_types = pattern_result
            logger.info(f"   Pattern result: {pattern_is_multiple} (confidence: {pattern_conf:.2f})")
            
            # Strategy 2: LLM-based detection
            logger.info("🤖 Running LLM-based detection...")
            llm_result = self._llm_based_detection(summary_text)
            if llm_result:
                logger.info(f"   LLM result: {llm_result.get('is_multiple')} (confidence: {llm_result.get('confidence')})")
            else:
                logger.warning("   LLM detection failed, using pattern-based only")
            
            # Strategy 3: Ensemble decision
            logger.info("🎯 Making ensemble decision...")
            final_result = self._ensemble_decision(pattern_result, llm_result)
            
            # Log final results
            logger.info("=" * 80)
            logger.info("📊 FINAL DETECTION RESULTS:")
            logger.info("=" * 80)
            logger.info(f"   Multiple Reports: {final_result['is_multiple']}")
            logger.info(f"   Confidence: {final_result['confidence']}")
            logger.info(f"   Report Count: {final_result['report_count']}")
            logger.info(f"   Report Types: {final_result['report_types']}")
            logger.info(f"   Detection Method: {final_result['detection_method']}")
            logger.info(f"   Reasoning: {final_result['reasoning'][:200]}...")
            logger.info("=" * 80)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "is_multiple": False,
                "confidence": "low",
                "reasoning": f"Detection failed: {str(e)}",
                "report_count": 1,
                "report_types": [],
                "detection_method": "error_fallback"
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
                        "report_types": [],
                        "detection_method": "initialization_failed"
                    }
            _detector_instance = DummyDetector()
    
    return _detector_instance


def detect_multiple_reports(summary_text: str) -> Dict[str, Any]:
    """Convenience function to detect multiple reports"""
    detector = get_multi_report_detector()
    return detector.detect_multiple_reports(summary_text)