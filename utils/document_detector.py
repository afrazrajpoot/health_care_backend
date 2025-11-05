"""
Document type detection with hybrid pattern + LLM approach
"""
import re
import logging
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import DocumentType

logger = logging.getLogger("document_ai")


class DocumentTypeDetector:
    """
    Hybrid document type detection with context-aware pattern matching.
    Prevents misclassification (e.g., PR-2 mentioning MRI classified as MRI report).
    """
    
    # [KEEP ALL THE DOC_TYPE_PATTERNS EXACTLY AS IN YOUR FILE - file:27]
    DOC_TYPE_PATTERNS = {
        # Med-Legal (highest priority)
        DocumentType.QME: {
            "primary": [
                r'\bQME\b.*(?:Report|Evaluation)',
                r'Qualified Medical Evaluator',
                r'QME.*(?:Exam|Evaluation)'
            ],
            "context": [
                r'apportionment',
                r'whole person impairment',
                r'medical-legal',
                r'permanent.*stationary',
                r'WPI',
                r'causation.*opinion'
            ],
            "exclude": [
                r'QME.*(?:requested|pending|scheduled)',
                r'request.*QME',
                r'awaiting.*QME'
            ]
        },
        # ... [COPY ALL OTHER PATTERNS FROM YOUR FILE - file:27] ...
    }
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def detect(self, text: str) -> DocumentType:
        """Hybrid detection with pattern + LLM fallback"""
        # Try pattern-based first (fast, accurate for 90% cases)
        detected = self._pattern_based_detection(text[:3000])
        if detected and detected != DocumentType.UNKNOWN:
            logger.info(f"✅ Pattern detected: {detected.value}")
            return detected
        
        # Fallback to LLM for edge cases
        logger.info("⚠️ Pattern unclear, using LLM classifier")
        return self._llm_based_detection(text[:4000])
    
    def _pattern_based_detection(self, text: str) -> DocumentType:
        """Context-aware pattern matching with scoring"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            score = 0
            
            # Check exclusions first
            excluded = False
            for exclude_pattern in patterns.get("exclude", []):
                if re.search(exclude_pattern, text_lower, re.IGNORECASE):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Primary patterns (weight: 10)
            for primary in patterns["primary"]:
                if re.search(primary, text_lower, re.IGNORECASE):
                    score += 10
            
            # Context patterns (weight: 3)
            for context in patterns["context"]:
                if re.search(context, text_lower, re.IGNORECASE):
                    score += 3
            
            if score > 0:
                scores[doc_type] = score
        
        if scores and max(scores.values()) >= 10:
            return max(scores, key=scores.get)
        
        return DocumentType.UNKNOWN
    
    def _llm_based_detection(self, text: str) -> DocumentType:
        """LLM fallback for ambiguous cases"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        prompt = PromptTemplate(
            template="""
Analyze this medical document and classify its PRIMARY type.

CRITICAL RULES:
1. If imaging (MRI/CT/X-ray) is only MENTIONED but not the PRIMARY SUBJECT → NOT an imaging report
2. QME/AME/IME = comprehensive med-legal evaluations (not simple progress notes)
3. PR-2 = progress reports; PR-4 = final/permanent & stationary reports
4. RFA = requests; UR = denials; Authorization = approvals

Document excerpt:
{text}

Classify into ONE type from this list:
MRI, CT, X-ray, Ultrasound, EMG, Labs, PR-2, PR-4, DFR, Consult, RFA, UR, Authorization,
Peer-to-Peer, QME, AME, IME, Adjuster, Attorney, NCM, Signature Request, Referral,
Discharge, Med Refill

Return JSON:
{{"doc_type": "exact_type_from_list", "confidence": "high|medium|low", "reason": "brief justification"}}

{format_instructions}
""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({"text": text})
            doc_type_str = result.get("doc_type", "Unknown")
            
            # Map string to enum
            for dt in DocumentType:
                if dt.value == doc_type_str:
                    return dt
            
            return DocumentType.UNKNOWN
        except Exception as e:
            logger.error(f"❌ LLM detection failed: {e}")
            return DocumentType.UNKNOWN
