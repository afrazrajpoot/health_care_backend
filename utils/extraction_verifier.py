"""
Post-extraction verification and correction layer
"""
import re
import json
import logging
from typing import List, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ExtractionVerifier:
    """
    Post-extraction verification layer that ensures:
    1. Summary matches required format for document type
    2. All required fields are present
    3. Data consistency (dates, names, abbreviations)
    4. Word limits enforced
    """
    
    # [KEEP ALL FORMAT_SPECS EXACTLY AS IN YOUR FILE - file:27]
    FORMAT_SPECS = {
        "QME": {
            "pattern": r"^\d{2}/\d{2}/\d{2}: QME",
            "max_words": 35,
            "required_elements": ["date", "QME/AME/IME", "body_part or diagnosis"],
            "format_template": "[DATE]: QME (Dr [LastName], [Specialty]) for [Body parts] = [Status] → [Recommendations]"
        },
        "AME": {
            "pattern": r"^\d{2}/\d{2}/\d{2}: AME",
            "max_words": 35,
            "required_elements": ["date", "QME/AME/IME", "body_part or diagnosis"],
            "format_template": "[DATE]: AME (Dr [LastName], [Specialty]) for [Body parts] = [Status] → [Recommendations]"
        },
        "IME": {
            "pattern": r"^\d{2}/\d{2}/\d{2}: IME",
            "max_words": 35,
            "required_elements": ["date", "QME/AME/IME", "body_part or diagnosis"],
            "format_template": "[DATE]: IME (Dr [LastName], [Specialty]) for [Body parts] = [Status] → [Recommendations]"
        },
        "MRI": {
            "pattern": r"^MRI .+ \d{2}/\d{2}/\d{2} =",
            "max_words": 20,
            "required_elements": ["MRI", "body_part", "date", "finding"],
            "format_template": "MRI [Body part] [DATE] = [Primary finding]"
        },
        "CT": {
            "pattern": r"^CT .+ \d{2}/\d{2}/\d{2} =",
            "max_words": 20,
            "required_elements": ["CT", "body_part", "date", "finding"],
            "format_template": "CT [Body part] [DATE] = [Primary finding]"
        },
        "X-ray": {
            "pattern": r"^X-ray .+ \d{2}/\d{2}/\d{2} =",
            "max_words": 20,
            "required_elements": ["X-ray", "body_part", "date", "finding"],
            "format_template": "X-ray [Body part] [DATE] = [Primary finding]"
        },
        "PR-2": {
            "pattern": r"^(Dr\.|MD|DO).+ PR-2 \d{2}/\d{2}/\d{2}",
            "max_words": 25,
            "required_elements": ["physician", "PR-2", "date", "status", "plan"],
            "format_template": "Dr [Name] PR-2 [DATE] [Body part] = [Status]; [Plan]"
        },
        "Consult": {
            "pattern": r"^(Dr\.|MD|DO).+ Consult \d{2}/\d{2}/\d{2}",
            "max_words": 20,
            "required_elements": ["physician", "specialty", "date", "recommendations"],
            "format_template": "Dr [Name] [Specialty] Consult [DATE] = [Recommendations]"
        },
        "RFA": {
            "pattern": r"^RFA \d{2}/\d{2}/\d{2} =",
            "max_words": 18,
            "required_elements": ["RFA", "date", "service"],
            "format_template": "RFA [DATE] = [Service] [Body part] requested"
        },
        "UR": {
            "pattern": r"^UR \d{2}/\d{2}/\d{2} =",
            "max_words": 20,
            "required_elements": ["UR", "date", "service", "reason"],
            "format_template": "UR [DATE] = [Service] denied; [Reason]"
        }
    }
   
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def verify_and_fix(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """
        Verify extraction result and fix if needed.
        Uses LLM to intelligently correct format issues.
        """
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        doc_type = extraction_result.document_type
        summary = extraction_result.summary_line
        
        # Stage 1: Basic validation (fast, no LLM)
        validation_issues = self._validate_format(summary, doc_type)
        
        if not validation_issues:
            logger.info(f"✅ Summary format validated: {doc_type}")
            return extraction_result
        
        # Stage 2: LLM-based correction (only if validation fails)
        logger.warning(f"⚠️ Format issues detected in {doc_type}: {validation_issues}")
        corrected_result = self._llm_correction(extraction_result, validation_issues)
        
        return corrected_result
    
    def _validate_format(self, summary: str, doc_type: str) -> List[str]:
        """Fast validation checks without LLM"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        issues = []
        spec = self.FORMAT_SPECS.get(doc_type, {})
        
        if not spec:
            return issues
        
        # Check 1: Pattern matching
        pattern = spec.get("pattern")
        if pattern and not re.search(pattern, summary):
            issues.append(f"Does not match expected pattern for {doc_type}")
        
        # Check 2: Word count
        word_count = len(summary.split())
        max_words = spec.get("max_words", 30)
        if word_count > max_words:
            issues.append(f"Exceeds word limit ({word_count}/{max_words} words)")
        
        # Check 3: Date format
        date_matches = re.findall(r'\d{2}/\d{2}/\d{2}', summary)
        if not date_matches:
            issues.append("Missing or invalid date format (expected MM/DD/YY)")
        
        # Check 4: Required elements
        required = spec.get("required_elements", [])
        summary_lower = summary.lower()
        for element in required:
            if element == "date" and not date_matches:
                issues.append(f"Missing required element: {element}")
            elif element in ["physician", "Dr"] and "dr" not in summary_lower and "md" not in summary_lower:
                issues.append(f"Missing required element: {element}")
        
        return issues
    
    def _llm_correction(self, extraction_result: ExtractionResult, issues: List[str]) -> ExtractionResult:
        """Use LLM to intelligently fix format issues"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        doc_type = extraction_result.document_type
        spec = self.FORMAT_SPECS.get(doc_type, {})
        
        prompt = PromptTemplate(
            template="""
You are a medical document summary formatter. Fix the summary to match the exact required format.

DOCUMENT TYPE: {doc_type}

CURRENT SUMMARY (with issues):
{current_summary}

DETECTED ISSUES:
{issues}

REQUIRED FORMAT:
{format_template}

RULES:
1. Maintain all factual information from current summary
2. Fix format to match required template exactly
3. Ensure date is in MM/DD/YY format
4. Enforce {max_words}-word maximum
5. Use standard abbreviations (R/L, PT, ESI, f/u, etc.)
6. Do NOT add information not in current summary
7. Do NOT remove key medical facts

RAW EXTRACTED DATA (for reference):
{raw_data}

Return JSON:
{{
  "corrected_summary": "Fixed summary in exact format",
  "changes_made": ["list of changes"],
  "confidence": "high|medium|low"
}}

{format_instructions}
""",
            input_variables=["doc_type", "current_summary", "issues", "format_template", "max_words", "raw_data"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "doc_type": doc_type,
                "current_summary": extraction_result.summary_line,
                "issues": "\n".join(f"- {issue}" for issue in issues),
                "format_template": spec.get("format_template", "Standard format"),
                "max_words": spec.get("max_words", 30),
                "raw_data": json.dumps(extraction_result.raw_data, indent=2)
            })
            
            corrected_summary = result.get("corrected_summary", extraction_result.summary_line)
            changes = result.get("changes_made", [])
            confidence = result.get("confidence", "low")
            
            logger.info(f"✅ Summary corrected for {doc_type} (confidence: {confidence})")
            logger.info(f"   Changes: {', '.join(changes)}")
            
            return ExtractionResult(
                document_type=extraction_result.document_type,
                document_date=extraction_result.document_date,
                summary_line=corrected_summary,
                examiner_name=extraction_result.examiner_name,
                specialty=extraction_result.specialty,
                body_parts=extraction_result.body_parts,
                raw_data=extraction_result.raw_data
            )
            
        except Exception as e:
            logger.error(f"❌ LLM correction failed: {e}")
            return extraction_result
