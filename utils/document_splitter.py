"""
Document Splitter - Splits multi-report documents into individual reports
"""
import logging
import re
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config.settings import CONFIG

logger = logging.getLogger("document_splitter")


class ReportSplit(BaseModel):
    """Model for a split report"""
    report_text: str = Field(description="The text content of this report")
    report_title: str = Field(description="The title/type of this report (e.g., QME, PR2, PR4)")
    start_index: int = Field(description="Starting character index in original document")
    end_index: int = Field(description="Ending character index in original document")


class DocumentSplitResult(BaseModel):
    """Result model for document splitting"""
    splits: List[ReportSplit] = Field(description="List of split reports")
    total_reports: int = Field(description="Total number of reports found")


class DocumentSplitter:
    """
    Splits multi-report documents into individual reports based on report titles.
    """
    
    def __init__(self):
        """Initialize the splitter with Azure OpenAI model"""
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                deployment_name=CONFIG.get("azure_openai_deployment"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.1,
                timeout=120,
            )
            self.parser = JsonOutputParser(pydantic_object=DocumentSplitResult)
            logger.info("✅ DocumentSplitter initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DocumentSplitter: {str(e)}")
            raise
    
    SYSTEM_PROMPT = """You are an expert document analyzer. Your task is to split a multi-report document into individual reports based on report titles/headers.

## YOUR TASK
Analyze the document text and identify where each report starts and ends based on report titles/headers.

## REPORT TITLES TO LOOK FOR
Common report titles include:
- "QUALIFIED MEDICAL EVALUATION" or "QME"
- "PROGRESS REPORT - PR2" or "PR-2" or "PR2"
- "PROGRESS REPORT - PR4" or "PR-4" or "PR4"
- "DOCTOR'S FIRST REPORT" or "DFR"
- "AGREED MEDICAL EVALUATION" or "AME"
- "INDEPENDENT MEDICAL EVALUATION" or "IME"
- "CONSULTATION REPORT" or "CONSULT"
- And other distinct report type titles

## SPLITTING RULES
1. Each report starts when you see a report title/header
2. Each report ends when you see the next report title/header OR the end of the document
3. Include the report title in the split report text
4. Preserve all content between report titles
5. If only one report title is found, return the entire document as a single split

## OUTPUT FORMAT
Provide your analysis in the following JSON format:
{format_instructions}
"""

    USER_PROMPT = """Split the following document into individual reports based on report titles/headers.

## DOCUMENT TEXT:
{document_text}

## YOUR ANALYSIS:
1. Identify all report titles/headers in the document
2. For each report title, determine where that report starts and ends
3. Split the document accordingly

Provide the splits in JSON format."""

    def split_document(self, document_text: str) -> Dict[str, Any]:
        """
        Split a multi-report document into individual reports.
        
        Args:
            document_text: The full document text containing multiple reports
            
        Returns:
            dict with:
                - splits: List of split reports with text, title, and indices
                - total_reports: Number of reports found
        """
        if not document_text or len(document_text.strip()) < 50:
            logger.warning("📄 Document too short for splitting")
            return {
                "splits": [{
                    "report_text": document_text,
                    "report_title": "Unknown",
                    "start_index": 0,
                    "end_index": len(document_text)
                }],
                "total_reports": 1
            }
        
        try:
            logger.info("🔪 Splitting document into individual reports...")
            logger.info(f"📝 Document text length: {len(document_text)} characters")
            
            # First, try pattern-based splitting
            pattern_splits = self._split_by_patterns(document_text)
            if pattern_splits and len(pattern_splits.get("splits", [])) > 1:
                logger.info(f"✅ Pattern-based splitting found {len(pattern_splits['splits'])} reports")
                return pattern_splits
            
            # If pattern splitting doesn't work well, use LLM
            logger.info("🤖 Using LLM to split document...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT)
            ])
            
            chain = prompt | self.llm | self.parser
            
            # Truncate very long documents
            max_chars = 30000
            truncated_text = document_text[:max_chars] if len(document_text) > max_chars else document_text
            if len(document_text) > max_chars:
                logger.info(f"⚠️ Document truncated from {len(document_text)} to {max_chars} chars for splitting")
            
            result = chain.invoke({
                "document_text": truncated_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Normalize result
            if isinstance(result, dict):
                split_result = result
            else:
                try:
                    split_result = result.dict()
                except Exception:
                    split_result = {
                        "splits": [{
                            "report_text": document_text,
                            "report_title": "Unknown",
                            "start_index": 0,
                            "end_index": len(document_text)
                        }],
                        "total_reports": 1
                    }
            
            logger.info(f"✅ Document split into {split_result.get('total_reports', 1)} reports")
            return split_result
            
        except Exception as e:
            logger.error(f"❌ Document splitting failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return safe default - entire document as single report
            return {
                "splits": [{
                    "report_text": document_text,
                    "report_title": "Unknown",
                    "start_index": 0,
                    "end_index": len(document_text)
                }],
                "total_reports": 1
            }
    
    def _split_by_patterns(self, text: str) -> Dict[str, Any]:
        """
        Pattern-based splitting using report title patterns.
        Returns splits if multiple report titles are found.
        """
        text_upper = text.upper()
        lines = text.split('\n')
        
        # Report title patterns with their positions
        title_patterns = [
            (r'^QUALIFIED\s+MEDICAL\s+EVALUATION', 'QME'),
            (r'^QME\s+(?:REPORT|EVALUATION|EXAM)', 'QME'),
            (r'PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*2', 'PR2'),
            (r'PROGRESS\s+REPORT\s*[-]?\s*2', 'PR2'),
            (r'^PR\s*[-]?\s*2\s+(?:PROGRESS\s+REPORT|REPORT)', 'PR2'),
            (r'PROGRESS\s+REPORT\s*[-]?\s*PR\s*[-]?\s*4', 'PR4'),
            (r'PROGRESS\s+REPORT\s*[-]?\s*4', 'PR4'),
            (r'PERMANENT\s+STATIONARY', 'PR4'),
            (r'^PR\s*[-]?\s*4\s+(?:PROGRESS\s+REPORT|REPORT)', 'PR4'),
            (r'DOCTOR[\'S]?\s+FIRST\s+REPORT', 'DFR'),
            (r'^DFR\s+(?:REPORT|FORM)', 'DFR'),
            (r'AGREED\s+MEDICAL\s+EVALUATION', 'AME'),
            (r'^AME\s+(?:REPORT|EVALUATION|EXAM)', 'AME'),
            (r'INDEPENDENT\s+MEDICAL\s+EVALUATION', 'IME'),
            (r'^IME\s+(?:REPORT|EVALUATION|EXAM)', 'IME'),
        ]
        
        # Find all report title positions
        title_positions = []
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            for pattern, title in title_patterns:
                if re.search(pattern, line_upper, re.IGNORECASE):
                    # Calculate character position
                    char_pos = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
                    title_positions.append({
                        "line_index": i,
                        "char_index": char_pos,
                        "title": title,
                        "line": line.strip()
                    })
                    break  # Only match first pattern per line
        
        # If we found 2+ titles, split the document
        if len(title_positions) >= 2:
            splits = []
            for i, pos in enumerate(title_positions):
                start_idx = pos["char_index"]
                end_idx = title_positions[i + 1]["char_index"] if i + 1 < len(title_positions) else len(text)
                report_text = text[start_idx:end_idx].strip()
                
                splits.append({
                    "report_text": report_text,
                    "report_title": pos["title"],
                    "start_index": start_idx,
                    "end_index": end_idx
                })
            
            logger.info(f"🔪 Pattern-based split found {len(splits)} reports: {[s['report_title'] for s in splits]}")
            return {
                "splits": splits,
                "total_reports": len(splits)
            }
        
        return None  # Not enough titles found for splitting


# Singleton instance
_splitter_instance = None

def get_document_splitter() -> DocumentSplitter:
    """Get singleton DocumentSplitter instance"""
    global _splitter_instance
    if _splitter_instance is None:
        logger.info("🚀 Initializing Document Splitter...")
        _splitter_instance = DocumentSplitter()
    return _splitter_instance

