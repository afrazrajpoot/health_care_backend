"""
Comprehensive Document Summarizer
Generates concise medical summaries (600-700 words) from extracted document data.
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from datetime import datetime

logger = logging.getLogger("document_ai")


class ComprehensiveDocumentSummarizer:
    """
    Generates comprehensive, physician-friendly summaries from extracted document data.
    Target: 600-700 words, includes all key clinical information.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        logger.info("✅ ComprehensiveDocumentSummarizer initialized")
    
    def generate_comprehensive_summary(
        self,
        extracted_data: Dict[str, Any],
        document_text: Optional[str] = None,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary from extracted document data.
        
        Args:
            extracted_data: Dictionary containing all extracted fields from the document
            document_text: Optional full document text for additional context
            page_zones: Optional page zones for additional context
        
        Returns:
            Dict with summary, word count, and metadata
        """
        logger.info("=" * 80)
        logger.info("🔍 STARTING COMPREHENSIVE DOCUMENT SUMMARIZATION")
        logger.info("=" * 80)
        
        try:
            # Log input data structure
            logger.info(f"📊 Input data keys: {list(extracted_data.keys())}")
            logger.info(f"📄 Document type: {extracted_data.get('document_type', 'Unknown')}")
            logger.info(f"📅 Document date: {extracted_data.get('document_date', 'Unknown')}")
            
            # Build structured input for LLM
            structured_input = self._build_structured_input(extracted_data)
            logger.info(f"✅ Structured input built ({len(structured_input)} characters)")
            
            # Generate summary
            logger.info("🤖 Invoking LLM for comprehensive summarization...")
            summary = self._generate_summary(structured_input, extracted_data)
            
            # Calculate word count
            word_count = len(summary.split())
            logger.info(f"📝 Summary generated: {word_count} words")
            
            # Validate word count
            if word_count < 500:
                logger.warning(f"⚠️ Summary too short ({word_count} words), expected 600-700")
            elif word_count > 750:
                logger.warning(f"⚠️ Summary too long ({word_count} words), expected 600-700")
            else:
                logger.info(f"✅ Summary length within target range (600-700 words)")
            
            # Build result
            result = {
                "summary": summary,
                "word_count": word_count,
                "generated_at": datetime.now().isoformat(),
                "document_type": extracted_data.get("document_type"),
                "document_date": extracted_data.get("document_date"),
                "patient_name": extracted_data.get("patient_name"),
                "physician_name": extracted_data.get("consulting_doctor") or extracted_data.get("examiner_name"),
            }
            
            # Log complete summary
            logger.info("=" * 80)
            logger.info("📋 COMPREHENSIVE SUMMARY GENERATED:")
            logger.info("=" * 80)
            logger.info(summary)
            logger.info("=" * 80)
            logger.info(f"✅ Summarization complete: {word_count} words")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive summarization failed: {str(e)}")
            return {
                "summary": "",
                "word_count": 0,
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
            }
    
    def _build_structured_input(self, data: Dict[str, Any]) -> str:
        """Build structured input string from extracted data"""
        sections = []
        
        # Document metadata
        sections.append("DOCUMENT INFORMATION:")
        sections.append(f"- Type: {data.get('document_type', 'Unknown')}")
        sections.append(f"- Date: {data.get('document_date', 'Unknown')}")
        
        # Patient information
        if data.get("patient_name"):
            sections.append(f"\nPATIENT:")
            sections.append(f"- Name: {data.get('patient_name')}")
            if data.get("dob"):
                sections.append(f"- DOB: {data.get('dob')}")
            if data.get("claim_number"):
                sections.append(f"- Claim: {data.get('claim_number')}")
        
        # Physician information
        physician = data.get("consulting_doctor") or data.get("examiner_name") or data.get("physician_name")
        if physician:
            sections.append(f"\nPHYSICIAN:")
            sections.append(f"- {physician}")
            if data.get("specialty"):
                sections.append(f"- Specialty: {data.get('specialty')}")
        
        # Clinical findings
        sections.append(f"\nCLINICAL DATA:")
        
        # Body parts
        if data.get("body_parts") or data.get("body_parts_evaluated") or data.get("body_part"):
            body_parts = data.get("body_parts") or data.get("body_parts_evaluated") or [data.get("body_part")]
            if body_parts:
                sections.append(f"- Body parts: {', '.join(str(bp) for bp in body_parts if bp)}")
        
        # Diagnoses
        if data.get("diagnoses_confirmed") or data.get("diagnosis") or data.get("findings"):
            diagnoses = data.get("diagnoses_confirmed") or [data.get("diagnosis") or data.get("findings")]
            if diagnoses:
                sections.append(f"- Diagnoses: {', '.join(str(dx) for dx in diagnoses if dx)}")
        
        # Current status
        if data.get("current_status"):
            sections.append(f"- Current status: {data.get('current_status')}")
        
        # MMI/Impairment
        if data.get("MMI_status"):
            sections.append(f"- MMI status: {data.get('MMI_status')}")
        if data.get("impairment_summary"):
            sections.append(f"- Impairment: {data.get('impairment_summary')}")
        if data.get("causation_opinion"):
            sections.append(f"- Causation: {data.get('causation_opinion')}")
        
        # Treatment
        sections.append(f"\nTREATMENT:")
        if data.get("treatment_recommendations"):
            sections.append(f"- Recommendations: {data.get('treatment_recommendations')}")
        if data.get("medication_recommendations"):
            sections.append(f"- Medications: {data.get('medication_recommendations')}")
        if data.get("future_medical_recommendations"):
            sections.append(f"- Future care: {data.get('future_medical_recommendations')}")
        
        # Work status
        if data.get("work_status") or data.get("work_restrictions"):
            sections.append(f"\nWORK STATUS:")
            if data.get("work_status"):
                sections.append(f"- Status: {data.get('work_status')}")
            if data.get("work_restrictions"):
                sections.append(f"- Restrictions: {data.get('work_restrictions')}")
        
        # Plan/Follow-up
        if data.get("next_plan") or data.get("recommendations"):
            sections.append(f"\nPLAN:")
            if data.get("next_plan"):
                sections.append(f"- Next steps: {data.get('next_plan')}")
            if data.get("recommendations"):
                sections.append(f"- Recommendations: {data.get('recommendations')}")
        
        # Raw data for additional context
        if data.get("raw_data") and isinstance(data["raw_data"], dict):
            sections.append(f"\nADDITIONAL DETAILS:")
            for key, value in data["raw_data"].items():
                if value and key not in ["physician_name", "patient_name", "document_date"]:
                    sections.append(f"- {key}: {value}")
        
        return "\n".join(sections)
    
    def _generate_summary(self, structured_input: str, original_data: Dict[str, Any]) -> str:
        """Generate comprehensive summary using LLM"""
        doc_type = original_data.get("document_type", "Medical Document")
        
        prompt = PromptTemplate(
            template="""
You are an expert medical documentation specialist creating a comprehensive summary for physicians and case managers.

TASK: Create a detailed, physician-friendly summary of this medical document in 600-700 words.

CRITICAL REQUIREMENTS:
1. **Length**: MUST be 600-700 words (strict requirement)
2. **Completeness**: Include ALL key information provided below
3. **Structure**: Use clear sections with headers
4. **Clarity**: Write for physicians - use medical terminology appropriately
5. **Conciseness**: Be thorough but avoid unnecessary repetition
6. **Objectivity**: Report findings factually, include both positive and negative findings

REQUIRED SECTIONS (organize logically):
1. **Document Overview**: Type, date, physician, patient (if available)
2. **Clinical Presentation**: Chief complaint, body parts, current status
3. **Findings/Diagnoses**: All diagnoses, test results, clinical impressions (include normal and abnormal)
4. **Treatment Plan**: Current treatments, medications, procedures
5. **Work Status**: Work restrictions or clearances (include "full duty" or "no restrictions" if applicable)
6. **Recommendations**: Future care, follow-up, referrals
7. **Key Outcomes**: MMI status, impairment ratings, apportionment (for QME/AME reports)

WRITING GUIDELINES:
- Start with document type and date: "{doc_type} dated [DATE]"
- Include physician name and specialty when available
- Use paragraph format (not bullet points)
- Include both positive findings (pathology) and negative findings (normal results, no restrictions)
- Mention specific body parts, diagnoses, medications with details
- Include work status even if "full duty" or "no restrictions"
- Include quantitative data (percentages, measurements, dates)
- End with clear next steps or conclusions

EXAMPLE OPENING:
"This Progress Report (PR-2) dated 09/26/2025 by Dr. John Smith, Orthopedic Surgery, documents continued treatment for the patient's right shoulder injury..."

EXTRACTED DOCUMENT DATA:
{structured_input}

Generate the 600-700 word comprehensive summary now:
""",
            input_variables=["structured_input", "doc_type"],
        )
        
        try:
            response = self.llm.invoke(prompt.format(
                structured_input=structured_input,
                doc_type=doc_type
            ))
            summary = response.content if hasattr(response, 'content') else str(response)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ LLM summarization failed: {str(e)}")
            # Fallback: create basic summary from structured input
            return self._create_fallback_summary(structured_input, original_data)
    
    def _create_fallback_summary(self, structured_input: str, data: Dict[str, Any]) -> str:
        """Create basic summary when LLM fails"""
        doc_type = data.get("document_type", "Medical Document")
        date = data.get("document_date", "Unknown date")
        physician = data.get("consulting_doctor") or data.get("examiner_name") or "Unknown physician"
        
        summary = f"SUMMARY OF {doc_type.upper()} (DATED {date})\n\n"
        summary += f"Physician: {physician}\n\n"
        summary += "EXTRACTED INFORMATION:\n\n"
        summary += structured_input
        
        return summary


# Usage function for easy integration
def generate_document_summary(
    extracted_data: Dict[str, Any],
    llm: AzureChatOpenAI,
    document_text: Optional[str] = None,
    page_zones: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Standalone function to generate comprehensive document summary.
    
    Args:
        extracted_data: All extracted document data
        llm: Azure ChatOpenAI instance
        document_text: Optional full document text
        page_zones: Optional page zones
    
    Returns:
        Dict with summary, word_count, and metadata
    """
    summarizer = ComprehensiveDocumentSummarizer(llm)
    return summarizer.generate_comprehensive_summary(
        extracted_data=extracted_data,
        document_text=document_text,
        page_zones=page_zones
    )
