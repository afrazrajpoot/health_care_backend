"""
Document Context Analyzer - Provides document-level understanding before extraction
Mimics Gemini's approach of understanding document structure first
"""
import logging
from typing import Dict, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger("document_ai")


class DocumentContextAnalyzer:
    """
    Analyzes document structure and identifies critical elements before extraction.
    This provides contextual understanding similar to Gemini's approach.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        logger.info("✅ DocumentContextAnalyzer initialized")
    
    def _sync_invoke_chain(self, chain, inputs: dict):
        """
        Wrapper to safely invoke LangChain in a synchronous context.
        Handles event loop issues when called from thread pools.
        """
        try:
            # Try direct invoke first (works in most cases)
            return chain.invoke(inputs)
        except RuntimeError as e:
            if "no current event loop" in str(e).lower() or "no running event loop" in str(e).lower():
                # If event loop issue, create a new one for this thread
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run invoke in the loop
                return loop.run_until_complete(chain.ainvoke(inputs))
            else:
                raise
    
    def analyze_document_structure(
        self,
        text: str,
        doc_type_hint: Optional[str] = None
    ) -> Dict:
        """
        Analyze document to understand structure and identify critical elements.
        This runs BEFORE detailed extraction to provide context.
        
        Returns:
            Dict with document structure, critical sections, physician roles, etc.
        """
        logger.info("🔍 Analyzing document structure and context...")
        
        # Use first 30K chars for analysis (enough to understand structure)
        analysis_text = text[:30000]
        
        prompt = PromptTemplate(
            template="""
You are a medical document structure analyst. Analyze this document to understand its structure and identify critical elements.

DOCUMENT TYPE HINT: {doc_type_hint}

ANALYSIS TASKS:

1. DOCUMENT STRUCTURE:
   - Identify main sections (e.g., "History of Present Illness", "Physical Examination", "Medical-Legal Analysis", "Conclusions")
   - Locate where critical medical-legal conclusions appear (usually near end)
   - Identify section boundaries and page structure

2. PHYSICIAN ROLE IDENTIFICATION (CRITICAL):
   Carefully distinguish between:
   
   a) PRIMARY PHYSICIAN (Report Author - HIGHEST PRIORITY):
      - Look for: "QUALIFIED MEDICAL EVALUATOR:", "QME:", "AME:", "Evaluating Physician:"
      - Check: Header, letterhead, signature block, "Electronically signed by:", "Respectfully submitted,"
      - This is THE PERSON who performed this examination and wrote this report
      - Example indicators:
        * "This evaluation was performed by Dr. John Smith, MD"
        * "QME: Dr. Sarah Johnson, Board Certified Orthopedic Surgery"
        * Signature: "Dr. Michael Chen, D.O."
   
   b) TREATING PHYSICIAN(S) (mentioned in history):
      - Look for: "Patient treated by Dr...", "Under the care of Dr...", "Referred by Dr..."
      - These are historical references to physicians who treated patient BEFORE this evaluation
      - Example: "Patient reports treatment by Dr. Anderson for 6 months"
   
   c) OTHER PHYSICIANS (mentioned in passing):
      - Surgeons who performed past procedures
      - Consultants mentioned in medical history
      - Example: "Dr. Williams performed arthroscopy in 2022"
   
   **RULE**: For QME/AME reports, the PRIMARY physician is the QME/AME who authored THIS report,
   NOT the treating physicians mentioned in the patient's history.

3. CRITICAL FINDINGS LOCATION:
   Identify WHERE (which sections/pages) contain:
   - MMI/P&S determination
   - WPI (Whole Person Impairment) percentage
   - Apportionment opinion
   - Future treatment recommendations
   - Work restrictions

4. SECTION IMPORTANCE RANKING:
   Rank sections by medical-legal importance:
   - Priority 1 (Critical): Medical-Legal Analysis, Conclusions, Apportionment
   - Priority 2 (High): Recommendations, Work Status
   - Priority 3 (Moderate): Physical Examination, Diagnoses
   - Priority 4 (Low): Administrative info, Records reviewed

5. AMBIGUITY DETECTION:
   Identify any ambiguities that need resolution:
   - Multiple physicians mentioned (clarify roles)
   - MMI stated in multiple places with different dates
   - Conflicting information across sections

DOCUMENT TEXT (first 30K chars):
{text}

Return structured analysis as JSON:
{{{{
  "document_structure": {{{{
    "total_sections": 0,
    "main_sections": [],
    "critical_sections": {{{{
      "medical_legal_analysis_section": "section name or null",
      "conclusions_section": "section name or null",
      "recommendations_section": "section name or null"
    }}}},
    "section_locations": {{{{
      "history": "approx page X",
      "physical_exam": "approx page Y",
      "conclusions": "approx page Z"
    }}}}
  }}}},
  "physician_analysis": {{{{
    "primary_physician": {{{{
      "name": "Dr. Full Name, MD/DO",
      "role": "QME/AME/Evaluating Physician",
      "confidence": "high/medium/low",
      "found_in": "header/signature/both",
      "reasoning": "why this is the primary physician"
    }}}},
    "treating_physicians": [
      {{{{"name": "Dr. X", "role": "historical treating physician", "context": "mentioned in history section"}}}}
    ],
    "other_physicians": [
      {{{{"name": "Dr. Y", "role": "consulting surgeon", "context": "performed surgery in 2021"}}}}
    ]
  }}}},
  "critical_findings_map": {{{{
    "mmi_location": "section name, approx location",
    "wpi_location": "section name, approx location",
    "apportionment_location": "section name, approx location",
    "work_restrictions_location": "section name, approx location"
  }}}},
  "section_priority_ranking": [
    {{{{"section": "Medical-Legal Analysis", "priority": 1, "reason": "contains MMI/WPI conclusions"}}}},
    {{{{"section": "Recommendations", "priority": 2, "reason": "contains future treatment"}}}}
  ],
  "ambiguities_detected": [
    {{{{"type": "physician_role", "description": "Multiple physicians mentioned, roles need clarification"}}}},
    {{{{"type": "mmi_date", "description": "MMI date appears in 2 sections with different dates"}}}}
  ],
  "extraction_guidance": {{{{
    "focus_on_sections": ["section1", "section2"],
    "physician_extraction_priority": "Use primary_physician identified above",
    "critical_info_locations": "Most critical info in final 30% of document"
  }}}}
}}}}
""",
            input_variables=["text", "doc_type_hint"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
          chain = prompt | self.llm
          logger.info("🤖 Invoking LLM for document context analysis...")
          # Get raw LLM output before parsing
          raw_llm_output = self._sync_invoke_chain(chain, {
            "text": analysis_text,
            "doc_type_hint": doc_type_hint or "Unknown"
          })
          logger.debug(f"📝 Raw LLM output: {raw_llm_output}")
          # Now parse the output
          try:
            analysis = self.parser.invoke(raw_llm_output)
          except Exception as parse_exc:
            logger.error(f"❌ JSON parsing failed: {parse_exc}")
            analysis = None
          # Check if analysis is None (parsing failed)
          if not analysis:
            logger.warning("⚠️ Analysis returned None (LLM parsing failed), using fallback")
            return self._get_fallback_analysis()
          logger.info("✅ Document context analysis completed successfully")
          # Log key findings
          primary_phys = analysis.get("physician_analysis", {}).get("primary_physician", {})
          logger.info(f"✅ Primary physician identified: {primary_phys.get('name', 'Unknown')}")
          logger.info(f"   Role: {primary_phys.get('role', 'Unknown')}")
          logger.info(f"   Confidence: {primary_phys.get('confidence', 'Unknown')}")
          critical_sections = analysis.get("document_structure", {}).get("critical_sections", {})
          logger.info(f"✅ Critical sections identified: {list(critical_sections.values())}")
          ambiguities = analysis.get("ambiguities_detected", [])
          if ambiguities:
            logger.warning(f"⚠️ Ambiguities detected: {len(ambiguities)}")
            for amb in ambiguities:
              logger.warning(f"   - {amb.get('type')}: {amb.get('description')}")
          return analysis
        except Exception as e:
          logger.error(f"❌ Document context analysis failed: {e}", exc_info=True)
          return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> Dict:
        """Return minimal fallback analysis"""
        return {
            "document_structure": {"main_sections": [], "critical_sections": {}},
            "physician_analysis": {"primary_physician": {}, "treating_physicians": []},
            "critical_findings_map": {},
            "extraction_guidance": {"focus_on_sections": []}
        }
