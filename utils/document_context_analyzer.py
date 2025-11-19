"""
Document Context Analyzer - Provides document-level understanding before extraction
Mimics Gemini's approach of understanding document structure first
Enhanced for universal document type support
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
    Enhanced to handle ANY document type including lab reports, imaging, etc.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        logger.info("✅ DocumentContextAnalyzer initialized with universal document support")
    
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
        Enhanced to handle all document types.
        
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
   - For lab reports: identify test panels, critical values section, ordering info
   - For imaging reports: identify technique, findings, impression sections
   - Locate where critical conclusions appear (usually near end)
   - Identify section boundaries and page structure

2. PHYSICIAN/AUTHOR IDENTIFICATION (CRITICAL):
   Carefully distinguish between:
   
   a) PRIMARY PHYSICIAN/AUTHOR (Report Creator - HIGHEST PRIORITY):
      - For QME/AME/IME: "QUALIFIED MEDICAL EVALUATOR:", "QME:", "AME:", "Evaluating Physician:"
      - For clinical notes: "Provider:", "Physician:", "Attending:"
      - For lab reports: "Ordering Physician:", "Requesting Provider:"
      - For imaging: "Interpreting Radiologist:", "Reading Physician:"
      - Check: Header, letterhead, signature block, "Electronically signed by:", "Respectfully submitted,"
      - This is THE PERSON who performed this examination/test and wrote this report
      - Example indicators:
        * "This evaluation was performed by Dr. John Smith, MD"
        * "QME: Dr. Sarah Johnson, Board Certified Orthopedic Surgery"
        * Signature: "Dr. Michael Chen, D.O."
        * "Interpreted by: Dr. Lisa Park, MD, Radiologist"
        * "Ordering Provider: Dr. James Wilson, MD"
   
   b) TREATING PHYSICIAN(S) (mentioned in history):
      - Look for: "Patient treated by Dr...", "Under the care of Dr...", "Referred by Dr..."
      - These are historical references to physicians who treated patient BEFORE this evaluation
      - Example: "Patient reports treatment by Dr. Anderson for 6 months"
   
   c) OTHER PHYSICIANS (mentioned in passing):
      - Surgeons who performed past procedures
      - Consultants mentioned in medical history
      - Example: "Dr. Williams performed arthroscopy in 2022"
   
   **RULE**: The PRIMARY physician is the author of THIS report (QME, radiologist, pathologist, ordering physician),
   NOT the treating physicians mentioned in the patient's history.

3. CRITICAL FINDINGS LOCATION:
   Identify WHERE (which sections/pages) contain:
   
   For Clinical/QME/IME Reports:
   - MMI/P&S determination
   - WPI (Whole Person Impairment) percentage
   - Apportionment opinion
   - Future treatment recommendations
   - Work restrictions
   
   For Lab Reports:
   - Critical/abnormal values (HIGH/LOW flags)
   - Out-of-range results
   - Ordering physician information
   - Collection date/time
   
   For Imaging Reports:
   - Key findings section
   - Impression/conclusion
   - Critical/urgent findings
   - Comparison to prior studies
   - Recommendations for follow-up
   
   For Any Document:
   - Most important clinical information
   - Actionable findings
   - Time-sensitive information

4. SECTION IMPORTANCE RANKING:
   Rank sections by medical importance (adapt to document type):
   
   Clinical/QME Reports:
   - Priority 1 (Critical): Medical-Legal Analysis, Conclusions, Apportionment
   - Priority 2 (High): Recommendations, Work Status
   - Priority 3 (Moderate): Physical Examination, Diagnoses
   - Priority 4 (Low): Administrative info, Records reviewed
   
   Lab Reports:
   - Priority 1 (Critical): Abnormal/critical values, out-of-range results
   - Priority 2 (High): All test results with reference ranges
   - Priority 3 (Moderate): Ordering information, collection details
   - Priority 4 (Low): Laboratory information, methodology
   
   Imaging Reports:
   - Priority 1 (Critical): Impression/conclusion, critical findings
   - Priority 2 (High): Key findings in body of report
   - Priority 3 (Moderate): Comparison to prior studies
   - Priority 4 (Low): Technique, patient information

5. AMBIGUITY DETECTION:
   Identify any ambiguities that need resolution:
   - Multiple physicians mentioned (clarify roles)
   - MMI stated in multiple places with different dates
   - Conflicting information across sections
   - Unclear test results or findings
   - Missing critical information

6. DOCUMENT TYPE SPECIFIC GUIDANCE:
   Provide extraction guidance based on document type:
   - Lab Report: Focus on test results, abnormal values, critical findings
   - Imaging: Focus on findings, impression, recommendations
   - Clinical Note: Focus on assessment, plan, diagnoses
   - QME/IME: Focus on medical-legal conclusions, MMI, restrictions

DOCUMENT TEXT (first 30K chars):
{text}

Return structured analysis as JSON:
{{{{
  "document_structure": {{{{
    "total_sections": 0,
    "main_sections": [],
    "document_type_detected": "QME/IME/Lab Report/Imaging/Clinical Note/etc",
    "critical_sections": {{{{
      "medical_legal_analysis_section": "section name or null",
      "conclusions_section": "section name or null",
      "recommendations_section": "section name or null",
      "test_results_section": "section name or null (for lab reports)",
      "findings_section": "section name or null (for imaging)",
      "impression_section": "section name or null (for imaging)"
    }}}},
    "section_locations": {{{{
      "history": "approx page X or null",
      "physical_exam": "approx page Y or null",
      "conclusions": "approx page Z or null",
      "results": "approx page N or null (for labs/imaging)"
    }}}}
  }}}},
  "physician_analysis": {{{{
    "primary_physician": {{{{
      "name": "Dr. Full Name, MD/DO or null",
      "role": "QME/AME/Radiologist/Pathologist/Ordering Physician/etc",
      "confidence": "high/medium/low",
      "found_in": "header/signature/ordering info/interpretation section",
      "reasoning": "why this is the primary physician/author"
    }}}},
    "treating_physicians": [
      {{{{"name": "Dr. X", "role": "historical treating physician", "context": "mentioned in history section"}}}}
    ],
    "other_physicians": [
      {{{{"name": "Dr. Y", "role": "consulting surgeon/radiologist/etc", "context": "performed surgery in 2021 or interpreted prior study"}}}}
    ]
  }}}},
  "critical_findings_map": {{{{
    "mmi_location": "section name, approx location or null",
    "wpi_location": "section name, approx location or null",
    "apportionment_location": "section name, approx location or null",
    "work_restrictions_location": "section name, approx location or null",
    "abnormal_results_location": "section name, approx location or null (for labs)",
    "critical_findings_location": "section name, approx location or null (for imaging)",
    "impression_location": "section name, approx location or null (for imaging)",
    "key_clinical_data_location": "section name, approx location or null (general)"
  }}}},
  "section_priority_ranking": [
    {{{{"section": "section name", "priority": 1, "reason": "contains most critical information"}}}},
    {{{{"section": "section name", "priority": 2, "reason": "contains important findings"}}}}
  ],
  "ambiguities_detected": [
    {{{{"type": "physician_role/mmi_date/test_result/finding/etc", "description": "description of ambiguity"}}}},
    {{{{"type": "missing_data", "description": "critical information not found"}}}}
  ],
  "extraction_guidance": {{{{
    "focus_on_sections": ["section1", "section2", "section3"],
    "physician_extraction_priority": "Use primary_physician identified above",
    "critical_info_locations": "Location description of most critical info",
    "document_type_guidance": "Specific extraction priorities for this document type"
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
          doc_structure = analysis.get("document_structure", {})
          detected_type = doc_structure.get("document_type_detected", "Unknown")
          logger.info(f"✅ Document type detected: {detected_type}")
          
          primary_phys = analysis.get("physician_analysis", {}).get("primary_physician", {})
          logger.info(f"✅ Primary physician/author identified: {primary_phys.get('name', 'Unknown')}")
          logger.info(f"   Role: {primary_phys.get('role', 'Unknown')}")
          logger.info(f"   Confidence: {primary_phys.get('confidence', 'Unknown')}")
          
          critical_sections = analysis.get("document_structure", {}).get("critical_sections", {})
          non_null_sections = {k: v for k, v in critical_sections.items() if v}
          if non_null_sections:
            logger.info(f"✅ Critical sections identified: {list(non_null_sections.values())}")
          
          ambiguities = analysis.get("ambiguities_detected", [])
          if ambiguities:
            logger.warning(f"⚠️ Ambiguities detected: {len(ambiguities)}")
            for amb in ambiguities[:3]:  # Log first 3
              logger.warning(f"   - {amb.get('type')}: {amb.get('description')}")
          
          # Log extraction guidance
          guidance = analysis.get("extraction_guidance", {})
          if guidance.get("document_type_guidance"):
            logger.info(f"📋 Document guidance: {guidance.get('document_type_guidance')}")
          
          return analysis
        except Exception as e:
          logger.error(f"❌ Document context analysis failed: {e}", exc_info=True)
          return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> Dict:
        """Return minimal fallback analysis"""
        return {
            "document_structure": {
                "main_sections": [], 
                "critical_sections": {},
                "document_type_detected": "Unknown"
            },
            "physician_analysis": {
                "primary_physician": {}, 
                "treating_physicians": [],
                "other_physicians": []
            },
            "critical_findings_map": {},
            "section_priority_ranking": [],
            "ambiguities_detected": [],
            "extraction_guidance": {
                "focus_on_sections": [],
                "document_type_guidance": "Extract all available information comprehensively"
            }
        }