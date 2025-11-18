"""
DecisionDocumentExtractor - Enhanced Extractor for UR/IMR Decisions, Appeals, Authorizations
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier

logger = logging.getLogger("document_ai")


class DecisionDocumentExtractor:
    """
    Enhanced Decision Document extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different decision types
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports UR/IMR, Appeals, Authorizations, RFA, DFR
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.decision_patterns = {
            'approved': re.compile(r'\b(approved|authorized|granted|allowed|certified)\b', re.IGNORECASE),
            'denied': re.compile(r'\b(denied|rejected|disallowed|not authorized|not approved)\b', re.IGNORECASE),
            'partially_approved': re.compile(r'\b(partially|partially approved|partially authorized)\b', re.IGNORECASE),
            'pending': re.compile(r'\b(pending|under review|being reviewed|in process)\b', re.IGNORECASE),
            'appeal': re.compile(r'\b(appeal|reconsideration|reevaluation|review requested)\b', re.IGNORECASE)
        }
        
        # Document type identifiers
        self.doc_type_patterns = {
            'ur_imr': re.compile(r'\b(UR|UM|Utilization Review|IMR|Independent Medical Review)\b', re.IGNORECASE),
            'appeal': re.compile(r'\b(Appeal|Reconsideration|Level I|Level II|Final Appeal)\b', re.IGNORECASE),
            'authorization': re.compile(r'\b(Authorization|RFA|Request for Authorization|Treatment Authorization)\b', re.IGNORECASE),
            'dfr': re.compile(r'\b(DFR|Doctor First Report|First Report|Initial Report)\b', re.IGNORECASE),
            'denial': re.compile(r'\b(Denial|Notice of Denial|Determination of Denial)\b', re.IGNORECASE)
        }
        
        logger.info("✅ DecisionDocumentExtractor initialized (Full Context + Context-Aware)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        context_analysis: Optional[Dict] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Extract Decision Document data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (UR/IMR/Appeal/Authorization/RFA/DFR)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("⚖️ STARTING DECISION DOCUMENT EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Auto-detect document type if not specified
        detected_type = self._detect_document_type(text, doc_type)
        logger.info(f"📋 Document Type: {detected_type} (original: {doc_type})")
        
        # Log context guidance if available
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            
            logger.info(f"🎯 Context Guidance Received:")
            logger.info(f"   Focus Sections: {focus_sections}")
            logger.info(f"   Critical Locations: {list(critical_locations.keys())}")
        else:
            logger.warning("⚠️ No context analysis provided - proceeding without guidance")
        
        # Check document size
        text_length = len(text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Extract with FULL CONTEXT and contextual guidance
        raw_result = self._extract_full_context_with_guidance(
            text=text,
            doc_type=detected_type,
            fallback_date=fallback_date,
            context_analysis=context_analysis
        )

        # Stage 2: Build long summary from ALL raw data
        long_summary = self._build_comprehensive_long_summary(raw_result, detected_type, fallback_date)

        # Stage 3: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)

        logger.info("=" * 80)
        logger.info("✅ DECISION DOCUMENT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_document_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect document type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for doc_type, pattern in self.doc_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[doc_type] = len(matches)
        
        # Also check for decision patterns
        for decision_type, pattern in self.decision_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Boost scores for documents with clear decision language
                if decision_type in ['approved', 'denied']:
                    for doc_type in ['ur_imr', 'authorization', 'appeal']:
                        type_scores[doc_type] = type_scores.get(doc_type, 0) + len(matches)
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].upper().replace('_', ' ')
                logger.info(f"🔍 Auto-detected document type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect document type, using: {original_type}")
        return original_type or "UNKNOWN"

    def _extract_full_context_with_guidance(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        context_analysis: Optional[Dict]
    ) -> Dict:
        """
        Extract with FULL document context + contextual guidance from DocumentContextAnalyzer.
        """
        logger.info("🔍 Processing ENTIRE decision document in single context window with guidance...")
        
        # Extract guidance from context analysis
        focus_sections = []
        critical_locations = {}
        ambiguities = []
        
        if context_analysis:
            focus_sections = context_analysis.get("extraction_guidance", {}).get("focus_on_sections", [])
            critical_locations = context_analysis.get("critical_findings_map", {})
            ambiguities = context_analysis.get("ambiguities_detected", [])
        
        # Build context-aware system prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE {doc_type} decision document with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing you to:
- Understand the complete decision rationale from start to finish
- Connect request details with decision criteria and justification
- Identify all services/treatments being decided upon
- Provide comprehensive extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate information
   - DO NOT fill in "typical" or "common" values
   
2. **DECISION STATUS - EXACT WORDING ONLY**
   - Extract decision status using EXACT wording from document
   - DO NOT interpret or categorize (e.g., if document says "not medically necessary", use that exact phrase)
   - For partial approvals, extract EXACTLY what was approved vs denied
   
3. **SERVICES/TREATMENTS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY services/treatments explicitly listed in the request/decision
   - Include quantities/durations ONLY if explicitly stated
   - DO NOT extract services mentioned as examples, comparisons, or historical context
   
4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
5. **CRITERIA AND REGULATIONS - EXACT REFERENCES**
   - Extract medical necessity criteria EXACTLY as stated
   - Include specific guideline references (e.g., "ODG", "MTUS", "ACOEM")
   - DO NOT add criteria not explicitly referenced

EXTRACTION FOCUS - 7 CRITICAL DECISION DOCUMENT CATEGORIES:

I. DOCUMENT IDENTITY & PARTIES
- Document type, dates, identification numbers
- All parties involved: patient, provider, reviewer, insurer
- Contact information for appeals/communication

II. REQUEST DETAILS (WHAT WAS REQUESTED)
- Requested services/treatments with SPECIFICS:
  * Procedure names, CPT codes if available
  * Medication names, dosages, durations
  * Therapy types, frequencies, durations
  * Diagnostic tests with body parts
- Dates of service requested
- Requesting provider details

III. DECISION STATUS & OUTCOME (MOST CRITICAL)
- Overall decision: APPROVED, DENIED, PARTIALLY APPROVED, PENDING
- **EXACT wording used in decision**
- Decision date and effective dates
- For partial decisions: EXACT breakdown of approved vs denied services

IV. MEDICAL NECESSITY DETERMINATION
- Medical necessity determination (Medically Necessary, Not Medically Necessary)
- Specific criteria applied (ODG, MTUS, ACOEM, etc.)
- Clinical rationale for decision
- Supporting evidence referenced

V. REVIEWER ANALYSIS & FINDINGS
- Clinical summary reviewed
- Key findings from records review
- Consultant/reviewer opinions if applicable
- Gaps in documentation noted

VI. APPEAL & NEXT STEPS INFORMATION
- Appeal deadlines and procedures
- Required documentation for appeals
- Contact information for questions
- Effective dates and expiration

VII. REGULATORY COMPLIANCE
- Regulatory references (California Code, Labor Code, etc.)
- Timeliness compliance statements
- Reviewer credentials and qualifications

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- DECISION STATUS: Use exact wording from document
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE {doc_type} decision document and extract ALL relevant information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} DECISION DOCUMENT TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical details:

{{
  "document_identity": {{
    "document_type": "{doc_type}",
    "document_date": "",
    "decision_date": "",
    "effective_date": "",
    "document_id": "",
    "claim_number": "",
    "case_number": "",
    "jurisdiction": ""
  }},
  
  "parties_involved": {{
    "patient": {{
      "name": "",
      "date_of_birth": "",
      "member_id": ""
    }},
    "requesting_provider": {{
      "name": "",
      "specialty": "",
      "npi": "",
      "contact_info": ""
    }},
    "reviewing_entity": {{
      "name": "",
      "reviewer_name": "",
      "reviewer_credentials": "",
      "contact_info": ""
    }},
    "claims_administrator": {{
      "name": "",
      "contact_info": ""
    }}
  }},
  
  "request_details": {{
    "date_of_service_requested": "",
    "request_received_date": "",
    "requested_services": [
      {{
        "service_type": "",
        "procedure_name": "",
        "cpt_code": "",
        "body_part": "",
        "frequency": "",
        "duration": "",
        "quantity": ""
      }}
    ],
    "clinical_reason": "",
    "supporting_documentation": []
  }},
  
  "decision_outcome": {{
    "overall_decision": "",
    "decision_date": "",
    "decision_details": "",
    "partial_decision_breakdown": [
      {{
        "service": "",
        "decision": "",
        "approved_quantity": "",
        "denied_quantity": ""
      }}
    ],
    "decision_effective_dates": {{
      "start_date": "",
      "end_date": ""
    }}
  }},
  
  "medical_necessity_determination": {{
    "medical_necessity": "",
    "criteria_applied": "",
    "clinical_rationale": "",
    "supporting_evidence": [],
    "guidelines_referenced": []
  }},
  
  "reviewer_analysis": {{
    "clinical_summary_reviewed": "",
    "key_findings": [],
    "consultant_opinions": [],
    "documentation_gaps": []
  }},
  
  "appeal_information": {{
    "appeal_deadline": "",
    "appeal_procedures": "",
    "required_documentation": [],
    "contact_information": "",
    "timeframe_for_response": ""
  }},
  
  "regulatory_compliance": {{
    "regulatory_references": [],
    "timeliness_compliance": "",
    "reviewer_qualifications": ""
  }},
  
  "critical_actions_required": []
}}

⚠️ CRITICAL REMINDERS:
1. For "overall_decision": Extract EXACT wording from document
   - If document says "not medically necessary", use: "not medically necessary"
   - If document says "authorized", use: "authorized"
   - DO NOT simplify or categorize

2. For "requested_services": Extract ONLY services explicitly listed in the REQUEST
   - Include details ONLY if explicitly stated
   - DO NOT include services mentioned as examples or comparisons

3. For "partial_decision_breakdown": Only populate if document explicitly breaks down partial approval
   - If no breakdown provided, leave empty

4. For "critical_actions_required": Include time-sensitive actions only
   - Appeal deadlines
   - Required response dates
   - Time-limited authorizations
""")

        # Build context guidance summary
        context_guidance_text = f"""
DOCUMENT TYPE: {doc_type}
FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- Decision Outcome: {critical_locations.get('decision_location', 'Search entire document')}
- Appeal Information: {critical_locations.get('appeal_location', 'Search entire document')}
- Requested Services: {critical_locations.get('services_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context decision extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "context_guidance": context_guidance_text
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context decision extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context decision extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(doc_type, fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted raw data with detailed headings.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted decision data...")
        
        sections = []
        
        # Section 1: DOCUMENT OVERVIEW
        sections.append("📋 DECISION DOCUMENT OVERVIEW")
        sections.append("-" * 50)
        
        document_identity = raw_data.get("document_identity", {})
        overview_lines = [
            f"Document Type: {document_identity.get('document_type', doc_type)}",
            f"Document Date: {document_identity.get('document_date', fallback_date)}",
            f"Decision Date: {document_identity.get('decision_date', 'Not specified')}",
            f"Document ID: {document_identity.get('document_id', 'Not specified')}",
            f"Claim/Case Number: {document_identity.get('claim_number', document_identity.get('case_number', 'Not specified'))}",
            f"Jurisdiction: {document_identity.get('jurisdiction', 'Not specified')}"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PARTIES INVOLVED
        sections.append("\n👥 PARTIES INVOLVED")
        sections.append("-" * 50)
        
        parties = raw_data.get("parties_involved", {})
        party_lines = []
        
        # Patient information
        patient = parties.get("patient", {})
        if patient.get("name"):
            party_lines.append(f"Patient: {patient.get('name')}")
            if patient.get("date_of_birth"):
                party_lines.append(f"  DOB: {patient.get('date_of_birth')}")
            if patient.get("member_id"):
                party_lines.append(f"  Member ID: {patient.get('member_id')}")
        
        # Requesting provider
        provider = parties.get("requesting_provider", {})
        if provider.get("name"):
            party_lines.append(f"\nRequesting Provider: {provider.get('name')}")
            if provider.get("specialty"):
                party_lines.append(f"  Specialty: {provider.get('specialty')}")
            if provider.get("npi"):
                party_lines.append(f"  NPI: {provider.get('npi')}")
        
        # Reviewing entity
        reviewer = parties.get("reviewing_entity", {})
        if reviewer.get("name"):
            party_lines.append(f"\nReviewing Entity: {reviewer.get('name')}")
            if reviewer.get("reviewer_name"):
                party_lines.append(f"  Reviewer: {reviewer.get('reviewer_name')}")
            if reviewer.get("reviewer_credentials"):
                party_lines.append(f"  Credentials: {reviewer.get('reviewer_credentials')}")
        
        sections.append("\n".join(party_lines) if party_lines else "No party information extracted")
        
        # Section 3: REQUEST DETAILS
        sections.append("\n📋 REQUEST DETAILS")
        sections.append("-" * 50)
        
        request_details = raw_data.get("request_details", {})
        request_lines = []
        
        # Dates
        if request_details.get("date_of_service_requested"):
            request_lines.append(f"Date of Service Requested: {request_details['date_of_service_requested']}")
        if request_details.get("request_received_date"):
            request_lines.append(f"Request Received: {request_details['request_received_date']}")
        
        # Requested services
        requested_services = request_details.get("requested_services", [])
        if requested_services:
            request_lines.append("\nRequested Services:")
            for service in requested_services[:10]:  # Limit to 10 services
                if isinstance(service, dict):
                    service_desc = []
                    if service.get("procedure_name"):
                        service_desc.append(service["procedure_name"])
                    if service.get("service_type"):
                        service_desc.append(f"({service['service_type']})")
                    if service.get("body_part"):
                        service_desc.append(f"- {service['body_part']}")
                    if service.get("frequency"):
                        service_desc.append(f"Frequency: {service['frequency']}")
                    if service.get("duration"):
                        service_desc.append(f"Duration: {service['duration']}")
                    
                    if service_desc:
                        request_lines.append(f"  • {' '.join(service_desc)}")
                elif service:
                    request_lines.append(f"  • {service}")
        else:
            request_lines.append("No specific services listed in request")
        
        # Clinical reason
        if request_details.get("clinical_reason"):
            request_lines.append(f"\nClinical Reason: {request_details['clinical_reason']}")
        
        sections.append("\n".join(request_lines))
        
        # Section 4: DECISION OUTCOME (MOST CRITICAL)
        sections.append("\n⚖️ DECISION OUTCOME")
        sections.append("-" * 50)
        
        decision_outcome = raw_data.get("decision_outcome", {})
        decision_lines = []
        
        # Overall decision
        overall_decision = decision_outcome.get("overall_decision", "")
        if overall_decision:
            decision_lines.append(f"Overall Decision: {overall_decision}")
        
        # Decision details
        decision_details = decision_outcome.get("decision_details", "")
        if decision_details:
            decision_lines.append(f"Decision Details: {decision_details}")
        
        # Partial decision breakdown
        partial_breakdown = decision_outcome.get("partial_decision_breakdown", [])
        if partial_breakdown:
            decision_lines.append("\nPartial Decision Breakdown:")
            for item in partial_breakdown[:5]:
                if isinstance(item, dict):
                    service = item.get("service", "")
                    decision = item.get("decision", "")
                    if service and decision:
                        decision_lines.append(f"  • {service}: {decision}")
                elif item:
                    decision_lines.append(f"  • {item}")
        
        # Effective dates
        effective_dates = decision_outcome.get("decision_effective_dates", {})
        if effective_dates.get("start_date") or effective_dates.get("end_date"):
            date_info = []
            if effective_dates.get("start_date"):
                date_info.append(f"Start: {effective_dates['start_date']}")
            if effective_dates.get("end_date"):
                date_info.append(f"End: {effective_dates['end_date']}")
            decision_lines.append(f"\nEffective Dates: {', '.join(date_info)}")
        
        sections.append("\n".join(decision_lines) if decision_lines else "No decision outcome extracted")
        
        # Section 5: MEDICAL NECESSITY DETERMINATION
        sections.append("\n🏥 MEDICAL NECESSITY DETERMINATION")
        sections.append("-" * 50)
        
        medical_necessity = raw_data.get("medical_necessity_determination", {})
        necessity_lines = []
        
        # Medical necessity
        med_necessity = medical_necessity.get("medical_necessity", "")
        if med_necessity:
            necessity_lines.append(f"Medical Necessity: {med_necessity}")
        
        # Criteria applied
        criteria = medical_necessity.get("criteria_applied", "")
        if criteria:
            necessity_lines.append(f"Criteria Applied: {criteria}")
        
        # Clinical rationale
        rationale = medical_necessity.get("clinical_rationale", "")
        if rationale:
            necessity_lines.append(f"Clinical Rationale: {rationale}")
        
        # Guidelines referenced
        guidelines = medical_necessity.get("guidelines_referenced", [])
        if guidelines:
            necessity_lines.append("\nGuidelines Referenced:")
            for guideline in guidelines[:5]:
                if isinstance(guideline, dict):
                    guideline_name = guideline.get("guideline", "")
                    if guideline_name:
                        necessity_lines.append(f"  • {guideline_name}")
                elif guideline:
                    necessity_lines.append(f"  • {guideline}")
        
        sections.append("\n".join(necessity_lines) if necessity_lines else "No medical necessity determination extracted")
        
        # Section 6: REVIEWER ANALYSIS
        sections.append("\n🔍 REVIEWER ANALYSIS")
        sections.append("-" * 50)
        
        reviewer_analysis = raw_data.get("reviewer_analysis", {})
        analysis_lines = []
        
        # Clinical summary
        clinical_summary = reviewer_analysis.get("clinical_summary_reviewed", "")
        if clinical_summary:
            analysis_lines.append(f"Clinical Summary Reviewed: {clinical_summary}")
        
        # Key findings
        key_findings = reviewer_analysis.get("key_findings", [])
        if key_findings:
            analysis_lines.append("\nKey Findings:")
            for finding in key_findings[:5]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("finding", "")
                    if finding_desc:
                        analysis_lines.append(f"  • {finding_desc}")
                elif finding:
                    analysis_lines.append(f"  • {finding}")
        
        # Documentation gaps
        doc_gaps = reviewer_analysis.get("documentation_gaps", [])
        if doc_gaps:
            analysis_lines.append("\nDocumentation Gaps:")
            for gap in doc_gaps[:3]:
                if isinstance(gap, dict):
                    gap_desc = gap.get("gap", "")
                    if gap_desc:
                        analysis_lines.append(f"  • {gap_desc}")
                elif gap:
                    analysis_lines.append(f"  • {gap}")
        
        sections.append("\n".join(analysis_lines) if analysis_lines else "No reviewer analysis extracted")
        
        # Section 7: APPEAL INFORMATION
        sections.append("\n🔄 APPEAL INFORMATION")
        sections.append("-" * 50)
        
        appeal_info = raw_data.get("appeal_information", {})
        appeal_lines = []
        
        # Appeal deadline
        appeal_deadline = appeal_info.get("appeal_deadline", "")
        if appeal_deadline:
            appeal_lines.append(f"Appeal Deadline: {appeal_deadline}")
        
        # Appeal procedures
        appeal_procedures = appeal_info.get("appeal_procedures", "")
        if appeal_procedures:
            appeal_lines.append(f"Appeal Procedures: {appeal_procedures}")
        
        # Required documentation
        required_docs = appeal_info.get("required_documentation", [])
        if required_docs:
            appeal_lines.append("\nRequired Documentation:")
            for doc in required_docs[:5]:
                if isinstance(doc, dict):
                    doc_desc = doc.get("document", "")
                    if doc_desc:
                        appeal_lines.append(f"  • {doc_desc}")
                elif doc:
                    appeal_lines.append(f"  • {doc}")
        
        # Timeframe for response
        response_timeframe = appeal_info.get("timeframe_for_response", "")
        if response_timeframe:
            appeal_lines.append(f"Timeframe for Response: {response_timeframe}")
        
        sections.append("\n".join(appeal_lines) if appeal_lines else "No appeal information extracted")
        
        # Section 8: CRITICAL ACTIONS REQUIRED
        sections.append("\n🚨 CRITICAL ACTIONS REQUIRED")
        sections.append("-" * 50)
        
        critical_actions = raw_data.get("critical_actions_required", [])
        if critical_actions:
            for action in critical_actions[:8]:
                if isinstance(action, dict):
                    action_desc = action.get("action", "")
                    action_deadline = action.get("deadline", "")
                    if action_desc:
                        if action_deadline:
                            sections.append(f"• {action_desc} (Deadline: {action_deadline})")
                        else:
                            sections.append(f"• {action_desc}")
                elif action:
                    sections.append(f"• {action}")
        else:
            # Extract critical actions from appeal information if no specific critical actions
            if appeal_info.get("appeal_deadline"):
                sections.append(f"• Appeal Deadline: {appeal_info['appeal_deadline']}")
            if not sections[-1].startswith("•"):
                sections.append("No time-critical actions identified")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Decision long summary built: {len(long_summary)} characters")
        
        return long_summary

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a comprehensive 60-word short summary covering all key decision aspects.
        """
        logger.info("🎯 Generating comprehensive 60-word decision short summary from long summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal specialist creating PRECISE 60-word summaries of {doc_type} decision documents.

CRITICAL REQUIREMENTS:
- EXACTLY 60 words (count carefully - this is mandatory)
- Cover ALL essential aspects in this order:
  1. Document type and decision date
  2. Requesting provider and patient context
  3. Services/treatments requested
  4. Final decision outcome (APPROVED/DENIED/PARTIAL)
  5. Medical necessity determination
  6. Key rationale for decision
  7. Appeal deadline if applicable

CONTENT RULES:
- MUST include the final decision status
- Include key services/treatments decided upon
- Mention medical necessity determination
- Include appeal deadlines if specified
- Be specific about partial approvals

WORD COUNT ENFORCEMENT:
- Count your words precisely before responding
- If over 60 words, remove less critical details
- If under 60 words, add more specific decision details
- Never exceed 60 words

FORMAT:
- Single paragraph, no bullet points
- Natural medical-legal narrative flow
- Use complete sentences
- Include quantitative data when available

EXAMPLES (60 words each):

✅ "UR Decision dated 10/15/2024 for Dr. Smith's request for lumbar epidural steroid injection for patient with chronic back pain. Decision: NOT MEDICALLY NECESSARY per ODG guidelines. Rationale: insufficient conservative treatment trial. Appeal deadline: 30 days from receipt. Required: documentation of failed physical therapy and medication management."

✅ "IMR Appeal determination approved partial request for pain management. Approved: 8 physical therapy sessions. Denied: continued opioid medication. Medical necessity: PT supported, opioids not indicated per MTUS. Decision effective 11/01/2024. No further appeal rights available for this determination."

Now create a PRECISE 60-word medical-legal decision summary from this long summary:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPREHENSIVE DECISION LONG SUMMARY:

{long_summary}

Create a PRECISE 60-word decision summary that includes:
1. Document type and decision date
2. Requesting provider context  
3. Key services/treatments decided
4. Final decision outcome
5. Medical necessity determination
6. Key rationale
7. Appeal information if applicable

60-WORD DECISION SUMMARY:
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        # Retry configuration (same as QME extractor)
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for decision short summary generation...")
                
                chain = chat_prompt | self.llm
                response = chain.invoke({
                    "doc_type": doc_type,
                    "long_summary": long_summary
                })
                
                short_summary = response.content.strip()
                end_time = time.time()
                
                # Clean and validate
                short_summary = self._clean_and_validate_short_summary(short_summary)
                word_count = len(short_summary.split())
                
                logger.info(f"⚡ Decision short summary generated in {end_time - start_time:.2f}s: {word_count} words")
                
                # Validate word count strictly
                if word_count == 60:
                    logger.info("✅ Perfect 60-word decision summary generated!")
                    return short_summary
                else:
                    logger.warning(f"⚠️ Decision summary has {word_count} words (expected 60), attempt {attempt + 1}")
                    
                    if attempt < max_retries - 1:
                        # Add word count feedback to next attempt
                        feedback_prompt = self._get_word_count_feedback_prompt(word_count, doc_type)
                        chat_prompt = ChatPromptTemplate.from_messages([feedback_prompt, user_prompt])
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"⚠️ Final decision summary has {word_count} words after {max_retries} attempts")
                        return short_summary
                        
            except Exception as e:
                logger.error(f"❌ Decision short summary generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"❌ All {max_retries} attempts failed for decision summary generation")
                    return self._create_decision_fallback_summary(long_summary, doc_type)
        
        return self._create_decision_fallback_summary(long_summary, doc_type)

    def _get_word_count_feedback_prompt(self, actual_word_count: int, doc_type: str) -> SystemMessagePromptTemplate:
        """Get feedback prompt for word count adjustment for decision documents"""
        
        if actual_word_count > 60:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO LONG). Remove less critical details to reach exactly 60 words. Prioritize: decision outcome, key services, medical necessity, appeal deadline."
        else:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO SHORT). Add more specific decision details to reach exactly 60 words. Include: specific service names, decision rationale, timeframe details."
        
        return SystemMessagePromptTemplate.from_template(f"""
You are a medical-legal specialist creating PRECISE 60-word summaries of {doc_type} documents.

CRITICAL FEEDBACK: {feedback}

REQUIREMENTS:
- EXACTLY 60 words
- Include: decision outcome, key services, medical necessity determination, appeal information
- Count words carefully before responding
- Adjust length by adding/removing specific decision details

""")

    def _clean_and_validate_short_summary(self, summary: str) -> str:
        """Clean and validate the 60-word short summary (same as QME version)"""
        # Remove excessive whitespace, quotes, and markdown
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = summary.replace('"', '').replace("'", "")
        summary = re.sub(r'[\*\#\-]', '', summary)
        
        # Remove common prefixes
        summary = re.sub(r'^(60-word summary:|summary:|decision summary:)\s*', '', summary, flags=re.IGNORECASE)
        
        # Count words
        words = summary.split()
        
        # Strict word count enforcement
        if len(words) != 60:
            logger.info(f"📝 Decision word count adjustment needed: {len(words)} words")
            
            if len(words) > 60:
                summary = self._trim_to_60_words(words)
            else:
                summary = self._expand_to_60_words(words, summary)
        
        return summary

    def _trim_to_60_words(self, words: List[str]) -> str:
        """Intelligently trim words to reach exactly 60 (same as QME version)"""
        if len(words) <= 60:
            return ' '.join(words)
        
        text = ' '.join(words)
        
        # Decision-specific reductions
        reductions = [
            (r'\b(and|with|including)\s+appropriate\s+', ' '),
            (r'\bfor\s+(a|the)\s+period\s+of\s+\w+\s+\w+', ' '),
            (r'\bwith\s+follow[- ]?up\s+in\s+\w+\s+\w+', ' with follow-up'),
            (r'\bmedical\s+necessity', 'med necessity'),
            (r'\brequested\s+services?\s*', 'requested '),
            (r'\bdetermination\s+', 'determ '),
        ]
        
        for pattern, replacement in reductions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        words = text.split()
        if len(words) > 60:
            excess = len(words) - 60
            mid_point = len(words) // 2
            start_remove = mid_point - excess // 2
            words = words[:start_remove] + words[start_remove + excess:]
        
        return ' '.join(words[:60])

    def _expand_to_60_words(self, words: List[str], original_text: str) -> str:
        """Intelligently expand text to reach exactly 60 words (same as QME version)"""
        if len(words) >= 60:
            return ' '.join(words)
        
        needed_words = 60 - len(words)
        
        # Decision-specific expansions
        expansions = []
        
        if 'appeal' in original_text.lower():
            expansions.append("with specified appeal procedures")
        
        if 'medical necessity' in original_text.lower():
            expansions.append("based on established guidelines")
        
        if 'partial' in original_text.lower():
            expansions.append("with specific service limitations")
        
        # Add generic decision context
        while len(words) + len(expansions) < 60 and len(expansions) < 5:
            expansions.extend([
                "per established medical guidelines",
                "with detailed clinical rationale", 
                "based on documentation review",
                "following utilization review",
                "with specified effective dates"
            ])
        
        # Add expansions to the text
        expanded_text = original_text
        for expansion in expansions[:needed_words]:
            expanded_text += f" {expansion}"
        
        words = expanded_text.split()
        return ' '.join(words[:60])

    def _create_decision_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback decision summary directly from long summary"""
        
        # Extract key decision information using regex patterns
        patterns = {
            'decision': r'Overall Decision:\s*([^\n]+)',
            'provider': r'Requesting Provider:\s*([^\n]+)',
            'services': r'Requested Services:(.*?)(?:\n\n|\n[A-Z]|$)',
            'medical_necessity': r'Medical Necessity:\s*([^\n]+)',
            'appeal_deadline': r'Appeal Deadline:\s*([^\n]+)',
            'rationale': r'Clinical Rationale:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and decision
        parts.append(f"{doc_type} decision")
        
        if 'decision' in extracted:
            parts.append(f"outcome: {extracted['decision']}")
        
        # Add provider context
        if 'provider' in extracted:
            parts.append(f"for {extracted['provider']}")
        
        # Add services
        if 'services' in extracted:
            first_service = extracted['services'].split('\n')[0].replace('•', '').strip()[:80]
            parts.append(f"regarding {first_service}")
        
        # Add medical necessity
        if 'medical_necessity' in extracted:
            parts.append(f"Medical necessity: {extracted['medical_necessity']}")
        
        # Add appeal information
        if 'appeal_deadline' in extracted:
            parts.append(f"Appeal deadline: {extracted['appeal_deadline']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["based on clinical documentation review", "following established guidelines", "with specified determination date"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used decision fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, doc_type: str, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for decision documents"""
        return {
            "document_identity": {
                "document_type": doc_type,
                "document_date": fallback_date,
                "decision_date": "",
                "effective_date": "",
                "document_id": "",
                "claim_number": "",
                "case_number": "",
                "jurisdiction": ""
            },
            "parties_involved": {
                "patient": {
                    "name": "",
                    "date_of_birth": "",
                    "member_id": ""
                },
                "requesting_provider": {
                    "name": "",
                    "specialty": "",
                    "npi": "",
                    "contact_info": ""
                },
                "reviewing_entity": {
                    "name": "",
                    "reviewer_name": "",
                    "reviewer_credentials": "",
                    "contact_info": ""
                },
                "claims_administrator": {
                    "name": "",
                    "contact_info": ""
                }
            },
            "request_details": {
                "date_of_service_requested": "",
                "request_received_date": "",
                "requested_services": [],
                "clinical_reason": "",
                "supporting_documentation": []
            },
            "decision_outcome": {
                "overall_decision": "",
                "decision_date": "",
                "decision_details": "",
                "partial_decision_breakdown": [],
                "decision_effective_dates": {
                    "start_date": "",
                    "end_date": ""
                }
            },
            "medical_necessity_determination": {
                "medical_necessity": "",
                "criteria_applied": "",
                "clinical_rationale": "",
                "supporting_evidence": [],
                "guidelines_referenced": []
            },
            "reviewer_analysis": {
                "clinical_summary_reviewed": "",
                "key_findings": [],
                "consultant_opinions": [],
                "documentation_gaps": []
            },
            "appeal_information": {
                "appeal_deadline": "",
                "appeal_procedures": "",
                "required_documentation": [],
                "contact_information": "",
                "timeframe_for_response": ""
            },
            "regulatory_compliance": {
                "regulatory_references": [],
                "timeliness_compliance": "",
                "reviewer_qualifications": ""
            },
            "critical_actions_required": []
        }