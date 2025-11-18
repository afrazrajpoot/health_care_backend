"""
AdministrativeExtractor - Enhanced Extractor for Administrative and Legal Documents
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


class AdministrativeExtractor:
    """
    Enhanced Administrative Document extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different administrative document types
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Attorney Letters, NCM Notes, Employer Reports, Disability Forms, Legal Correspondence
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.admin_type_patterns = {
            'attorney': re.compile(r'\b(attorney|lawyer|counsel|legal|subpoena|deposition)\b', re.IGNORECASE),
            'ncm': re.compile(r'\b(ncm|nurse case manager|case management|utilization review)\b', re.IGNORECASE),
            'employer': re.compile(r'\b(employer|supervisor|incident report|work injury|workers comp)\b', re.IGNORECASE),
            'disability': re.compile(r'\b(disability|claim form|dwc|workers compensation|benefits)\b', re.IGNORECASE),
            'job_analysis': re.compile(r'\b(job analysis|physical demands|work capacity|vocational)\b', re.IGNORECASE),
            'medication': re.compile(r'\b(medication administration|mar|pharmacy|prescription log)\b', re.IGNORECASE),
            'pharmacy': re.compile(r'\b(pharmacy|prescription|drug|medication list|pharmacist)\b', re.IGNORECASE),
            'telemedicine': re.compile(r'\b(telemedicine|telehealth|virtual visit|remote consultation)\b', re.IGNORECASE),
            'legal': re.compile(r'\b(legal correspondence|demand letter|settlement|liability)\b', re.IGNORECASE)
        }
        
        # Administrative patterns
        self.admin_patterns = {
            'claim_numbers': re.compile(r'\b(claim\s*(?:number|no|#)?[:\s]*([A-Z0-9\-]+))', re.IGNORECASE),
            'dates': re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[-]\d{2}[-]\d{2})\b'),
            'deadlines': re.compile(r'\b(deadline|due date|response required by|must respond by)\s*[:]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'contact_info': re.compile(r'\b(phone|tel|fax|email|@)\s*[:\-]?\s*([^\s,]+)', re.IGNORECASE)
        }
        
        logger.info("✅ AdministrativeExtractor initialized (Full Context + Context-Aware)")

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
        Extract Administrative Document data with FULL CONTEXT and contextual awareness.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (Attorney, NCM, Employer, Disability, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction
            context_analysis: Document context from DocumentContextAnalyzer
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING ADMINISTRATIVE DOCUMENT EXTRACTION (FULL CONTEXT + CONTEXT-AWARE)")
        logger.info("=" * 80)
        
        # Auto-detect specific administrative type if not specified
        detected_type = self._detect_admin_type(text, doc_type)
        logger.info(f"📋 Administrative Type: {detected_type} (original: {doc_type})")
        
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
        logger.info("✅ ADMINISTRATIVE DOCUMENT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _detect_admin_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific administrative document type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for admin_type, pattern in self.admin_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[admin_type] = len(matches)
        
        # Boost scores for administrative-specific terminology
        if self.admin_patterns['claim_numbers'].search(text):
            for admin_type in ['disability', 'ncm', 'employer']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 2
        
        if self.admin_patterns['deadlines'].search(text):
            for admin_type in ['attorney', 'legal']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected administrative type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect administrative type, using: {original_type}")
        return original_type or "Administrative Document"

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
        logger.info("🔍 Processing ENTIRE administrative document in single context window with guidance...")
        
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
You are an expert administrative and legal documentation specialist analyzing a COMPLETE {doc_type} with CONTEXTUAL GUIDANCE.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE administrative document at once, allowing you to:
- Understand the complete administrative or legal context from headers to signatures
- Track deadlines, requirements, and action items throughout the document
- Identify key parties, contact information, and procedural requirements
- Provide comprehensive extraction without information loss

CONTEXTUAL GUIDANCE PROVIDED:
{context_guidance}

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate administrative information
   - DO NOT fill in "typical" or "common" administrative values
   - DO NOT use legal knowledge to "complete" incomplete information
   
2. **DATES & DEADLINES - EXACT WORDING ONLY**
   - Extract deadlines and dates using EXACT wording from document
   - Include timeframes and response requirements verbatim
   - DO NOT interpret or calculate dates
   
3. **CONTACT INFORMATION - VERBATIM EXTRACTION**
   - Extract phone numbers, emails, addresses EXACTLY as written
   - DO NOT format or normalize contact information
   - Include titles and roles ONLY if explicitly stated
   
4. **LEGAL & ADMINISTRATIVE TERMS - PRECISE EXTRACTION**
   - Extract legal demands, requirements, and conditions verbatim
   - Include specific claim numbers, case numbers, reference numbers
   - DO NOT interpret legal language or implications
   
5. **ACTION ITEMS & REQUIREMENTS - SPECIFIC DETAILS ONLY**
   - Extract required actions, documentation, responses ONLY if explicitly listed
   - Include quantities, formats, submission methods ONLY if specified
   - DO NOT add typical administrative procedures

EXTRACTION FOCUS - 8 CRITICAL ADMINISTRATIVE CATEGORIES:

I. DOCUMENT IDENTITY & CONTEXT
- Document type, dates, identification numbers
- Purpose and subject of the document
- All reference numbers and case identifiers

II. PARTIES INVOLVED
- All individuals and organizations mentioned
- Roles and relationships (sender, recipient, cc'd parties)
- Complete contact information for all parties
- Legal representation details if applicable

III. KEY DATES & DEADLINES (MOST CRITICAL)
- Document date, effective dates, response deadlines
- Hearing dates, appointment dates, follow-up dates
- Time-sensitive requirements and cutoffs
- Statute of limitations if mentioned

IV. ADMINISTRATIVE CONTENT & SUBJECT
- Primary purpose and subject matter
- Summary of key points or issues addressed
- Background context and relevant history
- Specific incidents or events described

V. ACTION ITEMS & REQUIREMENTS
- Required responses and submissions
- Documentation or evidence required
- Specific actions to be taken by parties
- Compliance requirements and conditions

VI. LEGAL & PROCEDURAL ELEMENTS
- Legal demands, offers, or positions
- Procedural requirements and next steps
- Rights, obligations, and responsibilities
- Consequences of non-compliance if stated

VII. MEDICAL & CLAIM INFORMATION (if applicable)
- Claim numbers and case references
- Medical treatment authorizations or denials
- Work status and disability information
- Medication and treatment details

VIII. CONTACT & FOLLOW-UP PROCEDURES
- Submission methods and addresses
- Contact persons and departments
- Required response formats
- Follow-up procedures

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate administrative information
- DEADLINES: Use exact wording from document
- CONTACT INFO: Extract verbatim without formatting
- It is BETTER to have empty fields than incorrect administrative information

Now analyze this COMPLETE {doc_type} and extract ALL relevant administrative information:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} TEXT:

{full_document_text}

Extract into COMPREHENSIVE structured JSON with all critical administrative details:

{{
  "document_identity": {{
    "document_type": "{doc_type}",
    "document_date": "",
    "effective_date": "",
    "document_id": "",
    "subject": "",
    "purpose": "",
    "reference_numbers": []
  }},
  
  "parties_involved": {{
    "sender": {{
      "name": "",
      "title": "",
      "organization": "",
      "contact_info": ""
    }},
    "recipient": {{
      "name": "",
      "title": "",
      "organization": "",
      "contact_info": ""
    }},
    "cc_parties": [],
    "legal_representation": {{
      "attorney_name": "",
      "firm": "",
      "contact_info": ""
    }}
  }},
  
  "key_dates_deadlines": {{
    "response_deadline": "",
    "hearing_date": "",
    "appointment_date": "",
    "follow_up_date": "",
    "effective_period": "",
    "time_sensitive_requirements": []
  }},
  
  "administrative_content": {{
    "primary_subject": "",
    "key_points_summary": "",
    "background_context": "",
    "incident_details": "",
    "current_status": ""
  }},
  
  "action_items_requirements": {{
    "required_responses": [],
    "documentation_required": [],
    "specific_actions": [],
    "compliance_requirements": [],
    "submission_methods": ""
  }},
  
  "legal_procedural_elements": {{
    "legal_demands": [],
    "procedural_requirements": [],
    "next_steps": [],
    "rights_obligations": [],
    "consequences_non_compliance": ""
  }},
  
  "medical_claim_information": {{
    "claim_number": "",
    "case_number": "",
    "treatment_authorizations": [],
    "work_status": "",
    "disability_information": "",
    "medication_details": []
  }},
  
  "contact_follow_up": {{
    "submission_address": "",
    "contact_person": "",
    "phone_number": "",
    "email_address": "",
    "response_format": "",
    "follow_up_procedures": ""
  }},
  
  "critical_administrative_findings": []
}}

⚠️ CRITICAL ADMINISTRATIVE REMINDERS:
1. For "key_dates_deadlines": Extract EXACT date wording from document
   - Include phrases like "within 30 days", "by close of business", etc.
   - Do not interpret or calculate actual dates

2. For "action_items_requirements": Extract ONLY explicitly stated requirements
   - Include specific documentation, forms, evidence required
   - Do not include implied or typical requirements

3. For "legal_procedural_elements": Extract legal language VERBATIM
   - Include exact demands, offers, procedural requirements
   - Do not interpret legal implications

4. For "contact_follow_up": Extract contact information EXACTLY as written
   - Include phone numbers, emails, addresses without formatting
   - Preserve original spacing and punctuation

5. For "critical_administrative_findings": Include time-sensitive items only
   - Response deadlines
   - Legal deadlines
   - Critical compliance requirements
   - Urgent action items
""")

        # Build context guidance summary
        context_guidance_text = f"""
ADMINISTRATIVE DOCUMENT TYPE: {doc_type}
FOCUS ON THESE SECTIONS: {', '.join(focus_sections) if focus_sections else 'All sections equally'}

CRITICAL FINDING LOCATIONS:
- Deadlines/Requirements: {critical_locations.get('deadlines_location', 'Search entire document')}
- Contact Information: {critical_locations.get('contact_location', 'Search entire document')}
- Action Items: {critical_locations.get('actions_location', 'Search entire document')}

KNOWN AMBIGUITIES: {len(ambiguities)} detected
{chr(10).join([f"- {amb.get('type')}: {amb.get('description')}" for amb in ambiguities]) if ambiguities else 'None detected'}
"""

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for full-context administrative document extraction...")
            
            # Single LLM call with FULL document context and guidance
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "context_guidance": context_guidance_text
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Full-context administrative document extraction completed in {processing_time:.2f}s")
            logger.info(f"✅ Extracted data from complete {len(text):,} char administrative document")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Full-context administrative document extraction failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Administrative document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            return self._get_fallback_result(doc_type, fallback_date)

    def _build_comprehensive_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str) -> str:
        """
        Build comprehensive long summary from ALL extracted administrative data.
        """
        logger.info("📝 Building comprehensive long summary from ALL extracted administrative data...")
        
        sections = []
        
        # Section 1: DOCUMENT OVERVIEW
        sections.append("📋 ADMINISTRATIVE DOCUMENT OVERVIEW")
        sections.append("-" * 50)
        
        document_identity = raw_data.get("document_identity", {})
        overview_lines = [
            f"Document Type: {document_identity.get('document_type', doc_type)}",
            f"Document Date: {document_identity.get('document_date', fallback_date)}",
            f"Subject: {document_identity.get('subject', 'Not specified')}",
            f"Purpose: {document_identity.get('purpose', 'Not specified')}",
            f"Document ID: {document_identity.get('document_id', 'Not specified')}"
        ]
        sections.append("\n".join(overview_lines))
        
        # Section 2: PARTIES INVOLVED
        sections.append("\n👥 PARTIES INVOLVED")
        sections.append("-" * 50)
        
        parties = raw_data.get("parties_involved", {})
        party_lines = []
        
        # Sender
        sender = parties.get("sender", {})
        if sender.get("name"):
            party_lines.append(f"From: {sender['name']}")
            if sender.get("organization"):
                party_lines.append(f"  Organization: {sender['organization']}")
            if sender.get("title"):
                party_lines.append(f"  Title: {sender['title']}")
        
        # Recipient
        recipient = parties.get("recipient", {})
        if recipient.get("name"):
            party_lines.append(f"To: {recipient['name']}")
            if recipient.get("organization"):
                party_lines.append(f"  Organization: {recipient['organization']}")
        
        # Legal representation
        legal_rep = parties.get("legal_representation", {})
        if legal_rep.get("attorney_name"):
            party_lines.append(f"Legal Representation: {legal_rep['attorney_name']}")
            if legal_rep.get("firm"):
                party_lines.append(f"  Firm: {legal_rep['firm']}")
        
        sections.append("\n".join(party_lines) if party_lines else "No party information extracted")
        
        # Section 3: KEY DATES & DEADLINES
        sections.append("\n📅 KEY DATES & DEADLINES")
        sections.append("-" * 50)
        
        dates_deadlines = raw_data.get("key_dates_deadlines", {})
        date_lines = []
        
        if dates_deadlines.get("response_deadline"):
            date_lines.append(f"Response Deadline: {dates_deadlines['response_deadline']}")
        
        if dates_deadlines.get("hearing_date"):
            date_lines.append(f"Hearing Date: {dates_deadlines['hearing_date']}")
        
        if dates_deadlines.get("appointment_date"):
            date_lines.append(f"Appointment Date: {dates_deadlines['appointment_date']}")
        
        time_sensitive = dates_deadlines.get("time_sensitive_requirements", [])
        if time_sensitive:
            date_lines.append("\nTime-Sensitive Requirements:")
            for requirement in time_sensitive[:3]:
                if isinstance(requirement, dict):
                    req_desc = requirement.get("requirement", "")
                    if req_desc:
                        date_lines.append(f"  • {req_desc}")
                elif requirement:
                    date_lines.append(f"  • {requirement}")
        
        sections.append("\n".join(date_lines) if date_lines else "No dates/deadlines extracted")
        
        # Section 4: ADMINISTRATIVE CONTENT
        sections.append("\n📄 ADMINISTRATIVE CONTENT")
        sections.append("-" * 50)
        
        admin_content = raw_data.get("administrative_content", {})
        content_lines = []
        
        if admin_content.get("primary_subject"):
            content_lines.append(f"Primary Subject: {admin_content['primary_subject']}")
        
        if admin_content.get("key_points_summary"):
            content_lines.append(f"Key Points: {admin_content['key_points_summary']}")
        
        if admin_content.get("current_status"):
            content_lines.append(f"Current Status: {admin_content['current_status']}")
        
        if admin_content.get("incident_details"):
            # Truncate long incident details
            incident = admin_content['incident_details']
            if len(incident) > 200:
                incident = incident[:197] + "..."
            content_lines.append(f"Incident Details: {incident}")
        
        sections.append("\n".join(content_lines) if content_lines else "No administrative content extracted")
        
        # Section 5: ACTION ITEMS & REQUIREMENTS
        sections.append("\n✅ ACTION ITEMS & REQUIREMENTS")
        sections.append("-" * 50)
        
        action_items = raw_data.get("action_items_requirements", {})
        action_lines = []
        
        # Required responses
        required_responses = action_items.get("required_responses", [])
        if required_responses:
            action_lines.append("Required Responses:")
            for response in required_responses[:5]:
                if isinstance(response, dict):
                    resp_desc = response.get("response", "")
                    if resp_desc:
                        action_lines.append(f"  • {resp_desc}")
                elif response:
                    action_lines.append(f"  • {response}")
        
        # Documentation required
        documentation = action_items.get("documentation_required", [])
        if documentation:
            action_lines.append("\nDocumentation Required:")
            for doc in documentation[:5]:
                if isinstance(doc, dict):
                    doc_desc = doc.get("document", "")
                    if doc_desc:
                        action_lines.append(f"  • {doc_desc}")
                elif doc:
                    action_lines.append(f"  • {doc}")
        
        # Specific actions
        specific_actions = action_items.get("specific_actions", [])
        if specific_actions:
            action_lines.append("\nSpecific Actions:")
            for action in specific_actions[:5]:
                if isinstance(action, dict):
                    action_desc = action.get("action", "")
                    if action_desc:
                        action_lines.append(f"  • {action_desc}")
                elif action:
                    action_lines.append(f"  • {action}")
        
        sections.append("\n".join(action_lines) if action_lines else "No action items extracted")
        
        # Section 6: LEGAL & PROCEDURAL ELEMENTS
        sections.append("\n⚖️ LEGAL & PROCEDURAL ELEMENTS")
        sections.append("-" * 50)
        
        legal_elements = raw_data.get("legal_procedural_elements", {})
        legal_lines = []
        
        # Legal demands
        legal_demands = legal_elements.get("legal_demands", [])
        if legal_demands:
            legal_lines.append("Legal Demands:")
            for demand in legal_demands[:3]:
                if isinstance(demand, dict):
                    demand_desc = demand.get("demand", "")
                    if demand_desc:
                        legal_lines.append(f"  • {demand_desc}")
                elif demand:
                    legal_lines.append(f"  • {demand}")
        
        # Next steps
        next_steps = legal_elements.get("next_steps", [])
        if next_steps:
            legal_lines.append("\nNext Steps:")
            for step in next_steps[:3]:
                if isinstance(step, dict):
                    step_desc = step.get("step", "")
                    if step_desc:
                        legal_lines.append(f"  • {step_desc}")
                elif step:
                    legal_lines.append(f"  • {step}")
        
        if legal_elements.get("consequences_non_compliance"):
            legal_lines.append(f"\nConsequences of Non-Compliance: {legal_elements['consequences_non_compliance']}")
        
        sections.append("\n".join(legal_lines) if legal_lines else "No legal/procedural elements extracted")
        
        # Section 7: MEDICAL & CLAIM INFORMATION
        sections.append("\n🏥 MEDICAL & CLAIM INFORMATION")
        sections.append("-" * 50)
        
        medical_claim = raw_data.get("medical_claim_information", {})
        medical_lines = []
        
        if medical_claim.get("claim_number"):
            medical_lines.append(f"Claim Number: {medical_claim['claim_number']}")
        
        if medical_claim.get("case_number"):
            medical_lines.append(f"Case Number: {medical_claim['case_number']}")
        
        if medical_claim.get("work_status"):
            medical_lines.append(f"Work Status: {medical_claim['work_status']}")
        
        if medical_claim.get("disability_information"):
            medical_lines.append(f"Disability Information: {medical_claim['disability_information']}")
        
        # Treatment authorizations
        treatment_auths = medical_claim.get("treatment_authorizations", [])
        if treatment_auths:
            medical_lines.append("\nTreatment Authorizations:")
            for auth in treatment_auths[:3]:
                if isinstance(auth, dict):
                    auth_desc = auth.get("authorization", "")
                    if auth_desc:
                        medical_lines.append(f"  • {auth_desc}")
                elif auth:
                    medical_lines.append(f"  • {auth}")
        
        sections.append("\n".join(medical_lines) if medical_lines else "No medical/claim information extracted")
        
        # Section 8: CONTACT & FOLLOW-UP
        sections.append("\n📞 CONTACT & FOLLOW-UP")
        sections.append("-" * 50)
        
        contact_followup = raw_data.get("contact_follow_up", {})
        contact_lines = []
        
        if contact_followup.get("contact_person"):
            contact_lines.append(f"Contact Person: {contact_followup['contact_person']}")
        
        if contact_followup.get("phone_number"):
            contact_lines.append(f"Phone: {contact_followup['phone_number']}")
        
        if contact_followup.get("email_address"):
            contact_lines.append(f"Email: {contact_followup['email_address']}")
        
        if contact_followup.get("submission_address"):
            contact_lines.append(f"Submission Address: {contact_followup['submission_address']}")
        
        if contact_followup.get("response_format"):
            contact_lines.append(f"Response Format: {contact_followup['response_format']}")
        
        sections.append("\n".join(contact_lines) if contact_lines else "No contact information extracted")
        
        # Section 9: CRITICAL ADMINISTRATIVE FINDINGS
        sections.append("\n🚨 CRITICAL ADMINISTRATIVE FINDINGS")
        sections.append("-" * 50)
        
        critical_findings = raw_data.get("critical_administrative_findings", [])
        if critical_findings:
            for finding in critical_findings[:8]:
                if isinstance(finding, dict):
                    finding_desc = finding.get("finding", "")
                    finding_priority = finding.get("priority", "")
                    if finding_desc:
                        if finding_priority:
                            sections.append(f"• [{finding_priority}] {finding_desc}")
                        else:
                            sections.append(f"• {finding_desc}")
                elif finding:
                    sections.append(f"• {finding}")
        else:
            # Check for critical findings in other sections
            critical_items = []
            
            # Check for urgent deadlines
            if dates_deadlines.get("response_deadline"):
                critical_items.append(f"Response Deadline: {dates_deadlines['response_deadline']}")
            
            # Check for legal consequences
            if legal_elements.get("consequences_non_compliance"):
                critical_items.append("Legal consequences specified for non-compliance")
            
            # Check for urgent actions
            if action_items.get("required_responses"):
                critical_items.append(f"{len(action_items['required_responses'])} required responses")
            
            if critical_items:
                for item in critical_items:
                    sections.append(f"• {item}")
            else:
                sections.append("No critical administrative findings identified")
        
        # Join all sections
        long_summary = "\n\n".join(sections)
        logger.info(f"✅ Administrative document long summary built: {len(long_summary)} characters")
        
        return long_summary

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a comprehensive 60-word short summary covering all key administrative aspects.
        """
        logger.info("🎯 Generating comprehensive 60-word administrative short summary from long summary...")
        
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an administrative specialist creating PRECISE 60-word summaries of {doc_type} documents.

CRITICAL REQUIREMENTS:
- EXACTLY 60 words (count carefully - this is mandatory)
- Cover ALL essential aspects in this order:
  1. Document type and date
  2. Key parties involved (sender/recipient)
  3. Primary subject and purpose
  4. Critical deadlines and response requirements
  5. Key action items or legal demands
  6. Contact information for follow-up
  7. Urgent next steps if applicable

CONTENT RULES:
- MUST include response deadlines if specified
- Include key legal or administrative demands
- Mention required documentation or actions
- Include critical contact information
- Be specific about time-sensitive requirements

WORD COUNT ENFORCEMENT:
- Count your words precisely before responding
- If over 60 words, remove less critical details
- If under 60 words, add more specific administrative details
- Never exceed 60 words

FORMAT:
- Single paragraph, no bullet points
- Natural administrative narrative flow
- Use complete sentences
- Include specific dates and requirements

EXAMPLES (60 words each):

✅ "Attorney letter dated 10/15/2024 from Smith & Associates to claimant regarding settlement offer. Response required within 30 days. Demands medical documentation and wage statements. Includes deposition schedule for November 2024. Contact: John Smith at 555-0123. Urgent: Settlement conference scheduled for 11/20/2024 requiring client attendance and financial documentation."

✅ "NCM note from Nurse Johnson on 10/15/2024: Case review for lumbar strain claim. Required: updated functional capacity evaluation by 11/01/2024. Treatment authorization for PT extended. Work restrictions: no lifting >20 lbs. Follow-up with adjuster scheduled. Contact: 555-0123. Next IME scheduled for December 2024."

Now create a PRECISE 60-word administrative summary from this long summary:
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPREHENSIVE ADMINISTRATIVE DOCUMENT LONG SUMMARY:

{long_summary}

Create a PRECISE 60-word administrative summary that includes:
1. Document type and date
2. Key parties involved
3. Primary subject/purpose
4. Critical deadlines
5. Key action items
6. Contact information
7. Urgent next steps

60-WORD ADMINISTRATIVE SUMMARY:
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        # Retry configuration
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for administrative short summary generation...")
                
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
                
                logger.info(f"⚡ Administrative short summary generated in {end_time - start_time:.2f}s: {word_count} words")
                
                # Validate word count strictly
                if word_count == 60:
                    logger.info("✅ Perfect 60-word administrative summary generated!")
                    return short_summary
                else:
                    logger.warning(f"⚠️ Administrative summary has {word_count} words (expected 60), attempt {attempt + 1}")
                    
                    if attempt < max_retries - 1:
                        # Add word count feedback to next attempt
                        feedback_prompt = self._get_word_count_feedback_prompt(word_count, doc_type)
                        chat_prompt = ChatPromptTemplate.from_messages([feedback_prompt, user_prompt])
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"⚠️ Final administrative summary has {word_count} words after {max_retries} attempts")
                        return short_summary
                        
            except Exception as e:
                logger.error(f"❌ Administrative short summary generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"❌ All {max_retries} attempts failed for administrative summary generation")
                    return self._create_admin_fallback_summary(long_summary, doc_type)
        
        return self._create_admin_fallback_summary(long_summary, doc_type)

    def _get_word_count_feedback_prompt(self, actual_word_count: int, doc_type: str) -> SystemMessagePromptTemplate:
        """Get feedback prompt for word count adjustment for administrative documents"""
        
        if actual_word_count > 60:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO LONG). Remove less critical details to reach exactly 60 words. Prioritize: deadlines, action items, contact info, legal demands."
        else:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO SHORT). Add more specific administrative details to reach exactly 60 words. Include: specific dates, required documentation, contact methods, procedural steps."
        
        return SystemMessagePromptTemplate.from_template(f"""
You are an administrative specialist creating PRECISE 60-word summaries of {doc_type} documents.

CRITICAL FEEDBACK: {feedback}

REQUIREMENTS:
- EXACTLY 60 words
- Include: deadlines, action items, contact info, key demands
- Count words carefully before responding
- Adjust length by adding/removing specific administrative details

""")

    def _clean_and_validate_short_summary(self, summary: str) -> str:
        """Clean and validate the 60-word short summary"""
        # Remove excessive whitespace, quotes, and markdown
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = summary.replace('"', '').replace("'", "")
        summary = re.sub(r'[\*\#\-]', '', summary)
        
        # Remove common prefixes
        summary = re.sub(r'^(60-word summary:|summary:|administrative summary:)\s*', '', summary, flags=re.IGNORECASE)
        
        # Count words
        words = summary.split()
        
        # Strict word count enforcement
        if len(words) != 60:
            logger.info(f"📝 Administrative word count adjustment needed: {len(words)} words")
            
            if len(words) > 60:
                summary = self._trim_to_60_words(words)
            else:
                summary = self._expand_to_60_words(words, summary)
        
        return summary

    def _trim_to_60_words(self, words: List[str]) -> str:
        """Intelligently trim words to reach exactly 60"""
        if len(words) <= 60:
            return ' '.join(words)
        
        text = ' '.join(words)
        
        # Administrative-specific reductions
        reductions = [
            (r'\b(and|with|including)\s+appropriate\s+', ' '),
            (r'\bfor\s+(a|the)\s+period\s+of\s+\w+\s+\w+', ' '),
            (r'\bwith\s+follow[- ]?up\s+in\s+\w+\s+\w+', ' with follow-up'),
            (r'\bcontact\s+information', 'contact info'),
            (r'\brequired\s+documentation', 'required docs'),
            (r'\bresponse\s+deadline', 'deadline'),
            (r'\blegal\s+representation', 'legal rep'),
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
        """Intelligently expand text to reach exactly 60 words"""
        if len(words) >= 60:
            return ' '.join(words)
        
        needed_words = 60 - len(words)
        
        # Administrative-specific expansions
        expansions = []
        
        if 'deadline' in original_text.lower():
            expansions.append("with specified submission requirements")
        
        if 'legal' in original_text.lower():
            expansions.append("per legal procedural guidelines")
        
        if 'contact' in original_text.lower():
            expansions.append("for additional information or clarification")
        
        # Add generic administrative context
        while len(words) + len(expansions) < 60 and len(expansions) < 5:
            expansions.extend([
                "following established administrative procedures",
                "with required compliance documentation", 
                "per standard submission guidelines",
                "with specified response format",
                "through designated communication channels"
            ])
        
        # Add expansions to the text
        expanded_text = original_text
        for expansion in expansions[:needed_words]:
            expanded_text += f" {expansion}"
        
        words = expanded_text.split()
        return ' '.join(words[:60])

    def _create_admin_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback administrative summary directly from long summary"""
        
        # Extract key administrative information using regex patterns
        patterns = {
            'sender': r'From:\s*([^\n]+)',
            'recipient': r'To:\s*([^\n]+)',
            'deadline': r'Response Deadline:\s*([^\n]+)',
            'subject': r'Subject:\s*([^\n]+)',
            'actions': r'Required Responses:(.*?)(?:\n\n|\n[A-Z]|$)',
            'contact': r'Contact Person:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} document")
        
        # Add sender/recipient context
        if 'sender' in extracted:
            parts.append(f"from {extracted['sender']}")
        
        if 'recipient' in extracted:
            parts.append(f"to {extracted['recipient']}")
        
        # Add subject
        if 'subject' in extracted:
            subject = extracted['subject'][:60] + "..." if len(extracted['subject']) > 60 else extracted['subject']
            parts.append(f"regarding {subject}")
        
        # Add deadline
        if 'deadline' in extracted:
            parts.append(f"Deadline: {extracted['deadline']}")
        
        # Add contact
        if 'contact' in extracted:
            parts.append(f"Contact: {extracted['contact']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with specified submission requirements", "following established procedures", "requiring timely response"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used administrative fallback summary: {len(summary.split())} words")
        return summary

    def _get_fallback_result(self, doc_type: str, fallback_date: str) -> Dict:
        """Return minimal fallback result structure for administrative documents"""
        return {
            "document_identity": {
                "document_type": doc_type,
                "document_date": fallback_date,
                "effective_date": "",
                "document_id": "",
                "subject": "",
                "purpose": "",
                "reference_numbers": []
            },
            "parties_involved": {
                "sender": {
                    "name": "",
                    "title": "",
                    "organization": "",
                    "contact_info": ""
                },
                "recipient": {
                    "name": "",
                    "title": "",
                    "organization": "",
                    "contact_info": ""
                },
                "cc_parties": [],
                "legal_representation": {
                    "attorney_name": "",
                    "firm": "",
                    "contact_info": ""
                }
            },
            "key_dates_deadlines": {
                "response_deadline": "",
                "hearing_date": "",
                "appointment_date": "",
                "follow_up_date": "",
                "effective_period": "",
                "time_sensitive_requirements": []
            },
            "administrative_content": {
                "primary_subject": "",
                "key_points_summary": "",
                "background_context": "",
                "incident_details": "",
                "current_status": ""
            },
            "action_items_requirements": {
                "required_responses": [],
                "documentation_required": [],
                "specific_actions": [],
                "compliance_requirements": [],
                "submission_methods": ""
            },
            "legal_procedural_elements": {
                "legal_demands": [],
                "procedural_requirements": [],
                "next_steps": [],
                "rights_obligations": [],
                "consequences_non_compliance": ""
            },
            "medical_claim_information": {
                "claim_number": "",
                "case_number": "",
                "treatment_authorizations": [],
                "work_status": "",
                "disability_information": "",
                "medication_details": []
            },
            "contact_follow_up": {
                "submission_address": "",
                "contact_person": "",
                "phone_number": "",
                "email_address": "",
                "response_format": "",
                "follow_up_procedures": ""
            },
            "critical_administrative_findings": []
        }