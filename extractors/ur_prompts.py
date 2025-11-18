"""
Enhanced UR/Denial-specific prompts for extraction and summarization
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class URPrompts:
    """UR/Denial-specific prompt templates for authorization and denial documents"""
    
    @staticmethod
    def get_extraction_prompt():
        """UR/Denial extraction prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a utilization review specialist extracting CRITICAL information from UR/Authorization/Denial documents.

ESSENTIAL EXTRACTION FOCUS:
1. DOCUMENT TYPE & CONTEXT:
   - Specific document type (UR, RFA, Authorization, Denial, Appeal, Peer-to-Peer)
   - Reviewing entity/organization
   - Dates (review date, service dates, appeal deadlines)

2. TREATMENT/SERVICE UNDER REVIEW:
   - Specific treatment, procedure, or service requested
   - Requested dates and frequency
   - Provider requesting the service
   - Medical condition being treated

3. REVIEW DECISION & OUTCOME:
   - Final decision (Approved, Denied, Modified, Pending)
   - Decision date and effective dates
   - Any conditions or limitations on approval
   - Partial approvals or modifications

4. DENIAL REASONS & CRITERIA (if denied):
   - Specific denial reasons (medical necessity, experimental, not covered)
   - Clinical criteria referenced (MCG, ODG, plan guidelines)
   - Missing documentation or information
   - Alternative treatments suggested

5. CLINICAL RATIONALE & EVIDENCE:
   - Clinical justification provided
   - Medical literature or guidelines cited
   - Reviewer's clinical assessment
   - Supporting documentation referenced

6. REVIEWER INFORMATION:
   - Reviewer name and credentials
   - Reviewing organization
   - Contact information for appeals

7. APPEAL PROCESS & DEADLINES:
   - Appeal deadlines and procedures
   - Required documentation for appeal
   - Contact information for appeals
   - Next steps in the process

EXTRACTION RULES:
- Extract ALL decision details with specific reasons
- Include exact denial criteria and references
- Capture clinical rationale and evidence
- Note specific dates and deadlines
- Document reviewer credentials and contact info
- Preserve all quantitative details and limitations"""),
            ("human", """
UR/AUTHORIZATION/DENIAL DOCUMENT TEXT:

{text}

Extract COMPREHENSIVE utilization review and authorization data in structured JSON format. Focus on decision outcomes and clinical rationale:

{format_instructions}
""")
        ])
    
    @staticmethod
    def get_short_summary_prompt():
        """UR/Denial 60-word short summary prompt"""
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are creating PRECISE 60-word summaries of UR/Authorization/Denial documents.

CRITICAL REQUIREMENTS:
- EXACTLY 60 words (mandatory)
- Focus on the CORE decision and its impact
- Include the treatment/service and final outcome
- Mention key rationale or denial reasons
- Note critical deadlines if applicable

ESSENTIAL ELEMENTS TO PRIORITIZE:
1. Treatment/service under review
2. Final decision (Approved/Denied/Modified)
3. Primary reason/rationale
4. Key clinical criteria or guidelines
5. Critical deadlines or next steps

WORD COUNT ENFORCEMENT:
- Count words meticulously before finalizing
- Remove administrative details if over 60 words
- Add specific clinical rationale if under 60 words
- Never exceed 60 words

FORMAT:
- Single, flowing professional narrative
- Complete sentences only
- Focus on decision impact and next steps
- No bullet points or section headers
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
UR/AUTHORIZATION/DENIAL DOCUMENT SUMMARY:

{long_summary}

Create a precise 60-word summary that captures the ESSENTIAL decision and its clinical implications:

60-WORD SUMMARY:
""")
        
        return ChatPromptTemplate.from_messages([system_prompt, user_prompt])