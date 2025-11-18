"""
Enhanced QME-specific prompts for comprehensive extraction and summarization
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class QMEPrompts:
    """Enhanced QME-specific prompt templates for comprehensive extraction"""
    
    @staticmethod
    def get_extraction_prompt():
        """Comprehensive QME extraction prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a medical-legal specialist extracting COMPREHENSIVE information from Qualified Medical Evaluation (QME) reports. Extract EVERY important detail for a thorough 400-500 word summary.

CRITICAL EXTRACTION FOCUS - EXTRACT ALL AVAILABLE INFORMATION:

1. PHYSICIAN & EVALUATION CONTEXT:
   - Physician name, credentials, specialty
   - Evaluation date, location, type (QME/AME/IME)
   - Date of injury, mechanism of injury
   - Claim number, case details

2. PATIENT DEMOGRAPHICS & HISTORY:
   - Patient age, occupation, work history
   - Prior injuries, pre-existing conditions
   - Treatment history, previous surgeries
   - Current work status

3. SUBJECTIVE COMPLAINTS:
   - Pain location, intensity (0-10 scale), quality, radiation
   - Aggravating and relieving factors
   - Functional limitations (ADLs, work, sleep)
   - Specific symptoms (numbness, weakness, stiffness)

4. OBJECTIVE FINDINGS:
   - Physical examination details (ROM, strength, sensation)
   - Specific clinical tests and results
   - Gait, posture, functional assessments
   - Waddell signs if present

5. DIAGNOSTIC STUDIES:
   - Imaging results (MRI, CT, X-ray findings with dates)
   - EMG/NCS results, laboratory findings
   - Interpretation and clinical correlation

6. DIAGNOSES:
   - Primary diagnosis and secondary conditions
   - Body parts affected with specificity
   - Causation analysis and apportionment

7. TREATMENT & MEDICATIONS:
   - Current medications with dosages and frequency
   - Previous treatments and responses
   - Recommended future treatments
   - Therapy, injections, surgical recommendations

8. WORK STATUS & RESTRICTIONS:
   - Specific work restrictions with quantitative limits
   - Modified duty recommendations
   - Permanent restrictions if applicable

9. MMI & DISABILITY:
   - MMI status and date
   - Permanent disability rating (if determined)
   - Basis for rating (AMA Guides, etc.)

10. PROGNOSIS & FUTURE CARE:
    - Long-term prognosis
    - Future medical treatment needs
    - Follow-up requirements

EXTRACTION RULES:
- Extract ALL available information, don't summarize prematurely
- Include quantitative measurements and specific findings
- Preserve medical terminology and clinical details
- Capture both positive and negative findings
- Include reasoning and clinical rationale"""),
            ("human", """
QME REPORT TEXT:

{text}

Extract COMPREHENSIVE medical-legal evaluation data. Include EVERY important detail that would be needed for a 400-500 word detailed summary:

{format_instructions}
""")
        ])
    
    @staticmethod
    def get_long_summary_prompt():
        """Prompt to build comprehensive 400-500 word long summary"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are creating a COMPREHENSIVE 400-500 word medical-legal summary of a QME/AME/IME report.

SUMMARY REQUIREMENTS:
- Length: 400-500 words (be precise)
- Structure: Detailed paragraphs with proper medical narrative
- Content: Include ALL significant findings and details
- Depth: Provide clinical context and implications

MANDATORY SECTIONS (develop each fully):

1. EVALUATION CONTEXT & PHYSICIAN DETAILS
   - Full physician credentials and specialty
   - Complete evaluation context and circumstances
   - Date of injury and mechanism in detail

2. CLINICAL PRESENTATION & HISTORY
   - Comprehensive subjective complaints
   - Detailed occupational history and impact
   - Complete treatment chronology

3. OBJECTIVE EXAMINATION FINDINGS
   - Thorough physical examination results
   - Specific measurements and clinical tests
   - Functional limitations and impairments

4. DIAGNOSTIC STUDIES & INTERPRETATION
   - Complete imaging and test results
   - Radiologist interpretations
   - Clinical correlation analysis

5. DIAGNOSIS & CAUSATION ANALYSIS
   - Comprehensive diagnostic formulation
   - Body part-specific findings
   - Apportionment and causation reasoning

6. TREATMENT ANALYSIS & RECOMMENDATIONS
   - Current medication regimen details
   - Previous treatment outcomes
   - Future treatment rationale

7. WORK CAPACITY & RESTRICTIONS
   - Specific functional limitations
   - Quantitative work restrictions
   - Modified duty specifications

8. MMI & DISABILITY ASSESSMENT
   - MMI determination and rationale
   - Permanent impairment assessment
   - Disability rating methodology

9. PROGNOSIS & FUTURE DIRECTIONS
   - Long-term clinical prognosis
   - Future medical needs
   - Follow-up and monitoring plan

WRITING STYLE:
- Professional medical narrative
- Flowing paragraphs (not bullet points)
- Connect findings to clinical implications
- Maintain medical-legal precision"""),
            ("human", """
COMPREHENSIVE EXTRACTED DATA:

{raw_data}

Create a detailed 400-500 word medical-legal summary covering ALL sections above. Ensure comprehensive coverage of all clinical and legal aspects:
""")
        ])
    
    @staticmethod
    def get_short_summary_prompt():
        """Enhanced 60-word short summary prompt"""
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal specialist creating PRECISE 60-word summaries from comprehensive QME reports.

CRITICAL REQUIREMENTS:
- EXACTLY 60 words (count carefully)
- Must be derived from the COMPREHENSIVE long summary
- Cover the most critical medical-legal elements
- Maintain clinical accuracy and legal relevance

ESSENTIAL ELEMENTS TO INCLUDE:
1. Physician identity and evaluation date
2. Primary work-related diagnoses
3. Key objective findings and limitations
4. Current treatment status
5. Work restrictions and MMI status
6. Significant prognosis or next steps

WORD COUNT RULES:
- Count meticulously before finalizing
- Remove less critical details if over 60 words
- Add specific clinical details if under 60 words
- Never exceed 60 words

FORMAT:
- Single, flowing medical narrative
- Complete sentences only
- Professional medical-legal language
- No bullet points or section headers
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPREHENSIVE LONG SUMMARY (400-500 words):

{long_summary}

Based on this detailed medical-legal summary, extract the MOST CRITICAL elements to create a precise 60-word summary that captures the essential medical-legal conclusions and recommendations.

60-WORD MEDICAL-LEGAL SUMMARY:
""")
        
        return ChatPromptTemplate.from_messages([system_prompt, user_prompt])