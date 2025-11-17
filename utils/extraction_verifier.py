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
                "pattern": r"QME",
                "max_words": 400,
                "required_elements": ["date", "QME"],
                "format_template": """
                QME Medical-Legal Summary (Detailed, up to 400 words):

                1. **Patient & Case Info:**
                    - Patient name, age, date of injury, claim number, employer
                    - QME physician (name, specialty, credentials)
                    - Report date, evaluation date, document type

                2. **Diagnosis & Body Parts:**
                    - Primary and secondary diagnoses (with ICD-10 if available)
                    - All affected body parts
                    - Historical conditions relevant to the case

                3. **Clinical Status:**
                    - Chief complaint, pain scores, functional limitations
                    - Past surgeries, objective findings (ROM, gait, positive tests)

                4. **Medications:**
                    - Current medications (name, dose, purpose)
                    - Future/recommended medications

                5. **Medical-Legal Conclusions:**
                    - MMI/P&S status (with date/reason)
                    - WPI impairment (percentages, method)
                    - Apportionment (industrial/non-industrial, reasoning)

                6. **Treatment & Recommendations:**
                    - Interventional procedures, diagnostic tests, therapy, specialist referrals, future surgical needs

                7. **Work Status & Restrictions:**
                    - Current work status, restrictions (exact wording), prognosis for return to work

                8. **Critical Findings & Action Items:**
                    - Main actionable points (MMI, procedures, QMEs, diagnostics)

                9. **Summary Paragraph:**
                    - Narrative summary of the case, key findings, and recommendations

                Format as a multi-section narrative or bullet points. Include all available details. Do NOT add placeholders for missing data."
        """},
        "AME": {
            "pattern": r"AME",
            "max_words": 350,
            "required_elements": ["date", "AME"],
            # "format_template": "AME{doctor_section} for [Body parts] : [DATE] = [MMI/Impairment] → [Treatment/Future medical] | [Restrictions/Causation]"
            "format_template": """
            AME Medical-Legal Summary (Detailed, up to 350 words):

            1. **Patient & Case Info:**
                - Patient name, age, date of injury, claim number, employer
                - AME physician (name, specialty, credentials)
                - Report date, evaluation date, document type

            2. **Diagnosis & Body Parts:**
                - Primary and secondary diagnoses (with ICD-10 if available)
                - All affected body parts
                - Historical conditions relevant to the case

            3. **Clinical Status:**
                - Chief complaint, pain scores, functional limitations
                - Past surgeries, objective findings (ROM, gait, positive tests)

            4. **Medications:**
                - Current medications (name, dose, purpose)
                - Future/recommended medications

            5. **Medical-Legal Conclusions:**
                - MMI/P&S status (with date/reason)
                - WPI impairment (percentages, method)
                - Apportionment (industrial/non-industrial, reasoning)

            6. **Treatment & Recommendations:**
                - Interventional procedures, diagnostic tests, therapy, specialist referrals, future surgical needs

            7. **Work Status & Restrictions:**
                - Current work status, restrictions (exact wording), prognosis for return to work

            8. **Critical Findings & Action Items:**
                - Main actionable points (MMI, procedures, QMEs, diagnostics)

            9. **Summary Paragraph:**
                - Narrative summary of the case, key findings, and recommendations

            Format as a multi-section narrative or bullet points. Include all available details. Do NOT add placeholders for missing data."
            """
        },
        "IME": {
            "pattern": r"IME",
            "max_words": 350,
            "required_elements": ["date", "IME"],
            "format_template": "IME{doctor_section} for [Body parts] : [DATE] = [MMI/Impairment] → [Treatment/Future medical] | [Restrictions/Causation]"
        },
        "MRI": {
            "pattern": r"MRI",
            "max_words": 400,
            "required_elements": ["date", "MRI", "body_part"],
          "format_template": """
          To create an accurate, concise, and actionable summary from any imaging report (MRI, X-ray, CT, etc.), the focus should be strictly on the identifying information, the clinical question, and the diagnostic conclusion.

Here are the six key fields that should be extracted, ensuring all critical details and context are captured:

| Key Field                               | Focus Area                        | Details to Extract                                                                                                                                                               | Why It Is Critical to Extract                                                                                          |
| :-------------------------------------: | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **1. Header & Context**                 | **Report Identity & Date**        | **Imaging Center, Date of Exam, Type of Exam** (e.g., MRI L-Spine w/o contrast), **Patient Name, DOB, Referring Physician.**                                                     | Establishes the authenticity and timeliness of the results and who ordered the study.                                  |
| **2. Clinical Data/Indication**         | **Reason for the Study**          | **The specific symptom or injury** that prompted the imaging (e.g., "Left knee pain, history of work injury," or "Rule out fracture").                                           | Provides the necessary context for the findings and links the imaging to the claim/injury.                             |
| **3. Technique/Prior Studies**          | **Methodology**                   | **Use of contrast/dye** (With or Without Dye), **Body part imaged**, and whether **Prior Studies** were available for comparison.                                                | The use of contrast affects the interpretation; knowing about prior studies helps gauge progression/regression.        |
| **4. Key Findings (Positive/Negative)** | **Evidence of Pathology**         | **Specific structural abnormalities related to the complaint:** (e.g., Disc herniation, fracture, nerve impingement, meniscal tear, fluid).                                      | This is the objective data that supports the physician's diagnosis and treatment plan. **Focus on positive findings.** |
| **5. Impression/Conclusion**            | **Radiologist's Final Diagnosis** | **The official diagnostic statement** (e.g., "Features of dorsal DRUJ dislocation/subluxation," "Acute L5-S1 disc herniation," "Severe tri-compartmental degenerative changes"). | This is the final, definitive conclusion; it is the most critical piece of information for the treating physician.     |
| **6. Recommendations/Follow-up**        | **Actionable Next Steps**         | **Any specific follow-up suggested by the Radiologist** (e.g., "Recommend clinical correlation," "Follow-up as clinically indicated," "Suggest further evaluation with CT").     | Guides the referring physician on the next steps in care.                                                              |

### Summary Strategy

When creating the concise summary, you should prioritize the following flow:

1.  **Who, What, When:** *(Field 1)* Dr. X performed an **(Field 1) MRI of the \[Area\]** on **(Field 1) \[Date\]**.
2.  **Why:** The exam was performed due to **(Field 2) \[Clinical Indication\]**.
3.  **The Diagnosis:** The **(Field 5) Impression** is **\[Final Diagnosis\]**.
4.  **Key Objective Evidence:** The **(Field 4) key findings** supporting this diagnosis included **\[Specific positive findings\]**.
"""
        },
        "CT": {
            "pattern": r"CT",
            "max_words": 400,
            "required_elements": ["date", "CT", "body_part"],
      "format_template": """
      To create an accurate, concise, and actionable summary from any imaging report (MRI, X-ray, CT, etc.), the focus should be strictly on the identifying information, the clinical question, and the diagnostic conclusion.

Here are the six key fields that should be extracted, ensuring all critical details and context are captured:

| Key Field                               | Focus Area                        | Details to Extract                                                                                                                                                               | Why It Is Critical to Extract                                                                                          |
| :-------------------------------------: | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **1. Header & Context**                 | **Report Identity & Date**        | **Imaging Center, Date of Exam, Type of Exam** (e.g., MRI L-Spine w/o contrast), **Patient Name, DOB, Referring Physician.**                                                     | Establishes the authenticity and timeliness of the results and who ordered the study.                                  |
| **2. Clinical Data/Indication**         | **Reason for the Study**          | **The specific symptom or injury** that prompted the imaging (e.g., "Left knee pain, history of work injury," or "Rule out fracture").                                           | Provides the necessary context for the findings and links the imaging to the claim/injury.                             |
| **3. Technique/Prior Studies**          | **Methodology**                   | **Use of contrast/dye** (With or Without Dye), **Body part imaged**, and whether **Prior Studies** were available for comparison.                                                | The use of contrast affects the interpretation; knowing about prior studies helps gauge progression/regression.        |
| **4. Key Findings (Positive/Negative)** | **Evidence of Pathology**         | **Specific structural abnormalities related to the complaint:** (e.g., Disc herniation, fracture, nerve impingement, meniscal tear, fluid).                                      | This is the objective data that supports the physician's diagnosis and treatment plan. **Focus on positive findings.** |
| **5. Impression/Conclusion**            | **Radiologist's Final Diagnosis** | **The official diagnostic statement** (e.g., "Features of dorsal DRUJ dislocation/subluxation," "Acute L5-S1 disc herniation," "Severe tri-compartmental degenerative changes"). | This is the final, definitive conclusion; it is the most critical piece of information for the treating physician.     |
| **6. Recommendations/Follow-up**        | **Actionable Next Steps**         | **Any specific follow-up suggested by the Radiologist** (e.g., "Recommend clinical correlation," "Follow-up as clinically indicated," "Suggest further evaluation with CT").     | Guides the referring physician on the next steps in care.                                                              |

### Summary Strategy

When creating the concise summary, you should prioritize the following flow:

1.  **Who, What, When:** *(Field 1)* Dr. X performed an **(Field 1) MRI of the \[Area\]** on **(Field 1) \[Date\]**.
2.  **Why:** The exam was performed due to **(Field 2) \[Clinical Indication\]**.
3.  **The Diagnosis:** The **(Field 5) Impression** is **\[Final Diagnosis\]**.
4.  **Key Objective Evidence:** The **(Field 4) key findings** supporting this diagnosis included **\[Specific positive findings\]**.
"""
        },
        "X-ray": {
            "pattern": r"X-ray",
            "max_words": 400,
            "required_elements": ["date", "X-ray", "body_part"],
            "format_template": """
            To create an accurate, concise, and actionable summary from any imaging report (MRI, X-ray, CT, etc.), the focus should be strictly on the identifying information, the clinical question, and the diagnostic conclusion.

Here are the six key fields that should be extracted, ensuring all critical details and context are captured:

| Key Field                               | Focus Area                        | Details to Extract                                                                                                                                                               | Why It Is Critical to Extract                                                                                          |
| :-------------------------------------: | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **1. Header & Context**                 | **Report Identity & Date**        | **Imaging Center, Date of Exam, Type of Exam** (e.g., MRI L-Spine w/o contrast), **Patient Name, DOB, Referring Physician.**                                                     | Establishes the authenticity and timeliness of the results and who ordered the study.                                  |
| **2. Clinical Data/Indication**         | **Reason for the Study**          | **The specific symptom or injury** that prompted the imaging (e.g., "Left knee pain, history of work injury," or "Rule out fracture").                                           | Provides the necessary context for the findings and links the imaging to the claim/injury.                             |
| **3. Technique/Prior Studies**          | **Methodology**                   | **Use of contrast/dye** (With or Without Dye), **Body part imaged**, and whether **Prior Studies** were available for comparison.                                                | The use of contrast affects the interpretation; knowing about prior studies helps gauge progression/regression.        |
| **4. Key Findings (Positive/Negative)** | **Evidence of Pathology**         | **Specific structural abnormalities related to the complaint:** (e.g., Disc herniation, fracture, nerve impingement, meniscal tear, fluid).                                      | This is the objective data that supports the physician's diagnosis and treatment plan. **Focus on positive findings.** |
| **5. Impression/Conclusion**            | **Radiologist's Final Diagnosis** | **The official diagnostic statement** (e.g., "Features of dorsal DRUJ dislocation/subluxation," "Acute L5-S1 disc herniation," "Severe tri-compartmental degenerative changes"). | This is the final, definitive conclusion; it is the most critical piece of information for the treating physician.     |
| **6. Recommendations/Follow-up**        | **Actionable Next Steps**         | **Any specific follow-up suggested by the Radiologist** (e.g., "Recommend clinical correlation," "Follow-up as clinically indicated," "Suggest further evaluation with CT").     | Guides the referring physician on the next steps in care.                                                              |

### Summary Strategy

When creating the concise summary, you should prioritize the following flow:

1.  **Who, What, When:** *(Field 1)* Dr. X performed an **(Field 1) MRI of the \[Area\]** on **(Field 1) \[Date\]**.
2.  **Why:** The exam was performed due to **(Field 2) \[Clinical Indication\]**.
3.  **The Diagnosis:** The **(Field 5) Impression** is **\[Final Diagnosis\]**.
4.  **Key Objective Evidence:** The **(Field 4) key findings** supporting this diagnosis included **\[Specific positive findings\]**.
"""
        },
        "PR-2": {
            "pattern": r"PR-2",
            "max_words": 400,
            "required_elements": ["date", "PR-2"],
            "format_template": """
             Generally, when analyzing a Physician's Progress Report (PR-2) for **any patient**, the focus should be on the following four core areas to ensure all actionable and critical details are captured:

### 1\. Work Status and Impairment

The primary goal of a PR-2 in workers' compensation is to define the patient's capacity to work.

  * **Key Focus:**
      * **Work Status:** Is the patient Temporarily Total Disabled (TTD), Temporarily Partially Disabled (TPD), Permanent and Stationary (P\&S)/Maximum Medical Improvement (MMI), or Released to Work (Full/Modified)?
      * **New/Changed Limitations:** What are the specific work restrictions (e.g., lifting, standing, repetitive tasks)?
      * **Rationale:** Does the report clearly explain *why* the work status or limitations have changed or remained the same (e.g., patient pain level, objective exam findings)?

### 2\. Treatment Authorization Requests

This is the most time-sensitive and actionable section for the claims administrator.

  * **Key Focus:**
      * **Specific Request:** What exact treatment or testing is being requested (e.g., 6 weeks of PT, MRI, surgery, medication refill)?
      * **Medical Necessity/Rationale:** Does the physician justify the request? Is the patient showing progress, or is the treatment needed to reach MMI? *Look for objective evidence of benefit from past treatment.*
      * **Prior Authorization Status:** Was this treatment previously authorized or denied? (This requires cross-referencing with past records, but the PR-2 should lay the foundation).

### 3\. Patient Progress and Current Status

This provides the medical context for the claims decision and determines if the treatment plan is effective.

  * **Key Focus:**
      * **Subjective Improvement:** Has the patient's pain score (e.g., VAS) or subjective symptoms improved, worsened, or remained static since the last visit?
      * **Objective Findings:** What are the current objective findings (e.g., range of motion, strength, physical exam results)? Compare these to the previous report to gauge progress.
      * **Medication Changes:** Were any new medications prescribed, discontinued, or changed?

### 4\. Next Steps and Planning

This dictates the future management of the claim and medical necessity timeline.

  * **Key Focus:**
      * **Follow-up Date:** When is the next scheduled appointment? This helps set the date for the next expected PR-2.
      * **Specialist Referrals:** Is a referral to a specialist (e.g., Ortho, Neuro, Pain Management) being recommended or requested?
      * **MMI Status/Timing:** If the patient is approaching or at MMI, is a Permanent and Stationary (P\&S) report (PR-3) being prepared?

"""
        },
        "Consult": {
            "pattern": r"Consult",
            "max_words": 400,
            "required_elements": ["date", "Consult"],
            "format_template": """
To create an accurate and concise summary of a general consultation report, the focus should be on the areas that define the patient's current medical status, the diagnostic findings, and the resulting change in the treatment plan.

The following eight key fields and their contents should be extracted and highlighted:

| Key Field                                  | Focus Area                                                                                                                                                                                                                                        | Why It Is Critical to Extract                                                 |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------- |
| **1. Header & Context**                    | **Report Identity**                                                                                                                                                                                                                               | Establishes the authority and relevance of the document.                      |
|                                            | **Details to Extract:** Consulting Physician/Specialty, Date of Service, Referring Physician, Patient Name, Date of Injury (DOI), and Claim Number.                                                                                               |                                                                               |
| **2. Chief Complaint (CC)**                | **Patient's Primary Issue**                                                                                                                                                                                                                       | Defines the core medical problem leading to the consultation.                 |
|                                            | **Details to Extract:** The patient's exact stated pain/problem, location(s), and duration.                                                                                                                                                       |                                                                               |
| **3. Diagnosis & Assessment**              | **Medical Conclusion**                                                                                                                                                                                                                            | The ultimate reason for the consult and the foundation for treatment.         |
|                                            | **Details to Extract:** All official ICD-10 diagnoses (e.g., Radiculopathy, Degenerative Disc Disease, Myofascial Pain Syndrome). Include any statement on **Causation** (e.g., "Industrial causal relationship," "non-industrial contributors"). |                                                                               |
| **4. History of Present Illness (HPI)**    | **Symptoms & Severity**                                                                                                                                                                                                                           | Provides the context for the diagnosis and justification for intervention.    |
|                                            | **Details to Extract:** Description of the pain (quality, location, radiation), aggravating/alleviating factors, and related functional deficits (e.g., poor sleep, limited driving).                                                             |                                                                               |
| **5. Prior Treatment & Efficacy**          | **Failure of Conservative Care**                                                                                                                                                                                                                  | Justifies the need for specialized intervention (e.g., injections, surgery).  |
|                                            | **Details to Extract:** Specific treatments received (e.g., PT, Chiro, Medications, Injections) and the **reported level of relief** (e.g., "short-term relief," "no meaningful change," "failure of conservative management").                   |                                                                               |
| **6. Objective Findings (Exam & Imaging)** | **Verifiable Evidence**                                                                                                                                                                                                                           | Connects the subjective complaint to objective proof and the final diagnosis. |
|                                            | **Details to Extract:**                                                                                                                                                                                                                           |                                                                               |
|                                            | **Physical Exam:** Key positive findings (e.g., Spasm, Reduced ROM, Positive Straight Leg Raise, Sensory deficits).                                                                                                                               |                                                                               |
|                                            | **Imaging Review (MRI/X-ray):** Specific structural findings that correlate with the symptoms (e.g., "C5-6 disc protrusion," "L4-5 disc bulge contacting nerve root").                                                                            |                                                                               |
| **7. Plan (Recommended Treatments)**       | **Actionable Medical Directives**                                                                                                                                                                                                                 | The most critical part for claims administration and utilization review.      |
|                                            | **Details to Extract:** **Specific Interventions Requested** (e.g., RFA, specific Epidural Steroid Injections, Surgical Consult). Note any suggested changes to **Medication** (e.g., transition from Gabapentin to Pregabalin).                  |                                                                               |
| **8. Work Status & Impairment**            | **Legal/Administrative Status**                                                                                                                                                                                                                   | Defines the patient's current capacity for work.                              |
|                                            | **Details to Extract:** Current **Work Status** (e.g., TTD, Modified Duty, MMI) and **Specific Restrictions** (e.g., "No lifting greater than X lbs," "Avoid repetitive bending," "Allow postural changes").                                      |                                                                               |

## General Extraction Strategy:

The strategy should be to move linearly through the report, asking "Is this information directly related to the current *diagnosis*, the justification for *new treatment*, or the patient's *work status*?"

  * **Focus Most Heavily On:** The **Assessment/Diagnosis** (Field 3), the **Plan/RFA Requests** (Field 7), and **Work Status** (Field 8). These are the actionable, decision-making components.
  * **Use Other Fields To:** Provide the essential context and medical necessity for the decisions made in Fields 3, 7, and 8. (For example, use *Prior Treatment Failure* to justify the *New RFA Request*).

"""
        },
        "RFA": {
            "pattern": r"RFA",
            "max_words": 60,
            "required_elements": ["RFA", "date", "service", "body_part"],
            "format_template": "[DATE]: RFA{doctor_section} | Service → [Service] | Body part → [Body part]"
        },
        "UR": {
            "pattern": r"UR",
            "max_words": 60,
            "required_elements": ["UR", "date", "service", "reason"],
            "format_template": "[DATE]: UR Decision{doctor_section} | Service denied → [Service] | Reason → [Reason]"
        },
        "Authorization": {
            "pattern": r"Authorization",
            "max_words": 60,
            "required_elements": ["Authorization", "date", "service", "body_part"],
            "format_template": "[DATE]: Authorization{doctor_section} | Service approved → [Service] | Body part → [Body part]"
        },
        "DFR": {
            "pattern": r"DFR",
            "max_words": 60,
            "required_elements": ["DFR", "date", "diagnosis"],
            "format_template": "[DATE]: DFR{doctor_section} | DOI → [DOI] | Diagnosis → [Diagnosis] | Plan → [Plan]"
        },
        "PR-4": {
            "pattern": r"PR-4",
            "max_words": 60,
            "required_elements": ["PR-4", "date"],
            "format_template": "[DATE]: PR-4{doctor_section} | MMI Status → [MMI Status] | Future care → [Future care]"
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
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. Fix the summary to match the exact required format.

DOCUMENT TYPE: {doc_type}

CURRENT SUMMARY (with issues):
{current_summary}

DETECTED ISSUES:
{issues}

REQUIRED FORMAT:
{format_template}

CRITICAL RULES:
1. Maintain all factual information from current summary
2. Fix format to match required template exactly
3. Ensure date is in MM/DD/YY format
4. Enforce {max_words}-word maximum
5. Use standard abbreviations (R/L, PT, ESI, f/u, etc.)
6. Do NOT add information not in current summary
7. Do NOT remove key medical facts
8. **MOST IMPORTANT: ONLY include fields that have actual data. Do NOT add placeholders like "not provided", "not specified", "not mentioned" for missing fields. Simply omit those fields entirely.**
9. If a field in the template has no corresponding data, skip it completely - do not include the field label or any placeholder text.

RAW EXTRACTED DATA (for reference):
{raw_data}

Return JSON:
{{
  "corrected_summary": "Fixed summary with ONLY fields that have actual data (no 'not provided' placeholders)",
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
