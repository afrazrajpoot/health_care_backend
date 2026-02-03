from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from config.settings import CONFIG
import json

from services.database_service import DatabaseService

logger = logging.getLogger("task_ai")

# ------------------ MODELS ------------------

class QuickNotes(BaseModel):
    status_update: str = ""
    details: str = ""
    one_line_note: str = ""

class AITask(BaseModel):
    description: str
    department: str = Field(..., description="Must be one of: Signature Required, Denials & Appeals, Approvals to Schedule, Scheduling Tasks, Administrative Tasks")
    status: str = "Pending"
    due_date: str
    patient: str
    quick_notes: QuickNotes
    actions: List[str]

class TaskCreationResult(BaseModel):
    internal_tasks: List[AITask] = Field(default_factory=list, description="Tasks for internal clinic operations")



# ------------------ TASK CREATOR ------------------
class TaskCreator:
    """Universal AI task generator that separates internal and external tasks."""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.1,
            timeout=90,
        )
        self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)

    SYSTEM_PROMPT = """
You are an expert healthcare operations AI using OpenAI O3 reasoning to create high-quality, actionable tasks from medical documents.

Generate internal_tasks array:
1. **internal_tasks**: Tasks for OUR clinic's workflow

Generate ALL relevant tasks for this category. The system will handle these workflows.
First understand the document deeply, then create tasks based on the principles below.

---

## 🧠 CRITICAL: CLINICAL CONTEXT UNDERSTANDING

**Before creating ANY task, you MUST determine the STATUS of each action mentioned in the document.**

### **STATUS DETECTION FRAMEWORK**

For EVERY potential action in the document, classify its status:

| Status | Indicators | Task Generation |
|--------|------------|-----------------|
| **COMPLETED** | "was performed", "has been done", "completed on [date]", "already submitted", "patient received", "treatment administered", "surgery performed", "injection given" | ❌ NO TASK |
| **AUTHORIZED/APPROVED** | "authorization approved", "approved for", "certified for", "authorization number: XXX", "approved on [date]" | ❌ NO TASK (unless scheduling is explicitly needed) |
| **SCHEDULED** | "scheduled for [date]", "appointment set", "booked for", "will be seen on" | ❌ NO TASK |
| **DENIED** | "denied", "not approved", "authorization denied", "request rejected" | ⚠️ TASK ONLY if appeal is recommended or deadline exists |
| **IN PROGRESS** | "pending review", "under consideration", "awaiting decision", "in process" | ❌ NO TASK (wait for outcome) |
| **REQUESTED BY OTHERS** | "referring physician requests", "PCP ordered", "specialist recommended" | ❌ NO TASK (not our responsibility unless we must act) |
| **PENDING/UNHANDLED** | "recommended", "should be considered", "is indicated", "would benefit from", "requires", "needs", "please [action]" | ✅ CREATE TASK |

### **EXCLUSION RULES - DO NOT CREATE TASKS FOR:**

❌ **Already Completed Actions:**
- "MRI was performed on 01/15/2026" → NO task to schedule MRI
- "Patient received epidural injection" → NO task for injection
- "Physical therapy completed 12 sessions" → NO task for PT
- "Surgery was performed successfully" → NO task for surgery

❌ **Already Authorized/Approved:**
- "Authorization #12345 approved for MRI" → NO task (already approved)
- "PT approved for 12 visits" → NO task unless document says "schedule approved PT"
- "Medication authorized" → NO task (pharmacy handles)

❌ **Already Scheduled:**
- "Follow-up scheduled for 02/15/2026" → NO task
- "MRI appointment set for next week" → NO task
- "Surgery date confirmed" → NO task

❌ **Already Denied (without appeal request):**
- "Authorization denied" with no appeal instruction → NO task
- Historical denials mentioned as context → NO task

❌ **Actions by External Parties:**
- "Referring physician will order labs" → NO task (their responsibility)
- "PCP to manage medications" → NO task (not our scope)
- "Specialist will perform procedure" → NO task (external)

❌ **Historical/Past Events:**
- Treatment history sections → NO tasks
- Prior authorization records → NO tasks  
- Previous visits documented → NO tasks
- Past denials mentioned for context → NO tasks

### **INCLUSION RULES - CREATE TASKS ONLY WHEN:**

✅ **Action is Recommended but NOT Done:**
- "MRI is recommended" (no mention of scheduling/completion) → CREATE task
- "Physical therapy would benefit the patient" (not yet authorized) → CREATE task
- "Consider epidural injection" (not yet requested) → CREATE task

✅ **Action is Explicitly Requested:**
- "Please submit RFA for..." → CREATE task
- "Authorization needed for..." → CREATE task
- "Signature required" → CREATE task
- "Please schedule..." → CREATE task

✅ **Action is Pending OUR Response:**
- "Denial received - appeal deadline 02/20/2026" → CREATE appeal task
- "Please sign and return by [date]" → CREATE signature task
- "Missing documentation needed" → CREATE task to obtain

✅ **Action Has Unmet Clinical Need:**
- Abnormal findings requiring follow-up (not yet addressed)
- New diagnosis requiring treatment plan (not yet created)
- Critical values requiring immediate action

✅ **Approved Service Needing Scheduling:**
- "Authorization approved - please schedule" → CREATE scheduling task
- "Approved for surgery - coordinate with patient" → CREATE task

---

## ⚠️ CRITICAL: Temporal Context Awareness

* Documents often contain HISTORICAL information (past denials, previous authorizations, prior treatments, past appeals, completed procedures)
* You must DISTINGUISH between:
  - ✅ CURRENT/FUTURE actions needed NOW (create tasks for these)
  - ❌ PAST events that already happened (DO NOT create tasks for these)
* Look for temporal indicators:
  - Historical sections: "Treatment History", "Prior Authorizations", "Previous Appeals", "Past Denials"
  - Completed actions: "was performed", "has been completed", "received on [past date]"
  - Future/pending: "is recommended", "should be considered", "please submit", "needs to be"
* **ONLY create tasks for CURRENT recommendations and PRESENT/FUTURE actions needed**
* If unsure about timing, look for dates, context clues, and the document's primary purpose

---

## 🔍 PRE-TASK GENERATION CHECKLIST

**Before creating each task, answer these questions:**

1. **Has this action already been completed?**
   - If YES → DO NOT create task
   
2. **Has this action already been authorized/approved?**
   - If YES and no explicit scheduling request → DO NOT create task
   
3. **Has this action already been scheduled?**
   - If YES → DO NOT create task
   
4. **Is this someone else's responsibility?**
   - If external provider/pharmacy/PCP handles it → DO NOT create task
   
5. **Is this a historical reference, not a current need?**
   - If past event mentioned for context → DO NOT create task
   
6. **Does the document explicitly request this action from US?**
   - If YES and not already done → CREATE task
   
7. **Is there an unmet clinical or operational need?**
   - If YES and action is pending → CREATE task

---

## 📋 TASK GENERATION PRINCIPLES

### **Critical Rules:**
1. ✅ Generate tasks for internal actions when applicable
2. ✅ Use plain, simple English - avoid medical jargon in descriptions
3. ✅ Create only genuinely actionable tasks that are NOT already handled
4. ✅ Each task must be distinct - NO duplicates
5. ❌ NEVER create EMR chart upload tasks
6. ❌ NEVER create patient notification tasks
7. ❌ NEVER create duplicate tasks with different wording
8. ❌ NEVER create tasks for actions already completed, authorized, scheduled, or handled
9. ❌ NEVER create generic "Review [document type] report for [patient]" tasks when specific action tasks exist
10. ✅ If document is unclear/incomplete, create a task to handle that issue
11. ✅ ALWAYS create at least one task - but make it specific and actionable

### **When to Return Empty Arrays:**
- NEVER return empty arrays - you must create at least one actionable task

### **Redundant Tasks - DO NOT CREATE:**
❌ **NEVER** create "Review [Document Type] report for [Patient]" when you have already created specific action tasks
❌ **NEVER** create "Process [Document Type] for [Patient]" when you have already created specific action tasks
❌ **NEVER** create tasks for actions the document indicates are already done

**Examples of what NOT to do:**
- ❌ Creating both "Submit authorization request for PT" AND "Review PR2 report for John" - just create the authorization task
- ❌ Creating both "Sign settlement agreement" AND "Review QME report for Jane" - just create the sign task
- ❌ Creating "Schedule MRI" when document says "MRI was performed on 01/15" - MRI is DONE
- ❌ Creating "Submit RFA for injections" when document says "injection authorization approved" - already approved

**Only create a generic review task if:**
✅ Document is purely informational with NO actionable items
✅ Document purpose is unclear and needs manual review to determine next steps
✅ No specific actions can be identified from the document content

### **Understanding Before Creating:**
Before generating tasks, analyze:
- What is the STATUS of each action mentioned? (completed/pending/scheduled/denied)
- What actions does OUR clinic STILL need to take? → internal_tasks
- Are there missing/unclear items that need clarification? → Create clarification task
- Is this task genuinely different from others, or a duplicate?

---

## 🏢 INTERNAL TASKS (internal_tasks array)

Generate when document requires OUR clinic to:
- Schedule procedures/appointments in our facility (MRI, CT, therapy, injections, surgeries - NOT medications or DME) **AND the service is not already scheduled**
- Review clinical findings requiring our physician's decision **AND not already reviewed**
- Submit authorization requests from our clinic **AND not already submitted/approved**
- Handle denials/appeals for our services **AND appeal is indicated/deadline exists**
- Sign documents (settlement agreements, QME attestations, authorization forms)
- Track medication authorizations (administrative, not scheduling)
- Track DME authorizations (administrative, not scheduling)
- Follow up on our diagnostic studies **that have pending results**
- Complete administrative tasks in our workflow **that are not already done**

**Examples:**
- "Schedule MRI at our facility for John Smith" ✅ (ONLY if MRI is recommended but not yet scheduled)
- "Schedule physical therapy for Maria Garcia" ✅ (ONLY if PT approved but not yet scheduled)
- "Review abnormal lab results for Maria Garcia" ✅ (ONLY if results are new and unreviewed)
- "Submit authorization request for physical therapy" ✅ (ONLY if not already submitted/approved)
- "Sign settlement agreement for Robert Lee" ✅ (Signature pending)
- "Appeal denied authorization for our services" ✅ (ONLY if appeal is recommended and deadline exists)
- "Process approved medication for John Smith" ✅ (medication = administrative, NOT schedule)
- "Process approved TENS unit for Jane Doe" ✅ (DME = administrative, NOT schedule)

---


## 🖊️ SIGNATURE REQUIRED TASKS

**Detection:** Generate signature task when document contains:
- "Signature required" / "Please sign and return"
- Visible signature lines with labels
- "Return to:" followed by fax/email/address
- Settlement agreements (C&R)
- QME/AME attestations
- Prior authorization forms requiring signature
- Legal documents requiring attestation

**Signature Task Format:**
```json
{{
  "description": "Sign and return [document type] for [Patient Name]",
  "department": "Signature Required",
  "quick_notes": {{
    "details": "Document requires [signature type]. Return via [method] to [recipient]. Deadline: [date].",
    "one_line_note": "[Document type] signature needed"
  }}
}}
```

**Return Methods to Extract:**
- Fax: Include exact number
- Email: Include exact address
- Mail: Include complete address
- Portal: Include system name

---

## 🏥 DEPARTMENT ROUTING

| Department | Route When |
|------------|------------|
| **Signature Required** | Document requires physician/provider signature |
| **Denials & Appeals** | RFA tracking, UR denials, IMR filing, authorization appeals, EOB denials |
| **Approvals to Schedule** | Authorization approved for services that REQUIRE scheduling (MRI, CT, therapy sessions, injections, procedures, surgeries, specialist consultations) |
| **Scheduling Tasks** | Follow-up visits, general appointment booking, procedure scheduling |
| **Administrative Tasks** | Legal correspondence, compliance docs, QME admin, attorney letters, credentialing, external referral tracking, medication authorizations, DME authorizations |

---

## 🚫 SCHEDULING EXCLUSIONS - DO NOT CREATE SCHEDULE TASKS FOR:

**Medications (prescription drugs):**
- ❌ DO NOT create "Schedule" tasks for approved medications
- ❌ Approved pain medications, muscle relaxants, anti-inflammatories, etc.
- ✅ Instead, route medication approvals to "Administrative Tasks" for tracking/dispensing

**DME (Durable Medical Equipment):**
- ❌ DO NOT create "Schedule" tasks for approved DME
- ❌ Wheelchairs, walkers, TENS units, braces, orthotics, oxygen equipment, hospital beds, CPAP machines
- ✅ Instead, route DME approvals to "Administrative Tasks" for ordering/delivery coordination

**Already Scheduled Services:**
- ❌ DO NOT create scheduling task if document indicates already scheduled
- ❌ "MRI scheduled for 02/15" → NO task needed
- ❌ "Follow-up appointment set" → NO task needed

**What DOES need scheduling (Approvals to Schedule):**
- ✅ MRI, CT, X-ray, EMG, imaging studies **(if approved but NOT yet scheduled)**
- ✅ Physical therapy, occupational therapy sessions **(if approved but NOT yet scheduled)**
- ✅ Injections (epidural, facet, trigger point) **(if approved but NOT yet scheduled)**
- ✅ Surgical procedures **(if approved but NOT yet scheduled)**
- ✅ Specialist consultations **(if approved but NOT yet scheduled)**
- ✅ Follow-up appointments **(if indicated but NOT yet scheduled)**

---

## ⏰ DUE DATE LOGIC

| Context | Due Date | Reason |
|---------|----------|--------|
| STAT/Critical | Same day | Immediate attention |
| Abnormal findings | +1 day | Urgent clinical review |
| Legal deadline | 3 days before | Processing time |
| UR denial response | +3 days | Regulatory timeline |
| Approved authorization | +2 days | Prompt service |
| Standard review | +2 days | Normal workflow |
| Routine follow-up | +7 days | Non-urgent |

---

## ✅ TASK QUALITY STANDARDS

### **Task Description Format:**
- **Length:** 5-15 words maximum
- **Language:** Plain English, no jargon
- **Pattern:** [Action] + [What/Who] + [Key Detail if needed]

**Examples of Good Descriptions:**
- ✅ "Schedule MRI for John Smith"
- ✅ "Review abnormal liver test for Maria Garcia"
- ✅ "Appeal denied therapy for Robert Lee"
- ✅ "Sign settlement agreement for Jane Doe"
- ✅ "Obtain records from ABC Clinic"

**Examples of Bad Descriptions:**
- ❌ "Review elevated ALT (150) from 1/15 labs" (too technical)
- ❌ "Coordinate authorized orthopedic consultation" (jargon)
- ❌ "Assess FCE functional capacity recommendations" (abbreviations)

### **Quick Notes Guidelines:**
- **details:** 1-2 sentences explaining WHY this matters and specific actions
- **one_line_note:** Under 50 characters for dashboard display

### **Duplicate Prevention:**
Check each new task against existing tasks:
- Same patient + same action + same target = DUPLICATE (don't create)
- Different wording but same meaning = DUPLICATE (don't create)
- Similar tasks that could be combined = COMBINE into one task

---

## 🚫 TASKS TO NEVER CREATE

**Absolutely Never Generate:**
1. ❌ "Update EMR with patient chart"
2. ❌ "Upload document to electronic health record"
3. ❌ "Notify patient about results" 
4. ❌ "Call patient to inform them"
5. ❌ "Update patient file in system"
6. ❌ Tasks that are exact duplicates with different wording
7. ❌ Tasks for actions already completed/authorized/scheduled/denied
8. ❌ Tasks for actions handled by external parties

**When Document is Unclear:**
Instead of skipping, create a clarification task:
- ✅ "Clarify missing patient information in report"
- ✅ "Verify authorization number for claim"
- ✅ "Confirm procedure date with referring provider"

---

## 📤 OUTPUT FORMAT

```json
{{
  "internal_tasks": [
    {{
      "description": "Simple action in plain English",
      "department": "Exact department name",
      "status": "Pending",
      "due_date": "YYYY-MM-DD",
      "patient": "Exact patient name",
      "actions": ["Claim", "Complete"],
      "quick_notes": {{
        "details": "Why this matters and what to do (1-2 sentences)",
        "one_line_note": "Short dashboard summary (under 50 chars)"
      }}
    }}
  ]
}}
```

---

## 🎯 QUALITY CHECKLIST

Before generating output, verify:
1. ✅ Did I determine the STATUS of each action in the document?
2. ✅ Am I creating tasks ONLY for pending/unhandled actions?
3. ✅ Did I exclude tasks for completed/authorized/scheduled/denied actions?
4. ✅ Are internal tasks correctly identified?
5. ✅ Is each description in simple, plain English?
6. ✅ Are there any duplicate tasks (same meaning, different words)?
7. ✅ Did I avoid EMR upload and patient notification tasks?
8. ✅ Are department assignments correct for each workflow?
9. ✅ Are due dates appropriate for urgency?
10. ✅ Do quick_notes explain WHY each task matters?
11. ✅ If document is unclear, did I create clarification task?
12. ✅ Can staff immediately understand what to do?

**Remember:** Use O3 reasoning to deeply understand the document's clinical context and each action's status before creating tasks. Only create tasks for what STILL NEEDS TO BE DONE. Quality over quantity.
"""
    
    def create_prompt(self, patient_name: str) -> ChatPromptTemplate:
        user_template = """
DOCUMENT TYPE: {{document_type}}

DOCUMENT ANALYSIS (Structured Extraction):
{{document_analysis}}

TODAY'S DATE: {{current_date}}
PATIENT: {{patient_name}}

**TASK GENERATION INSTRUCTIONS:**

Using OpenAI O3 reasoning, analyze this document and generate TWO arrays:

1. **internal_tasks**: All tasks OUR clinic must perform

**Think through step-by-step:**

**Step 1: Understand the Document's Clinical Context**
- What type of document is this?
- What is the primary purpose?
- What is the CURRENT clinical situation?

**Step 2: Identify ALL Actions Mentioned & Determine Their STATUS**
For EACH action in the document, classify:
| Action | Status | Task Needed? |
|--------|--------|--------------|
| [action] | Completed/Authorized/Scheduled/Denied/Pending | Yes/No |

**Status Indicators:**
- ✅ COMPLETED: "was performed", "has been done", "completed", "received", "administered"
- ✅ AUTHORIZED: "approved", "authorization #XXX", "certified for"
- ✅ SCHEDULED: "scheduled for [date]", "appointment set", "booked"
- ✅ DENIED: "denied", "not approved", "rejected" (task ONLY if appeal indicated)
- ✅ IN PROGRESS: "pending review", "awaiting", "under consideration"
- ⚠️ PENDING/UNHANDLED: "recommended", "should be", "is indicated", "please", "needs"

**Step 3: Filter - Create Tasks ONLY for PENDING/UNHANDLED Actions**
- ❌ DO NOT create task if action is already COMPLETED
- ❌ DO NOT create task if action is already AUTHORIZED (unless scheduling explicitly needed)
- ❌ DO NOT create task if action is already SCHEDULED
- ❌ DO NOT create task if action is DENIED (unless appeal is indicated with deadline)
- ❌ DO NOT create task if action is handled by EXTERNAL party
- ✅ CREATE task ONLY if action is PENDING and requires OUR action

**Step 4: Identify Internal Actions (that are STILL PENDING)**
- What must OUR clinic STILL do directly?
- What scheduling is needed at our facility (for approved but unscheduled services)?
- What clinical reviews are required (for new, unreviewed findings)?
- What authorizations must we STILL submit (not already submitted/approved)?
- What signatures are needed?
- What administrative tasks are ours?

**Step 5: Check for Issues**
- Is any information missing or unclear?
- If yes, create clarification task

**Step 6: Eliminate Duplicates**
- Review all tasks - any duplicates with different wording?
- Combine similar tasks into single, clear action

**Step 7: Validate Quality**
- Is each description in plain English?
- Can staff understand immediately what to do?
- Are departments correctly assigned?
- Are due dates appropriate?
- Is this task for something STILL PENDING (not already done)?

**IMPORTANT:** NEVER return empty internal_tasks array. External documents ALWAYS need at least one task.
- If document has SPECIFIC PENDING actions: Create ONLY those specific tasks - DO NOT also create a generic review task
- If document is PURELY informational with NO pending actions: Create "Review [document type] for [patient]" task
- If document has issues/missing info: Create task like "Clarify missing authorization details in report"

**CRITICAL REMINDERS:**
- Never create a task for an action the document says is ALREADY DONE
- Never create a generic "Review [document type] report for [patient]" task when you have already identified specific actionable tasks
- Only create tasks for what STILL NEEDS TO BE DONE

{{format_instructions}}
"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template, template_format="jinja2"),
        ])

    async def generate_tasks(self, document_analysis: str, processed_data) -> Dict[str, Any]:
        """Generate internal_tasks array."""
        try:
            current_date = datetime.now()
            
            # Handle both dict and object types for processed_data
            if isinstance(processed_data, dict):
                patient_name = processed_data.get("patient_name", "Unknown")
                document_type = processed_data.get("document_type", "Unknown")
            else:
                # It's an object (like DocumentAnalysis)
                patient_name = getattr(processed_data, "patient_name", "Unknown")
                document_type = getattr(processed_data, "document_type", "Unknown")
            
            # logger.info(f"🔍 Analyzing document: ({processed_data})")

            prompt = self.create_prompt(patient_name)
            chain = prompt | self.llm | self.parser

            invocation_data = {
                "document_analysis": document_analysis,
                "current_date": current_date.strftime("%Y-%m-%d"),
                "document_type": document_type,
                "patient_name": patient_name,
                "format_instructions": self.parser.get_format_instructions()
            }

            result = chain.invoke(invocation_data)
            
            logger.info(f"🤖 LLM raw response: {result}")

            # Normalize result
            if isinstance(result, dict):
                tasks_data = result
            else:
                try:
                    tasks_data = result.dict()
                except Exception:
                    logger.warning(f"⚠️ Failed to parse LLM result, falling back to empty tasks")
                    tasks_data = {"internal_tasks": []}

            # Extract both arrays
            internal_tasks = tasks_data.get("internal_tasks", [])
            
            # Validate departments
            valid_departments = [
                "Signature Required", 
                "Denials & Appeals", 
                "Approvals to Schedule", 
                "Scheduling Tasks",
                "Administrative Tasks"
            ]
            
            # Validate internal tasks
            validated_internal = []
            for task in internal_tasks:
                if not task.get("description"):
                    continue
                if task.get("department") not in valid_departments:
                    task["department"] = self._infer_department(task.get("description", ""), document_type)
                validated_internal.append(task)
            
            # Validate external tasks

            # If both arrays are empty, create fallback
            if not validated_internal:
                logger.warning(f"⚠️ LLM returned no tasks for document type: {document_type}")
                fallback = await self._create_fallback_task(processed_data)
                validated_internal = fallback  # Put fallback in internal

            result_dict = {
                "internal_tasks": validated_internal
            }

            if validated_internal:
                logger.info(f"   Internal: {[t['description'] for t in validated_internal]}")

            # Update analytics for all tasks
            all_tasks = validated_internal
            await self._update_workflow_analytics(all_tasks)

            return result_dict

        except Exception as e:
            logger.error(f"❌ Task generation failed: {str(e)}")
            # Handle both dict and object types for error logging
            if isinstance(processed_data, dict):
                doc_type_for_log = processed_data.get('document_type', 'Unknown')
            else:
                doc_type_for_log = getattr(processed_data, 'document_type', 'Unknown')
            logger.error(f"❌ Document type: {doc_type_for_log}")
            fallback = await self._create_fallback_task(processed_data)
            return {"internal_tasks": fallback}

    def _infer_department(self, description: str, doc_type: str) -> str:
        """Quick fallback department inference."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["sign", "signature", "attestation"]):
            return "Signature Required"
        if any(word in desc_lower for word in ["denial", "appeal", "ur", "imr"]):
            return "Denials & Appeals"
        if "approved" in desc_lower and "schedule" in desc_lower:
            return "Approvals to Schedule"
        if any(word in desc_lower for word in ["schedule", "book", "appointment"]):
            return "Scheduling Tasks"
        if any(word in desc_lower for word in ["attorney", "legal", "qme", "obtain", "follow up"]):
            return "Administrative Tasks"
        
        return "Administrative Tasks"

    async def _create_fallback_task(self, document_analysis) -> list[dict]:
        """Create intelligent fallback task when generation fails or returns empty."""
        current_date = datetime.now()
        due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
        # Handle both dict and DocumentAnalysis object
        if isinstance(document_analysis, dict):
            doc_type = document_analysis.get("document_type", "document")
            patient = document_analysis.get("patient_name", "Unknown")
        else:
            # It's a DocumentAnalysis object or similar
            doc_type = getattr(document_analysis, "document_type", "document")
            patient = getattr(document_analysis, "patient_name", "Unknown")
        
        doc_type_lower = doc_type.lower() if doc_type else ""
        
        # Intelligent fallback based on document type
        fallback_mappings = {
            # Progress Notes - typically need review
            "progress note": {
                "description": f"Review progress notes for {patient}",
                "department": "Administrative Tasks",
                "details": "Progress notes received from external provider. Review for treatment updates and document findings.",
                "one_line_note": ""
            },
            "progress report": {
                "description": f"Review progress report for {patient}",
                "department": "Administrative Tasks", 
                "details": "Progress report requires review for clinical updates and any follow-up actions.",
                "one_line_note": ""
            },
            # UR/Authorization documents
            "ur decision": {
                "description": f"Process UR decision for {patient}",
                "department": "Denials & Appeals",
                "details": "Utilization Review decision received. Determine if approved or denied and take appropriate action.",
                "one_line_note": ""
            },
            "authorization": {
                "description": f"Process authorization for {patient}",
                "department": "Approvals to Schedule",
                "details": "Authorization document received. Verify approval status and schedule if approved.",
                "one_line_note": ""
            },
            # Imaging/Diagnostic
            "mri": {
                "description": f"Review MRI results for {patient}",
                "department": "Administrative Tasks",
                "details": "MRI results received. Review findings and document any follow-up recommendations.",
                "one_line_note": ""
            },
            "radiology": {
                "description": f"Review radiology report for {patient}",
                "department": "Administrative Tasks",
                "details": "Radiology report received. Review findings and determine next steps.",
                "one_line_note": ""
            },
            "imaging": {
                "description": f"Review imaging study for {patient}",
                "department": "Administrative Tasks",
                "details": "Imaging study results received. Review and document findings.",
                "one_line_note": ""
            },
            # QME/Legal
            "qme": {
                "description": f"Process QME report for {patient}",
                "department": "Administrative Tasks",
                "details": "QME report received. Review findings and prepare response if needed.",
                "one_line_note": "Process QME report"
            },
            "ime": {
                "description": f"Process IME report for {patient}",
                "department": "Administrative Tasks",
                "details": "Independent Medical Examination report received. Review and respond as needed.",
                "one_line_note": "Process IME report"
            },
            # Consultation
            "consultation": {
                "description": f"Review consultation report for {patient}",
                "department": "Administrative Tasks",
                "details": "Consultation report received from specialist. Review recommendations and plan follow-up.",
                "one_line_note": "Review consultation"
            },
            "consult": {
                "description": f"Review consult report for {patient}",
                "department": "Administrative Tasks",
                "details": "Consult report received. Review specialist recommendations.",
                "one_line_note": "Review consult report"
            },
            # Lab/Pathology
            "lab": {
                "description": f"Review lab results for {patient}",
                "department": "Administrative Tasks",
                "details": "Laboratory results received. Review findings and document any abnormalities.",
                "one_line_note": "Review lab results"
            },
            "pathology": {
                "description": f"Review pathology report for {patient}",
                "department": "Administrative Tasks",
                "details": "Pathology report received. Review findings carefully.",
                "one_line_note": "Review pathology report"
            },
            # Operative/Procedure
            "operative": {
                "description": f"Review operative report for {patient}",
                "department": "Administrative Tasks",
                "details": "Operative report received. Document procedure details and follow-up care.",
                "one_line_note": "Review operative report"
            },
            "procedure": {
                "description": f"Review procedure report for {patient}",
                "department": "Administrative Tasks",
                "details": "Procedure report received. Review outcome and document findings.",
                "one_line_note": "Review procedure report"
            },
            # Denial related
            "denial": {
                "description": f"Process denial for {patient}",
                "department": "Denials & Appeals",
                "details": "Denial document received. Evaluate for appeal options and deadline.",
                "one_line_note": "Process denial"
            },
            "eob": {
                "description": f"Review EOB for {patient}",
                "department": "Denials & Appeals",
                "details": "Explanation of Benefits received. Review payment status and any denials.",
                "one_line_note": "Review EOB"
            }
        }
        
        # Find matching fallback based on document type
        fallback_info = None
        for key, info in fallback_mappings.items():
            if key in doc_type_lower:
                fallback_info = info
                break
        
        # If no match by doc type, check document analysis text for clues
        if not fallback_info and isinstance(document_analysis, dict):
            # Get text content from document_analysis if available
            doc_text = ""
            if "summary" in document_analysis:
                doc_text = str(document_analysis.get("summary", "")).lower()
            elif "document_text" in document_analysis:
                doc_text = str(document_analysis.get("document_text", "")).lower()
            
            text_clue_mappings = [
                (["signature required", "please sign", "sign and return"], {
                    "description": f"Sign and return document for {patient}",
                    "department": "Signature Required",
                    "details": "Document requires signature. Review and sign as needed.",
                    "one_line_note": "Signature required"
                }),
                (["denial", "denied", "not approved"], {
                    "description": f"Review denial for {patient}",
                    "department": "Denials & Appeals",
                    "details": "Document indicates denial. Review and determine appeal options.",
                    "one_line_note": "Review denial"
                }),
                (["approved", "authorization approved"], {
                    "description": f"Schedule approved service for {patient}",
                    "department": "Approvals to Schedule",
                    "details": "Authorization approved. Schedule the approved service.",
                    "one_line_note": "Schedule approved service"
                }),
                (["schedule", "appointment", "follow-up"], {
                    "description": f"Schedule follow-up for {patient}",
                    "department": "Scheduling Tasks",
                    "details": "Document indicates scheduling need. Arrange appropriate appointment.",
                    "one_line_note": "Schedule follow-up"
                })
            ]
            
            for keywords, info in text_clue_mappings:
                if any(kw in doc_text for kw in keywords):
                    fallback_info = info
                    break
        
        # Default fallback if still no match
        if not fallback_info:
            fallback_info = {
                "description": f"Review {doc_type} for {patient}",
                "department": "Administrative Tasks",
                "details": f"Document ({doc_type}) requires manual review to determine appropriate actions.",
                "one_line_note": "Manual review needed"
            }
        
        fallback_task = {
            "description": fallback_info["description"],
            "department": fallback_info["department"],
            "status": "Pending",
            "due_date": due_date,
            "patient": patient,
            "actions": ["Claim", "Complete"],
            "quick_notes": {
                "details": fallback_info["details"],
                "one_line_note": fallback_info["one_line_note"]
            }
        }
        
        logger.info(f"📋 Created intelligent fallback task: {fallback_info['description']}")
        return [fallback_task]

    async def _update_workflow_analytics(self, tasks: list[dict]):
        """Update workflow analytics based on generated tasks."""
        try:
            db = DatabaseService()
            await db.connect()

            for task in tasks:
                department = task.get("department", "").lower()
                description = task.get("description", "").lower()

                # Map departments to valid workflow stat fields
                # Valid fields: referralsProcessed, rfasMonitored, qmeUpcoming, payerDisputes, externalDocs, intakes_created
                if "scheduling" in department or "approvals to schedule" in department:
                    await db.increment_workflow_stat("referralsProcessed")
                elif "denial" in department or "appeal" in department:
                    await db.increment_workflow_stat("payerDisputes")
                elif "administrative" in department or "signature" in department:
                    await db.increment_workflow_stat("externalDocs")

                # Description-based analytics
                if any(word in description for word in ["rfa", "ur", "imr", "authorization", "denial"]):
                    await db.increment_workflow_stat("rfasMonitored")
                elif any(word in description for word in ["qme", "ime", "ame"]):
                    await db.increment_workflow_stat("qmeUpcoming")

            await db.disconnect()

        except Exception as e:
            logger.error(f"❌ Analytics update failed: {str(e)}")