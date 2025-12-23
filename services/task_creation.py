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
    department: str = Field(..., description="Must be one of: Signature Required, Denials & Appeals, Approvals to Schedule, Scheduling Tasks, Administrative Tasks, Provider Action Required")
    status: str = "Pending"
    due_date: str
    patient: str
    quick_notes: QuickNotes
    actions: List[str]
    source_document: str = ""

class TaskCreationResult(BaseModel):
    internal_tasks: List[AITask] = Field(default_factory=list, description="Tasks for internal clinic operations")
    external_tasks: List[AITask] = Field(default_factory=list, description="Tasks related to external coordination")

# ------------------ TASK CREATOR ------------------
class TaskCreator:
    """Universal AI task generator that separates internal and external tasks."""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_o3_model"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.1,
            timeout=90,
        )
        self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)

    SYSTEM_PROMPT = """
You are an expert healthcare operations AI using OpenAI O3 reasoning to create high-quality, actionable tasks from medical documents.

## 🎯 CORE MISSION
Generate TWO separate task arrays:
1. **internal_tasks**: Tasks for OUR clinic's workflow
2. **external_tasks**: Tasks for external coordination/communication

Generate ALL relevant tasks for BOTH categories - don't choose between them. The system will handle both types of workflows.

---

## 📋 TASK GENERATION PRINCIPLES

### **Critical Rules:**
1. ✅ Generate tasks for BOTH internal and external actions when applicable
2. ✅ Use plain, simple English - avoid medical jargon in descriptions
3. ✅ Create only genuinely actionable tasks
4. ✅ Each task must be distinct - NO duplicates
5. ❌ NEVER create EMR chart upload tasks
6. ❌ NEVER create patient notification tasks
7. ❌ NEVER create duplicate tasks with different wording
8. ✅ If document is unclear/incomplete, create a task to handle that issue

### **When to Return Empty Arrays:**
- Document contains no actionable items
- Document is purely informational with no follow-up needed
- All actions are already completed and documented

### **Understanding Before Creating:**
Before generating tasks, analyze:
- What actions does OUR clinic need to take? → internal_tasks
- What external coordination is needed? → external_tasks
- Are there missing/unclear items that need clarification? → Create clarification task
- Is this task genuinely different from others, or a duplicate?

---

## 🏢 INTERNAL TASKS (internal_tasks array)

Generate when document requires OUR clinic to:
- Schedule procedures/appointments in our facility
- Review clinical findings requiring our physician's decision
- Submit authorization requests from our clinic
- Handle denials/appeals for our services
- Sign documents (settlement agreements, QME attestations, authorization forms)
- Manage medications/treatment plans for our patients
- Follow up on our diagnostic studies
- Complete administrative tasks in our workflow

**Examples:**
- "Schedule MRI at our facility for John Smith"
- "Review abnormal lab results for Maria Garcia"
- "Submit authorization request for physical therapy"
- "Sign settlement agreement for Robert Lee"
- "Appeal denied authorization for our services"

---

## 🌐 EXTERNAL TASKS (external_tasks array)

Generate when coordination with external entities is needed:
- Obtain records from external providers
- Confirm appointments at external facilities
- Coordinate care transitions with outside specialists
- Follow up on external referrals we made
- Communicate with external adjusters/attorneys
- Track external authorization status
- Verify external procedures were completed

**Examples:**
- "Obtain records from ABC Orthopedics for John Smith"
- "Confirm MRI appointment at XYZ Imaging for Maria Garcia"
- "Follow up with external cardiologist referral for Robert Lee"
- "Verify physical therapy completion at external facility"

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
{
  "description": "Sign and return [document type] for [Patient Name]",
  "department": "Signature Required",
  "quick_notes": {
    "details": "Document requires [signature type]. Return via [method] to [recipient]. Deadline: [date].",
    "one_line_note": "[Document type] signature needed"
  }
}
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
| **Approvals to Schedule** | Authorization approved and needs appointment scheduling |
| **Scheduling Tasks** | Follow-up visits, general appointment booking, procedure scheduling |
| **Administrative Tasks** | Legal correspondence, compliance docs, QME admin, attorney letters, credentialing, external referral tracking |
| **Provider Action Required** | Clinical findings needing physician review, treatment decisions, abnormal results, medication management |

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

**When Document is Unclear:**
Instead of skipping, create a clarification task:
- ✅ "Clarify missing patient information in report"
- ✅ "Verify authorization number for claim"
- ✅ "Confirm procedure date with referring provider"

---

## 📤 OUTPUT FORMAT

```json
{
  "internal_tasks": [
    {
      "description": "Simple action in plain English",
      "department": "Exact department name",
      "status": "Pending",
      "due_date": "YYYY-MM-DD",
      "patient": "Exact patient name",
      "actions": ["Claim", "Complete"],
      "source_document": "{{source_document}}",
      "quick_notes": {
        "details": "Why this matters and what to do (1-2 sentences)",
        "one_line_note": "Short dashboard summary (under 50 chars)"
      }
    }
  ],
  "external_tasks": [
    {
      "description": "External coordination action",
      "department": "Administrative Tasks",
      "status": "Pending",
      "due_date": "YYYY-MM-DD",
      "patient": "Exact patient name",
      "actions": ["Claim", "Complete"],
      "source_document": "{{source_document}}",
      "quick_notes": {
        "details": "External entity and coordination details",
        "one_line_note": "External action summary"
      }
    }
  ]
}
```

---

## 🎯 QUALITY CHECKLIST

Before generating output, verify:
1. ✅ Did I identify ALL distinct actions in the document?
2. ✅ Are internal vs external tasks correctly categorized?
3. ✅ Is each description in simple, plain English?
4. ✅ Are there any duplicate tasks (same meaning, different words)?
5. ✅ Did I avoid EMR upload and patient notification tasks?
6. ✅ Are department assignments correct for each workflow?
7. ✅ Are due dates appropriate for urgency?
8. ✅ Do quick_notes explain WHY each task matters?
9. ✅ If document is unclear, did I create clarification task?
10. ✅ Can staff immediately understand what to do?

**Remember:** Use O3 reasoning to deeply understand the document before creating tasks. Quality over quantity.
"""
    
    def create_prompt(self, patient_name: str, source_document: str, matched_doctor_name: str) -> ChatPromptTemplate:
        user_template = """
DOCUMENT TYPE: {{document_type}}

FULL DOCUMENT TEXT:
{{full_text}}

DOCUMENT ANALYSIS (Structured Extraction):
{{document_analysis}}

SOURCE DOCUMENT: {{source_document}}
TODAY'S DATE: {{current_date}}
PATIENT: {{patient_name}}
MATCHED DOCTOR: {{matched_doctor_name}}

**TASK GENERATION INSTRUCTIONS:**

Using OpenAI O3 reasoning, analyze this document and generate TWO arrays:

1. **internal_tasks**: All tasks OUR clinic must perform
2. **external_tasks**: All external coordination tasks

**Think through step-by-step:**

**Step 1: Understand the Document**
- What type of document is this?
- What is the primary purpose?
- What actions are explicitly requested?
- What actions are implicitly needed?

**Step 2: Identify Internal Actions**
- What must OUR clinic do directly?
- What scheduling is needed at our facility?
- What clinical reviews are required?
- What authorizations must we submit?
- What signatures are needed?
- What administrative tasks are ours?

**Step 3: Identify External Actions**
- What coordination with outside entities is needed?
- What records must we obtain?
- What external appointments need verification?
- What communication with adjusters/attorneys is required?
- What external referrals need follow-up?

**Step 4: Check for Issues**
- Is any information missing or unclear?
- If yes, create clarification task

**Step 5: Eliminate Duplicates**
- Review all tasks - any duplicates with different wording?
- Combine similar tasks into single, clear action

**Step 6: Validate Quality**
- Is each description in plain English?
- Can staff understand immediately what to do?
- Are departments correctly assigned?
- Are due dates appropriate?

**If no actionable tasks exist:** Return empty arrays for both internal_tasks and external_tasks.

**If document has issues:** Create task like "Clarify missing authorization details in report"

{{format_instructions}}
"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template, template_format="jinja2"),
        ])

    async def generate_tasks(self, document_analysis: dict, source_document: str = "", full_text: str = "", matched_doctor_name: str = "") -> dict:
        """Generate dual-array task output: internal_tasks and external_tasks."""
        try:
            current_date = datetime.now()
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            
            logger.info(f"🔍 Analyzing document: {source_document} ({document_type})")
            logger.info(f"📝 Full text length: {len(full_text) if full_text else 0} characters")
            logger.info(f"👨‍⚕️ Matched doctor: {matched_doctor_name if matched_doctor_name else 'Not specified'}")

            prompt = self.create_prompt(patient_name, source_document, matched_doctor_name)
            chain = prompt | self.llm | self.parser

            invocation_data = {
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date.strftime("%Y-%m-%d"),
                "source_document": source_document or "Unknown",
                "document_type": document_type,
                "patient_name": patient_name,
                "matched_doctor_name": matched_doctor_name or "Not specified",
                "full_text": full_text if full_text else "Not available",
                "format_instructions": self.parser.get_format_instructions()
            }

            result = chain.invoke(invocation_data)

            # Normalize result
            if isinstance(result, dict):
                tasks_data = result
            else:
                try:
                    tasks_data = result.dict()
                except Exception:
                    tasks_data = {"internal_tasks": [], "external_tasks": []}

            # Extract both arrays
            internal_tasks = tasks_data.get("internal_tasks", [])
            external_tasks = tasks_data.get("external_tasks", [])
            
            # Validate departments
            valid_departments = [
                "Signature Required", 
                "Denials & Appeals", 
                "Approvals to Schedule", 
                "Scheduling Tasks",
                "Administrative Tasks",
                "Provider Action Required"
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
            validated_external = []
            for task in external_tasks:
                if not task.get("description"):
                    continue
                if task.get("department") not in valid_departments:
                    task["department"] = "Administrative Tasks"  # External tasks typically admin
                validated_external.append(task)

            # If both arrays are empty, create fallback
            if not validated_internal and not validated_external:
                fallback = await self._create_fallback_task(document_analysis, source_document)
                validated_internal = fallback  # Put fallback in internal

            result_dict = {
                "internal_tasks": validated_internal,
                "external_tasks": validated_external
            }

            logger.info(f"✅ Generated {len(validated_internal)} internal task(s), {len(validated_external)} external task(s)")
            if validated_internal:
                logger.info(f"   Internal: {[t['description'] for t in validated_internal]}")
            if validated_external:
                logger.info(f"   External: {[t['description'] for t in validated_external]}")

            # Update analytics for all tasks
            all_tasks = validated_internal + validated_external
            await self._update_workflow_analytics(all_tasks)

            return result_dict

        except Exception as e:
            logger.error(f"❌ Task generation failed: {str(e)}")
            fallback = await self._create_fallback_task(document_analysis, source_document)
            return {"internal_tasks": fallback, "external_tasks": []}

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
        
        return "Provider Action Required"

    async def _create_fallback_task(self, document_analysis: dict, source_document: str) -> list[dict]:
        """Create intelligent fallback task when generation fails."""
        current_date = datetime.now()
        due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
        doc_type = document_analysis.get("document_type", "document")
        patient = document_analysis.get("patient_name", "Unknown")
        
        fallback_task = {
            "description": f"Review and route {doc_type}",
            "department": "Provider Action Required",
            "status": "Pending",
            "due_date": due_date,
            "patient": patient,
            "actions": ["Claim", "Complete"],
            "source_document": source_document or "Unknown",
            "quick_notes": {
                "details": f"Document requires manual review to determine appropriate actions and routing.",
                "one_line_note": "Manual review needed"
            }
        }
        
        return [fallback_task]

    async def _update_workflow_analytics(self, tasks: list[dict]):
        """Update workflow analytics based on generated tasks."""
        try:
            db = DatabaseService()
            await db.connect()

            for task in tasks:
                department = task.get("department", "").lower()
                description = task.get("description", "").lower()

                if "provider action" in department:
                    await db.increment_workflow_stat("clinicalReviews")
                elif "scheduling" in department:
                    await db.increment_workflow_stat("schedulingTasks")
                elif "approvals to schedule" in department:
                    await db.increment_workflow_stat("approvalsToSchedule")
                elif "administrative" in department:
                    await db.increment_workflow_stat("adminTasks")
                elif "denial" in department or "appeal" in department:
                    await db.increment_workflow_stat("authTasks")
                elif "signature" in department:
                    await db.increment_workflow_stat("signatureTasks")

                if any(word in description for word in ["rfa", "ur", "imr", "authorization", "denial"]):
                    await db.increment_workflow_stat("rfasMonitored")
                elif any(word in description for word in ["qme", "ime", "ame"]):
                    await db.increment_workflow_stat("qmeUpdating")
                elif any(word in description for word in ["attorney", "legal", "settlement"]):
                    await db.increment_workflow_stat("legalDocs")

            await db.disconnect()

        except Exception as e:
            logger.error(f"❌ Analytics update failed: {str(e)}")