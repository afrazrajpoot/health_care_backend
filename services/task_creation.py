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
    department: str = Field(..., description="Must be one of: Medical/Clinical, Scheduling & Coordination, Administrative/Compliance, Authorizations & Denials")
    status: str = "Pending"
    due_date: str
    patient: str
    quick_notes: QuickNotes
    actions: List[str]
    source_document: str = ""

class TaskCreationResult(BaseModel):
    tasks: List[AITask]

# ------------------ TASK CREATOR ------------------
class TaskCreator:
    """Universal AI task generator for any medical document type."""
    
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
You are an expert healthcare operations AI that converts ANY medical document into the necessary actionable tasks.

## 🎯 CORE MISSION
Read the document deeply. Identify ALL DISTINCT ACTIONABLE ITEMS that require separate follow-up. Generate one task per distinct action required.

---

## 📋 UNIVERSAL DOCUMENT UNDERSTANDING

You will receive ANY type of medical document:
- Clinical reports (labs, imaging, pathology, consultations, progress notes, H&P, discharge summaries, operative notes)
- Administrative documents (referrals, authorizations, denials, legal correspondence, insurance communications)
- Coordination documents (therapy notes, follow-up recommendations, care plans, medication requests)
- Compliance documents (QME/AME/IMR notices, attorney letters, settlement documents)

**Your job:** Extract EACH distinct action that must happen based on the document's content.

---

## 🔢 MULTIPLE TASKS RULES

**Generate MULTIPLE tasks when document contains:**
- Multiple procedure authorizations (each needs separate scheduling)
- Multiple treatment recommendations (each requiring different department actions)
- Multiple abnormal results requiring different clinical reviews
- Authorization + scheduling + clinical review all needed
- Legal response + clinical action + administrative update required
- Multiple referrals to different specialists
- Multiple medications needing different actions (refill, review, change)

**Generate SINGLE task when:**
- Document has one clear primary action
- Multiple items belong to same workflow step (e.g., "Review all lab results")
- Sub-items are part of one larger task

**CRITICAL:** Each task must be genuinely distinct and actionable separately. Don't artificially split one action into multiple tasks.

---

## 🧠 INTELLIGENT TASK EXTRACTION LOGIC

### **Step 1: Identify Document Intent**
Ask yourself:
- What are ALL the things this document tells us to do?
- Are these separate actions requiring different departments or workflows?
- Can these be combined into one task or must they be separate?

### **Step 2: Detect Action Triggers**

**Clinical Triggers → Medical/Clinical Department:**
- Abnormal/critical results requiring physician review
- Treatment recommendations from specialists
- Medication management needs
- Symptom changes requiring clinical assessment
- Functional capacity evaluations
- Progress reports needing clinical interpretation

**Scheduling Triggers → Scheduling & Coordination:**
- Approved authorizations needing appointments
- Referrals requiring coordination
- Diagnostic studies needing scheduling
- Follow-up appointments recommended
- Procedure/surgery scheduling needs

**Authorization Triggers → Authorizations & Denials:**
- RFA submissions needing tracking
- UR denials requiring response/appeal
- IMR processes needing management
- Peer-to-peer coordination requests
- Authorization expirations requiring renewal

**Administrative Triggers → Administrative/Compliance:**
- Legal communications requiring response
- Compliance documentation needs
- QME/AME administrative coordination
- Attorney correspondence requiring file updates
- Record requests or document distribution

### **Step 3: Extract Specifics**
From the document, capture:
- **Patient name** (exact)
- **Specific findings** (test results, diagnoses, recommendations)
- **Deadlines** (legal dates, authorization expiries, clinical urgency)
- **Provider names** (ordering physician, specialist, reviewer)
- **Actionable details** (what test, what procedure, what response needed)

---

## 🏢 DEPARTMENT ROUTING RULES

**Medical/Clinical** - Route when:
- Document contains clinical findings requiring physician interpretation
- Treatment decisions or modifications needed
- Abnormal results needing clinical follow-up
- Medication management required
- Patient care plan updates needed

**Scheduling & Coordination** - Route when:
- Authorization approved and appointment needs scheduling
- Referral received requiring coordination
- Diagnostic study approved and needs booking
- Follow-up visit recommended with timeframe

**Authorizations & Denials** - Route when:
- RFA submitted and needs tracking
- UR denial received requiring response
- IMR eligible and needs filing
- Authorization about to expire
- Peer-to-peer review requested

**Administrative/Compliance** - Route when:
- Legal document received (attorney, adjuster, court)
- QME/AME notice requiring administrative action
- Compliance documentation needed
- Record request or document distribution
- Settlement or C&R documents

---

## ⏰ INTELLIGENT DUE DATE ASSIGNMENT

Base due dates on document content analysis:

| Document Context | Due Date | Rationale |
|------------------|----------|-----------|
| STAT/Critical results | Same day | Immediate physician review required |
| Abnormal findings | +1 day | Urgent clinical attention needed |
| Authorization expiring soon | 2 days before expiry | Prevent lapse in coverage |
| Legal deadline specified | 3 days before deadline | Allow processing time |
| UR denial (response required) | +3 days | Meet regulatory timelines |
| Approved authorization | +2 days | Prompt patient service |
| Standard clinical review | +2 days | Normal workflow |
| Administrative tasks | +3 days | Standard processing |
| Routine follow-up | +7 days | Non-urgent coordination |

---

## ✅ TASK GENERATION RULES

**CRITICAL PRINCIPLES:**
1. **GENERATE AS MANY AS NEEDED** - One task per distinct action
2. **SPECIFIC & ACTIONABLE** - Staff must know exactly what to do
3. **DOCUMENT-DRIVEN** - Based only on actual content, no assumptions
4. **CONCISE DESCRIPTION** - 5-10 words maximum
5. **CONTEXT IN DETAILS** - Why this task matters (quick_notes.details)

**Task Description Formula:**
[Action Verb] + [Specific Item from Document] + [Critical Context]

**Examples:**
- ❌ Bad: "Review document"
- ✅ Good: "Review elevated ALT (150) from 1/15 labs"

- ❌ Bad: "Schedule appointment"
- ✅ Good: "Schedule authorized orthopedic consult within 7 days"

- ❌ Bad: "Handle denial"
- ✅ Good: "Prepare IMR appeal for denied PT authorization"

- ❌ Bad: "Process legal letter"
- ✅ Good: "Review attorney deposition request by 1/20"

**Multi-Task Example:**
Authorization report approves MRI, PT, and orthopedic consult:
Task 1: "Schedule authorized lumbar MRI within 7 days" → Scheduling
Task 2: "Schedule approved physical therapy sessions" → Scheduling
Task 3: "Schedule orthopedic specialist consultation" → Scheduling

---

## 🚫 ANTI-HALLUCINATION RULES

**NEVER:**
- Invent clinical findings not in the document
- Create tasks for missing information (instead: task = "Clarify missing X")
- Assume urgency without evidence
- Generate generic "review document" tasks
- Route to wrong department based on keywords alone
- Create duplicate tasks for the same action

**ALWAYS:**
- Use exact patient names from document
- Reference specific dates, values, findings from document
- Base urgency on explicit indicators (STAT, deadlines, critical values)
- Match task complexity to document content
- Ensure each task is genuinely distinct

---

## 📤 OUTPUT FORMAT

```json
{{
  "tasks": [
    {{
      "description": "Concise 5-10 word action statement with specific details",
      "department": "Exact department name from the four options",
      "status": "Pending",
      "due_date": "YYYY-MM-DD",
      "patient": "Exact patient name from document",
      "actions": ["Claim", "Complete"],
      "source_document": "{{source_document}}",
      "quick_notes": {{
        "details": "Brief explanation of WHY this task is critical based on document content (1-2 sentences)",
        "one_line_note": "Ultra-concise summary for dashboard (under 50 chars)"
      }}
    }}
  ]
}}
```

---

## 🎯 QUALITY CHECKLIST (Before Generating)

Ask yourself:
1. ✅ Have I identified ALL distinct actions in this document?
2. ✅ Is each task SPECIFIC with actual document details?
3. ✅ Is the department routing CORRECT for workflow ownership?
4. ✅ Is the due date APPROPRIATE for urgency level?
5. ✅ Does quick_notes.details explain WHY this matters?
6. ✅ Would a staff member know EXACTLY what to do?
7. ✅ Am I using ONLY information from the document?
8. ✅ Are multiple tasks truly distinct or should they be combined?

---

**Remember:** Generate the APPROPRIATE number of tasks - no more, no less. Quality and specificity matter more than quantity.
"""
    
    def create_prompt(self, patient_name: str, source_document: str) -> ChatPromptTemplate:
        user_template = """
DOCUMENT TYPE: {document_type}

FULL DOCUMENT TEXT:
{full_text}

DOCUMENT ANALYSIS (Structured Extraction):
{document_analysis}

SOURCE DOCUMENT: {source_document}
TODAY'S DATE: {current_date}
PATIENT: {patient_name}

**Task:** Generate ALL necessary actionable tasks from this document. Use the FULL DOCUMENT TEXT as the primary source of information, and the DOCUMENT ANALYSIS for structured data reference.

**Think through:**
1. What are ALL the distinct actions this document requires?
2. Should these be separate tasks or combined into one?
3. Which department owns each action in our workflow?
4. What is the appropriate timeline based on urgency for each?
5. What specific details make each task actionable?

{format_instructions}
"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template),
        ])

    async def generate_tasks(self, document_analysis: dict, source_document: str = "", full_text: str = "") -> list[dict]:
        """Generate intelligent, context-aware task from any medical document."""
        try:
            current_date = datetime.now()
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            
            logger.info(f"🔍 Analyzing document: {source_document} ({document_type})")
            logger.info(f"📝 Full text length: {len(full_text)} characters")

            prompt = self.create_prompt(patient_name, source_document)
            chain = prompt | self.llm | self.parser

            # Build invocation data with full text
            invocation_data = {
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date.strftime("%Y-%m-%d"),
                "source_document": source_document or "Unknown",
                "document_type": document_type,
                "patient_name": patient_name,
                "format_instructions": self.parser.get_format_instructions()
            }
            
            # Add full text if available
            if full_text:
                invocation_data["full_text"] = full_text
                logger.info("✅ Full document text included in task generation")
            else:
                invocation_data["full_text"] = "Not available"
                logger.warning("⚠️ No full text available for task generation")

            result = chain.invoke(invocation_data)

            # Normalize result
            if isinstance(result, dict):
                tasks_data = result
            else:
                try:
                    tasks_data = result.dict()
                except Exception:
                    tasks_data = {"tasks": []}

            # Extract tasks (now expecting list)
            tasks = tasks_data.get("tasks", [])
            
            if not tasks:
                return await self._create_fallback_task(document_analysis, source_document)

            # Validate each task
            valid_departments = [
                "Medical/Clinical", 
                "Scheduling & Coordination", 
                "Administrative/Compliance", 
                "Authorizations & Denials"
            ]
            
            validated_tasks = []
            for task in tasks:
                if not task.get("description"):
                    continue
                    
                if task.get("department") not in valid_departments:
                    task["department"] = self._infer_department(task.get("description", ""), document_type)
                
                validated_tasks.append(task)
            
            if not validated_tasks:
                return await self._create_fallback_task(document_analysis, source_document)

            logger.info(f"✅ Generated {len(validated_tasks)} task(s): {[t['description'] for t in validated_tasks]}")

            await self._update_workflow_analytics(validated_tasks)

            return validated_tasks

        except Exception as e:
            logger.error(f"❌ Task generation failed: {str(e)}")
            return await self._create_fallback_task(document_analysis, source_document)

    def _infer_department(self, description: str, doc_type: str) -> str:
        """Quick fallback department inference."""
        desc_lower = description.lower()
        type_lower = doc_type.lower()
        
        # Clinical indicators
        if any(word in desc_lower for word in ["review", "assess", "evaluate", "lab", "result", "finding", "clinical", "treatment", "medication"]):
            return "Medical/Clinical"
        
        # Scheduling indicators
        if any(word in desc_lower for word in ["schedule", "book", "appointment", "coordinate", "arrange"]):
            return "Scheduling & Coordination"
        
        # Authorization indicators
        if any(word in desc_lower for word in ["rfa", "authorization", "denial", "ur", "imr", "appeal", "peer-to-peer"]):
            return "Authorizations & Denials"
        
        # Administrative indicators
        if any(word in desc_lower for word in ["attorney", "legal", "qme", "compliance", "correspondence", "document"]):
            return "Administrative/Compliance"
        
        # Default to Medical/Clinical
        return "Medical/Clinical"

    async def _create_fallback_task(self, document_analysis: dict, source_document: str) -> list[dict]:
        """Create intelligent fallback task when generation fails."""
        current_date = datetime.now()
        due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
        doc_type = document_analysis.get("document_type", "document")
        patient = document_analysis.get("patient_name", "Unknown")
        
        fallback_task = {
            "description": f"Review {doc_type} and route appropriately",
            "department": "Medical/Clinical",
            "status": "Pending",
            "due_date": due_date,
            "patient": patient,
            "actions": ["Claim", "Complete"],
            "source_document": source_document or "Unknown",
            "quick_notes": {
                "details": f"System-generated task for {doc_type} requiring manual review and routing",
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

                # Department-based analytics
                if "medical" in department or "clinical" in department:
                    await db.increment_workflow_stat("clinicalReviews")
                elif "scheduling" in department or "coordination" in department:
                    await db.increment_workflow_stat("schedulingTasks")
                elif "administrative" in department or "compliance" in department:
                    await db.increment_workflow_stat("adminTasks")
                elif "authorization" in department or "denial" in department:
                    await db.increment_workflow_stat("authTasks")

                # Specific workflow tracking
                if any(word in description for word in ["rfa", "ur", "imr", "authorization", "denial"]):
                    await db.increment_workflow_stat("rfasMonitored")
                elif any(word in description for word in ["qme", "ime", "ame"]):
                    await db.increment_workflow_stat("qmeUpdating")
                elif any(word in description for word in ["attorney", "legal", "settlement"]):
                    await db.increment_workflow_stat("legalDocs")

            await db.disconnect()

        except Exception as e:
            logger.error(f"❌ Analytics update failed: {str(e)}")