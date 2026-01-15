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
    source_document: str = ""

class TaskCreationResult(BaseModel):
    internal_tasks: List[AITask] = Field(default_factory=list, description="Tasks for internal clinic operations")

# ------------------ TASK CREATOR ------------------
class TaskCreator:
    """Universal AI task generator with deduplication and context-aware task creation."""
    
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
        self.db = DatabaseService()

    SYSTEM_PROMPT = """
You are an expert healthcare operations AI using OpenAI O3 reasoning to create high-quality, actionable tasks from medical documents.

## 🚨 CRITICAL ANTI-DUPLICATION RULES

### **NEVER Create Generic Review Tasks**
❌ FORBIDDEN: "Review [document type] for [patient]"
❌ FORBIDDEN: "Review progress notes for [patient]"
❌ FORBIDDEN: "Review report for [patient]"
❌ FORBIDDEN: "Process [document type] for [patient]"

These generic tasks provide NO VALUE and will be rejected.

### **Only Create Tasks When:**
1. ✅ Document explicitly requests a specific action (schedule, sign, appeal, authorize)
2. ✅ Clinical findings require physician decision or intervention
3. ✅ Authorization/denial requires response with specific deadline
4. ✅ Missing critical information that blocks workflow (patient ID, auth number, etc.)

### **If Document is Purely Informational:**
Return EMPTY array `{{"internal_tasks": []}}` - do NOT create placeholder review tasks.

---

## 📋 TASK SPECIFICITY REQUIREMENTS

### **Scheduling Tasks - MUST Include:**
- **What body part/area**: "lumbar spine", "right knee", "cervical region"
- **What procedure**: "MRI", "physical therapy", "epidural injection"
- **Any constraints**: "with contrast", "bilateral", "12 sessions"

**Example:**
✅ "Schedule lumbar spine MRI with contrast for John Smith"
❌ "Schedule MRI for John Smith" (missing body part)

### **Denial/Appeal Tasks - MUST Include:**
- **What was denied**: specific service, procedure, medication
- **Why task matters**: "to restore treatment", "required for surgery clearance"
- **Actionable hint**: body part, diagnosis code, missing documentation

**Example:**
✅ "Appeal denied lumbar epidural injection - prior conservative care documented"
❌ "Appeal denial for John Smith" (missing what was denied and why)

### **Signature Tasks - MUST Include:**
- **Document type**: "settlement agreement", "QME attestation", "auth form"
- **Return method**: exact fax/email/address
- **Deadline**: if specified in document

**Example:**
✅ "Sign QME attestation for Jane Doe - fax to (555) 123-4567 by 1/20"
❌ "Sign document for Jane Doe" (missing document type and return method)

---

## 🔍 DUPLICATE DETECTION LOGIC

Before creating ANY task, check against these patterns:

### **Pattern 1: Same Patient + Same Action + Same Target**
- "Schedule MRI for John Smith" = "Book MRI for John Smith" → DUPLICATE
- "Appeal denied therapy" = "File appeal for therapy denial" → DUPLICATE

### **Pattern 2: Generic vs Specific (Keep Specific)**
- If you have "Schedule lumbar MRI" → DO NOT also create "Schedule imaging"
- If you have "Appeal denied injection" → DO NOT also create "Review denial"

### **Pattern 3: Parent Task Covers Child Tasks**
- If "Schedule 12 PT sessions" exists → DO NOT create "Schedule PT follow-up"
- If "Sign and return settlement" exists → DO NOT create "Review settlement"

---

## 🎯 TASK GENERATION DECISION TREE

**Step 1: Analyze Document Purpose**
```
Is this document requesting a specific action?
├─ YES → Go to Step 2
└─ NO → Is it purely informational?
    ├─ YES → Return empty array {{"internal_tasks": []}}
    └─ NO → Go to Step 2
```

**Step 2: Identify Specific Actions**
```
What specific actions are requested?
├─ Schedule → Extract: body part, procedure, constraints
├─ Sign → Extract: document type, return method, deadline
├─ Appeal → Extract: denied service, reason, deadline
├─ Authorize → Extract: service, diagnosis, urgency
└─ Clarify → ONLY if missing critical data blocks workflow
```

**Step 3: Check for Duplicates**
```
For each task:
1. Does this exact action already exist for this patient?
   ├─ YES → Skip this task
   └─ NO → Continue
2. Is this a generic version of a specific task?
   ├─ YES → Skip this task
   └─ NO → Continue
3. Does a parent task already cover this?
   ├─ YES → Skip this task
   └─ NO → Create task
```

---

## ✅ VALID TASK EXAMPLES

### **Scheduling Tasks (with specificity):**
```json
{{
  "description": "Schedule lumbar spine MRI with contrast for John Smith",
  "department": "Approvals to Schedule",
  "quick_notes": {{
    "details": "Authorization approved for MRI L-spine. Schedule within 2 weeks per auth requirements.",
    "one_line_note": "MRI L-spine approved - schedule ASAP"
  }}
}}
```

### **Appeal Tasks (with context):**
```json
{{
  "description": "Appeal denied lumbar epidural injection for Maria Garcia",
  "department": "Denials & Appeals",
  "quick_notes": {{
    "details": "Denial reason: insufficient conservative care. Have documented 8 weeks PT and medication trials. Deadline: 1/25.",
    "one_line_note": "ESI denied - PT documented, appeal by 1/25"
  }}
}}
```

### **Signature Tasks (with details):**
```json
{{
  "description": "Sign settlement agreement for Robert Lee - return via fax",
  "department": "Signature Required",
  "quick_notes": {{
    "details": "C&R settlement agreement requires signature. Fax to (555) 123-4567 attn: Claims Adjuster by 1/22.",
    "one_line_note": "Settlement signature - fax by 1/22"
  }}
}}
```

---

## 🚫 INVALID TASK EXAMPLES (DO NOT CREATE)

❌ **Generic Review Tasks:**
```json
{{
  "description": "Review progress notes for John Smith",  // NO VALUE
  "department": "Administrative Tasks"
}}
```

❌ **Missing Specificity:**
```json
{{
  "description": "Schedule MRI for patient",  // Missing body part
  "department": "Scheduling Tasks"
}}
```

❌ **Duplicate of Existing:**
```json
// If "Schedule lumbar MRI" exists, DO NOT create:
{{
  "description": "Book imaging study",  // Duplicate
  "department": "Scheduling Tasks"
}}
```

---

## 🏥 DEPARTMENT ROUTING

| Department | Route When | Requirements |
|------------|------------|--------------|
| **Signature Required** | Document has signature line + return instructions | Must include: doc type, return method, deadline |
| **Denials & Appeals** | Explicit denial with appeal deadline | Must include: denied service, reason, deadline |
| **Approvals to Schedule** | Authorization approved + needs scheduling | Must include: service, body part, timeframe |
| **Scheduling Tasks** | Follow-up or non-authorized scheduling | Must include: appointment type, reason |
| **Administrative Tasks** | Legal docs, records requests, QME coordination | Must include: specific action needed |

---

## ⏰ DUE DATE LOGIC

| Context | Due Date | Reason |
|---------|----------|--------|
| STAT/Critical | Same day | Immediate attention |
| Denial response deadline | 3 days before deadline | Processing time |
| Approved authorization | +2 days | Prompt service |
| Signature with deadline | 2 days before deadline | Review + return time |
| Standard scheduling | +2 days | Normal workflow |
| Routine follow-up | +7 days | Non-urgent |

---

## 📤 OUTPUT FORMAT

**If document has actionable items:**
```json
{{{{
  "internal_tasks": [
    {{{{
      "description": "[Specific action] + [what/body part] + [for patient]",
      "department": "Exact department name",
      "status": "Pending",
      "due_date": "YYYY-MM-DD",
      "patient": "Exact patient name",
      "actions": ["Claim", "Complete"],
      "source_document": "{{source_document}}",
      "quick_notes": {{{{
        "details": "Why this matters + specific context (deadline, body part, reason)",
        "one_line_note": "Short summary with key detail (under 50 chars)"
      }}}}
    }}}}
  ]
}}}}
```

**If document is purely informational:**
```json
{{{{
  "internal_tasks": []
}}}}
```

---

## 🎯 QUALITY CHECKLIST

Before generating output, verify:
1. ✅ Did I check if this document requires ANY action at all?
2. ✅ If no actions needed, did I return empty array?
3. ✅ Does each task description include specific details (body part, service, reason)?
4. ✅ Did I check for duplicates against existing tasks?
5. ✅ Did I avoid creating generic "Review [document]" tasks?
6. ✅ Are scheduling tasks specific about what's being scheduled?
7. ✅ Do appeal tasks explain what was denied and why it matters?
8. ✅ Do signature tasks include return method and deadline?
9. ✅ Can staff immediately understand the specific action needed?
10. ✅ Did I eliminate redundant or overlapping tasks?

**Remember:** Quality over quantity. Empty array is better than useless generic tasks.
"""
    
    def create_prompt(self, patient_name: str, source_document: str, matched_doctor_name: str) -> ChatPromptTemplate:
        user_template = """
DOCUMENT TYPE: {{document_type}}

FULL DOCUMENT TEXT:
{{full_text}}

DOCUMENT ANALYSIS (Structured Extraction):
{{document_analysis}}

EXISTING TASKS FOR THIS PATIENT:
{{existing_tasks}}

SOURCE DOCUMENT: {{source_document}}
TODAY'S DATE: {{current_date}}
PATIENT: {{patient_name}}
MATCHED DOCTOR: {{matched_doctor_name}}

**TASK GENERATION INSTRUCTIONS:**

Using OpenAI O3 reasoning, analyze this document and determine if ANY actionable tasks are needed.

**CRITICAL: Review existing tasks above. DO NOT create duplicates or similar tasks.**

**🚫 ANTI-PATTERN CHECK (PERFORM FIRST):**
- Does your task start with "Review", "Process", "Analyze"? **DROP IT immediately.**
- Is the task just "Review [Report Type]"? **DROP IT.**
- UNLESS it is "Review and Sign" or "Review for Surgery Clearance" (specific goal).

**Step 1: Is This Document Actionable?**
- Does it explicitly request scheduling, signing, appealing, or authorizing?
- Does it contain clinical findings requiring physician decision?
- Is critical information missing that blocks our workflow?

**If NO to all above → Return empty array: {{"internal_tasks": []}}**

**Step 2: Extract Specific Action Details**
For each action needed, you MUST include specific details in the description:
- **Scheduling**: 
    - WHAT: Body part? Procedure/Service? (e.g., "Lumbar MRI", "PT for Right Knee")
    - DETAILS: Contrast? Number of sessions?
- **Appeals / Denials**: 
    - WHAT: What was denied? (e.g., "Cervical Epidural", "Tylenol prescription")
    - WHY: Hint/Guidance for appeal (e.g., "Provide conservative care notes", "Attach MRI report")
- **Signatures**: 
    - WHAT: Specific document? (e.g., "QME Attestation", "C&R Agreement")
    - HOW: Fax number? Email?
- **Other**: What specific action? Why does it matter?

**Step 3: Deduplication Check**
Before adding ANY task:
1. Check EXISTING TASKS above - does similar task exist?
2. Check your own generated tasks - any duplicates?
3. Is this a generic version of a specific task?
4. If YES to any → DO NOT create this task

**Step 4: Validate Specificity (Self-Correction)**
- "Schedule MRI" -> ❌ REJECT. Fix to: "Schedule Lumbar Spine MRI w/ Contrast"
- "Appeal Denial" -> ❌ REJECT. Fix to: "Appeal Denial for Right Knee Surgery - Submit PT notes"
- "Review Report" -> ❌ REJECT. (Unless explicit physician review requested for specific clearance).

**If task is vague or generic → DO NOT create it**

**Step 5: Final Quality Check**
- Can staff execute this without asking "what exactly do I do?"
- Does this provide value beyond just acknowledging the document exists?
- Is this genuinely different from existing tasks?

**If NO to any → DO NOT create this task**

{format_instructions}
"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template),  # Removed template_format="jinja2" to avoid conflict
        ])

    async def _get_existing_tasks(self, patient_name: str) -> str:
        """Fetch existing tasks for the patient to prevent duplicates."""
        try:
            await self.db.connect()
            # Query tasks for this patient using Prisma ORM via DB service accessor (or direct if exposed)
            # Since DatabaseService wraps Prisma but doesn't expose a generic 'query' method for collections like Mongo,
            # we need to use the prisma client directly if available or add a method.
            # Assuming 'task' table exists in schema.
            
            if not self.db.prisma:
                 return "Unable to access database."

            tasks = await self.db.prisma.task.find_many(
                where={
                    "patient": {"contains": patient_name, "mode": "insensitive"},
                    "status": {"not": "Completed"}
                }
            )
            
            await self.db.disconnect()
            
            if not tasks:
                return "No existing tasks for this patient."
            
            # Format existing tasks for context
            task_list = []
            for task in tasks:
                # Access attributes directly from Prisma object
                desc = getattr(task, 'description', 'Unknown')
                dept = getattr(task, 'department', 'Unknown')
                task_list.append(f"- {desc} ({dept})")
            
            return "\n".join(task_list)
        except Exception as e:
            logger.error(f"❌ Failed to fetch existing tasks: {str(e)}")
            # Don't crash creation if fetch fails, just return empty context
            return "Unable to retrieve existing tasks (Database Error)."

    async def generate_tasks(self, document_analysis: dict, source_document: str = "", full_text: str = "", matched_doctor_name: str = "") -> dict:
        """Generate internal_tasks array with deduplication."""
        try:
            current_date = datetime.now()
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            
            logger.info(f"🔍 Analyzing document: {source_document} ({document_type})")
            logger.info(f"📝 Full text length: {len(full_text) if full_text else 0} characters")
            logger.info(f"👨‍⚕️ Matched doctor: {matched_doctor_name if matched_doctor_name else 'Not specified'}")

            # Get existing tasks for deduplication
            existing_tasks = await self._get_existing_tasks(patient_name)
            logger.info(f"📋 Existing tasks retrieved for deduplication check")

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
                "existing_tasks": existing_tasks,
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

            # Extract tasks
            internal_tasks = tasks_data.get("internal_tasks", [])
            
            # Validate departments
            valid_departments = [
                "Signature Required", 
                "Denials & Appeals", 
                "Approvals to Schedule", 
                "Scheduling Tasks",
                "Administrative Tasks"
            ]
            
            # Validate and deduplicate tasks
            validated_internal = []
            seen_task_signatures = set()
            
            for task in internal_tasks:
                if not task.get("description"):
                    continue
                
                # Create task signature for deduplication
                task_signature = self._create_task_signature(task)
                
                # Skip if duplicate
                if task_signature in seen_task_signatures:
                    logger.info(f"⚠️ Skipping duplicate task: {task['description']}")
                    continue
                
                # Skip if generic review task
                if self._is_generic_review_task(task.get("description", "")):
                    logger.info(f"⚠️ Skipping generic review task: {task['description']}")
                    continue
                
                # Validate department
                if task.get("department") not in valid_departments:
                    task["department"] = self._infer_department(task.get("description", ""), document_type)
                
                seen_task_signatures.add(task_signature)
                validated_internal.append(task)
            
            # Only create fallback if document truly requires action
            if not validated_internal and self._requires_fallback(document_type, full_text):
                logger.warning(f"⚠️ Creating fallback for actionable document: {document_type}")
                fallback = await self._create_fallback_task(document_analysis, source_document, full_text)
                validated_internal = fallback
            elif not validated_internal:
                logger.info(f"✅ No tasks needed for informational document: {document_type}")

            result_dict = {
                "internal_tasks": validated_internal
            }

            if validated_internal:
                logger.info(f"✅ Generated {len(validated_internal)} tasks:")
                logger.info(f"   Internal: {[t['description'] for t in validated_internal]}")

            # Update analytics for all tasks
            if validated_internal:
                await self._update_workflow_analytics(validated_internal)

            return result_dict

        except Exception as e:
            logger.error(f"❌ Task generation failed: {str(e)}")
            logger.error(f"❌ Document type: {document_analysis.get('document_type', 'Unknown')}")
            # Only create fallback if truly needed
            if self._requires_fallback(document_analysis.get('document_type', ''), full_text):
                fallback = await self._create_fallback_task(document_analysis, source_document, full_text)
                return {"internal_tasks": fallback}
            return {"internal_tasks": []}

    def _create_task_signature(self, task: dict) -> str:
        """Create unique signature for task deduplication."""
        # Normalize description for comparison
        desc = task.get("description", "").lower()
        patient = task.get("patient", "").lower()
        department = task.get("department", "").lower()
        
        # Remove common words for better matching
        common_words = ["the", "a", "an", "for", "to", "and", "or", "in", "on", "at"]
        desc_words = [w for w in desc.split() if w not in common_words]
        
        # Create signature from key components
        signature = f"{patient}:{department}:{' '.join(sorted(desc_words[:5]))}"
        return signature

    def _is_generic_review_task(self, description: str) -> bool:
        """Detect generic review tasks that should be avoided."""
        desc_lower = description.lower()
        
        # Forbidden patterns
        forbidden_patterns = [
            "review progress note",
            "review report for",
            "process report for",
            "review document for",
            "review progress report",
            "process progress note",
            "manual review needed"
        ]
        
        return any(pattern in desc_lower for pattern in forbidden_patterns)

    def _requires_fallback(self, doc_type: str, full_text: str) -> bool:
        """Determine if document type requires fallback task or can be empty."""
        doc_type_lower = (doc_type or "").lower()
        full_text_lower = (full_text or "").lower()
        
        # Documents that truly require action
        actionable_indicators = [
            "signature required",
            "please sign",
            "denial",
            "denied",
            "appeal",
            "authorization approved",
            "schedule",
            "appointment",
            "urgent",
            "stat"
        ]
        
        # Check if any actionable indicators present
        return any(indicator in full_text_lower or indicator in doc_type_lower 
                  for indicator in actionable_indicators)

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

    async def _create_fallback_task(self, document_analysis: dict, source_document: str, full_text: str = "") -> list[dict]:
        """Create intelligent fallback task ONLY when truly needed."""
        current_date = datetime.now()
        due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
        doc_type = document_analysis.get("document_type", "document")
        patient = document_analysis.get("patient_name", "Unknown")
        doc_type_lower = doc_type.lower() if doc_type else ""
        full_text_lower = (full_text or "").lower()
        
        # Only create specific, actionable fallback tasks
        fallback_mappings = {
            "denial": {
                "description": f"Evaluate denial response options for {patient}",
                "department": "Denials & Appeals",
                "details": "Denial received. Review denial reason, supporting documentation, and determine if appeal is warranted.",
                "one_line_note": "Evaluate denial for appeal"
            },
            "authorization": {
                "description": f"Verify authorization status for {patient}",
                "department": "Approvals to Schedule",
                "details": "Authorization document received. Verify approval status and schedule if approved.",
                "one_line_note": "Verify auth and schedule"
            }
        }
        
        # Check for signature requirements
        if any(kw in full_text_lower for kw in ["signature required", "please sign", "sign and return"]):
            return [{
                "description": f"Sign and return document for {patient}",
                "department": "Signature Required",
                "status": "Pending",
                "due_date": due_date,
                "patient": patient,
                "actions": ["Claim", "Complete"],
                "source_document": source_document or "Unknown",
                "quick_notes": {
                    "details": "Document requires signature. Review document for signature location and return instructions.",
                    "one_line_note": "Signature needed"
                }
            }]
        
        # Find matching specific fallback
        for key, info in fallback_mappings.items():
            if key in doc_type_lower or key in full_text_lower:
                fallback_task = {
                    "description": info["description"],
                    "department": info["department"],
                    "status": "Pending",
                    "due_date": due_date,
                    "patient": patient,
                    "actions": ["Claim", "Complete"],
                    "source_document": source_document or "Unknown",
                    "quick_notes": {
                        "details": info["details"],
                        "one_line_note": info["one_line_note"]
                    }
                }
                
                logger.info(f"📋 Created specific fallback task: {info['description']}")
                return [fallback_task]
        
        # If truly unclear, return empty rather than generic review
        logger.info(f"ℹ️ No fallback needed for informational document: {doc_type}")
        return []

    async def _update_workflow_analytics(self, tasks: list[dict]):
        """Update workflow analytics based on generated tasks."""
        try:
            await self.db.connect()

            for task in tasks:
                department = task.get("department", "").lower()
                description = task.get("description", "").lower()

                # Map departments to valid workflow stat fields
                if "scheduling" in department or "approvals to schedule" in department:
                    await self.db.increment_workflow_stat("referralsProcessed")
                elif "denial" in department or "appeal" in department:
                    await self.db.increment_workflow_stat("payerDisputes")
                elif "administrative" in department or "signature" in department:
                    await self.db.increment_workflow_stat("externalDocs")

                # Description-based analytics
                if any(word in description for word in ["rfa", "ur", "imr", "authorization", "denial"]):
                    await self.db.increment_workflow_stat("rfasMonitored")
                elif any(word in description for word in ["qme", "ime", "ame"]):
                    await self.db.increment_workflow_stat("qmeUpcoming")

            await self.db.disconnect()

        except Exception as e:
            logger.error(f"❌ Analytics update failed: {str(e)}")