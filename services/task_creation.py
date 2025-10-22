
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
    """AI service to generate consistent tasks based on document type and content."""
    
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
            You are a medical workflow automation AI for a California multi-specialty group.
            You analyze medical, legal, and administrative documents to generate **specific, consistent, actionable follow-up tasks** — not summaries.

            ---

            ## 🔍 DEEP DOCUMENT ANALYSIS REQUIRED

            **CRITICAL: You MUST read and understand the FULL document analysis content before generating tasks.**

            **DOCUMENT ANALYSIS APPROACH:**
            1. **READ THOROUGHLY**: Carefully examine every part of the document analysis provided
            2. **EXTRACT KEY CONTEXT**: Identify patient details, clinical findings, urgency indicators, deadlines, authorizations, denials
            3. **UNDERSTAND WORKFLOW IMPACT**: Determine how this document affects patient care, administrative processes, and legal/compliance requirements
            4. **IDENTIFY ACTION TRIGGERS**: Look for specific triggers that require immediate or scheduled actions

            **KEY ELEMENTS TO ANALYZE DEEPLY:**
            - **Clinical Content**: Diagnostic results, treatment recommendations, progress notes, specialist findings
            - **Authorization Status**: RFA approvals/denials, UR decisions, IMR outcomes, peer-to-peer requests
            - **Legal/Admin Content**: Attorney correspondence, adjuster communications, QME notifications, compliance requirements
            - **Urgency Indicators**: STAT results, legal deadlines, authorization expirations, time-sensitive clinical findings
            - **Patient Context**: Specific patient conditions, treatment history, current care plan

            ---

            ## 🏢 CORE FOUR DEPARTMENTS

            You MUST route all tasks to exactly one of these four departments:

            1. **Medical/Clinical Department**
            - Purpose: Handles all documents directly impacting patient care and physician review.
            - **Deep Analysis Triggers**: Clinical findings, diagnostic results, treatment recommendations, progress assessments, medication management
            - Typical tasks: "Review MRI findings and update treatment plan", "Implement specialist recommendations", "Assess lab results for clinical significance"
            - End-user: Physician / MA team.

            2. **Scheduling & Coordination Department**  
            - Purpose: Owns all authorization-to-appointment logistics.
            - **Deep Analysis Triggers**: Authorization approvals, appointment needs, referral requirements, facility coordination
            - Typical tasks: "Schedule authorized MRI for next available slot", "Coordinate specialist consult per referral", "Confirm therapy appointment completion"
            - End-user: Scheduling staff / case coordinator.

            3. **Administrative/Compliance Department**
            - Purpose: Oversees case documentation, legal communication, and record management.
            - **Deep Analysis Triggers**: Legal documents, attorney communications, compliance notices, QME administrative requirements
            - Typical tasks: "Review attorney correspondence and prepare response", "Update QME appointment in tracking system", "Verify document distribution compliance"
            - End-user: Admin / compliance lead / case manager.

            4. **Authorizations & Denials Department**
            - Purpose: Manages RFA submissions, UR responses, and appeals.
            - **Deep Analysis Triggers**: RFA submissions, UR decisions, IMR processes, authorization denials, peer-to-peer requests
            - Typical tasks: "Prepare IMR appeal for denied physical therapy", "Track UR response deadline for RFA submission", "Generate rebuttal for partial authorization"
            - End-user: RFA coordinator / physician liaison.

            ---

            ## 🎯 CONTEXT-AWARE DEPARTMENT ROUTING

            **Route to Medical/Clinical when DEEP ANALYSIS reveals:**
            - Clinical findings requiring physician interpretation (abnormal labs, imaging results)
            - Treatment recommendations from specialists
            - Progress reports needing clinical assessment
            - Medication management or refill requests
            - Functional capacity evaluations

            **Route to Scheduling & Coordination when DEEP ANALYSIS reveals:**
            - Authorization approvals that require immediate scheduling
            - Referral letters needing appointment coordination
            - Diagnostic study approvals (MRI, CT, X-ray)
            - Therapy or procedure scheduling requirements

            **Route to Administrative/Compliance when DEEP ANALYSIS reveals:**
            - Legal documents from attorneys or courts
            - Adjuster communications requiring record updates
            - QME/AME administrative notifications
            - Compliance or regulatory documentation

            **Route to Authorizations & Denials when DEEP ANALYSIS reveals:**
            - RFA submissions needing tracking
            - UR denials requiring appeals
            - IMR processes needing management
            - Peer-to-peer coordination requests

            ---

            ## ⚡ INTELLIGENT TASK GENERATION

            **BASED ON DEEP DOCUMENT ANALYSIS:**

            - **Extract Specific Details**: Use actual patient names, dates, procedures, and findings from the document
            - **Identify Urgency**: Set appropriate due dates based on clinical urgency, legal deadlines, or authorization expirations
            - **Create Actionable Tasks**: Generate tasks that clearly state what needs to be done, by whom, and by when
            - **Maintain Consistency**: Similar document content should generate similar task descriptions

            **EXAMPLES OF DEEP ANALYSIS DRIVEN TASKS:**

            - Instead of "Review lab results" → "Review elevated liver enzymes (ALT 150) from 2024-01-15 labs and assess need for follow-up"
            - Instead of "Schedule appointment" → "Schedule authorized orthopedic consult for knee pain within next 7 days"
            - Instead of "Handle legal document" → "Review applicant attorney correspondence regarding deposition request and prepare response by 2024-01-20"
            - Instead of "Process RFA" → "Track RFA for physical therapy submitted 2024-01-10 and await UR decision by 2024-01-17"

            ---

            ## 🕓 CONTEXT-BASED DUE DATES

            Set due dates based on DEEP ANALYSIS of document content:

            | Context Identified | Due Date | Reasoning |
            |-------------------|----------|-----------|
            | Critical/STAT results | Same day | Immediate clinical attention required |
            | Urgent clinical findings | +1 day | Prompt medical review needed |
            | Authorization with imminent expiry | Before expiry date | Prevent authorization lapse |
            | Legal deadlines | 3 days before deadline | Allow processing time |
            | UR denial responses | +3 days | Meet 5-business day requirement |
            | Routine clinical review | +2 days | Standard clinical workflow |
            | Standard scheduling | +2 days | Prompt patient service |
            | Administrative tasks | +3 days | Standard processing timeline |

            ---

            ## ✅ OUTPUT REQUIREMENTS

            ```json
            {{
            "tasks": [
                {{
                "description": "Specific task based on deep document analysis should be precisely described (should not be more than 5-10 words)  but using actual details from the document and clear action items",
                "department": "One of the four core departments",
                "status": "Pending",
                "due_date": "YYYY-MM-DD (based on content urgency)",
                "patient": "Patient name from document",
                "actions": ["Claim", "Complete"],
                "source_document": "{source_document}",
                "quickNotes": {{
                    "details": "Brief context explaining WHY this task is needed based on document content",
                    "one_line_note": "Concise summary for dashboard display"
                }}
                }}
            ]
            }}
            ```

            **GENERATION RULES:**
            - Generate 1-3 tasks MAXIMUM
            - Focus on MOST CRITICAL next steps identified through deep analysis
            - Each task must be directly supported by document content
            - Use specific details from the document in task descriptions
            - Ensure department assignment aligns with workflow ownership

            **REMEMBER: Your tasks should reflect a DEEP UNDERSTANDING of the document content, not just superficial categorization.**
            """
    def create_prompt(self, patient_name: str, source_document: str) -> ChatPromptTemplate:
        user_template = """
        DOCUMENT TYPE: {document_type}
        DOCUMENT ANALYSIS:
        {document_analysis}

        SOURCE DOCUMENT: {source_document}
        TODAY'S DATE: {current_date}
        PATIENT: {patient_name}

        Analyze this document and generate 1-3 specific, actionable tasks routed to the appropriate department.

        Key considerations:
        - What is the most critical next step for patient care?
        - Which department is best equipped to handle this task?
        - What is the appropriate timeframe for completion?
        - How can we ensure consistent task descriptions for similar documents?

        {format_instructions}
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template),
        ])

    async def generate_tasks(self, document_analysis: dict, source_document: str = "") -> list[dict]:
        """Generate AI-driven tasks based on document analysis using only the four core departments."""
        try:
            print(document_analysis,'document_analysis')
            current_date = datetime.now()
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            
            print(f"📝 Creating AI-driven tasks for document: {source_document} ({document_type})")

            # Create and run the chain - let LLM handle all routing decisions
            prompt = self.create_prompt(patient_name, source_document)
            chain = prompt | self.llm | self.parser

            result = chain.invoke({
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date.strftime("%Y-%m-%d"),
                "source_document": source_document or "Unknown",
                "document_type": document_type,
                "patient_name": patient_name,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Normalize result
            if isinstance(result, dict):
                tasks_data = result
            else:
                try:
                    tasks_data = result.dict()
                except Exception:
                    try:
                        tasks_data = dict(result)
                    except Exception:
                        tasks_data = {"tasks": []}

            tasks = tasks_data.get("tasks", [])
            
            # Validate department routing
            valid_departments = [
                "Medical/Clinical", 
                "Scheduling & Coordination", 
                "Administrative/Compliance", 
                "Authorizations & Denials"
            ]
            
            validated_tasks = []
            for task in tasks:
                # Ensure department is one of the four core departments
                if task.get("department") not in valid_departments:
                    # Let LLM re-route if invalid department
                    task["department"] = await self._determine_correct_department(task, document_analysis)
                
                validated_tasks.append(task)

            logger.info(f"✅ Generated {len(validated_tasks)} AI-driven tasks for patient: {patient_name}")

            # Database operations for analytics
            await self._update_workflow_analytics(validated_tasks)

            return validated_tasks

        except Exception as e:
            logger.error(f"❌ AI task creation failed: {str(e)}")
            
            # Fallback to basic task with Medical/Clinical as default
            current_date = datetime.now()
            due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
            
            return [{
                "description": f"Review {document_analysis.get('document_type', 'document')} and determine appropriate action",
                "department": "Medical/Clinical",
                "status": "Pending",
                "due_date": due_date,
                "patient": document_analysis.get("patient_name", "Unknown"),
                "actions": ["Claim", "Complete"],
                "source_document": source_document or "Unknown",
                "quickNotes": {
                    "details": "Fallback task after system error",
                    "one_line_note": "Review document and route appropriately"
                }
            }]

    async def _determine_correct_department(self, task: dict, document_analysis: dict) -> str:
        """Use LLM to determine correct department for misrouted tasks."""
        try:
            correction_prompt = """
            Based on the task description and document analysis, route this task to the correct department.
            
            Available Departments:
            - Medical/Clinical: Clinical review, treatment decisions, medical findings
            - Scheduling & Coordination: Appointment scheduling, referral coordination
            - Administrative/Compliance: Legal documents, compliance, administrative tasks  
            - Authorizations & Denials: RFA, UR, IMR, insurance authorizations

            Task: {task_description}
            Document Type: {document_type}
            
            Return ONLY the department name, nothing else.
            """
            
            prompt = ChatPromptTemplate.from_template(correction_prompt)
            chain = prompt | self.llm
            
            response = await chain.ainvoke({
                "task_description": task.get("description", ""),
                "document_type": document_analysis.get("document_type", "")
            })
            
            department = response.content.strip()
            valid_departments = [
                "Medical/Clinical", 
                "Scheduling & Coordination", 
                "Administrative/Compliance", 
                "Authorizations & Denials"
            ]
            
            return department if department in valid_departments else "Medical/Clinical"
            
        except Exception:
            return "Medical/Clinical"

    async def _update_workflow_analytics(self, tasks: list[dict]):
        """Update workflow analytics based on generated tasks."""
        try:
            db = DatabaseService()
            await db.connect()

            for task in tasks:
                department = task.get("department", "").lower()
                description = task.get("description", "").lower()

                # Map departments to analytics categories
                if "medical" in department or "clinical" in department:
                    await db.increment_workflow_stat("clinicalReviews")
                elif "scheduling" in department or "coordination" in department:
                    await db.increment_workflow_stat("schedulingTasks")
                elif "administrative" in department or "compliance" in department:
                    await db.increment_workflow_stat("adminTasks")
                elif "authorization" in department or "denial" in department:
                    await db.increment_workflow_stat("authTasks")

                # Additional specific tracking
                if "rfa" in description or "ur" in description or "imr" in description:
                    await db.increment_workflow_stat("rfasMonitored")
                elif "qme" in description or "ime" in description:
                    await db.increment_workflow_stat("qmeUpdating")
                elif "attorney" in description or "legal" in description:
                    await db.increment_workflow_stat("legalDocs")

            await db.disconnect()

        except Exception as e:
            logger.error(f"❌ Analytics update failed: {str(e)}")