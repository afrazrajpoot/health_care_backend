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
    department: str
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

        ## ⚕️ CORE CONTEXT

        You handle two major modes:
        - **Workers' Compensation (WC)**
        - **General Medicine (GM)**

        Your job is to:
        1. Read the `document_analysis` text.
        2. Infer **what step** in the patient care lifecycle it represents.
        3. Generate up to **3 next-step tasks** (short, precise, and department-specific).
        4. Each task must mirror *real-world medical office workflows* in California.

        ---

        ## 🎯 CONSISTENCY RULES

        **CRITICAL: For the same document type, generate CONSISTENT task descriptions**

        ### Document-Type → Task Mapping (MUST FOLLOW):

        - **MRI Report / Imaging Study** → ALWAYS: "Review imaging findings and update treatment plan"
        - **Lab Results** → ALWAYS: "Review lab results and assess clinical significance"  
        - **Consult Note** → ALWAYS: "Implement specialist recommendations"
        - **Progress Note** → ALWAYS: "Physician follow-up based on progress findings"
        - **RFA Submission** → ALWAYS: "Track RFA & await UR decision"
        - **UR Approval** → ALWAYS: "Schedule authorized service"
        - **UR Denial** → ALWAYS: "Prepare IMR appeal packet"
        - **Attorney Letter** → ALWAYS: "Review legal correspondence and respond as needed"
        - **Referral Letter** → ALWAYS: "Review referral & schedule consult"
        - **Prior Auth** → ALWAYS: "Track prior auth approval"
        - **QME/AME Report** → ALWAYS: "Review findings & update treatment plan"
        - **Billing Document** → ALWAYS: "Review billing and reconcile as needed"

        ### Department Consistency:
        - Imaging/Lab Reports → "Physician Review"
        - Scheduling Tasks → "Scheduling" 
        - Legal Documents → "Admin/Legal"
        - RFA/UR/IMR → "RFA/IMR"
        - Referrals → "Referrals / Coordination"
        - Billing → "Billing/Compliance"

        **DO NOT create variations for the same document type. Use EXACT task descriptions above.**

        ---

        ## 🩻 WORKERS' COMP LIFECYCLE (CA FLOW)

        | Step | Event / Document | Task Description | Department |
        |------|------------------|------------------|-------------|
        | 1️⃣ | DWC 5021 (Doctor's First Report) | Chart prep & initial scheduling | Intake / Scheduling |
        | 2️⃣ | PR-1 (Initial PTP Report) | Physician Review — Establish PTP relationship | Physician Review |
        | 3️⃣ | RFA submitted | Track RFA & await UR decision | RFA/IMR |
        | 4️⃣ | UR Decision: Approved | Schedule authorized service | Scheduling |
        |    | UR Decision: Denied | Prepare IMR appeal packet | RFA/IMR |
        | 5️⃣ | PR-2 (Progress Report) | Physician follow-up based on progress findings | Physician Review |
        | 6️⃣ | Stalled improvement | Request FCE or Specialist Consult | Physician Review |
        | 7️⃣ | Diagnostic Auth | Schedule diagnostic study | Scheduling |
        | 8️⃣ | Specialist Consult | Implement specialist recommendations | Physician Review |
        | 9️⃣ | PR-3 / PR-4 | Determine P&S / MMI status | Physician Review |
        | 🔟 | QME / AME Report | Review findings & update treatment plan | Physician Review |
        | 1️⃣1️⃣ | IMR Appeal Outcome | Update case per IMR decision | RFA/IMR | Physician Review |
        | 1️⃣2️⃣ | Legal / Attorney Letter | Review legal correspondence and respond as needed | Admin/Legal |
        | 1️⃣3️⃣ | EOR / Billing Docs | Review billing and reconcile as needed | Billing/Compliance |

        ---

        ## 🏥 GENERAL MEDICINE (GM) FLOW

        | Document | Task Description | Department |
        |-----------|----------------|-------------|
        | Referral Letter | Review referral & schedule consult | Referrals / Coordination |
        | Prior Auth Form | Track prior auth approval | Prior Authorization |
        | Approval Notice | Schedule authorized service | Scheduling |
        | Lab or Imaging Report | Review results & update chart | Physician Review |
        | Progress Note | Physician follow-up based on progress findings | Physician Review |
        | Patient Outreach or Missed Appointment | Call patient / reschedule | Patient Outreach |
        | Compliance Notice | Review & log compliance task | Quality & Compliance |
        | Billing Statement | Review billing and reconcile as needed | Billing / Revenue Cycle |

        ---

        ## ⚙️ CONDITIONAL LOGIC & CONTEXTUAL RULES

        - **UR / RFA / IMR FLOW**
        - If "RFA submitted" → create "Track RFA & await UR decision".
        - If UR says "Approved" → create "Schedule authorized service".
        - If UR says "Denied" → create "Prepare IMR appeal packet".

        - **Attorney Letters**
        - "Applicant Attorney" → "Review legal correspondence and respond as needed" + optional "Acknowledge in writing".
        - "Defense Attorney" → "Review legal correspondence and respond as needed" (FYI only).

        - **Adjuster or Nurse Case Manager**
        - "Review adjuster communication" (Admin/Legal).
        - If it includes approval → also create "Schedule service" (Scheduling).
        - "Review NCM notes & confirm plan" (due 2 days).

        - **Diagnostic / Consult Authorization**
        - "Schedule diagnostic study" or "Schedule specialist consult".

        - **P2P Request**
        - Step 1: "Verify & forward written P2P request" (1 day)
        - Step 2: "Physician drafts written response" (2 days)
        - Step 3: "Confirm submission & log receipt" (3 days)

        ---

        ## 🕓 PRIORITY & DUE DATES

        | Department | Default Due |
        |-------------|-------------|
        | Physician Review | +2 days |
        | Scheduling | +2 days |
        | RFA/IMR | +5 days |
        | Admin/Legal | +3 days |
        | Intake | same-day |
        | Billing | +3 days |
        | Referrals / Coordination | +2 days |

        ---

        ## 🧩 OUTPUT RULES

        - Always produce **valid JSON**.
        - Limit to **1–3 tasks**.
        - Tasks must be **consistent**, **context-aware**, and **department-routed**.
        - Description must use EXACT predefined descriptions for document types.
        - Each task includes a short `details` and a concise `one_line_note` for dashboard display.

        ---

        ## ✅ OUTPUT FORMAT

        ```json
        {{
        "tasks": [
            {{
            "description": "Exact predefined task description for document type",
            "department": "Relevant department",
            "status": "Pending",
            "due_date": "YYYY-MM-DD",
            "patient": "Patient name",
            "actions": ["Claim", "Complete"],
            "source_document": "{source_document}",
            "quickNotes": {{
                "details": "Brief explanation or context",
                "one_line_note": "Short dashboard note"
            }}
            }}
        ]
        }}
        ```
        """

    def _get_consistent_task_template(self, document_type: str, content: str = "") -> dict:
        """Return consistent task templates for specific document types."""
        document_type_lower = document_type.lower()
        content_lower = content.lower()
        
        # MRI/Imaging consistency
        if any(term in document_type_lower or term in content_lower for term in ['mri', 'imaging', 'radiology', 'x-ray', 'ct', 'scan']):
            return {
                "description": "Review imaging findings and update treatment plan",
                "department": "Physician Review",
                "due_days": 2
            }
        
        # Lab results consistency
        elif any(term in document_type_lower or term in content_lower for term in ['lab', 'blood', 'test', 'result', 'chemistry', 'hematology']):
            return {
                "description": "Review lab results and assess clinical significance",
                "department": "Physician Review", 
                "due_days": 2
            }
        
        # RFA consistency
        elif 'rfa' in document_type_lower or 'rfa' in content_lower:
            if 'denied' in content_lower or 'denial' in content_lower:
                return {
                    "description": "Prepare IMR appeal packet",
                    "department": "RFA/IMR",
                    "due_days": 5
                }
            elif 'approved' in content_lower or 'approval' in content_lower:
                return {
                    "description": "Schedule authorized service",
                    "department": "Scheduling",
                    "due_days": 2
                }
            else:
                return {
                    "description": "Track RFA & await UR decision",
                    "department": "RFA/IMR",
                    "due_days": 5
                }
        
        # Progress reports consistency
        elif any(term in document_type_lower for term in ['pr-', 'progress', 'follow-up']):
            return {
                "description": "Physician follow-up based on progress findings",
                "department": "Physician Review",
                "due_days": 2
            }
        
        # Attorney letters consistency
        elif any(term in document_type_lower or term in content_lower for term in ['attorney', 'legal', 'lawyer', 'counsel']):
            return {
                "description": "Review legal correspondence and respond as needed",
                "department": "Admin/Legal", 
                "due_days": 3
            }
        
        # Referral consistency
        elif any(term in document_type_lower or term in content_lower for term in ['referral', 'consult', 'specialist']):
            return {
                "description": "Review referral & schedule consult",
                "department": "Referrals / Coordination",
                "due_days": 2
            }
        
        # QME/AME consistency
        elif any(term in document_type_lower or term in content_lower for term in ['qme', 'ame', 'independent medical']):
            return {
                "description": "Review findings & update treatment plan",
                "department": "Physician Review",
                "due_days": 2
            }
        
        # Billing consistency
        elif any(term in document_type_lower or term in content_lower for term in ['billing', 'invoice', 'payment', 'eob', 'eor']):
            return {
                "description": "Review billing and reconcile as needed",
                "department": "Billing/Compliance",
                "due_days": 3
            }
        
        # Prior auth consistency
        elif any(term in document_type_lower or term in content_lower for term in ['prior auth', 'authorization', 'pre-cert']):
            return {
                "description": "Track prior auth approval",
                "department": "Prior Authorization",
                "due_days": 3
            }
        
        # Default fallback
        return {
            "description": "Review document and determine next steps",
            "department": "Physician Review",
            "due_days": 2
        }

    def create_prompt(self, consistent_template: dict, patient_name: str, source_document: str) -> ChatPromptTemplate:
        user_template = """
        DOCUMENT TYPE: {document_type}
        DOCUMENT ANALYSIS:
        {document_analysis}

        MODE: {mode}
        SOURCE DOCUMENT: {source_document}
        TODAY'S DATE: {current_date}
        PATIENT: {patient_name}

        **CONSISTENCY REQUIREMENTS:**
        - PRIMARY TASK MUST BE: "{consistent_description}"
        - DEPARTMENT MUST BE: "{consistent_department}"
        - DUE IN: {consistent_due_days} days

        Generate 1-3 tasks (preferably 1) using the primary task above as the main task.
        You may add additional tasks ONLY if clearly warranted by the document content.

        {{
            "tasks": [
                {{
                    "description": "{consistent_description}",
                    "department": "{consistent_department}",
                    "status": "Pending",
                    "due_date": "{consistent_due_date}",
                    "patient": "{patient_name}",
                    "actions": ["Claim", "Complete"],
                    "source_document": "{source_document}",
                    "quickNotes": {{
                        "details": "Brief context from document analysis",
                        "one_line_note": "Short note for dashboard"
                    }}
                }}
            ]
        }}

        {format_instructions}
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(user_template),
        ])

    async def generate_tasks(self, document_analysis: dict, source_document: str = "") -> list[dict]:
        """Generate consistent AI-driven tasks based on document analysis."""
        try:
            current_date = datetime.now()
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            
            print(f"📝 Creating consistent tasks for document: {source_document} ({document_type})")

            # Get text content for analysis
            text_content = " ".join(
                str(v) for v in document_analysis.values() if isinstance(v, (str, list))
            ).lower()

            # Get consistent task template FIRST
            consistent_template = self._get_consistent_task_template(document_type, text_content)
            
            # Calculate due date
            due_date = (current_date + timedelta(days=consistent_template["due_days"])).strftime("%Y-%m-%d")
            
            # Determine mode
            wc_keywords = ["rfa", "ur denial", "attorney", "qme", "workers", "claim", "pr-"]
            gm_keywords = ["prior auth", "referral", "consult", "intake", "specialist"]

            if any(k in text_content for k in wc_keywords):
                mode = "WC"
            elif any(k in text_content for k in gm_keywords):
                mode = "GM"
            else:
                mode = "WC" if "pr-" in document_type.lower() else "GM"

            # Create and run the chain with consistency enforcement
            prompt = self.create_prompt(consistent_template, patient_name, source_document)
            chain = prompt | self.llm | self.parser

            result = chain.invoke({
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date.strftime("%Y-%m-%d"),
                "source_document": source_document or "Unknown",
                "mode": mode,
                "document_type": document_type,
                "patient_name": patient_name,
                "consistent_description": consistent_template["description"],
                "consistent_department": consistent_template["department"],
                "consistent_due_days": consistent_template["due_days"],
                "consistent_due_date": due_date,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Normalize result
            if isinstance(result, dict):
                out = result
            else:
                try:
                    out = result.dict()
                except Exception:
                    try:
                        out = dict(result)
                    except Exception:
                        out = {}

            # Process tasks
            ai_tasks = out.get("tasks", [])
            if not isinstance(ai_tasks, list):
                ai_tasks = [ai_tasks] if ai_tasks else []

            tasks = []
            for t in ai_tasks:
                if not isinstance(t, dict):
                    continue

                # Use consistent template as base, allow AI to enhance quickNotes
                task = {
                    "description": consistent_template["description"],  # Always use consistent description
                    "department": consistent_template["department"],    # Always use consistent department
                    "status": t.get("status", "Pending"),
                    "due_date": due_date,  # Use calculated due date
                    "patient": patient_name,
                    "actions": t.get("actions", ["Claim", "Complete"]),
                    "source_document": source_document or "Unknown",
                    "quickNotes": t.get("quickNotes", {
                        "details": f"AI generated follow-up task for {document_type}",
                        "one_line_note": consistent_template["description"]
                    })
                }

                # Ensure due_date is valid datetime
                try:
                    task["due_date"] = datetime.strptime(task["due_date"], "%Y-%m-%d")
                except Exception:
                    task["due_date"] = current_date + timedelta(days=consistent_template["due_days"])

                tasks.append(task)

            # Fallback if AI failed
            if not tasks:
                tasks = [{
                    "description": consistent_template["description"],
                    "department": consistent_template["department"],
                    "status": "Pending",
                    "due_date": current_date + timedelta(days=consistent_template["due_days"]),
                    "patient": patient_name,
                    "actions": ["Claim", "Complete"],
                    "source_document": source_document or "Unknown",
                    "quickNotes": {
                        "details": f"Consistent task for {document_type}",
                        "one_line_note": consistent_template["description"]
                    }
                }]

            logger.info(f"✅ Generated {len(tasks)} consistent task(s) for patient: {patient_name}")

            # Database operations
            db = DatabaseService()
            await db.connect()

            for t in tasks:
                desc = t["description"].lower()
                dept = t["department"].lower()

                if "referral" in desc or "consult" in desc:
                    await db.increment_workflow_stat("referralsProcessed")
                elif "rfa" in desc or "ur decision" in desc:
                    await db.increment_workflow_stat("rfasMonitored")
                elif "qme" in desc or "ime" in desc:
                    await db.increment_workflow_stat("qmeUpcoming")
                elif "dispute" in desc or "appeal" in desc or "claim" in desc:
                    await db.increment_workflow_stat("payerDisputes")
                else:
                    await db.increment_workflow_stat("externalDocs")

            await db.disconnect()

            return tasks

        except Exception as e:
            logger.error(f"❌ Consistent task creation failed: {str(e)}")
            
            # Even in error, return consistent task
            current_date = datetime.now()
            document_type = document_analysis.get("document_type", "Unknown")
            patient_name = document_analysis.get("patient_name", "Unknown")
            
            consistent_template = self._get_consistent_task_template(document_type)
            
            return [{
                "description": consistent_template["description"],
                "department": consistent_template["department"],
                "status": "Pending",
                "due_date": current_date + timedelta(days=consistent_template["due_days"]),
                "patient": patient_name,
                "actions": ["Claim", "Complete"],
                "source_document": source_document or "Unknown",
                "quickNotes": {
                    "details": "Fallback consistent task after error",
                    "one_line_note": consistent_template["description"]
                }
            }]