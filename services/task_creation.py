from langchain_openai import ChatOpenAI
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
    """AI service to generate ONE most possible task based on document type and content."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.2,
            timeout=90,
        )
        self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)

    SYSTEM_PROMPT = """
        You are a medical workflow automation AI for a California multi-specialty group.
        You analyze medical, legal, and administrative documents to generate **specific, actionable follow-up tasks** — not summaries.

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

        ## 🩻 WORKERS’ COMP LIFECYCLE (CA FLOW)

        | Step | Event / Document | Task Description | Department |
        |------|------------------|------------------|-------------|
        | 1️⃣ | DWC 5021 (Doctor’s First Report) | Chart prep & initial scheduling | Intake / Scheduling |
        | 2️⃣ | PR-1 (Initial PTP Report) | Physician Review — Establish PTP relationship | Physician Review |
        | 3️⃣ | RFA submitted | Track RFA & await UR decision | RFA/IMR |
        | 4️⃣ | UR Decision: Approved | Schedule authorized service | Scheduling |
        |    | UR Decision: Denied | Prepare IMR appeal packet | RFA/IMR |
        | 5️⃣ | PR-2 (Progress Report) | Physician Review — Progress findings | Physician Review |
        | 6️⃣ | Stalled improvement | Request FCE or Specialist Consult | Physician Review |
        | 7️⃣ | Diagnostic Auth | Schedule diagnostic study | Scheduling |
        | 8️⃣ | Specialist Consult | Implement specialist recommendations | Physician Review |
        | 9️⃣ | PR-3 / PR-4 | Determine P&S / MMI status | Physician Review |
        | 🔟 | QME / AME Report | Review findings & update plan | Physician Review |
        | 1️⃣1️⃣ | IMR Appeal Outcome | Update case per IMR decision | RFA/IMR | Physician Review |
        | 1️⃣2️⃣ | Legal / Attorney Letter | Review correspondence | Admin/Legal |
        | 1️⃣3️⃣ | EOR / Billing Docs | Billing Review / Appeal as needed | Billing/Compliance |

        ---

        ## 🏥 GENERAL MEDICINE (GM) FLOW

        | Document | Example Tasks | Department |
        |-----------|----------------|-------------|
        | Referral Letter | Review referral & schedule consult | Referrals / Coordination |
        | Prior Auth Form | Track prior auth approval | Prior Authorization |
        | Approval Notice | Schedule authorized service | Scheduling |
        | Lab or Imaging Report | Review results & update chart | Physician Review |
        | Progress Note | Physician follow-up or referral review | Physician Review |
        | Patient Outreach or Missed Appointment | Call patient / reschedule | Patient Outreach |
        | Compliance Notice | Review & log compliance task | Quality & Compliance |
        | Billing Statement | Review charge or reconcile denial | Billing / Revenue Cycle |

        ---

        ## ⚙️ CONDITIONAL LOGIC & CONTEXTUAL RULES

        - **UR / RFA / IMR FLOW**
        - If “RFA submitted” → create “Track RFA & await UR decision”.
        - If UR says “Approved” → create “Schedule authorized service”.
        - If UR says “Denied” → create “Prepare IMR appeal packet”.

        - **Attorney Letters**
        - “Applicant Attorney” → “Physician Review — Attorney Letter” + optional “Acknowledge in writing”.
        - “Defense Attorney” → “Physician Review — Defense Correspondence” (FYI only).

        - **Adjuster or Nurse Case Manager**
        - “Review adjuster communication” (Admin/Legal).
        - If it includes approval → also create “Schedule service” (Scheduling).
        - “Review NCM notes & confirm plan” (due 2 days).

        - **Diagnostic / Consult Authorization**
        - “Schedule diagnostic study” or “Schedule specialist consult”.

        - **P2P Request**
        - Step 1: “Verify & forward written P2P request” (1 day)
        - Step 2: “Physician drafts written response” (2 days)
        - Step 3: “Confirm submission & log receipt” (3 days)

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

        ---

        ## 🧩 OUTPUT RULES

        - Always produce **valid JSON**.
        - Limit to **1–3 tasks**.
        - Tasks must be **unique**, **context-aware**, and **department-routed**.
        - Description must be a clear one-liner.
        - Each task includes a short `details` and a concise `one_line_note` for dashboard display.

        ---

        ## ✅ OUTPUT FORMAT

        ```json
        {{
        "tasks": [
            {{
            "description": "Specific one-line actionable step",
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


    def create_prompt(self) -> ChatPromptTemplate:
        user_template = """
        DOCUMENT TYPE: {document_type}
        DOCUMENT ANALYSIS:
        {document_analysis}

        MODE: {mode}
        SOURCE DOCUMENT: {source_document}
        TODAY'S DATE: {current_date}

                Generate upto 1-3 (preferably 1) most important task in this format:

                {{
                    "tasks": [
                        {{
                            "description": "Single most important task description",
                            "department": "Most relevant department",
                            "status": "Pending",
                            "due_date": "YYYY-MM-DD",
                            "patient": "Patient name",
                            "actions": ["Unclaim", "Pending"],
                            "source_document": "{source_document}"
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
        """Generate one or more AI-driven tasks based on document analysis."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            patient_name = document_analysis.get("patient_name", "Unknown")
            document_type = document_analysis.get("document_type", "Unknown")
            print(f"📝 Creating tasks for document: {source_document} ({document_type})")

            # 🔍 Combine all text content for keyword mode detection
            text_content = " ".join(
                str(v) for v in document_analysis.values() if isinstance(v, (str, list))
            ).lower()

            wc_keywords = ["rfa", "ur denial", "attorney", "qme", "workers", "claim"]
            gm_keywords = ["prior auth", "referral", "consult", "intake", "specialist"]

            if any(k in text_content for k in wc_keywords):
                mode = "WC"
            elif any(k in text_content for k in gm_keywords):
                mode = "GM"
            else:
                mode = "WC" if "pr-" in document_type.lower() else "GM"

            # 🔗 Build and run the chain
            prompt = self.create_prompt()
            chain = prompt | self.llm | self.parser

            result = chain.invoke({
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date,
                "source_document": source_document or "Unknown",
                "mode": mode,
                "document_type": document_type,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Normalize result: some parsers return a dict, others return pydantic objects
            if isinstance(result, dict):
                out = result
            else:
                try:
                    out = result.dict()
                except Exception:
                    # Fallback: try treating as mapping-like
                    try:
                        out = dict(result)
                    except Exception:
                        out = {}

            # 🧩 Handle result with multiple tasks
            ai_tasks = out.get("tasks", [])
            if not isinstance(ai_tasks, list):
                ai_tasks = [ai_tasks] if ai_tasks else []

            tasks = []
            for t in ai_tasks:
                if not isinstance(t, dict):
                    continue

                # Default fallbacks for robustness
                task = {
                    "description": t.get("description", "Review document content"),
                    "department": t.get("department", "Physician Review"),
                    "status": t.get("status", "Pending"),
                    "due_date": t.get("due_date", (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")),
                    "patient": patient_name,
                    "actions": t.get("actions", ["Claim", "Complete"]),
                    "source_document": source_document or "Unknown",
                    "quickNotes": t.get("quickNotes", {
                        "details": "AI generated follow-up task.",
                        "one_line_note": t.get("description", "Review document content")
                    })
                }

                # Ensure due_date is valid datetime
                try:
                    task["due_date"] = datetime.strptime(task["due_date"], "%Y-%m-%d")
                except Exception:
                    task["due_date"] = datetime.now() + timedelta(days=2)

                tasks.append(task)

            # 🧠 If AI failed to generate anything
            if not tasks:
                tasks = [{
                    "description": "Review and process document",
                    "department": "Physician Review",
                    "status": "Pending",
                    "due_date": datetime.now() + timedelta(days=2),
                    "patient": patient_name,
                    "actions": ["Claim", "Complete"],
                    "source_document": source_document or "Unknown",
                    "quickNotes": {
                        "details": "Fallback task due to missing AI output.",
                        "one_line_note": "Review document manually."
                    }
                }]

            logger.info(f"✅ Generated {tasks} task(s) for patient: {patient_name}")

            db = DatabaseService()
            # Use DatabaseService.connect which is concurrency-safe and handles Prisma availability
            await db.connect()

            # Classify and increment the right counter(s)
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
            logger.error(f"❌ Task creation failed: {str(e)}")
            return [{
                "description": "Review and process document",
                "department": "Physician Review",
                "status": "Pending",
                "due_date": datetime.now() + timedelta(days=2),
                "patient": document_analysis.get("patient_name", "Unknown"),
                "actions": ["Claim", "Complete"],
                "source_document": source_document or "Unknown",
                "quickNotes": {
                    "details": "Fallback task after error.",
                    "one_line_note": "Manual review required."
                }
            }]
