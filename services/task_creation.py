from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from config.settings import CONFIG
import json

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
    """AI service to generate realistic, department-specific tasks based on document type and content."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.2,
            timeout=90,
        )
        self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)

    # 🧠 SYSTEM PROMPT — defines professional behavior
    SYSTEM_PROMPT = """
    You are a workflow automation AI for a California multi-specialty medical group handling
    Workers’ Compensation (WC) and General Medicine (GM) cases.

    You create precise, department-specific, actionable follow-up tasks **only when appropriate**
    based on the type of document uploaded and its context.

    --- KEY RULES ---
    1. Base task creation on **document type** logic.
       Examples:
       • PR-1 / 5021 → Intake or Admin/Compliance tasks (“Verify claim info”, “Confirm employer data”)
       • PR-2 → Review or RFA tasks *only if new treatment requests appear*
       • RFA → “Submit to UR” or “Track UR response”
       • UR Denial → “Prepare IMR package for physician”
       • Authorization → “Schedule approved MRI” or “Notify patient of approval”
       • Ortho / Neuro Consult → “Physician Review” (doctor should review consult)
       • QME / AME → “Physician Review” and “Send to Attorney”
       • PR-3 / PR-4 → “Finalize MMI report” or “Billing review”
       • IMR Determination → “Implement IMR decision” (Approved / Denied)
       • Supplemental Report → “Update claim records”
       • Attorney Correspondence → “Legal/Attorney Liaison follow-up”
       • Billing / Lien → “Billing/Compliance” or “Legal/Attorney Liaison”
       • FCE → “Physician to review FCE results and update work restrictions”

    2. DO NOT create scheduling tasks (e.g., MRI, surgery) unless:
       - The document is an **Authorization** or **RFA Approval**, NOT a PR-2 or DFR.
    
    3. DO NOT create duplicate or unnecessary tasks.
       - Skip tasks that would already have been triggered by previous reports.

    4. ALWAYS use professional, concise 1-liners.
       - Examples: “Submit UR denial for IMR review”, “Notify patient of MRI approval”, “Physician to review Ortho Consult”.

    5. DEPARTMENTS (by mode)
       Workers’ Comp (WC):
         ["Scheduling","RFA/IMR","Physician Review","Intake","Admin/Compliance","Billing/Compliance","Legal/Attorney Liaison"]
       General Medicine (GM):
         ["Scheduling","Prior Authorization","Physician Review","Intake/Registration","Referrals/Coordination","Billing/Revenue Cycle","Quality & Compliance","Patient Outreach"]

    Output only structured JSON per the schema.
    """

    def create_prompt(self) -> ChatPromptTemplate:
        user_template = """
        DOCUMENT TYPE: {document_type}
        DOCUMENT ANALYSIS:
        {document_analysis}

        MODE: {mode}
        SOURCE DOCUMENT: {source_document}
        TODAY’S DATE: {current_date}

        Return structured JSON strictly matching this format:

        {{
          "tasks": [
            {{
              "description": "...",
              "department": "...",
              "status": "Pending",
              "due_date": "YYYY-MM-DD",
              "patient": "...",
              "quick_notes": {{
                "status_update": "",
                "details": "",
                "one_line_note": ""
              }},
              "actions": ["Claim", "Complete"],
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



    # ----------------------------------------------------------
    # 🔹 2. EXTENDED generate_tasks()
    # ----------------------------------------------------------
    def generate_tasks(self, document_analysis: dict, source_document: str = "") -> list[dict]:
        """Generate structured tasks — combining rule-based + AI-generated ones."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            text_content = " ".join(
                str(v) for v in document_analysis.values() if isinstance(v, (str, list))
            ).lower()

            # --- Step 1: Detect mode automatically ---
            wc_keywords = ["rfa", "ur denial", "attorney", "qme", "workers", "claim"]
            gm_keywords = ["prior auth", "referral", "consult", "intake", "specialist"]

            if any(k in text_content for k in wc_keywords):
                mode = "WC"
            elif any(k in text_content for k in gm_keywords):
                mode = "GM"
            else:
                mode = "WC" if "pr-" in document_analysis.get("document_type", "").lower() else "GM"

            # --- Step 2: Rule-based baseline tasks ---
            baseline_tasks = []
            doc_type = document_analysis.get("document_type", "").lower()
            DOCUMENT_RULES = {
                "pr-2": [
                    {
                        "if_contains": ["pending authorization", "mri", "imaging"],
                        "create_task": "Submit RFA for imaging authorization",
                        "department": "RFA/IMR"
                    },
                    {
                        "if_contains": ["referral", "consult"],
                        "create_task": "Send referral to specialist",
                        "department": "Scheduling"
                    },
                    {
                        "if_contains": ["modified duty", "restrictions"],
                        "create_task": "Notify employer of work restrictions",
                        "department": "Admin/Compliance"
                    }
                ],
                "rfa": [
                    {"if_contains": ["approved"], "create_task": "Schedule approved treatment", "department": "Scheduling"},
                    {"if_contains": ["denied"], "create_task": "Prepare IMR packet", "department": "RFA/IMR"}
                ],
                "ur denial": [
                    {"create_task": "Prepare IMR packet for denied treatment", "department": "RFA/IMR"}
                ],
                "authorization": [
                    {"create_task": "Schedule approved procedure", "department": "Scheduling"}
                ],
                "ortho consult": [
                    {"create_task": "Physician to review Ortho Consult", "department": "Physician Review"}
                ],
                "attorney letter": [
                    {"create_task": "Forward correspondence to legal liaison", "department": "Legal/Attorney Liaison"}
                ],
            }


            for key, rules in DOCUMENT_RULES.items():
                if key in doc_type:
                    for rule in rules:
                        if "if_contains" not in rule or any(term in text_content for term in rule["if_contains"]):
                            baseline_tasks.append({
                                "description": rule["create_task"],
                                "department": rule["department"],
                                "status": "Pending",
                                "due_date": datetime.now() + timedelta(days=2),
                                "patient": document_analysis.get("patient_name", "Unknown"),
                                "quick_notes": {
                                    "status_update": "",
                                    "details": f"Auto-generated from {doc_type.upper()}",
                                    "one_line_note": rule["create_task"]
                                },
                                "actions": ["Claim", "Complete"],
                                "source_document": source_document or "Unknown"
                            })

            # --- Step 3: AI-driven refinement / additions ---
            prompt = self.create_prompt()
            chain = prompt | self.llm | self.parser

            result = chain.invoke({
                "document_analysis": json.dumps(document_analysis, indent=2),
                "current_date": current_date,
                "source_document": source_document or "Unknown",
                "mode": mode,
                "document_type": document_analysis.get("document_type", "Unknown"),
                "format_instructions": self.parser.get_format_instructions()
            })


            ai_tasks = []
            for t in result["tasks"]:
                try:
                    t["due_date"] = datetime.strptime(t["due_date"], "%Y-%m-%d")
                except Exception:
                    t["due_date"] = datetime.now() + timedelta(days=3)
                ai_tasks.append(t)

            # --- Step 4: Merge intelligently (avoid duplicates) ---
            def similar(desc1, desc2):
                return desc1.lower()[:25] in desc2.lower() or desc2.lower()[:25] in desc1.lower()

            merged_tasks = baseline_tasks[:]
            for ai_t in ai_tasks:
                if not any(similar(ai_t["description"], bt["description"]) for bt in baseline_tasks):
                    merged_tasks.append(ai_t)

            logger.info(f"✅ Generated {merged_tasks} tasks for patient: {document_analysis.get('patient_name', 'Unknown')}")
            return merged_tasks

        except Exception as e:
            logger.error(f"❌ Task creation failed: {str(e)}")
            return []
