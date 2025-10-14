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
    """AI service to generate ONE most possible task based on document type and content."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=CONFIG.get("openai_api_key"),
            temperature=0.2,
            timeout=90,
        )
        self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)

    # 🧠 SYSTEM PROMPT - Updated: actions default to ["Unclaim", "Pending"]
    SYSTEM_PROMPT = """
    You are a workflow automation AI for a California multi-specialty medical group handling
    Workers' Compensation (WC) and General Medicine (GM) cases.

    You create ONE most important, actionable follow-up task based on the document content.

    --- KEY RULES ---
    1. Analyze the document and select the SINGLE most critical task that needs to be done.
    2. Choose the task that is most urgent, highest priority, or most relevant to the document type.
    3. Consider what is the most important next step in the workflow process.
    4. If multiple tasks seem equally important, choose the one that is most time-sensitive.

    COMMON PRIORITY TASKS:
    • Clinical Reports: "Physician to review [document type] findings"
    • RFAs & Authorizations: "Process authorization for [treatment]"
    • Consult Reports: "Implement specialist recommendations"
    • Legal Documents: "Attorney liaison review required"
    • Urgent/Time-sensitive items always take priority

    DEPARTMENTS (by mode):
    Workers' Comp (WC):
        ["Scheduling","RFA/IMR","Physician Review","Intake","Admin/Compliance","Billing/Compliance","Legal/Attorney Liaison"]
    General Medicine (GM):
        ["Scheduling","Prior Authorization","Physician Review","Intake/Registration","Referrals/Coordination","Billing/Revenue Cycle","Quality & Compliance","Patient Outreach"]

    When generating the task, the 'actions' field must always start as ["Unclaim", "Pending"] 
    for new tasks, not ["Claim", "Complete"].

    Output only ONE task in structured JSON format.
    """

    def create_prompt(self) -> ChatPromptTemplate:
        user_template = """
        DOCUMENT TYPE: {document_type}
        DOCUMENT ANALYSIS:
        {document_analysis}

        MODE: {mode}
        SOURCE DOCUMENT: {source_document}
        TODAY'S DATE: {current_date}

        Generate ONLY ONE most important task in this format:

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

    def generate_tasks(self, document_analysis: dict, source_document: str = "") -> list[dict]:
        """Generate ONE most possible task - default actions are Unclaim and Pending."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            print(f"📝 Creating ONE task for document: {source_document}")
            
            text_content = " ".join(
                str(v) for v in document_analysis.values() if isinstance(v, (str, list))
            ).lower()

            # Detect mode
            wc_keywords = ["rfa", "ur denial", "attorney", "qme", "workers", "claim"]
            gm_keywords = ["prior auth", "referral", "consult", "intake", "specialist"]

            if any(k in text_content for k in wc_keywords):
                mode = "WC"
            elif any(k in text_content for k in gm_keywords):
                mode = "GM"
            else:
                mode = "WC" if "pr-" in document_analysis.get("document_type", "").lower() else "GM"

            # AI-driven single task generation
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

            # Process the single AI task
            single_task = result["tasks"][0] if result["tasks"] else {
                "description": "Review document content",
                "department": "Physician Review",
                "status": "Pending", 
                "due_date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "patient": document_analysis.get("patient_name", "Unknown"),
                "actions": ["Unclaim", "Pending"],
                "source_document": source_document or "Unknown"
            }

            # Ensure due_date is datetime object
            try:
                single_task["due_date"] = datetime.strptime(single_task["due_date"], "%Y-%m-%d")
            except Exception:
                single_task["due_date"] = datetime.now() + timedelta(days=2)

            logger.info(f"✅ Generated ONE task for patient: {document_analysis.get('patient_name', 'Unknown')}")
            return [single_task]

        except Exception as e:
            logger.error(f"❌ Single task creation failed: {str(e)}")
            # Fallback single task
            return [{
                "description": "Review and process document",
                "department": "Physician Review",
                "status": "Pending",
                "due_date": datetime.now() + timedelta(days=2),
                "patient": document_analysis.get("patient_name", "Unknown"),
                "actions": ["Unclaim", "Pending"],
                "source_document": source_document or "Unknown"
            }]
