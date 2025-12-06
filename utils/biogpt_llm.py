# # """
# # BioGPT LLM Wrapper for Medical Task Generation
# # Provides a LangChain-compatible interface for BioGPT model
# # """
# # import logging
# # from typing import Any, List, Optional
# # from langchain_core.language_models.llms import LLM
# # from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# # from transformers import AutoTokenizer, AutoModelForCausalLM, BioGptTokenizer, BioGptForCausalLM
# # import torch

# # logger = logging.getLogger("biogpt")


# # class BioGPTLLM(LLM):
# #     """
# #     BioGPT Language Model for medical text generation.
# #     Optimized for biomedical domain with better accuracy than general LLMs.
# #     """
    
# #     tokenizer: Any = None
# #     model: Any = None
# #     device: str = "cuda" if torch.cuda.is_available() else "cpu"
# #     max_length: int = 512
# #     temperature: float = 0.7
# #     top_p: float = 0.9
    
# #     def __init__(self, **kwargs):
# #         super().__init__(**kwargs)
# #         self._initialize_model()
    
# #     def _initialize_model(self):
# #         """Load BioGPT model and tokenizer"""
# #         try:
# #             logger.info(f"🔬 Initializing BioGPT model on {self.device}...")
            
# #             # Load BioGPT tokenizer and model
# #             self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# #             self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
            
# #             # Move model to appropriate device
# #             self.model.to(self.device)
# #             self.model.eval()  # Set to evaluation mode
            
# #             logger.info(f"✅ BioGPT model loaded successfully on {self.device}")
            
# #         except Exception as e:
# #             logger.error(f"❌ Failed to initialize BioGPT: {e}")
# #             raise
    
# #     @property
# #     def _llm_type(self) -> str:
# #         """Return identifier for this LLM type"""
# #         return "biogpt"
    
# #     @property
# #     def _identifying_params(self):
# #         """Return identifying parameters"""
# #         return {
# #             "model": "microsoft/biogpt",
# #             "max_length": self.max_length,
# #             "temperature": self.temperature,
# #             "device": self.device
# #         }
    
# #     def _call(
# #         self,
# #         prompt: str,
# #         stop: Optional[List[str]] = None,
# #         run_manager: Optional[CallbackManagerForLLMRun] = None,
# #         **kwargs: Any,
# #     ) -> str:
# #         """
# #         Generate text using BioGPT model
        
# #         Args:
# #             prompt: Input text prompt
# #             stop: Stop sequences (not used in current implementation)
# #             run_manager: Callback manager
# #             **kwargs: Additional generation parameters
        
# #         Returns:
# #             Generated text
# #         """
# #         try:
# #             # Tokenize input
# #             inputs = self.tokenizer(
# #                 prompt,
# #                 return_tensors="pt",
# #                 max_length=self.max_length,
# #                 truncation=True,
# #                 padding=True
# #             )
            
# #             # Move inputs to device
# #             inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
# #             # Generate with no gradient computation
# #             with torch.no_grad():
# #                 outputs = self.model.generate(
# #                     **inputs,
# #                     max_length=self.max_length,
# #                     temperature=self.temperature,
# #                     top_p=self.top_p,
# #                     do_sample=True,
# #                     pad_token_id=self.tokenizer.eos_token_id,
# #                     num_return_sequences=1,
# #                     **kwargs
# #                 )
            
# #             # Decode generated text
# #             generated_text = self.tokenizer.decode(
# #                 outputs[0],
# #                 skip_special_tokens=True
# #             )
            
# #             # Remove the input prompt from output if present
# #             if generated_text.startswith(prompt):
# #                 generated_text = generated_text[len(prompt):].strip()
            
# #             return generated_text
            
# #         except Exception as e:
# #             logger.error(f"❌ BioGPT generation error: {e}")
# #             raise


# # class BioGPTChatLLM(BioGPTLLM):
# #     """
# #     Chat-style wrapper for BioGPT to maintain compatibility with existing code.
# #     Converts chat-style prompts to text generation format.
# #     """
    
# #     def _call(
# #         self,
# #         prompt: str,
# #         stop: Optional[List[str]] = None,
# #         run_manager: Optional[CallbackManagerForLLMRun] = None,
# #         **kwargs: Any,
# #     ) -> str:
# #         """
# #         Process chat-style prompt and generate response
        
# #         Chat prompts are expected in format:
# #         System: [system message]
# #         Human: [user message]
# #         Assistant:
# #         """
# #         # Extract the actual prompt from chat format if present
# #         if "Human:" in prompt:
# #             # Extract the human part as the main prompt
# #             parts = prompt.split("Human:")
# #             if len(parts) > 1:
# #                 actual_prompt = parts[-1].strip()
# #                 if "Assistant:" in actual_prompt:
# #                     actual_prompt = actual_prompt.split("Assistant:")[0].strip()
# #                 prompt = actual_prompt
        
# #         # Call parent generation method
# #         return super()._call(prompt, stop, run_manager, **kwargs)


# # def get_biogpt_llm(
# #     max_length: int = 512,
# #     temperature: float = 0.7,
# #     use_chat: bool = True
# # ) -> LLM:
# #     """
# #     Factory function to create BioGPT LLM instance
    
# #     Args:
# #         max_length: Maximum generation length
# #         temperature: Sampling temperature
# #         use_chat: Use chat-style wrapper (recommended for task generation)
    
# #     Returns:
# #         BioGPT LLM instance
# #     """
# #     if use_chat:
# #         return BioGPTChatLLM(
# #             max_length=max_length,
# #             temperature=temperature
# #         )
# #     else:
# #         return BioGPTLLM(
# #             max_length=max_length,
# #             temperature=temperature
# #         )




# # task creation 

# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any
# from datetime import datetime, timedelta
# import logging
# from config.settings import CONFIG
# import json
# import os

# from services.database_service import DatabaseService
# from utils.biogpt_llm import get_biogpt_llm

# logger = logging.getLogger("task_ai")

# # ------------------ MODELS ------------------

# class QuickNotes(BaseModel):
#     status_update: str = ""
#     details: str = ""
#     one_line_note: str = ""

# class AITask(BaseModel):
#     description: str
#     department: str = Field(..., description="Must be one of: Medical/Clinical, Scheduling & Coordination, Administrative/Compliance, Authorizations & Denials")
#     status: str = "Pending"
#     due_date: str
#     patient: str
#     quick_notes: QuickNotes
#     actions: List[str]
#     source_document: str = ""

# class TaskCreationResult(BaseModel):
#     tasks: List[AITask]

# # ------------------ TASK CREATOR ------------------
# class TaskCreator:
#     """Universal AI task generator for any medical document type using BioGPT."""
    
#     def __init__(self, use_biogpt: bool = None):
#         """
#         Initialize TaskCreator with either BioGPT or Azure OpenAI
        
#         Args:
#             use_biogpt: If True, use BioGPT. If False, use Azure. If None, check env variable.
#         """
#         # Determine which model to use
#         if use_biogpt is None:
#             use_biogpt = os.getenv("USE_BIOGPT", "true").lower() == "true"
        
#         self.use_biogpt = use_biogpt
        
#         if self.use_biogpt:
#             logger.info("🔬 Initializing TaskCreator with BioGPT (biomedical specialist)")
#             try:
#                 self.llm = get_biogpt_llm(
#                     max_length=1024,  # Longer for task generation
#                     temperature=0.1,  # Low temperature for consistent medical tasks
#                     use_chat=True
#                 )
#                 logger.info("✅ BioGPT initialized successfully")
#             except Exception as e:
#                 logger.error(f"❌ BioGPT initialization failed: {e}")
#                 logger.info("🔄 Falling back to Azure OpenAI")
#                 self.use_biogpt = False
#                 self._initialize_azure()
#         else:
#             logger.info("☁️ Initializing TaskCreator with Azure OpenAI")
#             self._initialize_azure()
        
#         self.parser = JsonOutputParser(pydantic_object=TaskCreationResult)
    
#     def _initialize_azure(self):
#         """Initialize Azure OpenAI as fallback"""
#         self.llm = AzureChatOpenAI(
#             azure_endpoint=CONFIG.get("azure_openai_endpoint"),
#             api_key=CONFIG.get("azure_openai_api_key"),
#             deployment_name=CONFIG.get("azure_openai_deployment"),
#             api_version=CONFIG.get("azure_openai_api_version"),
#             temperature=0.1,
#             timeout=90,
#         )

#     SYSTEM_PROMPT = """
# You are an expert healthcare operations AI that converts ANY medical document into the necessary actionable tasks.

# ## 🎯 CORE MISSION
# Read the document deeply. Identify ALL DISTINCT ACTIONABLE ITEMS that require separate follow-up. Generate one task per distinct action required.

# ---

# ## 📋 UNIVERSAL DOCUMENT UNDERSTANDING

# You will receive ANY type of medical document:
# - Clinical reports (labs, imaging, pathology, consultations, progress notes, H&P, discharge summaries, operative notes)
# - Administrative documents (referrals, authorizations, denials, legal correspondence, insurance communications)
# - Coordination documents (therapy notes, follow-up recommendations, care plans, medication requests)
# - Compliance documents (QME/AME/IMR notices, attorney letters, settlement documents)

# **Your job:** Extract EACH distinct action that must happen based on the document's content.

# ### 👩‍💼 ADMINISTRATIVE DOCUMENT TASK FOCUS

# For administrative documents (authorizations, EOBs, denials, referrals, credentialing), generate **Staff Action Tasks** focused on Revenue Cycle Management (RCM), Scheduling, and administrative workflows. These are actionable follow-up items for staff, not clinical summaries for physicians.

# **Key Administrative Document Types & Task Generation:**

# | Document Type | Staff Task Generated | Focus |
# |---------------|---------------------|-------|
# | **Prior Authorization (PA) Approval** | Schedule service and attach PA number to claim | RCM: Ensure authorized service is delivered and properly billed |
# | **Prior Authorization Denial** | File appeal by [date] or obtain peer-to-peer review | RCM: Recover denied authorization through proper channels |
# | **EOB with Denial Codes** | - CARC 29: File timely filing appeal (High Priority)<br>- PR-1/PR-3: Bill patient (Medium Priority)<br>- CARC 97/16: Recode and resubmit (Medium Priority) | RCM: Resolve claim denials through appropriate action |
# | **External Referral (Outgoing)** | Follow up with specialist to confirm appointment booked and kept | Coordination: Track continuity of care |
# | **Credentialing/Contracting Renewals** | Renew payer contract by [date] or update provider credentials | Compliance: Ensure continued payment eligibility |


# Breakdown of Generated Approval Tasks | UR Approval Task Generation
# Scheduling/Confirmation Task (T-UR-1A/3A): This is the most critical staff-first action for an approval. It tells the staff to immediately schedule the approved service (like the orthosis fitting or confirming the medication was sent).
# Patient Notification Task (T-UR-1B/3B): This task ensures the patient is informed of the approval, which is a required administrative step.
# Document Upload Task: This is the common final step for all documents, ensuring the approval is filed in the EMR.

# **Administrative Task Characteristics:**
# - Focus on billing, scheduling, compliance workflows
# - Include specific action items (appeal, bill, schedule, follow-up)
# - Prioritize based on financial/compliance impact
# - Reference specific codes, dates, authorization numbers
# - Route to appropriate administrative department (not clinical)

# ---

# ## 🔢 MULTIPLE TASKS RULES

# **Generate MULTIPLE tasks when document contains:**
# - Multiple procedure authorizations (each needs separate scheduling)
# - Multiple treatment recommendations (each requiring different department actions)
# - Multiple abnormal results requiring different clinical reviews
# - Authorization + scheduling + clinical review all needed
# - Legal response + clinical action + administrative update required
# - Multiple referrals to different specialists
# - Multiple medications needing different actions (refill, review, change)

# **Generate SINGLE task when:**
# - Document has one clear primary action
# - Multiple items belong to same workflow step (e.g., "Review all lab results")
# - Sub-items are part of one larger task

# **CRITICAL:** Each task must be genuinely distinct and actionable separately. Don't artificially split one action into multiple tasks.

# ---

# ## 🧠 INTELLIGENT TASK EXTRACTION LOGIC

# ### **Step 1: Identify Document Intent**
# Ask yourself:
# - What are ALL the things this document tells us to do?
# - Are these separate actions requiring different departments or workflows?
# - Can these be combined into one task or must they be separate?

# ### **Step 2: Detect Action Triggers**

# **Clinical Triggers → Medical/Clinical Department:**
# - Abnormal/critical results requiring physician review
# - Treatment recommendations from specialists
# - Medication management needs
# - Symptom changes requiring clinical assessment
# - Functional capacity evaluations
# - Progress reports needing clinical interpretation

# **Scheduling Triggers → Scheduling & Coordination:**
# - Approved authorizations needing appointments
# - Referrals requiring coordination
# - Diagnostic studies needing scheduling
# - Follow-up appointments recommended
# - Procedure/surgery scheduling needs

# **Authorization Triggers → Authorizations & Denials:**
# - RFA submissions needing tracking
# - UR denials requiring response/appeal
# - IMR processes needing management
# - Peer-to-peer coordination requests
# - Authorization expirations requiring renewal
# - Prior authorization approvals needing claim attachment

# **Administrative Triggers → Administrative/Compliance:**
# - Legal communications requiring response
# - Compliance documentation needs
# - QME/AME administrative coordination
# - Attorney correspondence requiring file updates
# - Record requests or document distribution
# - EOB processing and denial code resolution
# - Credentialing and payer contracting renewals
# - External referral tracking and follow-up

# ### **Step 3: Extract Specifics**
# From the document, capture:
# - **Patient name** (exact)
# - **Specific findings** (test results, diagnoses, recommendations)
# - **Deadlines** (legal dates, authorization expiries, clinical urgency)
# - **Provider names** (ordering physician, specialist, reviewer)
# - **Actionable details** (what test, what procedure, what response needed)

# ---

# ## 🏢 DEPARTMENT ROUTING RULES

# **Medical/Clinical** - Route when:
# - Document contains clinical findings requiring physician interpretation
# - Treatment decisions or modifications needed
# - Abnormal results needing clinical follow-up
# - Medication management required
# - Patient care plan updates needed

# **Scheduling & Coordination** - Route when:
# - Authorization approved and appointment needs scheduling
# - Referral received requiring coordination
# - Diagnostic study approved and needs booking
# - Follow-up visit recommended with timeframe

# **Authorizations & Denials** - Route when:
# - RFA submitted and needs tracking
# - UR denial received requiring response
# - IMR eligible and needs filing
# - Authorization about to expire
# - Peer-to-peer review requested

# **Administrative/Compliance** - Route when:
# - Legal document received (attorney, adjuster, court)
# - QME/AME notice requiring administrative action
# - Compliance documentation needed
# - Record request or document distribution
# - Settlement or C&R documents

# ---

# ## ⏰ INTELLIGENT DUE DATE ASSIGNMENT

# Base due dates on document content analysis:

# | Document Context | Due Date | Rationale |
# |------------------|----------|-----------|
# | STAT/Critical results | Same day | Immediate physician review required |
# | Abnormal findings | +1 day | Urgent clinical attention needed |
# | Authorization expiring soon | 2 days before expiry | Prevent lapse in coverage |
# | Legal deadline specified | 3 days before deadline | Allow processing time |
# | UR denial (response required) | +3 days | Meet regulatory timelines |
# | Approved authorization | +2 days | Prompt patient service |
# | Standard clinical review | +2 days | Normal workflow |
# | Administrative tasks | +3 days | Standard processing |
# | Routine follow-up | +7 days | Non-urgent coordination |

# ---

# ## ✅ TASK GENERATION RULES

# **CRITICAL PRINCIPLES:**
# 1. **GENERATE AS MANY AS NEEDED** - One task per distinct action
# 2. **SIMPLE & CLEAR** - Use plain language anyone can understand
# 3. **DOCUMENT-DRIVEN** - Based only on actual content, no assumptions
# 4. **CONCISE DESCRIPTION** - 5-8 words maximum, avoid jargon
# 5. **CONTEXT IN DETAILS** - Technical details go in quick_notes.details

# **Task Description Formula:**
# [Simple Action] + [What/Who] + [Key Detail if critical]

# **Plain Language Guidelines:**
# - Use everyday words: "Review" not "Evaluate", "Schedule" not "Coordinate"
# - Avoid medical codes and abbreviations in description
# - Keep it conversational and direct
# - Technical details → quick_notes.details (not description)

# **Examples - Before & After:**

# **Clinical Tasks:**
# - ❌ Complex: "Review elevated ALT (150) from 1/15 labs"
# - ✅ Simple: "Review abnormal liver test for [Patient Name]"

# - ❌ Complex: "Assess FCE functional capacity recommendations"
# - ✅ Simple: "Review work capacity report for [Patient Name]"

# **Scheduling Tasks:**
# - ❌ Complex: "Schedule authorized orthopedic consult within 7 days"
# - ✅ Simple: "Schedule orthopedic appointment for [Patient Name]"

# - ❌ Complex: "Schedule approved lumbar MRI and attach PA #12345"
# - ✅ Simple: "Schedule MRI for [Patient Name]"

# **Authorization Tasks:**
# - ❌ Complex: "Prepare IMR appeal for denied PT authorization"
# - ✅ Simple: "Appeal denied physical therapy for [Patient Name]"

# - ❌ Complex: "Submit RFA for cervical epidural injections"
# - ✅ Simple: "Request authorization for injections - [Patient Name]"

# **Administrative Tasks:**
# - ❌ Complex: "Review attorney deposition request by 1/20"
# - ✅ Simple: "Respond to attorney letter for [Patient Name]"

# - ❌ Complex: "File timely filing appeal for CARC 29 denial by 1/25"
# - ✅ Simple: "Appeal late filing denial for [Patient Name]"

# - ❌ Complex: "Confirm orthopedic specialist appointment kept on 2/10"
# - ✅ Simple: "Confirm specialist visit for [Patient Name]"

# **Multi-Task Example:**
# Authorization report approves MRI, PT, and orthopedic consult:
# - Task 1: "Schedule MRI for [Patient Name]" → Scheduling
# - Task 2: "Schedule physical therapy for [Patient Name]" → Scheduling
# - Task 3: "Schedule orthopedic visit for [Patient Name]" → Scheduling

# **EOB with Multiple Denials Example:**
# - Task 1: "Appeal late filing denial for [Patient Name]" → Authorizations & Denials (High Priority)
# - Task 2: "Bill patient for service - [Patient Name]" → Administrative/Compliance (Medium Priority)
# - Task 3: "Resubmit corrected claim for [Patient Name]" → Administrative/Compliance (Medium Priority)

# **Key Principle:** 
# The description should be understandable by office staff, patients, or anyone without medical training. Save the technical details (codes, values, specific dates) for quick_notes.details.
# ---

# ## 🚫 ANTI-HALLUCINATION RULES

# **NEVER:**
# - Invent clinical findings not in the document
# - Create tasks for missing information (instead: task = "Clarify missing X")
# - Assume urgency without evidence
# - Generate generic "review document" tasks
# - Route to wrong department based on keywords alone
# - Create duplicate tasks for the same action

# **ALWAYS:**
# - Use exact patient names from document
# - Reference specific dates, values, findings from document
# - Base urgency on explicit indicators (STAT, deadlines, critical values)
# - Match task complexity to document content
# - Ensure each task is genuinely distinct

# ---

# ## 📤 OUTPUT FORMAT

# ```json
# {{
#   "tasks": [
#     {{
#       "description": "Concise 5-10 word action statement with specific details",
#       "department": "Exact department name from the four options",
#       "status": "Pending",
#       "due_date": "YYYY-MM-DD",
#       "patient": "Exact patient name from document",
#       "actions": ["Claim", "Complete"],
#       "source_document": "{{source_document}}",
#       "quick_notes": {{
#         "details": "Brief explanation of WHY this task is critical based on document content (1-2 sentences)",
#         "one_line_note": "Ultra-concise summary for dashboard (under 50 chars)"
#       }}
#     }}
#   ]
# }}
# ```

# ---

# ## 🎯 QUALITY CHECKLIST (Before Generating)

# Ask yourself:
# 1. ✅ Have I identified ALL distinct actions in this document?
# 2. ✅ Is each task SPECIFIC with actual document details?
# 3. ✅ Is the department routing CORRECT for workflow ownership?
# 4. ✅ Is the due date APPROPRIATE for urgency level?
# 5. ✅ Does quick_notes.details explain WHY this matters?
# 6. ✅ Would a staff member know EXACTLY what to do?
# 7. ✅ Am I using ONLY information from the document?
# 8. ✅ Are multiple tasks truly distinct or should they be combined?

# ---

# **Remember:** Generate the APPROPRIATE number of tasks - no more, no less. Quality and specificity matter more than quantity.
# """
    
#     def create_prompt(self, patient_name: str, source_document: str) -> ChatPromptTemplate:
#         user_template = """
# DOCUMENT TYPE: {document_type}

# FULL DOCUMENT TEXT:
# {full_text}

# DOCUMENT ANALYSIS (Structured Extraction):
# {document_analysis}

# SOURCE DOCUMENT: {source_document}
# TODAY'S DATE: {current_date}
# PATIENT: {patient_name}

# **Task:** Generate ALL necessary actionable tasks from this document. Use the FULL DOCUMENT TEXT as the primary source of information, and the DOCUMENT ANALYSIS for structured data reference.

# **Think through:**
# 1. What are ALL the distinct actions this document requires?
# 2. Should these be separate tasks or combined into one?
# 3. Which department owns each action in our workflow?
# 4. What is the appropriate timeline based on urgency for each?
# 5. What specific details make each task actionable?

# {format_instructions}
# """
        
#         return ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
#             HumanMessagePromptTemplate.from_template(user_template),
#         ])

#     async def generate_tasks(self, document_analysis: dict, source_document: str = "", full_text: str = "") -> list[dict]:
#         """Generate intelligent, context-aware task from any medical document."""
#         try:
#             current_date = datetime.now()
#             patient_name = document_analysis.get("patient_name", "Unknown")
#             document_type = document_analysis.get("document_type", "Unknown")
            
#             logger.info(f"🔍 Analyzing document: {source_document} ({document_type})")
#             logger.info(f"📝 Full text length: {len(full_text)} characters")
#             logger.info(f"🤖 Using model: {'BioGPT' if self.use_biogpt else 'Azure OpenAI'}")

#             # Use BioGPT-optimized approach if enabled
#             if self.use_biogpt:
#                 return await self._generate_tasks_with_biogpt(
#                     document_analysis, source_document, full_text, patient_name, document_type, current_date
#                 )
#             else:
#                 return await self._generate_tasks_with_azure(
#                     document_analysis, source_document, full_text, patient_name, document_type, current_date
#                 )

#         except Exception as e:
#             logger.error(f"❌ Task generation failed: {str(e)}")
#             return await self._create_fallback_task(document_analysis, source_document)
    
#     async def _generate_tasks_with_biogpt(
#         self, document_analysis: dict, source_document: str, full_text: str,
#         patient_name: str, document_type: str, current_date: datetime
#     ) -> list[dict]:
#         """
#         Generate tasks using BioGPT with medical-specific prompt optimization.
#         BioGPT works better with focused medical prompts and structured medical context.
#         """
#         try:
#             # Create medically-focused prompt for BioGPT
#             medical_prompt = f"""Medical Document Analysis Task:

# Document Type: {document_type}
# Patient: {patient_name}
# Date: {current_date.strftime("%Y-%m-%d")}

# Clinical Context:
# {json.dumps(document_analysis, indent=2)}

# Full Medical Report:
# {full_text[:2000] if full_text else 'Not available'}  

# Task: Generate actionable medical workflow tasks from this document.

# For each distinct actionable item, create a task with:
# 1. Clear description (5-8 words, plain language)
# 2. Department: Medical/Clinical, Scheduling & Coordination, Administrative/Compliance, or Authorizations & Denials
# 3. Due date based on clinical urgency
# 4. Patient name
# 5. Required actions
# 6. Clinical details

# Focus on:
# - Abnormal findings requiring physician review
# - Treatment recommendations needing implementation
# - Medication changes requiring action
# - Test orders requiring scheduling
# - Authorization requests requiring processing
# - Legal/compliance items requiring response

# Return JSON format:
# {{
#   "tasks": [
#     {{
#       "description": "Review abnormal lab results",
#       "department": "Medical/Clinical",
#       "status": "Pending",
#       "due_date": "2025-01-15",
#       "patient": "{patient_name}",
#       "quick_notes": {{
#         "status_update": "",
#         "details": "Clinical context and specifics",
#         "one_line_note": "Quick summary"
#       }},
#       "actions": ["Claim", "Review", "Complete"],
#       "source_document": "{source_document}"
#     }}
#   ]
# }}

# Generate all necessary tasks now:"""

#             # Invoke BioGPT
#             logger.info("🔬 Invoking BioGPT for medical task generation...")
#             result_text = self.llm._call(medical_prompt)
            
#             # Parse JSON from BioGPT output
#             import re
#             json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
#             if json_match:
#                 tasks_data = json.loads(json_match.group())
#             else:
#                 logger.warning("⚠️ BioGPT did not return valid JSON, using fallback")
#                 return await self._create_fallback_task(document_analysis, source_document)
            
#             tasks = tasks_data.get("tasks", [])
            
#             if not tasks:
#                 return await self._create_fallback_task(document_analysis, source_document)

#             # Validate and normalize tasks
#             validated_tasks = await self._validate_and_normalize_tasks(tasks, document_type)
            
#             if not validated_tasks:
#                 return await self._create_fallback_task(document_analysis, source_document)

#             logger.info(f"✅ BioGPT generated {len(validated_tasks)} medically accurate task(s)")
#             await self._update_workflow_analytics(validated_tasks)
            
#             return validated_tasks
            
#         except Exception as e:
#             logger.error(f"❌ BioGPT task generation error: {e}")
#             return await self._create_fallback_task(document_analysis, source_document)
    
#     async def _generate_tasks_with_azure(
#         self, document_analysis: dict, source_document: str, full_text: str,
#         patient_name: str, document_type: str, current_date: datetime
#     ) -> list[dict]:
#         """Generate tasks using Azure OpenAI (original method)"""
#         prompt = self.create_prompt(patient_name, source_document)
#         chain = prompt | self.llm | self.parser

#         # Build invocation data with full text
#         invocation_data = {
#             "document_analysis": json.dumps(document_analysis, indent=2),
#             "current_date": current_date.strftime("%Y-%m-%d"),
#             "source_document": source_document or "Unknown",
#             "document_type": document_type,
#             "patient_name": patient_name,
#             "format_instructions": self.parser.get_format_instructions()
#         }
        
#         # Add full text if available
#         if full_text:
#             invocation_data["full_text"] = full_text
#             logger.info("✅ Full document text included in task generation")
#         else:
#             invocation_data["full_text"] = "Not available"
#             logger.warning("⚠️ No full text available for task generation")

#         result = chain.invoke(invocation_data)

#         # Normalize result
#         if isinstance(result, dict):
#             tasks_data = result
#         else:
#             try:
#                 tasks_data = result.dict()
#             except Exception:
#                 tasks_data = {"tasks": []}

#         # Extract tasks
#         tasks = tasks_data.get("tasks", [])
        
#         if not tasks:
#             return await self._create_fallback_task(document_analysis, source_document)

#         # Validate and normalize tasks
#         validated_tasks = await self._validate_and_normalize_tasks(tasks, document_type)
        
#         if not validated_tasks:
#             return await self._create_fallback_task(document_analysis, source_document)

#         logger.info(f"✅ Azure generated {len(validated_tasks)} task(s)")
#         await self._update_workflow_analytics(validated_tasks)

#         return validated_tasks
    
#     async def _validate_and_normalize_tasks(self, tasks: list, document_type: str) -> list:
#         """Validate and normalize task data"""
#         valid_departments = [
#             "Medical/Clinical", 
#             "Scheduling & Coordination", 
#             "Administrative/Compliance", 
#             "Authorizations & Denials"
#         ]
        
#         validated_tasks = []
#         for task in tasks:
#             if not task.get("description"):
#                 continue
                
#             if task.get("department") not in valid_departments:
#                 task["department"] = self._infer_department(task.get("description", ""), document_type)
            
#             validated_tasks.append(task)
        
#         return validated_tasks

#     def _infer_department(self, description: str, doc_type: str) -> str:
#         """Quick fallback department inference."""
#         desc_lower = description.lower()
#         type_lower = doc_type.lower()
        
#         # Clinical indicators
#         if any(word in desc_lower for word in ["review", "assess", "evaluate", "lab", "result", "finding", "clinical", "treatment", "medication"]):
#             return "Medical/Clinical"
        
#         # Scheduling indicators
#         if any(word in desc_lower for word in ["schedule", "book", "appointment", "coordinate", "arrange"]):
#             return "Scheduling & Coordination"
        
#         # Authorization indicators
#         if any(word in desc_lower for word in ["rfa", "authorization", "denial", "ur", "imr", "appeal", "peer-to-peer"]):
#             return "Authorizations & Denials"
        
#         # Administrative indicators
#         if any(word in desc_lower for word in ["attorney", "legal", "qme", "compliance", "correspondence", "document"]):
#             return "Administrative/Compliance"
        
#         # Default to Medical/Clinical
#         return "Medical/Clinical"

#     async def _create_fallback_task(self, document_analysis: dict, source_document: str) -> list[dict]:
#         """Create intelligent fallback task when generation fails."""
#         current_date = datetime.now()
#         due_date = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
        
#         doc_type = document_analysis.get("document_type", "document")
#         patient = document_analysis.get("patient_name", "Unknown")
        
#         fallback_task = {
#             "description": f"Review {doc_type} and route appropriately",
#             "department": "Medical/Clinical",
#             "status": "Pending",
#             "due_date": due_date,
#             "patient": patient,
#             "actions": ["Claim", "Complete"],
#             "source_document": source_document or "Unknown",
#             "quick_notes": {
#                 "details": f"System-generated task for {doc_type} requiring manual review and routing",
#                 "one_line_note": "Manual review needed"
#             }
#         }
        
#         return [fallback_task]

#     async def _update_workflow_analytics(self, tasks: list[dict]):
#         """Update workflow analytics based on generated tasks."""
#         try:
#             db = DatabaseService()
#             await db.connect()

#             for task in tasks:
#                 department = task.get("department", "").lower()
#                 description = task.get("description", "").lower()

#                 # Department-based analytics
#                 if "medical" in department or "clinical" in department:
#                     await db.increment_workflow_stat("clinicalReviews")
#                 elif "scheduling" in department or "coordination" in department:
#                     await db.increment_workflow_stat("schedulingTasks")
#                 elif "administrative" in department or "compliance" in department:
#                     await db.increment_workflow_stat("adminTasks")
#                 elif "authorization" in department or "denial" in department:
#                     await db.increment_workflow_stat("authTasks")

#                 # Specific workflow tracking
#                 if any(word in description for word in ["rfa", "ur", "imr", "authorization", "denial"]):
#                     await db.increment_workflow_stat("rfasMonitored")
#                 elif any(word in description for word in ["qme", "ime", "ame"]):
#                     await db.increment_workflow_stat("qmeUpdating")
#                 elif any(word in description for word in ["attorney", "legal", "settlement"]):
#                     await db.increment_workflow_stat("legalDocs")

#             await db.disconnect()

#         except Exception as e:
#             logger.error(f"❌ Analytics update failed: {str(e)}")