import json
import os
from datetime import datetime, timezone, timedelta
from prisma import Prisma
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import CONFIG

# 🧩 Use main app's Prisma instance through dependency injection
class FollowUpService:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=CONFIG["azure_openai_deployment"],
            azure_endpoint=CONFIG["azure_openai_endpoint"],
            api_key=CONFIG["azure_openai_api_key"],
            api_version=CONFIG["azure_openai_api_version"],
            temperature=0.3,
            model_name="gpt-4o-mini"
        )
        
        self.followup_prompt_template = PromptTemplate.from_template("""
You are a medical task management AI. Analyze the following SINGLE overdue task and generate ONE appropriate follow-up task.
OVERDUE TASK:
{overdue_task}
CRITICAL INFORMATION:
- Current Date: {current_date}
INSTRUCTIONS:
1. Analyze the overdue task and create ONE logical follow-up
2. Consider medical urgency and workflow dependencies
3. Set reasonable due dates (1-7 days from now)
4. Assign appropriate departments based on task context
5. Use the original physician ID if available
RESPONSE FORMAT - JSON object (not array):
{{
  "description": "Specific follow-up action",
  "department": "Relevant department",
  "reason": "Why this follow-up is needed based on the overdue task",
  "due_in_days": 2,
  "related_physician_id": "physician_id_if_available",
  "priority": "High/Medium/Low",
  "original_task_id": "reference_to_original_task"
}}
EXAMPLES:
- Overdue: "Review patient lab results" → Follow-up: "Contact patient with lab results update"
- Overdue: "Submit insurance claim" → Follow-up: "Follow up on insurance claim status"
- Overdue: "Patient follow-up appointment" → Follow-up: "Schedule make-up appointment"
""")

    async def generate_followup_for_task(self, overdue_task, db):
        """Generate automatic follow-up task for a SINGLE overdue item"""
        formatted_task = (
            f"- Task ID: {overdue_task.id} | Description: {overdue_task.description} | "
            f"Dept: {overdue_task.department} | Due: {overdue_task.dueDate} | "
            f"Physician: {overdue_task.physicianId or 'Not assigned'} | Patient: {overdue_task.patient}"
        )
        try:
            response = self.llm.invoke(self.followup_prompt_template.format(
                overdue_task=formatted_task,
                current_date=datetime.now().strftime("%Y-%m-%d")
            ))
            if hasattr(response, 'content'):
                content = response.content.replace('```json', '').replace('```', '').strip()
                followup_suggestion = json.loads(content)
               
                if isinstance(followup_suggestion, dict):
                    # Validate and enrich with original data
                    validated_suggestion = {
                        "description": followup_suggestion.get("description", f"Follow-up for: {overdue_task.description}"),
                        "department": followup_suggestion.get("department", overdue_task.department),
                        "reason": followup_suggestion.get("reason", f"Auto-generated follow-up for overdue task: {overdue_task.description}"),
                        "due_in_days": max(1, min(7, followup_suggestion.get("due_in_days", 2))),
                        "related_physician_id": followup_suggestion.get("related_physician_id", overdue_task.physicianId),
                        "priority": followup_suggestion.get("priority", "Medium"),
                        "patient": overdue_task.patient,
                        "status": "Open",
                        "sourceDocument": overdue_task.sourceDocument,
                        "claimNumber": overdue_task.claimNumber,
                        "documentId": overdue_task.document.id if hasattr(overdue_task, 'document') and overdue_task.document else None,
                        "originalTaskId": overdue_task.id  # Link back to original overdue task
                    }
                    return [validated_suggestion]  # Return as list for consistency
        except Exception as e:
            print(f"❌ Follow-up task generation failed for task {overdue_task.id}: {e}")
        return []

# Create global service instance
followup_service = FollowUpService()

# ⏰ ENHANCED Cron job: check ALL overdue tasks (process all, one by one)
async def check_all_overdue_tasks():
    """Check ALL overdue tasks and generate follow-ups for EVERY overdue task, one by one"""
    print("=" * 60)
    print("🔄 AUTO FOLLOW-UP SCHEDULED JOB STARTED")
    print(f" ⏰ Time: {datetime.now()}")
    print("=" * 60)
   
    # Create a new database connection for this job
    db = Prisma()
    try:
        await db.connect()
        
        now = datetime.now(timezone.utc)
       
        # Find ALL overdue tasks regardless of physician
        # EXCLUDE tasks with status "Closed" or "Done"
        overdue_tasks = await db.task.find_many(
            where={
                "dueDate": {"lt": now},
                "status": {"notIn": ["Closed", "Done"]}  # 🆕 EXCLUDE Closed and Done tasks
            },
            include={
                "document": True
            }
        )
        if not overdue_tasks:
            print("✅ No overdue tasks found in the system.")
            return

        print(f"⚠️ Found {len(overdue_tasks)} total overdue tasks in the system")
        print(f"📋 Processing ALL {len(overdue_tasks)} overdue tasks (one by one)")
       
        # Print overdue tasks summary
        for i, task in enumerate(overdue_tasks, 1):
            overdue_days = (now - task.dueDate).days
            physician = task.physicianId or 'Not assigned'
            print(f" {i}. [{task.patient}] {task.description} - "
                  f"Overdue: {overdue_days} days - Physician: {physician} - Status: {task.status}")
        
        # Generate follow-up tasks one by one for ALL tasks
        created_count = 0
        processed_count = 0
        for overdue_task in overdue_tasks:
            try:
                followup_suggestions = await followup_service.generate_followup_for_task(overdue_task, db)
               
                if not followup_suggestions:
                    print(f"❌ No follow-up generated for task {overdue_task.id}")
                    continue
               
                # Take the first (and only) suggestion
                suggestion = followup_suggestions[0]
               
                # Determine assigned physician ID based on role (enhanced for all tasks)
                original_task_id = suggestion.get("original_task_id", overdue_task.id)
                assigned_id = suggestion.get("related_physician_id", overdue_task.physicianId)
                if overdue_task.physicianId:
                    assignee = await db.user.find_unique(where={"id": overdue_task.physicianId})
                    if assignee and assignee.role != "Physician" and assignee.physicianId:
                        assigned_id = assignee.physicianId
                    # else keep as is
                suggestion["related_physician_id"] = assigned_id

                due_date = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=suggestion["due_in_days"])
               
                # 🆕 FIXED: Build data dict conditionally
                create_data = {
                    "description": suggestion["description"],
                    "department": suggestion["department"],
                    "reason": suggestion["reason"],
                    "dueDate": due_date,
                    "patient": suggestion["patient"],
                    "status": "Open",
                    "physicianId": suggestion["related_physician_id"],
                    "sourceDocument": suggestion["sourceDocument"],
                    "claimNumber": suggestion["claimNumber"],
                    "followUpTaskId": overdue_task.id  # 🆕 Store the overdue task ID as followUpTaskId
                    # 🆕 REMOVED quickNotes entirely - let it use the default value from schema
                }
               
                # Only add documentId if not None
                document_id = suggestion.get("documentId")
                if document_id:
                    create_data["documentId"] = document_id
               
                # Create the follow-up task
                followup_task = await db.task.create(data=create_data)
                
                # 🆕 UPDATE ORIGINAL TASK STATUS TO "Closed"
                await db.task.update(
                    where={"id": overdue_task.id},
                    data={"status": "Closed"}  # 🆕 Set overdue task status to Closed
                )
                
                created_count += 1
                processed_count += 1
                print(f" ✅ Created follow-up task {followup_task.id} for overdue task {overdue_task.id}")
                print(f"    Follow-up: {suggestion['description']}")
                print(f"    Updated overdue task status: {overdue_task.status} → Closed")
               
            except Exception as e:
                print(f" ❌ Failed to create follow-up for task {overdue_task.id}: {e}")
                processed_count += 1
       
        print("=" * 60)
        print(f"✅ SCHEDULED JOB COMPLETED")
        print(f" 📊 Created {created_count} follow-up tasks")
        print(f" 📊 Processed {processed_count} overdue tasks")
        print(f" 📊 Closed {created_count} overdue tasks")
        print("=" * 60)
           
    except Exception as e:
        print(f"❌ Error in scheduled job: {e}")
    finally:
        # Always disconnect the database connection
        await db.disconnect()