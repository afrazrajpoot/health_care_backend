import json
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from prisma import Prisma
from openai import AsyncAzureOpenAI
from config.settings import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class TreatmentEvent:
    """Data class for individual treatment events"""
    date: str
    event: str
    details: str

class TreatmentHistoryGenerator:
    """Generates structured treatment history from documents"""
    
    def __init__(self):
        # Initialize Azure OpenAI client using your existing settings
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            api_version=CONFIG.get("azure_openai_api_version")
        )
        self.prisma = Prisma()
        self.deployment_name = CONFIG.get("azure_openai_deployment")
    
    async def connect(self):
        """Connect to database"""
        if not self.prisma.is_connected():
            await self.prisma.connect()
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.prisma.is_connected():
            await self.prisma.disconnect()
    
    async def get_patient_documents(self, patient_name: str, dob: Optional[str], 
                                   claim_number: Optional[str], physician_id: str, 
                                   exclude_document_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve all previous documents for a patient
        """
        try:
            await self.connect()
            
            # Build query conditions
            where_conditions = {
                "patientName": patient_name,
                "physicianId": physician_id,
                "status": {"not": "failed"}  # Exclude failed documents
            }
            
            if dob:
                where_conditions["dob"] = dob
            if claim_number:
                where_conditions["claimNumber"] = claim_number
            
            # Exclude current document if specified
            if exclude_document_id:
                where_conditions["id"] = {"not": exclude_document_id}
            
            # Fetch documents with related data
            documents = await self.prisma.document.find_many(
                where=where_conditions,
                include={
                    "summarySnapshot": True,
                    "bodyPartSnapshots": True,
                    "adl": True,
                    "documentSummary": True
                },
                order={"createdAt": "desc"}  # Most recent first
            )
            
            logger.info(f"📋 Found {len(documents)} previous documents for patient: {patient_name}")
            
            # Convert to dict and enrich with text data
            enriched_docs = []
            for doc in documents:
                doc_dict = doc.dict()
                
                # Try to get document text from various sources
                try:
                    # Try to get whatsNew data which contains summaries
                    if doc_dict.get("whatsNew"):
                        whats_new = json.loads(doc_dict["whatsNew"]) if isinstance(doc_dict["whatsNew"], str) else doc_dict["whatsNew"]
                        doc_dict["long_summary"] = whats_new.get("long_summary", "")
                        doc_dict["short_summary"] = whats_new.get("short_summary", "")
                    else:
                        doc_dict["long_summary"] = ""
                        doc_dict["short_summary"] = doc_dict.get("briefSummary", "")
                except:
                    doc_dict["long_summary"] = ""
                    doc_dict["short_summary"] = doc_dict.get("briefSummary", "")
                
                enriched_docs.append(doc_dict)
            
            return enriched_docs
            
        except Exception as e:
            logger.error(f"❌ Error fetching patient documents: {str(e)}")
            return []
    
    async def extract_treatment_history_from_docs(self, documents: List[Dict], 
                                                current_document: Dict = None) -> str:
        """
        Extract treatment history context from documents for LLM processing
        """
        context_parts = []
        
        # Add current document first if available
        if current_document:
            context_parts.append("=== CURRENT DOCUMENT ===")
            context_parts.append(f"Date: {current_document.get('createdAt', datetime.now().isoformat())}")
            context_parts.append(f"Summary: {current_document.get('short_summary', '')}")
            if current_document.get('long_summary'):
                context_parts.append(f"Details: {current_document.get('long_summary')[:1000]}...")
            
            # Add body part snapshots if available
            if current_document.get('bodyPartSnapshots'):
                context_parts.append("\nBody Part Details:")
                for snapshot in current_document['bodyPartSnapshots']:
                    if isinstance(snapshot, dict):
                        context_parts.append(f"- {snapshot.get('bodyPart') or snapshot.get('condition')}: "
                                           f"DX: {snapshot.get('dx')}, "
                                           f"Concern: {snapshot.get('keyConcern')}, "
                                           f"Next Step: {snapshot.get('nextStep')}")
        
        # Add previous documents
        if documents:
            context_parts.append("\n=== PREVIOUS DOCUMENTS ===")
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"\nDocument {i}:")
                context_parts.append(f"Date: {doc.get('createdAt')}")
                context_parts.append(f"Summary: {doc.get('short_summary', '')}")
                
                # Add document summary if available
                if doc.get('documentSummary'):
                    if isinstance(doc['documentSummary'], dict):
                        context_parts.append(f"Document Summary: {doc['documentSummary'].get('summary', '')}")
                
                # Add key findings from body parts
                if doc.get('bodyPartSnapshots'):
                    context_parts.append("Key Findings:")
                    for snapshot in doc['bodyPartSnapshots']:
                        if isinstance(snapshot, dict):
                            context_parts.append(f"- {snapshot.get('dx', 'Unknown')}: {snapshot.get('keyConcern', '')}")
        
        return "\n".join(context_parts)
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string into datetime for comparison.
        Returns datetime.min if parsing fails.
        """
        if not date_str:
            return datetime.min
        
        date_str = str(date_str).strip()
        
        # Remove any time portion if present
        if 'T' in date_str:
            date_str = date_str.split('T')[0]
        
        # Try different date formats in order of specificity
        formats = [
            "%m/%d/%Y",  # 03/10/2025
            "%m-%d-%Y",  # 03-10-2025
            "%Y-%m-%d",  # 2025-03-10
            "%m/%d/%y",  # 03/10/25
            "%m-%d-%y",  # 03-10-25
            "%m/%Y",     # 03/2025
            "%m-%Y",     # 03-2025
            "%Y/%m",     # 2025/03
            "%Y",        # 2025
            "%B %Y",     # March 2025
            "%b %Y",     # Mar 2025
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If standard parsing fails, try to extract year and month using regex
        try:
            # Look for year pattern (19xx or 20xx)
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = int(year_match.group())
                
                # Look for month pattern (1-12, with or without leading zero)
                month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', date_str)
                if month_match:
                    month = int(month_match.group())
                else:
                    # Try month names
                    month_names = {
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    for name, month_num in month_names.items():
                        if name in date_str.lower():
                            month = month_num
                            break
                    else:
                        month = 1  # Default to January if month not found
                
                return datetime(year, month, 1)
        except Exception:
            pass
        
        # If all parsing fails, return minimum datetime
        return datetime.min
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """
        Remove duplicate events based on key fields (date, event, and first 100 chars of details).
        Preserves the most recent entry if duplicates are found.
        """
        if not events:
            return []
        
        # Create a dictionary with composite key
        unique_events = {}
        
        for event in events:
            if not isinstance(event, dict):
                continue
                
            # Create composite key
            event_key = (
                str(event.get('date', '')).strip().lower(),
                str(event.get('event', '')).strip().lower(),
                str(event.get('details', '')).strip()[:100].lower()
            )
            
            # Keep the event with the most recent date or the first encountered
            if event_key not in unique_events:
                unique_events[event_key] = event
            else:
                # If duplicate found, compare dates and keep the one with more recent/later date
                existing_date = self._parse_date(unique_events[event_key].get('date', ''))
                new_date = self._parse_date(event.get('date', ''))
                
                if new_date > existing_date:
                    unique_events[event_key] = event
        
        return list(unique_events.values())
    
    async def generate_treatment_history_with_llm(self, patient_name: str, 
                                                 context: str, 
                                                 current_document_analysis: Any = None) -> Dict[str, List[Dict]]:
        """
        Use LLM to generate structured treatment history
        """
        try:
            # Prepare prompt for LLM
            prompt = f"""
            You are a medical history analyzer. Create a structured treatment history timeline from the provided document context.
            
            PATIENT: {patient_name}
            
            DOCUMENT CONTEXT:
            {context}
            
            CURRENT DOCUMENT ANALYSIS (if available):
            {str(current_document_analysis) if current_document_analysis else 'Not available'}
            
            INSTRUCTIONS:
            1. Analyze all provided documents and extract treatment events chronologically (most recent first)
            2. Organize events by body system/organ system categories
            3. For each event, include:
               - Date (extract from document, use format like "03/2024", "2023", "Q2 2023" if exact date not available)
               - Event type (e.g., "MRI", "PT Session", "Medication Change", "Consultation", "Surgery")
               - Details (specific findings, treatments, outcomes)
            4. Group events into logical categories such as:
               - musculoskeletal_system (for orthopedic, PT, spine, joint issues)
               - cardiovascular_system (for heart, BP, circulation)
               - pulmonary_respiratory (for lungs, breathing)
               - neurological (for brain, nerves, headaches)
               - gastrointestinal (for stomach, digestion)
               - metabolic_endocrine (for diabetes, thyroid, hormones)
               - other_systems (for anything else)
            5. If no specific system is mentioned, use "general_treatments"
            6. Only include information explicitly mentioned in the documents
            7. Format as JSON with system categories as keys and arrays of events as values
            
            OUTPUT FORMAT EXAMPLE:
            {{
              "musculoskeletal_system": [
                {{
                  "date": "03/10/2025",
                  "event": "MRI Lumbar Spine",
                  "details": "L4–5 disc protrusion, Mild canal stenosis, Consider ESI"
                }}
              ],
              "cardiovascular_system": [],
              "pulmonary_respiratory": []
            }}
            
            Return ONLY valid JSON, no additional text.
            """
            
            # Call Azure OpenAI
            response = await self.openai_client.chat.completions.create(
                model=self.deployment_name,  # Use the deployment name from settings
                messages=[
                    {"role": "system", "content": "You are a medical data analyst that extracts and structures treatment history information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            history_json = json.loads(response.choices[0].message.content)
            
            # Validate structure
            if not isinstance(history_json, dict):
                logger.warning("⚠️ LLM returned non-dict response, creating empty history")
                return self._get_empty_history_template()
            
            # Deduplicate events in each category
            for category, events in history_json.items():
                if isinstance(events, list):
                    history_json[category] = self._deduplicate_events(events)
            
            # Ensure all required categories exist
            required_categories = [
                "musculoskeletal_system", "cardiovascular_system", 
                "pulmonary_respiratory", "neurological", "gastrointestinal",
                "metabolic_endocrine", "other_systems", "general_treatments"
            ]
            
            for category in required_categories:
                if category not in history_json:
                    history_json[category] = []
                elif not isinstance(history_json[category], list):
                    history_json[category] = []
            
            # Sort events in each category by date (newest first)
            for category in history_json:
                if isinstance(history_json[category], list):
                    history_json[category].sort(
                        key=lambda x: self._parse_date(x.get('date', '')),
                        reverse=True
                    )
            
            total_events = sum(len(v) for v in history_json.values() if isinstance(v, list))
            logger.info(f"✅ Generated treatment history with {total_events} events across {len(history_json)} categories")
            
            return history_json
            
        except Exception as e:
            logger.error(f"❌ Error generating treatment history with LLM: {str(e)}")
            return self._get_empty_history_template()
    
    def _get_empty_history_template(self) -> Dict[str, Dict[str, List]]:
        """Return empty treatment history template with current and archive structure"""
        return {
            "musculoskeletal_system": {"current": [], "archive": []},
            "cardiovascular_system": {"current": [], "archive": []},
            "pulmonary_respiratory": {"current": [], "archive": []},
            "neurological": {"current": [], "archive": []},
            "gastrointestinal": {"current": [], "archive": []},
            "metabolic_endocrine": {"current": [], "archive": []},
            "other_systems": {"current": [], "archive": []},
            "general_treatments": {"current": [], "archive": []}
        }
    
    def merge_history_data(self, existing_data: Dict, new_data: Dict) -> Dict:
        """
        Merge new treatment history into existing history with intelligent date-based archiving.
        Older events are moved to archive based on date comparison.
        """
        def create_event_key(event):
            """Create a unique key for event deduplication"""
            if not isinstance(event, dict):
                return ""
            return (
                str(event.get('date', '')).strip().lower(),
                str(event.get('event', '')).strip().lower(),
                str(event.get('details', '')).strip()[:100].lower()
            )
        
        merged = existing_data.copy()
        
        for system, new_events in new_data.items():
            if not new_events:
                # No new events for this system
                continue
            
            # If system doesn't exist in merged data, initialize it
            if system not in merged:
                merged[system] = {
                    "current": new_events,
                    "archive": []
                }
                continue
            
            # Ensure system has proper structure
            if isinstance(merged[system], list):
                # Convert old format (list) to new format (dict with current/archive)
                merged[system] = {
                    "current": merged[system],
                    "archive": []
                }
            
            existing_current = merged[system].get("current", [])
            existing_archive = merged[system].get("archive", [])
            
            # Create sets to track unique events
            seen_events = set()
            final_current = []
            final_archive = existing_archive.copy()  # Start with existing archive
            
            # Process new events first (these should be the most recent)
            for new_event in new_events:
                if not isinstance(new_event, dict):
                    continue
                    
                event_key = create_event_key(new_event)
                if event_key and event_key not in seen_events:
                    seen_events.add(event_key)
                    
                    # Check if this event already exists in archive
                    archive_exists = False
                    for archived_event in final_archive:
                        if create_event_key(archived_event) == event_key:
                            archive_exists = True
                            break
                    
                    if not archive_exists:
                        final_current.append(new_event)
            
            # Process existing current events
            for existing_event in existing_current:
                if not isinstance(existing_event, dict):
                    continue
                    
                event_key = create_event_key(existing_event)
                if event_key and event_key not in seen_events:
                    seen_events.add(event_key)
                    
                    # Check if this event is older than the newest new event
                    if new_events:
                        # Get the most recent date from new events
                        new_event_dates = []
                        for ne in new_events:
                            if isinstance(ne, dict):
                                date = self._parse_date(ne.get('date', ''))
                                if date != datetime.min:
                                    new_event_dates.append(date)
                        
                        if new_event_dates:
                            newest_new_date = max(new_event_dates)
                            existing_date = self._parse_date(existing_event.get('date', ''))
                            
                            # If existing event is older than 6 months compared to newest new event,
                            # move it to archive instead of keeping in current
                            if existing_date < newest_new_date:
                                # Calculate 6 months ago from the newest date
                                six_months_ago = newest_new_date.replace(
                                    month=newest_new_date.month - 6 if newest_new_date.month > 6 else 12 - (6 - newest_new_date.month),
                                    year=newest_new_date.year if newest_new_date.month > 6 else newest_new_date.year - 1
                                )
                                
                                if existing_date < six_months_ago:
                                    # Check if it already exists in archive
                                    exists_in_archive = False
                                    for archived_event in final_archive:
                                        if create_event_key(archived_event) == event_key:
                                            exists_in_archive = True
                                            break
                                    
                                    if not exists_in_archive:
                                        final_archive.append(existing_event)
                                    continue
                    
                    # If not moved to archive, keep in current
                    final_current.append(existing_event)
            
            # Sort current events by date (newest first)
            final_current.sort(
                key=lambda x: self._parse_date(x.get('date', '')),
                reverse=True
            )
            
            # Sort archive events by date (oldest first)
            final_archive.sort(
                key=lambda x: self._parse_date(x.get('date', ''))
            )
            
            # Limit current events to last 15 per system to avoid overflow
            if len(final_current) > 15:
                # Move oldest events to archive
                events_to_archive = final_current[15:]
                
                # Add to archive (avoiding duplicates)
                for event in events_to_archive:
                    event_key = create_event_key(event)
                    if event_key:
                        # Check if already in archive
                        exists = False
                        for archived in final_archive:
                            if create_event_key(archived) == event_key:
                                exists = True
                                break
                        
                        if not exists:
                            final_archive.append(event)
                
                final_current = final_current[:15]
            
            # Remove duplicates from archive
            unique_archive = []
            seen_archive = set()
            for event in final_archive:
                event_key = create_event_key(event)
                if event_key and event_key not in seen_archive:
                    seen_archive.add(event_key)
                    unique_archive.append(event)
            
            merged[system] = {
                "current": final_current,
                "archive": unique_archive
            }
        
        return merged
    
    async def generate_treatment_history(self, 
                                        patient_name: str,
                                        dob: Optional[str],
                                        claim_number: Optional[str],
                                        physician_id: str,
                                        current_document_id: Optional[str] = None,
                                        current_document_analysis: Any = None,
                                        current_document_data: Dict = None,
                                        only_current: bool = False) -> Dict:
        """
        Main function to generate treatment history
        Returns the treatment history data
        """
        try:
            logger.info(f"🔄 Generating treatment history for patient: {patient_name} (only_current={only_current})")
            
            # Get previous documents (skip if only_current is True)
            previous_docs = []
            if not only_current:
                previous_docs = await self.get_patient_documents(
                    patient_name=patient_name,
                    dob=dob,
                    claim_number=claim_number,
                    physician_id=physician_id,
                    exclude_document_id=current_document_id
                )
            
            # Prepare context for LLM
            context = await self.extract_treatment_history_from_docs(
                documents=previous_docs,
                current_document=current_document_data
            )
            
            # If no documents found, create minimal context from current analysis
            if not context.strip():
                logger.info(f"📝 No previous documents found for {patient_name}, creating initial history")
                context = f"Initial document for {patient_name}. "
                if current_document_analysis:
                    context += f"Current findings: {current_document_analysis.diagnosis if hasattr(current_document_analysis, 'diagnosis') else 'Not specified'}"
            
            # Generate treatment history with LLM
            treatment_history = await self.generate_treatment_history_with_llm(
                patient_name=patient_name,
                context=context,
                current_document_analysis=current_document_analysis
            )
            
            logger.info(f"✅ Treatment history generated for patient: {patient_name}")
            return treatment_history
            
        except Exception as e:
            logger.error(f"❌ Error creating treatment history: {str(e)}")
            return self._get_empty_history_template()
    
    async def save_treatment_history(self,
                                   patient_name: str,
                                   dob: Optional[str],
                                   claim_number: Optional[str],
                                   physician_id: str,
                                   history_data: Dict,
                                   document_id: Optional[str] = None):
        """
        Save treatment history to database with intelligent date-based archiving
        """
        try:
            await self.connect()
            
            # Check if treatment history already exists
            existing = await self.prisma.treatmenthistory.find_unique(
                where={
                    "patientName_dob_claimNumber_physicianId": {
                        "patientName": patient_name,
                        "dob": dob or "",
                        "claimNumber": claim_number or "",
                        "physicianId": physician_id
                    }
                }
            )
            
            if existing:
                # Merge new history with existing history using date-based logic
                existing_data = existing.historyData
                if isinstance(existing_data, str):
                    existing_data = json.loads(existing_data)
                
                # If existing data is in old format (just lists), convert it first
                if isinstance(existing_data, dict):
                    # Check if any system is still in old format (list instead of dict with current/archive)
                    for system in existing_data:
                        if isinstance(existing_data[system], list):
                            existing_data[system] = {
                                "current": existing_data[system],
                                "archive": []
                            }
                else:
                    # If existing_data is not a dict, create empty structure
                    existing_data = self._get_empty_history_template()
                
                merged_data = self.merge_history_data(existing_data, history_data)
                
                # Update existing record
                await self.prisma.treatmenthistory.update(
                    where={"id": existing.id},
                    data={
                        "historyData": json.dumps(merged_data),
                        "documentId": document_id,
                        "updatedAt": datetime.now()
                    }
                )
                logger.info(f"📝 Merged and updated treatment history for {patient_name}")
            else:
                # For new records, format the LLM output
                formatted_data = self._get_empty_history_template()
                for system, events in history_data.items():
                    if system in formatted_data and events:
                        # Sort new events by date (newest first) and deduplicate
                        unique_events = self._deduplicate_events(events)
                        formatted_data[system]["current"] = sorted(
                            unique_events, 
                            key=lambda x: self._parse_date(x.get('date', '')), 
                            reverse=True
                        )
                
                # Create new record
                await self.prisma.treatmenthistory.create(
                    data={
                        "patientName": patient_name,
                        "dob": dob,
                        "claimNumber": claim_number,
                        "physicianId": physician_id,
                        "historyData": json.dumps(formatted_data),
                        "documentId": document_id
                    }
                )
                logger.info(f"📝 Created new treatment history for {patient_name}")
                
        except Exception as e:
            logger.error(f"❌ Error saving treatment history to database: {str(e)}")
            raise
    
    async def get_treatment_history(self,
                                  patient_name: str,
                                  dob: Optional[str],
                                  claim_number: Optional[str],
                                  physician_id: str) -> Optional[Dict]:
        """
        Retrieve treatment history from database
        """
        try:
            await self.connect()
            
            history = await self.prisma.treatmenthistory.find_unique(
                where={
                    "patientName_dob_claimNumber_physicianId": {
                        "patientName": patient_name,
                        "dob": dob or "",
                        "claimNumber": claim_number or "",
                        "physicianId": physician_id
                    }
                }
            )
            
            if history and history.historyData:
                if isinstance(history.historyData, str):
                    data = json.loads(history.historyData)
                else:
                    data = history.historyData
                
                # Ensure all systems have proper structure
                if isinstance(data, dict):
                    for system in data:
                        if isinstance(data[system], list):
                            data[system] = {"current": data[system], "archive": []}
                
                return data
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving treatment history: {str(e)}")
            return None