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
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string into datetime for comparison.
        Returns datetime.min if parsing fails or if date is empty/not specified.
        """
        if not date_str:
            return datetime.min
        
        date_str = str(date_str).strip()
        
        # Handle empty strings, "Date not specified" or similar placeholders - should sort to the end
        if not date_str or date_str.lower() in ["date not specified", "not specified", "unknown", "n/a", "none", ""]:
            return datetime.min
        
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
    
    async def extract_treatment_history_from_docs(self, documents: List[Dict], 
                                            current_document: Dict = None) -> str:
        """
        Extract treatment history context from documents for LLM processing
        Enhanced to include more comprehensive medical details
        """
        context_parts = []
        
        # Add current document first if available
        if current_document:
            context_parts.append("=== CURRENT DOCUMENT ===")
            # Only include date if it's actually available from the document
            doc_date = current_document.get('createdAt') or current_document.get('documentDate') or current_document.get('date')
            if doc_date:
                context_parts.append(f"Date: {doc_date}")
            
            # Include both summaries for more context
            if current_document.get('short_summary'):
                context_parts.append(f"Summary: {current_document.get('short_summary')}")
            if current_document.get('long_summary'):
                context_parts.append(f"Detailed Summary: {current_document.get('long_summary')[:2000]}...")
            
            # Include brief summary as fallback
            if current_document.get('briefSummary') and not current_document.get('short_summary'):
                context_parts.append(f"Summary: {current_document.get('briefSummary')}")
            
            # Add body part snapshots with full details
            if current_document.get('bodyPartSnapshots'):
                context_parts.append("\nBody Part Details:")
                for snapshot in current_document['bodyPartSnapshots']:
                    if isinstance(snapshot, dict):
                        parts = []
                        if snapshot.get('bodyPart') or snapshot.get('condition'):
                            parts.append(f"Body Part/Condition: {snapshot.get('bodyPart') or snapshot.get('condition')}")
                        if snapshot.get('dx'):
                            parts.append(f"Diagnosis: {snapshot.get('dx')}")
                        if snapshot.get('keyConcern'):
                            parts.append(f"Key Concern: {snapshot.get('keyConcern')}")
                        if snapshot.get('nextStep'):
                            parts.append(f"Next Step: {snapshot.get('nextStep')}")
                        if snapshot.get('status'):
                            parts.append(f"Status: {snapshot.get('status')}")
                        if parts:
                            context_parts.append("  - " + " | ".join(parts))
            
            # Add ADL information if available
            if current_document.get('adl'):
                adl = current_document['adl']
                if isinstance(adl, dict):
                    context_parts.append("\nActivities of Daily Living:")
                    if adl.get('limitations'):
                        context_parts.append(f"Limitations: {adl.get('limitations')}")
                    if adl.get('restrictions'):
                        context_parts.append(f"Restrictions: {adl.get('restrictions')}")
        
        # Add previous documents with enhanced detail extraction
        if documents:
            context_parts.append("\n=== PREVIOUS DOCUMENTS (CHRONOLOGICAL) ===")
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"\n--- Document {i} ---")
                
                # Date
                doc_date = doc.get('createdAt') or doc.get('documentDate')
                if doc_date:
                    context_parts.append(f"Date: {doc_date}")
                
                # Summaries - include both for comprehensive context
                if doc.get('short_summary'):
                    context_parts.append(f"Summary: {doc.get('short_summary')}")
                if doc.get('long_summary'):
                    # Include more of the long summary for better context
                    context_parts.append(f"Details: {doc.get('long_summary')[:1500]}...")
                elif doc.get('briefSummary'):
                    context_parts.append(f"Summary: {doc.get('briefSummary')}")
                
                # Document summary if available
                if doc.get('documentSummary'):
                    if isinstance(doc['documentSummary'], dict):
                        if doc['documentSummary'].get('summary'):
                            context_parts.append(f"Document Analysis: {doc['documentSummary'].get('summary')}")
                
                # Body part findings with full details
                if doc.get('bodyPartSnapshots'):
                    context_parts.append("Clinical Findings:")
                    for snapshot in doc['bodyPartSnapshots']:
                        if isinstance(snapshot, dict):
                            finding_parts = []
                            if snapshot.get('bodyPart') or snapshot.get('condition'):
                                finding_parts.append(snapshot.get('bodyPart') or snapshot.get('condition'))
                            if snapshot.get('dx'):
                                finding_parts.append(f"DX: {snapshot.get('dx')}")
                            if snapshot.get('keyConcern'):
                                finding_parts.append(f"Concern: {snapshot.get('keyConcern')}")
                            if snapshot.get('nextStep'):
                                finding_parts.append(f"Plan: {snapshot.get('nextStep')}")
                            if finding_parts:
                                context_parts.append(f"  • {' - '.join(finding_parts)}")
                
                # ADL from previous documents
                if doc.get('adl'):
                    adl = doc['adl']
                    if isinstance(adl, dict) and (adl.get('limitations') or adl.get('restrictions')):
                        context_parts.append("Functional Status:")
                        if adl.get('limitations'):
                            context_parts.append(f"  • Limitations: {adl.get('limitations')}")
                        if adl.get('restrictions'):
                            context_parts.append(f"  • Restrictions: {adl.get('restrictions')}")
        
        full_context = "\n".join(context_parts)
        
        # Log context length for debugging
        logger.debug(f"📝 Generated context with {len(full_context)} characters from {len(documents)} previous docs + current doc")
        
        return full_context

    
    async def generate_treatment_history_with_llm(self, patient_name: str, 
                                            context: str, 
                                            current_document_analysis: Any = None,
                                            max_retries: int = 3) -> Dict[str, List[Dict]]:
        """
        Use LLM to generate structured treatment history with retry logic.
        Follows DocLatch Treatment History Timeline canonical specification.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt)
                    logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries} after {wait_time}s delay...")
                    await asyncio.sleep(wait_time)
                
                # Prepare enhanced prompt for LLM
                prompt = f"""
    You are a medical document extraction system. Your ONLY task is to extract explicitly documented events from external medical documents into a chronological timeline. You MUST NOT interpret, infer, summarize, or generate any content not explicitly present in the source documents.

    PATIENT: {patient_name}

    DOCUMENT CONTEXT:
    {context}

    CURRENT DOCUMENT ANALYSIS (if available):
    {str(current_document_analysis) if current_document_analysis else 'Not available'}

    === CRITICAL RULES (MANDATORY) ===

    🔴 RULE 1: NO FABRICATION - ZERO TOLERANCE
    - Extract ONLY what is explicitly documented
    - NEVER infer, assume, or generate details
    - NEVER summarize in your own words
    - NEVER add clinical reasoning or interpretation
    - If a detail is not explicitly stated → DO NOT include it
    - When in doubt → EXCLUDE the information

    🔴 RULE 2: DATE HANDLING (STRICTLY ENFORCED)
    - ONLY use dates EXPLICITLY stated in the documents
    - NEVER use today's date, current date, or system date
    - NEVER guess or approximate dates
    - If NO date is mentioned → use empty string ""
    - Acceptable date formats: "03/10/2025", "March 2025", "2025", "Q1 2024"
    - DO NOT use: "Date not specified", "Unknown", placeholder text
    - DO NOT use dates from document receipt/processing

    🔴 RULE 3: SOURCE ATTRIBUTION (MANDATORY)
    - Every entry MUST include document source
    - Format: "Source: [Document Type] dated [Document Date]"
    - Example: "Source: Orthopedic consult report dated 11/12/2025"
    - If document type unclear → use "External document dated [date]"

    🔴 RULE 4: LANGUAGE REQUIREMENTS
    - Use ONLY past tense or present-perfect tense
    - Use neutral, factual, descriptive language
    - State what was "documented", "noted", "referenced", "completed"
    - NEVER use: recommended, planned, required, indicated, needed, failed, succeeded, appropriate, unnecessary

    🔴 RULE 5: EXTRACTION SCOPE
    Extract ONLY these types of documented events:
    - Diagnostic tests completed (with explicit results if stated)
    - Consultations documented (with explicit findings if stated)
    - Treatments administered/completed
    - Utilization review decisions
    - Therapy courses completed
    - Surgical procedures performed
    - Medication changes documented

    DO NOT extract:
    - Treatment plans or future recommendations
    - Clinical reasoning or interpretation
    - Physician opinions about appropriateness
    - Hypothetical scenarios ("if patient fails...")
    - Internal EMR documentation

    === VERBATIM EXTRACTION RULES ===

    When extracting details:
    1. Use exact medical terminology from source
    2. Include specific measurements/values ONLY if explicitly stated
    3. Preserve exact phrasing for diagnoses and findings
    4. If a finding has qualifiers (mild, moderate, severe) → include them ONLY if stated
    5. Do NOT paraphrase medical findings
    6. Do NOT combine information from multiple sentences into interpretations

    Example - CORRECT extraction:
    Document states: "MRI shows 3mm disc protrusion at L4-L5"
    Extract: "MRI documented 3mm disc protrusion at L4-L5"

    Example - INCORRECT extraction:
    Document states: "MRI shows 3mm disc protrusion at L4-L5"
    Extract: "Patient has significant disc herniation requiring surgical evaluation"
    (This adds interpretation, severity judgment, and treatment direction not stated)

    === BODY SYSTEM CATEGORIZATION ===

    Organize events into these categories based on PRIMARY body system:

    - musculoskeletal_system: bones, joints, spine, muscles, tendons, ligaments, orthopedic conditions
    - cardiovascular_system: heart, blood vessels, circulation, cardiac conditions
    - pulmonary_respiratory: lungs, airways, breathing disorders
    - neurological: brain, spinal cord, nerves, headaches, neuropathy
    - gastrointestinal: stomach, intestines, liver, digestive system
    - metabolic_endocrine: diabetes, thyroid, hormonal disorders
    - genitourinary_renal: kidneys, bladder, urinary system
    - reproductive_obstetric_gynecologic: pregnancy, menstrual, fertility, pelvic conditions
    - dermatological: skin conditions, wounds, rashes
    - ophthalmologic: eyes, vision conditions
    - ent_head_neck: ears, nose, throat, sinuses
    - dental_oral: teeth, gums, jaw, oral cavity
    - hematologic_lymphatic: blood disorders, lymph system
    - immune_allergy: autoimmune diseases, allergic reactions
    - psychiatric_mental_health: mental health conditions, psychological disorders
    - sleep_disorders: sleep-related conditions
    - other_systems: conditions not fitting above categories

    === OUTPUT FORMAT ===

    Return ONLY valid JSON with this EXACT structure:

    {{
    "musculoskeletal_system": [
        {{
        "date": "11/12/2025",
        "event_type": "Orthopedic consultation documented",
        "details": "Shoulder pathology documented. Injection therapy referenced as treatment option. Surgical intervention referenced as contingent option.",
        "source": "Orthopedic consult report dated 11/12/2025",
        "author": "Consulting orthopedist"
        }},
        {{
        "date": "11/22/2025",
        "event_type": "Utilization review decision received",
        "details": "Injection request denied. Rationale documented: insufficient objective findings per UR determination.",
        "source": "UR determination dated 11/22/2025",
        "author": "UR physician"
        }},
        {{
        "date": "",
        "event_type": "Physical therapy course completed",
        "details": "Therapy course completed per facility documentation.",
        "source": "Physical therapy discharge summary",
        "author": "Treating facility"
        }}
    ],
    "neurological": [
        {{
        "date": "03/10/2025",
        "event_type": "Nerve conduction study completed",
        "details": "Right L5 radiculopathy documented. Prolonged F-wave latency noted.",
        "source": "NCS report dated 03/10/2025",
        "author": "Neurologist"
        }}
    ],
    "cardiovascular_system": [],
    "pulmonary_respiratory": []
    }}

    === FIELD DEFINITIONS ===

    - date: Exact date from document (empty string "" if not stated)
    - event_type: Type of documented event (consultation documented, test completed, decision received, etc.)
    - details: VERBATIM extraction of documented findings/outcomes (NO interpretation)
    - source: Document type and date
    - author: External source author (orthopedist, radiologist, UR physician, facility)

    === QUALITY CHECKLIST ===

    Before returning JSON, verify:
    □ Every date is from source document or empty string
    □ Every entry has source attribution
    □ Language is past tense and factual
    □ No interpretive or predictive language used
    □ No fabricated details included
    □ Details match verbatim or near-verbatim from source
    □ Author is external source, never treating physician
    □ No treatment plans or recommendations included

    === CRITICAL REMINDERS ===

    ✅ DOCUMENT what was done, when, by whom
    ✅ USE past tense: "documented", "completed", "received", "noted"
    ✅ EXTRACT verbatim findings when present
    ✅ ATTRIBUTE every entry to source document

    ❌ NEVER interpret or infer
    ❌ NEVER use: "recommended", "should", "planned", "indicated"
    ❌ NEVER fabricate measurements or findings
    ❌ NEVER combine sources into conclusions
    ❌ NEVER add clinical reasoning

    If a document contains NO extractable events → return empty arrays
    If uncertain about extraction → EXCLUDE the information

    Return ONLY the JSON object, no additional text.
                """
                
                # Call Azure OpenAI
                logger.info(f"📤 Calling LLM for treatment history generation (patient: {patient_name})")
                logger.debug(f"📝 Context length: {len(context)} chars")
                
                response = await self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a strict medical document extraction system. You extract ONLY explicitly documented events from external medical documents with zero fabrication, zero interpretation, and zero summarization. You follow the DocLatch Treatment History Timeline specification exactly. You NEVER infer, assume, or generate content not explicitly present in source documents."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Zero temperature for strict extraction
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                
                # Log raw response
                raw_response = response.choices[0].message.content
                logger.info(f"📥 LLM Response received (length: {len(raw_response)} chars)")
                logger.debug(f"📥 Raw LLM Response: {raw_response[:500]}..." if len(raw_response) > 500 else f"📥 Raw LLM Response: {raw_response}")
                
                # Parse response
                history_json = json.loads(raw_response)
                
                # Validate structure
                if not isinstance(history_json, dict):
                    logger.warning(f"⚠️ LLM returned non-dict response for patient {patient_name}, type: {type(history_json)}")
                    logger.warning(f"⚠️ Response content: {raw_response[:300]}")
                    return self._get_empty_history_template()
                
                # Validate required fields in each entry
                for category, events in history_json.items():
                    if isinstance(events, list):
                        validated_events = []
                        for event in events:
                            # Check required fields
                            if not isinstance(event, dict):
                                logger.warning(f"⚠️ Skipping invalid event (not a dict): {event}")
                                continue
                            
                            required_fields = ['date', 'event_type', 'details', 'source', 'author']
                            if all(field in event for field in required_fields):
                                validated_events.append(event)
                            else:
                                missing = [f for f in required_fields if f not in event]
                                logger.warning(f"⚠️ Skipping event missing required fields {missing}: {event}")
                        
                        history_json[category] = validated_events
                
                # Deduplicate events in each category
                for category, events in history_json.items():
                    if isinstance(events, list):
                        history_json[category] = self._deduplicate_events(events)
                
                # Ensure all required categories exist
                required_categories = [
                    "musculoskeletal_system", "cardiovascular_system", 
                    "pulmonary_respiratory", "neurological", "gastrointestinal",
                    "metabolic_endocrine", "genitourinary_renal", "reproductive_obstetric_gynecologic",
                    "dermatological", "ophthalmologic", "ent_head_neck", "dental_oral",
                    "hematologic_lymphatic", "immune_allergy", "psychiatric_mental_health",
                    "sleep_disorders", "other_systems"
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
                logger.info(f"✅ Generated treatment history for patient {patient_name} with {total_events} events across {len(history_json)} categories")
                
                # Log categories with events for debugging
                categories_with_events = {k: len(v) for k, v in history_json.items() if isinstance(v, list) and len(v) > 0}
                if categories_with_events:
                    logger.info(f"📊 Events per category: {categories_with_events}")
                else:
                    logger.warning(f"⚠️ No events extracted for patient {patient_name}. Context provided: {context[:200]}...")
                
                return history_json
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Log detailed error information
                import traceback
                logger.error(f"❌ Treatment history LLM error (attempt {attempt + 1}/{max_retries}) for patient {patient_name}: {str(e)}")
                logger.error(f"❌ Error type: {type(e).__name__}")
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                
                # Check if it's a retryable error
                is_retryable = any(err in error_msg for err in [
                    "connection", "timeout", "rate limit", "429", "503", "502", 
                    "service unavailable", "gateway", "network", "reset"
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    logger.warning(f"⚠️ LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    continue
                else:
                    logger.error(f"❌ Error generating treatment history with LLM: {str(e)}")
                    break
        
        # All retries exhausted or non-retryable error
        logger.error(f"❌ All {max_retries} attempts failed for treatment history generation. Last error: {str(last_error)}")
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
        "genitourinary_renal": {"current": [], "archive": []},
        "reproductive_obstetric_gynecologic": {"current": [], "archive": []},
        "dermatological": {"current": [], "archive": []},
        "ophthalmologic": {"current": [], "archive": []},
        "ent_head_neck": {"current": [], "archive": []},
        "dental_oral": {"current": [], "archive": []},
        "hematologic_lymphatic": {"current": [], "archive": []},
        "immune_allergy": {"current": [], "archive": []},
        "psychiatric_mental_health": {"current": [], "archive": []},
        "sleep_disorders": {"current": [], "archive": []},
        "other_systems": {"current": [], "archive": []},
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
                                        current_document_data: Dict = None,
                                        only_current: bool = False) -> Dict:
        """
        Main function to generate treatment history
        Returns the treatment history data
        """
        logger.info(f"🔄 Treatment history generation is currently DISABLED for patient: {patient_name}")
        return self._get_empty_history_template()
        # try:
        #     logger.info(f"🔄 Generating treatment history for patient: {patient_name} (only_current={only_current})")
        #     
        #     # Get previous documents (skip if only_current is True)
        #     previous_docs = []
        #     if not only_current:
        #         previous_docs = await self.get_patient_documents(
        #             patient_name=patient_name,
        #             dob=dob,
        #             claim_number=claim_number,
        #             physician_id=physician_id,
        #             exclude_document_id=current_document_id
        #         )
        #     
        #     # Prepare context for LLM
        #     context = await self.extract_treatment_history_from_docs(
        #         documents=previous_docs,
        #         current_document=current_document_data
        #     )
        #     
        #     # If no documents found, create minimal context from current analysis
        #     if not context.strip():
        #         logger.info(f"📝 No previous documents found for {patient_name}, creating initial history")
        #         context = f"Initial document for {patient_name}. "
        #         # Removed current_document_analysis dependency
        #     
        #     # Generate treatment history with LLM
        #     treatment_history = await self.generate_treatment_history_with_llm(
        #         patient_name=patient_name,
        #         context=context,
        #         current_document_analysis=current_document_data  # Explicitly None as parameter was removed
        #     )
        #     
        #     logger.info(f"✅ Treatment history generated for patient: {patient_name}")
        #     return treatment_history
        #     
        # except Exception as e:
        #     logger.error(f"❌ Error creating treatment history: {str(e)}")
        #     return self._get_empty_history_template()
    
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
        logger.info(f"💾 Treatment history saving is currently DISABLED for {patient_name}")
        return
        # try:
        #     await self.connect()
        #     
        #     # Check if treatment history already exists
        #     existing = await self.prisma.treatmenthistory.find_unique(
        #         where={
        #             "patientName_dob_claimNumber_physicianId": {
        #                 "patientName": patient_name,
        #                 "dob": dob or "",
        #                 "claimNumber": claim_number or "",
        #                 "physicianId": physician_id
        #             }
        #         }
        #     )
        #     
        #     if existing:
        #         # Merge new history with existing history using date-based logic
        #         existing_data = existing.historyData
        #         if isinstance(existing_data, str):
        #             existing_data = json.loads(existing_data)
        #         
        #         # If existing data is in old format (just lists), convert it first
        #         if isinstance(existing_data, dict):
        #             # Check if any system is still in old format (list instead of dict with current/archive)
        #             for system in existing_data:
        #                 if isinstance(existing_data[system], list):
        #                     existing_data[system] = {
        #                         "current": existing_data[system],
        #                         "archive": []
        #                     }
        #         else:
        #             # If existing_data is not a dict, create empty structure
        #             existing_data = self._get_empty_history_template()
        #         
        #         merged_data = self.merge_history_data(existing_data, history_data)
        #         
        #         # Update existing record
        #         await self.prisma.treatmenthistory.update(
        #             where={"id": existing.id},
        #             data={
        #                 "historyData": json.dumps(merged_data),
        #                 "documentId": document_id,
        #                 "updatedAt": datetime.now()
        #             }
        #         )
        #         logger.info(f"📝 Merged and updated treatment history for {patient_name}")
        #     else:
        #         # For new records, format the LLM output
        #         formatted_data = self._get_empty_history_template()
        #         for system, events in history_data.items():
        #             if system in formatted_data and events:
        #                 # Sort new events by date (newest first) and deduplicate
        #                 unique_events = self._deduplicate_events(events)
        #                 formatted_data[system]["current"] = sorted(
        #                     unique_events, 
        #                     key=lambda x: self._parse_date(x.get('date', '')), 
        #                     reverse=True
        #                 )
        #         
        #         # Create new record
        #         await self.prisma.treatmenthistory.create(
        #             data={
        #                 "patientName": patient_name,
        #                 "dob": dob,
        #                 "claimNumber": claim_number,
        #                 "physicianId": physician_id,
        #                 "historyData": json.dumps(formatted_data),
        #                 "documentId": document_id
        #             }
        #         )
        #         logger.info(f"📝 Created new treatment history for {patient_name}")
        #         
        # except Exception as e:
        #     logger.error(f"❌ Error saving treatment history to database: {str(e)}")
        #     raise
    
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