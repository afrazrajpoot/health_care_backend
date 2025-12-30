import json
import asyncio
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
            
            logger.info(f"✅ Generated treatment history with {sum(len(v) for v in history_json.values())} events across {len(history_json)} categories")
            return history_json
            
        except Exception as e:
            logger.error(f"❌ Error generating treatment history with LLM: {str(e)}")
            return self._get_empty_history_template()
    
    def _get_empty_history_template(self) -> Dict[str, List]:
        """Return empty treatment history template"""
        return {
            "musculoskeletal_system": [],
            "cardiovascular_system": [],
            "pulmonary_respiratory": [],
            "neurological": [],
            "gastrointestinal": [],
            "metabolic_endocrine": [],
            "other_systems": [],
            "general_treatments": []
        }
    
    async def create_or_update_treatment_history(self, 
                                                patient_name: str,
                                                dob: Optional[str],
                                                claim_number: Optional[str],
                                                physician_id: str,
                                                current_document_id: Optional[str] = None,
                                                current_document_analysis: Any = None,
                                                current_document_data: Dict = None) -> Dict:
        """
        Main function to create or update treatment history
        Returns the treatment history data
        """
        try:
            logger.info(f"🔄 Creating/updating treatment history for patient: {patient_name}")
            
            # Get previous documents
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
            
            # Save to database
            await self.save_treatment_history(
                patient_name=patient_name,
                dob=dob,
                claim_number=claim_number,
                physician_id=physician_id,
                history_data=treatment_history,
                document_id=current_document_id
            )
            
            logger.info(f"✅ Treatment history saved/updated for patient: {patient_name}")
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
        Save treatment history to database
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
                # Update existing record
                await self.prisma.treatmenthistory.update(
                    where={"id": existing.id},
                    data={
                        "historyData": json.dumps(history_data),
                        "documentId": document_id,
                        "updatedAt": datetime.now()
                    }
                )
                logger.info(f"📝 Updated existing treatment history for {patient_name}")
            else:
                # Create new record
                await self.prisma.treatmenthistory.create(
                    data={
                        "patientName": patient_name,
                        "dob": dob,
                        "claimNumber": claim_number,
                        "physicianId": physician_id,
                        "historyData": json.dumps(history_data),
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
                    return json.loads(history.historyData)
                return history.historyData
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving treatment history: {str(e)}")
            return None