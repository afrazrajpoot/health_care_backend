"""
Universal Prompt Manager - One Smart Prompt for ALL Document Types
"""

import logging
from typing import Dict, Optional, Union, List, Tuple
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from extractors.document_type_groups import DocumentType, DocumentTypeGroups
from extractors.section_mappings import SectionMappings

logger = logging.getLogger("document_ai")

class PromptManager:
    """Universal prompt manager - one smart prompt for ALL 70+ document types"""
    
    def __init__(self):
        self.document_groups = DocumentTypeGroups()
        self.section_mappings = SectionMappings()
    
    def get_extraction_prompt(self, doc_type: str) -> ChatPromptTemplate:
        """Get universal extraction prompt for ALL types"""
        logger.info(f"🎯 Using universal extraction prompt for {doc_type}")
        return self._get_universal_extraction_prompt()
    
    def get_short_summary_prompt(self, doc_type: str) -> ChatPromptTemplate:
        """Get universal short summary prompt for ALL types"""
        logger.info(f"🎯 Using universal short summary prompt for {doc_type}")
        return self._get_universal_short_summary_prompt()
    
    def get_long_summary_prompt(self, doc_type: str) -> ChatPromptTemplate:
        """Get universal long summary prompt for ALL types"""
        logger.info(f"🎯 Using universal long summary prompt for {doc_type}")
        return self._get_universal_long_summary_prompt()
    
    def get_section_mapping(self, doc_type: str) -> List[Tuple[str, str]]:
        """Get type-specific section mapping"""
        normalized_type = self._normalize_doc_type(doc_type)
        group_name = self._find_document_group(normalized_type)
        
        if group_name:
            sections = self.section_mappings.get_sections_for_group(group_name)
            logger.info(f"📋 Using {len(sections)} {group_name} sections for {doc_type}")
            return sections
        else:
            logger.info(f"📋 Using base sections for {doc_type}")
            return self.section_mappings.BASE_SECTIONS

    def _get_universal_long_summary_prompt(self) -> ChatPromptTemplate:
        """Universal long summary prompt for ALL document types"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a medical documentation specialist creating comprehensive structured summaries for all types of medical documents.

UNIVERSAL DOCUMENT PROCESSING FOR ALL TYPES:

DOCUMENT CATEGORIES COVERED:
- Medical Evaluations: QME, AME, IME, PR2, PR4, DFR, Consult, Progress Notes
- Authorization & Utilization: UR, RFA, Authorization, Denial, Appeal, Peer-to-Peer
- Imaging & Diagnostics: MRI, CT, X-ray, Ultrasound, Labs, Pathology, Biopsy
- Specialty Reports: Cardiology, Neurology, Orthopedics, Psychiatry, Psychology, etc.
- Therapy & Treatment: Physical Therapy, Occupational Therapy, Chiropractic, etc.
- Administrative & Legal: Adjuster, Attorney, NCM, Correspondence, etc.
- Work & Vocational: Work Status, Disability, FCE, Vocational Rehab, etc.

YOUR TASK:
Create a comprehensive structured medical summary using the provided section headings.
Use only the information from the extracted data.

WRITING GUIDELINES:
- Use the exact section headings provided
- Write detailed, comprehensive content under each heading
- Include quantitative data and specific findings
- Maintain professional medical terminology
- Connect findings to clinical/administrative implications
- Create a flowing, professional medical narrative

ADAPTIVE CONTENT FOCUS:
- Medical Evaluations: Focus on clinical findings, diagnoses, treatment plans
- Authorization Docs: Focus on decisions, rationale, clinical criteria
- Imaging Reports: Focus on technical aspects, findings, interpretations
- Therapy Notes: Focus on treatment progress, functional assessments
- Administrative Docs: Focus on case details, communications, decisions
- Work Status: Focus on functional capacity, restrictions, work abilities

OUTPUT FORMAT:
- Use the exact section headings provided
- Write comprehensive content under each heading
- Maintain professional structure and medical narrative style"""),

            ("human", """
DOCUMENT TYPE: {doc_type}
SECTION HEADINGS TO USE:
{section_headings}

EXTRACTED DATA:

{raw_data}

Create a comprehensive structured medical summary using the section headings above.
Focus on creating a detailed, professional medical narrative:

STRUCTURED MEDICAL SUMMARY:
""")
        ])

    def _get_universal_short_summary_prompt(self) -> ChatPromptTemplate:
        """Universal short summary prompt for ALL document types"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a medical specialist creating concise summaries from comprehensive medical summaries.

TASK:
Create a concise summary by extracting the most important information from the comprehensive medical summary.

GUIDELINES:
- Extract the key points and essential information
- Focus on the most critical clinical findings and decisions
- Create a flowing, single-paragraph summary
- Maintain professional medical language
- Preserve the essential meaning and facts
- Keep it concise and to the point

ADAPTIVE FOCUS BY DOCUMENT TYPE:
- Clinical Documents: Key diagnoses, significant findings, main recommendations
- Authorization Docs: Primary decision, main rationale, key criteria
- Diagnostic Reports: Significant findings, clinical implications
- Administrative Docs: Key decisions, important actions, next steps
- Work Status: Functional capacity, main restrictions, work ability

OUTPUT:
- Single, flowing paragraph
- Concise and professional
- Complete medical summary
- Natural length based on content"""),

            ("human", """
DOCUMENT TYPE: {doc_type}
COMPREHENSIVE MEDICAL SUMMARY:

{long_summary}

Create a concise summary by extracting the most important information from the comprehensive summary above.
Focus on the key points and essential clinical information:

CONCISE SUMMARY:
""")
        ])

    def _get_universal_extraction_prompt(self) -> ChatPromptTemplate:
        """Universal extraction prompt for ALL document types"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a medical documentation specialist extracting comprehensive information from all types of medical documents.

UNIVERSAL EXTRACTION FOR ALL DOCUMENT TYPES:

DOCUMENT CATEGORIES COVERED:
- Medical Evaluations & Clinical Notes
- Authorization & Utilization Review  
- Imaging & Diagnostic Reports
- Specialty Consultations & Reports
- Therapy & Treatment Notes
- Administrative & Legal Documents
- Work Status & Vocational Assessments

EXTRACTION FOCUS:
- Extract all clinically and administratively relevant information
- Preserve specific medical terminology and measurements
- Include both positive and negative findings
- Capture quantitative data and specific details
- Maintain context and clinical significance

EXTRACTION RULES:
1. Extract explicitly stated information
2. Include quantitative data (measurements, dosages, frequencies)
3. Preserve medical specificity and terminology
4. Return empty for missing information
5. Capture clinical reasoning and implications
6. Include relevant dates, timelines, and contexts

OUTPUT:
- Return structured JSON with all available information
- Use consistent field names across document types
- Include enough detail for comprehensive medical summaries"""),

            ("human", """
DOCUMENT TYPE: {doc_type}
DOCUMENT TEXT:

{text}

Extract comprehensive medical information in structured JSON format.
Focus on the most relevant information for this specific document type:

{format_instructions}
""")
        ])

    def _find_document_group(self, doc_type: str) -> Optional[str]:
        """Find which group a document type belongs to"""
        all_groups = self.document_groups.get_all_groups()
        for group_name, group_types in all_groups.items():
            if doc_type in group_types:
                return group_name
        return None

    def _normalize_doc_type(self, doc_type: str) -> str:
        """Normalize document type for section mapping"""
        if not doc_type:
            return DocumentType.UNKNOWN.value
            
        doc_type_upper = doc_type.upper().strip()
        
        # Exact match first
        for doc_enum in DocumentType:
            if doc_enum.value.upper() == doc_type_upper:
                return doc_enum.value
        
        # Partial matching
        for doc_enum in DocumentType:
            if doc_enum.value.upper() in doc_type_upper or doc_type_upper in doc_enum.value.upper():
                return doc_enum.value
        
        return DocumentType.UNKNOWN.value

    def get_prompt_info(self, doc_type: str) -> Dict:
        """Get prompt information for a document type"""
        normalized_type = self._normalize_doc_type(doc_type)
        group_name = self._find_document_group(normalized_type)
        
        return {
            "original_type": doc_type,
            "normalized_type": normalized_type,
            "document_group": group_name,
            "prompt_strategy": "Universal Adaptive Prompt",
            "coverage": "All Document Types"
        }

    def get_section_info(self, doc_type: str) -> Dict:
        """Get section information for a document type"""
        normalized_type = self._normalize_doc_type(doc_type)
        sections = self.get_section_mapping(doc_type)
        group_name = self._find_document_group(normalized_type)
        
        return {
            "original_type": doc_type,
            "normalized_type": normalized_type,
            "document_group": group_name,
            "total_sections": len(sections),
            "sections": [section[0] for section in sections]
        }