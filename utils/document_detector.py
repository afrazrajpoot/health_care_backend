"""
Document type detection with hybrid pattern + LLM approach
"""
import re
import logging
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import DocumentType

logger = logging.getLogger("document_ai")


class DocumentTypeDetector:
    """
    Hybrid document type detection with context-aware pattern matching.
    Prevents misclassification (e.g., PR-2 mentioning MRI classified as MRI report).
    """
    
    # [KEEP ALL THE DOC_TYPE_PATTERNS EXACTLY AS IN YOUR FILE - file:27]
    DOC_TYPE_PATTERNS = {
        # Med-Legal (highest priority - comprehensive reports)
        DocumentType.QME: {
            "primary": [
                r'\bQME\b.*(?:Report|Evaluation)',
                r'Qualified Medical Evaluator',
                r'QME.*(?:Exam|Evaluation)'
            ],
            "context": [
                r'apportionment',
                r'whole person impairment',
                r'medical-legal',
                r'permanent.*stationary',
                r'WPI',
                r'causation.*opinion'
            ],
            "exclude": [
                r'QME.*(?:requested|pending|scheduled)',
                r'request.*QME',
                r'awaiting.*QME'
            ]
        },
        DocumentType.AME: {
            "primary": [
                r'\bAME\b.*(?:Report|Evaluation)',
                r'Agreed Medical Evaluator',
                r'AME.*(?:Exam|Evaluation)'
            ],
            "context": [
                r'agreed.*evaluation',
                r'impairment.*rating',
                r'permanent.*stationary'
            ],
            "exclude": [
                r'AME.*(?:requested|pending)',
                r'request.*AME'
            ]
        },
        DocumentType.IME: {
            "primary": [
                r'\bIME\b.*Report',
                r'Independent Medical Exam',
                r'IME.*Evaluation'
            ],
            "context": [
                r'independent.*evaluation',
                r'examining.*physician'
            ],
            "exclude": [
                r'IME.*(?:requested|pending)'
            ]
        },
        
        # Authorization Workflow
        DocumentType.UR: {
            "primary": [
                r'\bUR\b.*(?:Decision|Report)',
                r'Utilization Review',
                r'(?:not|non).*certified',
                r'non-certification'
            ],
            "context": [
                r'denied',
                r'medical necessity.*not',
                r'UR.*determination',
                r'request.*not.*approved'
            ],
            "exclude": []
        },
        DocumentType.RFA: {
            "primary": [
                r'\bRFA\b',
                r'Request for Authorization',
                r'DWC.*Form.*RFA'
            ],
            "context": [
                r'requesting.*authorization',
                r'treatment.*requested',
                r'authorization.*for'
            ],
            "exclude": [
                r'RFA.*(?:denied|approved)',
                r'UR.*decision'
            ]
        },
        DocumentType.AUTHORIZATION: {
            "primary": [
                r'authorization.*approved',
                r'certified.*approved',
                r'auth.*granted'
            ],
            "context": [
                r'approval.*date',
                r'authorized.*(?:visits|sessions)',
                r'treatment.*authorized'
            ],
            "exclude": [
                r'request.*authorization',
                r'denied'
            ]
        },
        
        # Progress Reports
        DocumentType.PR2: {
            "primary": [
                r'\bPR-2\b',
                r'Progress Report',
                r'Primary Treating Physician.*Report'
            ],
            "context": [
                r'work status',
                r'temporary disability',
                r'treatment.*plan',
                r'current.*status'
            ],
            "exclude": []
        },
        DocumentType.PR4: {
            "primary": [
                r'\bPR-4\b',
                r'Permanent.*(?:and|&).*Stationary',
                r'Final.*Report'
            ],
            "context": [
                r'permanent.*impairment',
                r'future.*medical.*care',
                r'MMI',
                r'P&S'
            ],
            "exclude": []
        },
        DocumentType.DFR: {
            "primary": [
                r'\bDFR\b',
                r"Doctor'?s First Report",
                r'First Report.*Injury'
            ],
            "context": [
                r'date.*of.*injury',
                r'injury.*occurred',
                r'initial.*evaluation',
                r'DOI'
            ],
            "exclude": []
        },
        
        # Specialist Consults
        DocumentType.CONSULT: {
            "primary": [
                r'consultation.*report',
                r'specialist.*consult',
                r'seen.*in.*consultation',
                r'consult.*note'
            ],
            "context": [
                r'referred.*by',
                r'consultation.*with',
                r'recommendations.*include',
                r'plan:.*(?:PT|therapy|surgery)'
            ],
            "exclude": [
                r'MRI.*Report',
                r'CT.*Report',
                r'X-ray.*Report',
                r'EMG.*Report'
            ]
        },
        
        # Imaging (lower priority - often mentioned in other docs)
        DocumentType.MRI: {
            "primary": [
                r'MRI.*Report',
                r'Magnetic Resonance.*Imaging.*Report',
                r'Impression:.*MRI',
                r'MRI.*(?:Study|Examination).*of'
            ],
            "context": [
                r'T[12].*weighted',
                r'sequences.*obtained',
                r'radiologist',
                r'contrast.*(?:enhanced|administered)',
                r'FINDINGS:'
            ],
            "exclude": [
                r'MRI.*(?:ordered|pending|requested|needed)',
                r'recommend.*MRI',
                r'will.*obtain.*MRI'
            ]
        },
        DocumentType.CT: {
            "primary": [
                r'CT.*Report',
                r'Computed Tomography.*Report',
                r'CT.*Scan.*of'
            ],
            "context": [
                r'contrast.*enhanced',
                r'axial.*images',
                r'radiologist',
                r'Hounsfield'
            ],
            "exclude": [
                r'CT.*(?:ordered|pending|requested)'
            ]
        },
        DocumentType.XRAY: {
            "primary": [
                r'X-ray.*Report',
                r'Radiograph.*of',
                r'Plain.*Film.*Report'
            ],
            "context": [
                r'AP.*view',
                r'lateral.*view',
                r'radiographic.*findings',
                r'projection'
            ],
            "exclude": [
                r'X-ray.*(?:ordered|requested)'
            ]
        },
        DocumentType.EMG: {
            "primary": [
                r'EMG.*Report',
                r'Electromyography.*Report',
                r'Nerve Conduction.*Study'
            ],
            "context": [
                r'motor.*nerve',
                r'sensory.*nerve',
                r'neurologist',
                r'electrodiagnostic'
            ],
            "exclude": [
                r'EMG.*(?:ordered|pending)'
            ]
        },
        DocumentType.ULTRASOUND: {
            "primary": [
                r'Ultrasound.*Report',
                r'Sonogram',
                r'Duplex.*Study'
            ],
            "context": [
                r'ultrasound.*examination',
                r'sonographic.*findings'
            ],
            "exclude": [
                r'ultrasound.*ordered'
            ]
        },
        DocumentType.LABS: {
            "primary": [
                r'Laboratory.*Report',
                r'Lab.*Results',
                r'Pathology.*Report'
            ],
            "context": [
                r'reference.*range',
                r'abnormal.*values',
                r'specimen',
                r'CBC|CMP|A1C|glucose'
            ],
            "exclude": [
                r'lab.*(?:ordered|pending)'
            ]
        },
        
        # Administrative
        DocumentType.ADJUSTER: {
            "primary": [
                r'claims.*adjuster',
                r'adjuster.*letter',
                r'administrator.*letter'
            ],
            "context": [
                r'request.*documents',
                r'provide.*information',
                r'claims.*department'
            ],
            "exclude": []
        },
        DocumentType.ATTORNEY: {
            "primary": [
                r'attorney.*letter',
                r'law.*office.*of',
                r'legal.*counsel'
            ],
            "context": [
                r'applicant.*attorney',
                r'defense.*attorney',
                r'lien.*claim'
            ],
            "exclude": []
        },
        DocumentType.NCM: {
            "primary": [
                r'\bNCM\b',
                r'Nurse Case Manager',
                r'Case Management.*(?:Letter|Report)'
            ],
            "context": [
                r'case.*manager',
                r'nurse.*coordination',
                r'care.*coordination'
            ],
            "exclude": []
        },
        DocumentType.SIGNATURE_REQUEST: {
            "primary": [
                r'signature.*(?:request|required)',
                r'please.*sign',
                r'authorization.*to.*treat'
            ],
            "context": [
                r'signed.*order',
                r'physician.*signature'
            ],
            "exclude": []
        },
        
        # General Medicine
        DocumentType.REFERRAL: {
            "primary": [
                r'Referral.*to',
                r'Consultation.*Request',
                r'Please.*evaluate'
            ],
            "context": [
                r'referred.*for',
                r'evaluation.*requested'
            ],
            "exclude": []
        },
        DocumentType.DISCHARGE: {
            "primary": [
                r'Discharge.*Summary',
                r'Hospital.*Discharge'
            ],
            "context": [
                r'admitted.*on',
                r'discharged.*on',
                r'hospital.*course'
            ],
            "exclude": []
        },
        DocumentType.MED_REFILL: {
            "primary": [
                r'prescription.*refill',
                r'medication.*renewal',
                r'Rx.*refill'
            ],
            "context": [
                r'refill.*request',
                r'pharmacy'
            ],
            "exclude": []
        }
    }
   
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def detect(self, text: str) -> DocumentType:
        """Hybrid detection with pattern + LLM fallback"""
        # Try pattern-based first (fast, accurate for 90% cases)
        detected = self._pattern_based_detection(text[:3000])
        if detected and detected != DocumentType.UNKNOWN:
            logger.info(f"✅ Pattern detected: {detected.value}")
            return detected
        
        # Fallback to LLM for edge cases
        logger.info("⚠️ Pattern unclear, using LLM classifier")
        return self._llm_based_detection(text[:4000])
    
    def _pattern_based_detection(self, text: str) -> DocumentType:
        """Context-aware pattern matching with scoring"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            score = 0
            
            # Check exclusions first
            excluded = False
            for exclude_pattern in patterns.get("exclude", []):
                if re.search(exclude_pattern, text_lower, re.IGNORECASE):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Primary patterns (weight: 10)
            for primary in patterns["primary"]:
                if re.search(primary, text_lower, re.IGNORECASE):
                    score += 10
            
            # Context patterns (weight: 3)
            for context in patterns["context"]:
                if re.search(context, text_lower, re.IGNORECASE):
                    score += 3
            
            if score > 0:
                scores[doc_type] = score
        
        if scores and max(scores.values()) >= 10:
            return max(scores, key=scores.get)
        
        return DocumentType.UNKNOWN
    
    def _llm_based_detection(self, text: str) -> DocumentType:
        """LLM fallback for ambiguous cases"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        prompt = PromptTemplate(
            template="""
Analyze this medical document and classify its PRIMARY type.

CRITICAL RULES:
1. If imaging (MRI/CT/X-ray) is only MENTIONED but not the PRIMARY SUBJECT → NOT an imaging report
2. QME/AME/IME = comprehensive med-legal evaluations (not simple progress notes)
3. PR-2 = progress reports; PR-4 = final/permanent & stationary reports
4. RFA = requests; UR = denials; Authorization = approvals

Document excerpt:
{text}

Classify into ONE type from this list:
MRI, CT, X-ray, Ultrasound, EMG, Labs, PR-2, PR-4, DFR, Consult, RFA, UR, Authorization,
Peer-to-Peer, QME, AME, IME, Adjuster, Attorney, NCM, Signature Request, Referral,
Discharge, Med Refill

Return JSON:
{{"doc_type": "exact_type_from_list", "confidence": "high|medium|low", "reason": "brief justification"}}

{format_instructions}
""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({"text": text})
            doc_type_str = result.get("doc_type", "Unknown")
            
            # Map string to enum
            for dt in DocumentType:
                if dt.value == doc_type_str:
                    return dt
            
            return DocumentType.UNKNOWN
        except Exception as e:
            logger.error(f"❌ LLM detection failed: {e}")
            return DocumentType.UNKNOWN
