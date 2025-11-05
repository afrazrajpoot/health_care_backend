"""
Data models for document extraction
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DocumentType(Enum):
    """All supported document types"""
    MRI = "MRI"
    CT = "CT"
    XRAY = "X-ray"
    ULTRASOUND = "Ultrasound"
    EMG = "EMG"
    LABS = "Labs"
    PR2 = "PR-2"
    PR4 = "PR-4"
    DFR = "DFR"
    CONSULT = "Consult"
    RFA = "RFA"
    UR = "UR"
    AUTHORIZATION = "Authorization"
    PEER_TO_PEER = "Peer-to-Peer"
    QME = "QME"
    AME = "AME"
    IME = "IME"
    ADJUSTER = "Adjuster"
    ATTORNEY = "Attorney"
    NCM = "NCM"
    SIGNATURE_REQUEST = "Signature Request"
    REFERRAL = "Referral"
    DISCHARGE = "Discharge"
    MED_REFILL = "Med Refill"
    UNKNOWN = "Unknown"


@dataclass
class ExtractionResult:
    """Structured output for all extractions"""
    document_type: str
    document_date: str
    summary_line: str
    examiner_name: Optional[str] = None
    specialty: Optional[str] = None
    body_parts: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
