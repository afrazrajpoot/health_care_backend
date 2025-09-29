# services/review_service.py
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.schemas import ExtractionResult, ComprehensiveAnalysis


class ReviewService:
    """
    Centralizes logic for deciding whether a parsed document should be routed
    to the Review Queue (missing required fields, low confidence, OCR issues).
    """

    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        # Minimal required fields per doc (can be extended or customized by clinic)
        self.required_fields = ["patient_name", "time_day"]  # claim_no considered optional but flagged

    def generate_review_tickets(self, extraction_result: ExtractionResult, analysis: Optional[ComprehensiveAnalysis] = None) -> List[Dict[str, Any]]:
        tickets: List[Dict[str, Any]] = []
        # 1) Confidence-based
        if extraction_result.confidence and extraction_result.confidence < self.confidence_threshold:
            tickets.append({
                "reason": "Low OCR/Extraction confidence",
                "confidence": extraction_result.confidence,
                "suggested_action": "Review text, rerun with higher-quality OCR or upload better scan",
                "created_at": datetime.utcnow().isoformat()
            })

        # 2) Missing required structured fields
        if analysis and analysis.report_json:
            for field in self.required_fields:
                value = getattr(analysis.report_json, field, None)
                if not value:
                    tickets.append({
                        "reason": f"Missing required field: {field}",
                        "field": field,
                        "suggested_action": f"Please confirm or enter {field}",
                        "created_at": datetime.utcnow().isoformat()
                    })

        # 3) Detection of "no text" handled somewhere else, but double-check
        if not extraction_result.text or not extraction_result.text.strip():
            tickets.append({
                "reason": "No readable text extracted",
                "suggested_action": "Run enhanced OCR, or request original electronic document",
                "created_at": datetime.utcnow().isoformat()
            })

        return tickets
