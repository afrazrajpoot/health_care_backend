# services/rule_engine.py
import difflib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from models.schemas import ComprehensiveAnalysis, AlertModel, ActionModel


class RuleEngine:
    """
    Deterministic rule engine to:
      - generate alerts (deterministically)
      - generate actions/tasks (deterministically)
      - compute What's New by comparing previous/current summary (deterministic diff)
    """

    # Simple keyword dictionaries (extendable/configurable)
    URGENT_KEYWORDS = {"fracture", "tumor", "mass", "acute", "severe", "life-threatening", "embolus"}
    DENIAL_KEYWORDS = {"denied", "denial", "not authorized", "coverage denied", "not covered"}
    REFERRAL_KEYWORDS = {"refer to", "referral", "referred to", "recommend referral", "refer for"}
    TTD_KEYWORDS = {"ttd", "temporary total disability", "temporary total", "total disability"}
    MISSING_DOC_KEYWORDS = {"missing", "not provided", "not present", "absent document", "incomplete documentation"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # deadlines defaults (configurable)
        self.rebuttal_days = int(self.config.get("rebuttal_days", 30))
        self.authorization_check_days = int(self.config.get("authorization_check_days", 7))

    def _has_keyword(self, text: str, keywords: set) -> bool:
        if not text:
            return False
        tl = text.lower()
        return any(kw in tl for kw in keywords)

    def generate_alerts(self, analysis: ComprehensiveAnalysis) -> List[Dict[str, Any]]:
        """
        Generate deterministic alerts based on parsed analysis (no AI).
        Returns list of AlertModel-compatible dicts.
        """
        alerts: List[Dict[str, Any]] = []
        text = (analysis.original_report or "") + " " + " ".join(analysis.summary or [])
        report_json = analysis.report_json or None

        now = datetime.utcnow().date().isoformat()

        # 1) Medical Urgency based on keywords
        if self._has_keyword(text, self.URGENT_KEYWORDS):
            alerts.append({
                "alert_type": "Medical Urgency",
                "title": "Critical finding detected in report (keyword match)",
                "date": now,
                "status": "urgent",
                "source": "rule_engine.keyword_medical_urgency",
                "rule_id": "RE_01"
            })

        # 2) Denial / Coverage issues
        if self._has_keyword(text, self.DENIAL_KEYWORDS):
            alerts.append({
                "alert_type": "Treatment Authorization",
                "title": "Denial or coverage decision detected",
                "date": now,
                "status": "urgent",
                "source": "rule_engine.keyword_denial",
                "rule_id": "RE_02"
            })

        # 3) Work Status Review (TTD)
        if self._has_keyword(text, self.TTD_KEYWORDS) or (report_json and getattr(report_json, "status", "") == "urgent"):
            alerts.append({
                "alert_type": "Work Status Review",
                "title": "Work status / TTD requires review",
                "date": now,
                "status": "normal",
                "source": "rule_engine.keyword_ttd",
                "rule_id": "RE_03"
            })

        # 4) Missing claim or required fields
        if report_json and not getattr(report_json, "claim_no", None):
            alerts.append({
                "alert_type": "Follow-Up Required",
                "title": "Missing claim number — request from uploader",
                "date": now,
                "status": "normal",
                "source": "rule_engine.missing_claim",
                "rule_id": "RE_04"
            })

        # 5) Incomplete documentation
        if self._has_keyword(text, self.MISSING_DOC_KEYWORDS):
            alerts.append({
                "alert_type": "Follow-Up Required",
                "title": "Document indicates missing/incomplete documentation",
                "date": now,
                "status": "normal",
                "source": "rule_engine.incomplete_docs",
                "rule_id": "RE_05"
            })

        if not alerts:
            alerts.append({
                "action_id": f"manualreview_{now.strftime('%Y%m%d%H%M%S')}",
                "type": "manual_review",
                "reason": "No deterministic rule matched - manual review required",
                "due_date": now.date().isoformat(),
                "assignee": "staff_claims",
                "source_doc_id": None,
                "deterministic_rule_id": "RE_ACT_03",
                "suggestion": False,
                "auto_create": True,
                "created_at": now.isoformat()
            })

        return alerts

    def generate_compliance_nudges(self, analysis: ComprehensiveAnalysis) -> List[Dict[str, Any]]:
        nudges = []
        text = (analysis.original_report or "") + " " + " ".join(analysis.summary or [])
        now = datetime.utcnow()

        # Example: Missing documentation → compliance nudge
        if self._has_keyword(text, self.MISSING_DOC_KEYWORDS):
            nudges.append({
                "nudge_id": f"nudge_{now.strftime('%Y%m%d%H%M%S')}",
                "type": "missing_documentation",
                "reason": "Report indicates missing or incomplete documentation",
                "assignee": "compliance_team",
                "due_date": (now + timedelta(days=5)).date().isoformat(),
                "created_at": now.isoformat()
            })

        # Example: Authorization mention → compliance nudge
        if "request for authorization" in text.lower():
            nudges.append({
                "nudge_id": f"nudge_{now.strftime('%Y%m%d%H%M%S')}",
                "type": "authorization_check",
                "reason": "Report contains RFA - check compliance to avoid denial",
                "assignee": "utilization_review",
                "due_date": (now + timedelta(days=self.authorization_check_days)).date().isoformat(),
                "created_at": now.isoformat()
            })

        return nudges

    def generate_referrals(self, analysis: ComprehensiveAnalysis) -> List[Dict[str, Any]]:
        referrals = []
        text = (analysis.original_report or "") + " " + " ".join(analysis.summary or [])
        now = datetime.utcnow()

        if self._has_keyword(text, self.REFERRAL_KEYWORDS):
            referrals.append({
                "referral_id": f"referral_{now.strftime('%Y%m%d%H%M%S')}",
                "specialty": "General",  # could refine if we parse specialty (e.g. ortho, neuro)
                "reason": "Referral explicitly mentioned in report",
                "assignee": "care_coordination",
                "due_date": (now + timedelta(days=7)).date().isoformat(),
                "created_at": now.isoformat()
            })
        return referrals


    def compute_whats_new(self, previous_summary: List[str], current_summary: List[str], min_items: int = 3, max_items: int = 6) -> List[str]:
        """
        Deterministic comparison algorithm producing 3-6 lines describing what changed.
        Logic:
          - For every item in current_summary, find best match in previous_summary.
            - if no good match -> mark as NEW
            - if good match but different -> mark as UPDATED
          - Also include removed items (present in previous, not in current) up to max_items.
        """
        prev = previous_summary or []
        cur = current_summary or []

        results: List[str] = []
        used_prev_idx = set()

        # 1) Compare current items to previous
        for c in cur:
            best_ratio = 0.0
            best_idx = None
            for i, p in enumerate(prev):
                ratio = difflib.SequenceMatcher(None, c, p).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            # decision thresholds
            if best_ratio < 0.45:
                results.append(f"⚠️ New: {c}")
            elif best_ratio < 0.85:
                # considered updated
                prev_text = prev[best_idx] if best_idx is not None else ""
                results.append(f"➝ Updated: {prev_text} → {c}")
                if best_idx is not None:
                    used_prev_idx.add(best_idx)
            else:
                # largely unchanged - skip or mark as unchanged later
                if best_idx is not None:
                    used_prev_idx.add(best_idx)
                    # optionally include minor updates if we need to fill lines
                    # We'll skip unchanged to prioritize new/updated lines

            if len(results) >= max_items:
                break

        # 2) Include removed items (present in prev not matched)
        if len(results) < max_items:
            for i, p in enumerate(prev):
                if i in used_prev_idx:
                    continue
                results.append(f"⬇️ Removed: {p}")
                if len(results) >= max_items:
                    break

        # 3) Ensure at least min_items (pad if necessary)
        if len(results) < min_items:
            # fill with 'No significant changes' or repeat key items
            while len(results) < min_items:
                results.append("✅ No significant changes detected")
        # Trim if exceeded
        return results[:max_items]
