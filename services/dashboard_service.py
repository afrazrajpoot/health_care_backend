# services/dashboard_service.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from services.rule_engine import RuleEngine
from services.database_service import get_database_service


class DashboardService:
    """
    Assembles the structured payload that the dashboard UI will bind to.
    The output is deterministic and uses persisted documents + rule-engine outputs.
    """

    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        self.rule_engine = rule_engine or RuleEngine()

    async def get_patient_dashboard(self, patient_name: str) -> Dict[str, Any]:
        db = await get_database_service()
        latest_doc = await db.get_last_document_for_patient(patient_name)
        if not latest_doc:
            return {"error": f"No documents found for patient: {patient_name}"}

        # Patient snapshot
        patient_snapshot = {
            "patient_name": latest_doc.get("patientName"),
            "claim_number": latest_doc.get("claimNumber"),
            "latest_report_title": latest_doc.get("reportTitle"),
            "latest_report_date": latest_doc.get("reportDate").isoformat() if latest_doc.get("reportDate") else None,
            "work_status": latest_doc.get("status"),
            "document_id": latest_doc.get("id")
        }

        # What's New: prefer stored lastchanges if present, else compute
        whats_new = latest_doc.get("lastchanges")
        if not whats_new:
            previous_summary = []
            # get previous doc
            prev_doc = None
            # Attempt to find the previous document
            # (Assume get_recent_documents returns ordered by createdAt)
            recent_docs = await db.get_recent_documents(limit=5)
            for d in recent_docs:
                if d.get("patientName") == patient_name and d.get("id") != latest_doc.get("id"):
                    prev_doc = d
                    break
            if prev_doc:
                previous_summary = prev_doc.get("summary", [])
            current_summary = latest_doc.get("summary", [])
            whats_new = self.rule_engine.compute_whats_new(previous_summary, current_summary)
        else:
            # If lastchanges is stored as a string, keep; if list, return as-is
            if isinstance(whats_new, str):
                whats_new = [whats_new]

        # Report summaries - latest doc summary as list
        report_summaries = latest_doc.get("summary", [])

        # Orders & referrals - try to read stored actions or fallback to empty
        actions = latest_doc.get("actions")
        if not actions:
            actions = []
        else:
            # If persisted as JSON string, ensure parsed
            if isinstance(actions, str):
                try:
                    actions = json.loads(actions)
                except:
                    actions = []

        # Compliance nudges - urgent alerts
        alerts = latest_doc.get("alerts", [])
        if isinstance(alerts, str):
            try:
                alerts = json.loads(alerts)
            except:
                alerts = []

        urgent_alerts = [a for a in alerts if a.get("status") == "urgent"]

        payload = {
            "patient_snapshot": patient_snapshot,
            "whats_new": whats_new,
            "report_summaries": report_summaries,
            "orders_referrals": actions,
            "compliance_nudges": urgent_alerts,
        }

        return payload
