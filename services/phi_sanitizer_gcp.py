# services/phi_sanitizer_gcp.py
from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types
import re
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _make_custom_info_types():
    claim_regex = r"(?i)(?:claim(?:\s*(?:number|#)?)\s*[:#]?\s*)([A-Z0-9-]+)"
    medical_regex = r"(?i)(diagnosis|injury|treatment|procedure|disability|fracture|therapy)"
    return [
        types.CustomInfoType(
            info_type=types.InfoType(name="CLAIM_NUMBER"),
            regex=types.CustomInfoType.Regex(pattern=claim_regex)
        ),
        types.CustomInfoType(
            info_type=types.InfoType(name="MEDICAL_CONTEXT"),
            regex=types.CustomInfoType.Regex(pattern=medical_regex)
        )
    ]



def redact_with_dlp(text: str, project_id: str) -> Tuple[str, Dict[str, str]]:
    """
    Uses Google Cloud DLP to detect and redact PHI from a text block.
    Returns (sanitized_text, phi_map).
    """
    client = dlp_v2.DlpServiceClient()

    # ✅ Always use global region for maximum infoType coverage
    parent = f"projects/{project_id}/locations/global"

    inspect_config = {
        "info_types": [
            {"name": "PERSON_NAME"},
            {"name": "DATE"},
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "MEDICAL_TERM"},
            # {"name": "HEALTH_INSURANCE_BENEFICIARY_NUMBER"},
        ],
        "custom_info_types": _make_custom_info_types(),
        "include_quote": True,
        "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
    }

    item = {"value": text}

    # 🔍 Call DLP inspection
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_config": inspect_config,
            "item": item,
        }
    )

    findings = response.result.findings or []
    logger.info(f"✅ DLP detected {len(findings)} PHI entities")

    detected: List[Tuple[str, str]] = []
    for f in findings:
        if not f.quote:
            continue
        detected.append((f.quote, f.info_type.name))

    detected = sorted(detected, key=lambda x: len(x[0]), reverse=True)

    sanitized_text = text
    phi_map: Dict[str, str] = {}
    counters: Dict[str, int] = {}

    for quote, info_name in detected:
        label = info_name.upper()

        # Contextual refinement for DOB / DOI
        if label in {"DATE", "PERSON_NAME"}:
            idx = text.find(quote)
            window = text[max(0, idx - 40): idx + len(quote) + 40].lower()
            if re.search(r"birth|dob|date of birth", window):
                label = "DOB"
            elif re.search(r"inju|injury|date of injury|doi", window):
                label = "DOI"

        if "CLAIM" in label:
            label = "CLAIM_NUMBER"

        counters[label] = counters.get(label, 0) + 1
        placeholder = f"<{label}_{counters[label]}>"

        # Safe replace (case-insensitive)
        pattern = re.escape(quote)
        new_text, nsub = re.subn(pattern, placeholder, sanitized_text, count=1)
        if nsub == 0:
            new_text, nsub = re.subn(pattern, placeholder, sanitized_text, count=1, flags=re.IGNORECASE)

        if nsub > 0:
            sanitized_text = new_text
            phi_map[placeholder] = quote
            logger.info(f"🔒 Replaced PHI: {quote} → {placeholder}")
        else:
            logger.warning(f"⚠️ Could not replace PHI quote: {quote}")

    logger.info(f"✅ Sanitized text length: {len(sanitized_text)}, PHI map size: {len(phi_map)}")
    return sanitized_text, phi_map


def relink_phi(text: str, phi_map: Dict[str, str]) -> str:
    """
    Restores original PHI into summarized text after AI processing.
    """
    for placeholder in sorted(phi_map.keys(), key=len, reverse=True):
        original = phi_map[placeholder]
        text = re.sub(re.escape(placeholder), original, text, flags=re.IGNORECASE)

        label = placeholder.strip("<>").split("_")[0].upper()
        if label == "CLAIM_NUMBER":
            text = re.sub(
                r"(claim\s*(number|#)?\s*:?\s*)" + re.escape(placeholder),
                lambda m: m.group(1) + original,
                text,
                flags=re.IGNORECASE,
            )
        elif label == "DOB":
            text = re.sub(
                r"(dob\s*:?\s*)" + re.escape(placeholder),
                lambda m: m.group(1) + original,
                text,
                flags=re.IGNORECASE,
            )
        elif label == "DOI":
            text = re.sub(
                r"(doi\s*:?\s*)" + re.escape(placeholder),
                lambda m: m.group(1) + original,
                text,
                flags=re.IGNORECASE,
            )

    return text
