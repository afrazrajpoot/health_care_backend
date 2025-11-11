"""
Enhanced Doctor Detector with Zone-Awareness (Relaxed Title Rule)
→ Keeps ALL your logic: zones, exclusions, priority, LLM + regex
→ BUT: Title (Dr./MD/DO) is NOT required — just find WHO signed
"""
import logging
import re
from typing import Optional, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger("document_ai")


class DoctorDetector:
    """
    Zone-aware signer extraction (relaxed):
    - Prioritizes LAST PAGE signature → FIRST PAGE header
    - Excludes: referral, ordering, PCP, dictated, CC
    - Returns ANY person who signed (Dr., PA, NP, resident, etc.)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = StrOutputParser()

        # Regex patterns for names (with or without title)
        self.name_patterns = [
            r"Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+",  # Dr. John A. Smith
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+,\s*(?:MD|DO|M\.D\.|D\.O\.|PA|NP|RN|PT)",  # John Smith, MD/PA/NP
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+\s*(?:MD|DO|M\.D\.|D\.O\.|MBBS|MBChB|PA|NP)",  # John Smith MD
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+",  # John Smith (no title)
        ]

        # Exclusion phrases
        self.exclusion_phrases = [
            "referred by", "referring physician", "referring provider", "referring doctor",
            "referred to", "referral to", "referral", "ordering physician", "ordering provider",
            "primary care", "pcp", "pcp:", "family physician", "dictated by", "transcribed by",
            "scribe", "cc:", "copied to", "according to dr.", "per dr.", "previously seen by",
            "seen by dr.", "evaluated by dr."
        ]

        # Positive context phrases (signing indicators)
        self.positive_context_phrases = [
            "electronically signed by", "signed by", "attending physician", "examining physician",
            "evaluating physician", "consulting physician", "treating physician",
            "primary treating physician", "qualified medical evaluator", "qme", "ame", "ime",
            "radiologist", "interpreting physician", "examiner", "consultant", "provider:",
            "signature:", "/s/", "signed:"
        ]

    def detect_doctor(
        self,
        text: str = None,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, str]:
        """
        Detect ANY signer with zone-awareness.
        """
        if page_zones:
            logger.info(f"Doctor detection: Checking page_zones (keys: {list(page_zones.keys())})")
            
            numeric_keys = [k for k in page_zones.keys() if str(k).isdigit()]
            if not numeric_keys:
                logger.warning("No numeric page keys found in page_zones")
            else:
                last_page_num = max(int(k) for k in numeric_keys)
                logger.info(f"Last page number: {last_page_num}")
                
                pages_to_check = min(5, last_page_num)
                start_page = max(1, last_page_num - pages_to_check + 1)
                logger.info(f"Checking signature zones in pages {start_page} to {last_page_num}")
                
                for page_num in range(last_page_num, start_page - 1, -1):
                    page = page_zones.get(str(page_num), {})
                    header = page.get("header", "").strip()
                    body = page.get("body", "").strip()
                    footer = page.get("footer", "").strip()
                    signature = page.get("signature", "").strip()
                    
                    logger.info(f"Page {page_num} zones: H:{len(header)}, B:{len(body)}, F:{len(footer)}, S:{len(signature)}")
                    
                    # Priority: signature → footer → body end
                    for zone_text, zone_name in [
                        (signature, "signature"),
                        (footer, "footer"),
                        (body[-500:] if len(body) > 500 else body, "body_end")
                    ]:
                        if not zone_text:
                            continue
                        logger.info(f"Checking {zone_name} on page {page_num}...")
                        name = self._detect_from_zone(zone_text, is_signature_zone=True)
                        if name:
                            return {
                                "doctor_name": name,
                                "confidence": "high" if zone_name in ["signature", "footer"] else "medium",
                                "source": f"page_{page_num}_{zone_name}",
                                "validation_notes": f"Signer found in {zone_name} on page {page_num}"
                            }
                
                logger.warning(f"No signer in last {pages_to_check} pages")

        # First 1-3 pages: header + body start
        if page_zones:
            numeric_keys = [k for k in page_zones.keys() if str(k).isdigit()]
            if numeric_keys:
                max_page = max(int(k) for k in numeric_keys)
                pages_to_check = min(3, max_page)
                logger.info(f"Checking first {pages_to_check} pages for treating provider")
                
                for page_num in range(1, pages_to_check + 1):
                    page = page_zones.get(str(page_num), {})
                    header = page.get("header", "").strip()
                    body = page.get("body", "").strip()
                    
                    if header:
                        name = self._detect_from_zone(header, is_signature_zone=False)
                        if name:
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_header",
                                "validation_notes": f"Treating provider in header on page {page_num}"
                            }
                    
                    if body:
                        body_start = body[:500] if len(body) > 500 else body
                        name = self._detect_from_zone(body_start, is_signature_zone=False)
                        if name:
                            return {
                                "doctor_name": name,
                                "confidence": "medium",
                                "source": f"page_{page_num}_body_start",
                                "validation_notes": f"Provider in body start on page {page_num}"
                            }

        # Full body fallback
        if text:
            name = self._detect_from_body(text)
            if name:
                return {
                    "doctor_name": name,
                    "confidence": "low",
                    "source": "body",
                    "validation_notes": "Signer inferred from body"
                }
        
        return {
            "doctor_name": "",
            "confidence": "none",
            "source": "none",
            "validation_notes": "No signer found"
        }

    def _detect_from_zone(self, text_block: str, is_signature_zone: bool = True) -> str:
        """
        Unified LLM extraction for signature/header/body.
        Title NOT required. Focus: who signed or is listed as provider.
        """
        if not text_block or len(text_block.strip()) < 5:
            return ""

        role = "SIGNER" if is_signature_zone else "TREATING PROVIDER"
        prompt = PromptTemplate(
            template=f"""
Extract the NAME of the {role} from the text.

RULES:
1. Look for name near: "signed by", "electronically signed", "signature:", "/s/", "provider:", "physician:"
2. Name can be: Dr. X, John Doe, Jane Smith, MD, PA, NP, PT, etc.
3. Title is OPTIONAL
4. REJECT:
   - "referred by", "referral", "ordering", "PCP", "dictated by", "transcribed by", "CC:"
5. If multiple names → pick the one in signing context
6. If unclear → "EMPTY"

Text:
{{text_block}}

Output one line: "Dr. John Smith" or "Jane Doe, NP" or "EMPTY"
""",
            input_variables=["text_block"],
        )

        try:
            logger.info(f"LLM analyzing {role.lower()} block ({len(text_block)} chars)...")
            chain = prompt | self.llm | self.parser
            candidate = chain.invoke({"text_block": text_block[:1500]}).strip()
            logger.info(f"LLM candidate: '{candidate}'")
            
            if candidate and candidate != "EMPTY":
                if self._is_valid_signer(text_block, candidate):
                    logger.info(f"Valid signer: {candidate}")
                    return candidate
                else:
                    logger.warning(f"Candidate rejected by context: {candidate}")
        except Exception as e:
            logger.warning(f"LLM detection failed: {e}")

        return ""

    def _detect_from_body(self, text: str) -> str:
        """Regex body scan — now includes untitled names"""
        candidates = []
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text)
            candidates.extend(matches)

        seen = set()
        for c in candidates:
            c = c.strip()
            if c not in seen and self._is_valid_signer(text, c):
                seen.add(c)
                return c
        return ""

    def _is_valid_signer(self, context: str, candidate: str) -> bool:
        """Context validation: exclude bad phrases, prefer signing indicators"""
        ctx = context.lower()
        cand = candidate.lower()
        idx = ctx.find(cand)
        if idx == -1:
            return False

        window = ctx[max(0, idx-120):min(len(ctx), idx+len(cand)+120)]

        # REJECT exclusion phrases
        for phrase in self.exclusion_phrases:
            if phrase in window:
                logger.debug(f"Rejected '{candidate}': exclusion '{phrase}'")
                return False

        # PREFER positive context (not required)
        positive_hits = sum(1 for p in self.positive_context_phrases if p in window)
        logger.debug(f"Positive context hits: {positive_hits}")
        
        return True

    # Removed _has_required_title() — no longer needed