"""
Enhanced Doctor Detector with Zone-Awareness (First & Last Page Priority)
Uses Document AI page zones for accurate main/treating doctor extraction.
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
    Zone-aware doctor extraction with strict validation:
    - Prioritizes FIRST PAGE header + LAST PAGE signature zones
    - Must have title: Dr., MD, DO, M.D., D.O., MBBS, MBChB
    - Excludes: referring/referral/ordering/PCP/dictated/transcribed/CC
    - Returns only main/treating/consulting doctor
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = StrOutputParser()

        # Regex patterns for titled physician names
        self.name_patterns = [
            r"Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+",  # Dr. John A. Smith
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+,\s*(?:MD|DO|M\.D\.|D\.O\.)",  # John Smith, MD
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+\s*(?:MD|DO|M\.D\.|D\.O\.)",  # John Smith MD
            r"[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)+\s*(?:MBBS|MBChB)",  # John Smith MBBS
        ]

        # Exclusion phrases (referring/referral contexts)
        self.exclusion_phrases = [
            "referred by",
            "referring physician",
            "referring provider",
            "referring doctor",
            "referred to",
            "referral to",
            "referral",  # NEW: catch "referral" alone
            "ordering physician",
            "ordering provider",
            "primary care",
            "pcp",
            "pcp:",
            "family physician",
            "dictated by",
            "transcribed by",
            "scribe",
            "cc:",
            "copied to",
            "according to dr.",
            "per dr.",
            "previously seen by",
            "seen by dr.",
            "evaluated by dr.",
        ]

        # Positive context phrases (main/treating doctor indicators)
        self.positive_context_phrases = [
            "electronically signed by",
            "signed by",
            "attending physician",
            "examining physician",
            "evaluating physician",
            "consulting physician",
            "treating physician",  # NEW
            "primary treating physician",  # NEW
            "qualified medical evaluator",
            "qme",
            "ame",
            "ime",
            "radiologist",
            "interpreting physician",
            "examiner",
            "consultant",
        ]

    def detect_doctor(
        self,
        text: str = None,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, str]:
        """
        Detect main/treating doctor with zone-awareness.
        
        Args:
            text: Full document text (fallback if zones not available)
            page_zones: Dict of {page_num: {header, body, footer, signature}}
        
        Returns:
            {
                "doctor_name": str,
                "confidence": "high" | "medium" | "low" | "none",
                "source": "last_page_signature" | "first_page_header" | "body" | "none",
                "validation_notes": str
            }
        """
        # Priority 1: Last 4-5 pages signature (most reliable)
        if page_zones:
            logger.info(f"🔍 Doctor detection: Checking page_zones (keys: {list(page_zones.keys())})")
            
            # Get last page number
            numeric_keys = [k for k in page_zones.keys() if str(k).isdigit()]
            if not numeric_keys:
                logger.warning("⚠️ No numeric page keys found in page_zones")
            else:
                last_page_num = max(int(k) for k in numeric_keys)
                logger.info(f"📄 Last page number: {last_page_num}")
                
                # Check last 5 pages for signature (or fewer if document is shorter)
                pages_to_check = min(5, last_page_num)
                start_page = max(1, last_page_num - pages_to_check + 1)
                logger.info(f"🔍 Checking signature zones in pages {start_page} to {last_page_num}")
                
                # Search from last page backwards (prioritize most recent pages)
                for page_num in range(last_page_num, start_page - 1, -1):
                    page = page_zones.get(str(page_num), {})
                    
                    # Log all zones for this page
                    header = page.get("header", "").strip()
                    body = page.get("body", "").strip()
                    footer = page.get("footer", "").strip()
                    signature = page.get("signature", "").strip()
                    
                    logger.info(f"📄 Page {page_num} zones:")
                    logger.info(f"   - Header: {len(header)} chars")
                    logger.info(f"   - Body: {len(body)} chars")
                    logger.info(f"   - Footer: {len(footer)} chars")
                    logger.info(f"   - Signature: {len(signature)} chars")
                    
                    if footer:
                        logger.info(f"   📝 Footer preview (first 200 chars): {footer[:200]}...")
                    
                    # Check signature zone first
                    if signature:
                        logger.info(f"   📝 Signature content: {signature[:200]}...")
                        name = self._detect_from_signature(signature)
                        if name:
                            logger.info(f"   ✅ Doctor found: {name}")
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_signature",
                                "validation_notes": f"Main doctor found in signature zone on page {page_num}"
                            }
                        else:
                            logger.warning(f"   ⚠️ Signature present but no valid doctor extracted")
                    
                    # Check footer zone as fallback
                    if footer:
                        logger.info(f"   🔍 Checking footer for doctor...")
                        name = self._detect_from_signature(footer)  # Use same strict logic
                        if name:
                            logger.info(f"   ✅ Doctor found in footer: {name}")
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_footer",
                                "validation_notes": f"Main doctor found in footer zone on page {page_num}"
                            }
                        else:
                            logger.warning(f"   ⚠️ Footer present but no valid doctor extracted")
                    
                    # Check body zone (third priority - sometimes signatures are in body)
                    if body:
                        logger.info(f"   🔍 Checking body for doctor signature...")
                        # Only check last 500 chars of body (where signatures typically appear)
                        body_end = body[-500:] if len(body) > 500 else body
                        logger.info(f"   📝 Body end preview (last 200 chars): ...{body_end[-200:]}")
                        name = self._detect_from_signature(body_end)
                        if name:
                            logger.info(f"   ✅ Doctor found in body: {name}")
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_body",
                                "validation_notes": f"Main doctor found in body zone on page {page_num}"
                            }
                        else:
                            logger.warning(f"   ⚠️ Body checked but no valid doctor extracted")
                
                logger.warning(f"⚠️ No doctor signature found in last {pages_to_check} pages ({start_page}-{last_page_num})")
        else:
            logger.warning("⚠️ No page_zones provided to doctor detector")
        
        # Priority 2: Check first 1-3 pages (treating physician info often in header/body)
        if page_zones:
            numeric_keys = [k for k in page_zones.keys() if str(k).isdigit()]
            if numeric_keys:
                max_page = max(int(k) for k in numeric_keys)
                pages_to_check_start = min(3, max_page)
                logger.info(f"🔍 Checking first {pages_to_check_start} pages for treating physician info")
                
                for page_num in range(1, pages_to_check_start + 1):
                    page = page_zones.get(str(page_num), {})
                    
                    header = page.get("header", "").strip()
                    body = page.get("body", "").strip()
                    
                    logger.info(f"📄 Page {page_num} (first pages check):")
                    logger.info(f"   - Header: {len(header)} chars")
                    logger.info(f"   - Body: {len(body)} chars")
                    
                    # Check header first (common location for treating physician)
                    if header:
                        logger.info(f"   📝 Header preview: {header[:200]}...")
                        name = self._detect_from_header(header)
                        if name:
                            logger.info(f"   ✅ Doctor found in header: {name}")
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_header",
                                "validation_notes": f"Treating doctor found in header on page {page_num}"
                            }
                    
                    # Check body first 500 chars (where treating physician info typically appears)
                    if body:
                        logger.info(f"   🔍 Checking body start for treating physician...")
                        body_start = body[:500] if len(body) > 500 else body
                        logger.info(f"   📝 Body start preview (first 200 chars): {body_start[:200]}...")
                        name = self._detect_from_signature(body_start)
                        if name:
                            logger.info(f"   ✅ Doctor found in body: {name}")
                            return {
                                "doctor_name": name,
                                "confidence": "high",
                                "source": f"page_{page_num}_body_start",
                                "validation_notes": f"Treating doctor found in body on page {page_num}"
                            }
                
                logger.info(f"⚠️ No treating physician found in first {pages_to_check_start} pages")
        
        # Priority 3: Body text fallback (strict exclusion)
        if text:
            name = self._detect_from_body(text)
            if name:
                return {
                    "doctor_name": name,
                    "confidence": "low",
                    "source": "body",
                    "validation_notes": "Main doctor inferred from body (strict filters applied)"
                }
        
        return {
            "doctor_name": "",
            "confidence": "none",
            "source": "none",
            "validation_notes": "No valid main/treating doctor found"
        }

    def _detect_from_signature(self, signature_text: str) -> str:
        """
        LLM-guided extraction from signature/footer with STRICT title validation.
        ONLY extracts if Dr./MD/DO/etc. present.
        """
        if not signature_text or len(signature_text.strip()) < 8:
            return ""

        prompt = PromptTemplate(
            template="""
Extract the MAIN/TREATING physician's name from the signature/footer block.

STRICT RULES:
1. Name MUST include a title: Dr., MD, DO, M.D., D.O., MBBS, or MBChB
2. IGNORE names without titles (administrative staff, nurses, PAs without MD/DO)
3. REJECT names in these contexts:
   - "Referring physician" or "Referral"
   - "Ordering physician"
   - "PCP" or "Primary Care Physician"
   - "Dictated by", "Transcribed by", "CC:"
4. PREFER names with these contexts:
   - "Electronically signed by Dr. X"
   - "Signed by X, MD"
   - "Attending physician: Dr. X"
   - "Examining/Evaluating/Consulting physician: Dr. X"
   - "Treating Physician: Dr. X"
   - "QME/AME/IME: Dr. X"
   - "Radiologist: Dr. X"

CRITICAL: If no name has a title (Dr./MD/DO/etc.) → return "EMPTY"

Signature block:
{signature_text}

Output exactly one line:
- Valid example: "Dr. Jane A. Smith" or "John B. Doe, MD"
- If no titled physician: "EMPTY"
""",
            input_variables=["signature_text"],
        )

        try:
            logger.info(f"🔍 LLM analyzing signature block ({len(signature_text)} chars)...")
            chain = prompt | self.llm | self.parser
            candidate = chain.invoke({"signature_text": signature_text[:1500]}).strip()
            logger.info(f"🤖 LLM extracted candidate: '{candidate}'")
            
            if candidate and candidate != "EMPTY":
                # Double-check title requirement
                has_title = self._has_required_title(candidate)
                is_valid = self._is_valid_treating_doctor(signature_text, candidate)
                logger.info(f"   ✓ Has title: {has_title}, Is valid: {is_valid}")
                
                if has_title and is_valid:
                    logger.info(f"   ✅ Valid doctor confirmed: {candidate}")
                    return candidate
                else:
                    logger.warning(f"   ⚠️ Candidate rejected - has_title: {has_title}, is_valid: {is_valid}")
            else:
                logger.warning(f"   ⚠️ LLM returned EMPTY or invalid candidate")
        except Exception as e:
            logger.warning(f"Signature detection failed: {e}")

        return ""

    def _detect_from_header(self, header_text: str) -> str:
        """
        LLM-guided extraction from first-page header with title validation.
        """
        if not header_text or len(header_text.strip()) < 8:
            return ""

        prompt = PromptTemplate(
            template="""
Extract the MAIN/TREATING physician's name from the first-page header.

STRICT RULES:
1. Name MUST include a title: Dr., MD, DO, M.D., D.O., MBBS, or MBChB
2. Look for:
   - "Requesting Physician:" (common in RFA forms)
   - "Treating Physician:"
   - "Primary Treating Physician:"
   - "Physician:" or "Provider:" followed by MD/DO
   - "Examining Physician:"
3. REJECT:
   - "Referring physician" or "Referral to"
   - "Ordering physician"
   - "PCP" or "Primary Care"
   - Names without titles
4. ONLY extract if title present (Dr./MD/DO/etc.)

Header section:
{header_text}

Output one line:
- "Dr. Jane Smith" or "Jane Smith, MD"
- Or "EMPTY" if no titled physician
""",
            input_variables=["header_text"],
        )

        try:
            chain = prompt | self.llm | self.parser
            candidate = chain.invoke({"header_text": header_text[:2000]}).strip()
            
            if candidate and candidate != "EMPTY":
                if self._has_required_title(candidate) and self._is_valid_treating_doctor(header_text, candidate):
                    return candidate
        except Exception as e:
            logger.warning(f"Header detection failed: {e}")

        return ""

    def _detect_from_body(self, text: str) -> str:
        """
        Regex-only body extraction with STRICT title and exclusion validation.
        Returns first strictly valid main/treating doctor.
        """
        candidates = []
        for pattern in self.name_patterns:
            candidates.extend(re.findall(pattern, text))

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            c = c.strip()
            if c not in seen:
                seen.add(c)
                unique.append(c)

        for candidate in unique:
            if self._is_valid_treating_doctor(text, candidate):
                return candidate

        return ""

    def _is_valid_treating_doctor(self, context: str, candidate: str) -> bool:
        """
        Strict validator:
        - Has required title (Dr./MD/DO/etc.)
        - NOT in excluded referral/ordering/PCP contexts
        - PREFER positive treating/consulting contexts
        """
        # Must have title
        if not self._has_required_title(candidate):
            return False

        # Context window check (±120 chars)
        ctx = context.lower()
        cand = candidate.lower()
        idx = ctx.find(cand)
        if idx == -1:
            return False

        start = max(0, idx - 120)
        end = min(len(ctx), idx + len(cand) + 120)
        window = ctx[start:end]

        # REJECT if any exclusion phrase present
        for phrase in self.exclusion_phrases:
            if phrase in window:
                logger.debug(f"Rejected '{candidate}': found exclusion phrase '{phrase}'")
                return False

        # PREFER if positive context present (but not required)
        positive_hits = sum(1 for p in self.positive_context_phrases if p in window)
        
        # Accept if no exclusions present (positive context is advisory, not mandatory)
        return True

    def _has_required_title(self, name: str) -> bool:
        """Check if name includes required medical title."""
        t = name.upper()
        required_titles = ["DR.", " MD", " DO", "M.D.", "D.O.", "MBBS", "MBCHB"]
        has_title = any(title in t for title in required_titles)
        
        if not has_title:
            logger.debug(f"Rejected '{name}': no required title (Dr./MD/DO/etc.)")
        
        return has_title
