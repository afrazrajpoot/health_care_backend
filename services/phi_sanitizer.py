from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import re
import logging

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import re
import logging

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Add custom recognizer for Claim Number
from presidio_analyzer import Pattern
claim_pattern = Pattern(name="claim", regex=r"(?i)claim\s*#?:?\s*[A-Z0-9-]+", score=0.8)
claim_recognizer = PatternRecognizer(
    supported_entity="CLAIM_NUMBER",
    patterns=[claim_pattern]
)
analyzer.registry.add_recognizer(claim_recognizer)

logger = logging.getLogger("phi_sanitizer")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def redact_phi(text: str):
    """
    Removes PHI from medical text and returns:
    - sanitized_text: redacted version for OpenAI
    - phi_map: mapping of placeholders to original PHI for re-linking
    """
    logger.info("Starting PHI redaction...")
    results = analyzer.analyze(text=text, language='en')

    phi_map = {}
    sanitized_text = text

    for i, r in enumerate(sorted(results, key=lambda x: x.start, reverse=True)):
        original_value = text[r.start:r.end]
        entity = r.entity_type.upper()

        # Normalize entity labels (improve relinking consistency)
        if "DATE_TIME" in entity:
            if re.search(r"dob|birth", original_value, re.I):
                entity = "DOB"
            elif re.search(r"doi|injury", original_value, re.I):
                entity = "DOI"
            else:
                entity = "DATE"
        elif "CLAIM" in entity or re.search(r"claim", original_value, re.I):
            entity = "CLAIM_NUMBER"

        placeholder = f"<{entity}_{i}>"
        phi_map[placeholder] = original_value
        sanitized_text = sanitized_text[:r.start] + placeholder + sanitized_text[r.end:]

        logger.info(f"🔒 PHI replaced: {entity} -> {placeholder} ({original_value})")

    logger.info(f"PHI map: {phi_map}")
    return sanitized_text, phi_map

