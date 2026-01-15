import re
from fastapi import File, UploadFile, HTTPException, Query
from typing import List, Optional, Any, Dict
from datetime import datetime
import traceback
import os
import logging
from datetime import datetime
from fastapi import HTTPException, UploadFile, File, Query
from prisma import Prisma
import traceback
from datetime import datetime

# Create Prisma client instance
db = Prisma()

# Module level logger
logger = logging.getLogger(__name__)


async def check_subscription(
    documents: List[UploadFile] = File(...),
    physicianId: str = Query(None),
    userId: str = Query(None)
):
    """Dependency to check subscription before processing documents"""
    
    try:
        # Ensure database is connected
        if not db.is_connected():
            await db.connect()
            print("✅ Database connected in subscription check")

        # Count documents
        document_count = len(documents)
        print(f"📄 Documents to process: {document_count}")
        
        # Use physicianId for subscription check (as per your schema)
        subscription_id = physicianId
        if not subscription_id:
            raise HTTPException(status_code=400, detail="physicianId is required for subscription check")
        
        print(f"🔍 Checking subscription for physicianId: {subscription_id}")

        # Query DB for active subscription using physicianId
        sub = await db.subscription.find_first(
            where={
                "physicianId": subscription_id,
                "status": "active"
            }
        )
        
        print(f"📊 Database query completed. Found subscription: {sub is not None}")
        
        if not sub:
            print("❌ No active subscription found")
            raise HTTPException(
                status_code=400,
                detail="No active subscription found. Please upgrade your plan."
            )
        
        # Get documentParse from subscription (as per your schema)
        remaining_parses = sub.documentParse
        print(f"📊 Subscription ID: {sub.id}, Remaining parses: {remaining_parses}")
        print(f"📄 Requested documents: {document_count}")
        
        if document_count > remaining_parses:
            print(f"❌ Document count ({document_count}) exceeds remaining parses ({remaining_parses})")
            raise HTTPException(
                status_code=400,
                detail=f"Not enough remaining parses. You requested {document_count} documents, but only {remaining_parses} parses available. Please upgrade your plan."
            )
        
        if remaining_parses <= 0:
            print("❌ Document parse limit exceeded")
            raise HTTPException(
                status_code=400,
                detail="Document parse limit exceeded. Please upgrade your plan."
            )
        
        print("✅ Subscription check passed")
        return {
            "subscription": sub,
            "document_count": document_count,
            "physician_id": subscription_id,
            "remaining_parses": remaining_parses
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Subscription check error: {e}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during subscription check")

def parse_date(date_str: Optional[str], field_name: str) -> datetime:
    """Parse a date string safely, supporting multiple formats."""
    if not date_str or date_str.strip() == "":
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty or missing")

    # Supported formats
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # If no format worked
    raise HTTPException(
        status_code=400,
        detail=f"Invalid date format for {field_name}. Expected one of: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY"
    )

def serialize_payload(payload: dict) -> dict:
    """Convert any non-JSON datatypes to ISO strings for Celery serialization."""
    def convert(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    return convert(payload)


# ============================
# 🔵 Normalization Helpers
# ============================

def clean_name_string(name: Optional[str]) -> str:
    """
    Cleans name string: removes titles, credentials, special chars, extra spaces.
    Preserves word order.
    """
    if not name:
        return ""
    name = str(name)
    # Combined regex for titles/suffixes
    pattern = r'\b(Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?|Doctor|M\.?D\.?|D\.?O\.?|D\.?P\.?M\.?|D\.?C\.?|N\.?P\.?|P\.?A\.?|QME|Sr\.?|Jr\.?|II|III|IV)\b'
    
    name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    name = name.replace(',', ' ').replace('.', ' ')
    name = re.sub(r'[^\w\s-]', '', name) # Keep hyphens, remove other punctuation
    return ' '.join(name.split()).lower()

def normalize_name(name: Optional[str]) -> str:
    """
    Normalize patient name for matching.
    Handles: comma-separated names, case differences, extra spaces.
    Returns: sorted lowercase name parts (e.g., "morales lorina")
    
    NOTE: For comparison, use get_name_variations() to get all possible formats.
    """
    if not name or str(name).lower() in ["not specified", "unknown", "n/a", "na", ""]:
        return ""
    
    # Remove commas, convert to lowercase, strip whitespace
    name = str(name).replace(",", " ").lower().strip()
    # Split into parts and remove empty strings
    parts = [p for p in name.split() if p]
    
    if not parts:
        return ""
    
    # For 2+ part names, use first and last name sorted
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        # Sort alphabetically for consistent comparison
        normalized = " ".join(sorted([first, last]))
    else:
        normalized = parts[0]
    
    return normalized.strip()

def get_name_variations(name: Optional[str]) -> list[str]:
    """
    Generate all possible name format variations for flexible matching.
    
    Examples:
    - "John Smith" → ["john smith", "smith john"]
    - "Lemus Guillen, Miguel" → ["lemus guillen miguel", "miguel lemus guillen", "lemus miguel", "miguel lemus", ...]
    
    Returns: List of all possible name orderings (lowercase, no commas)
    """
    if not name or str(name).lower() in ["not specified", "unknown", "n/a", "na", ""]:
        return [""]
    
    # Remove commas, convert to lowercase, strip whitespace
    name = str(name).replace(",", " ").lower().strip()
    # Split into parts and remove empty strings
    parts = [p for p in name.split() if p]
    
    if not parts:
        return [""]
    
    # Single name - only one variation
    if len(parts) == 1:
        return [parts[0]]
    
    variations = set()
    
    if len(parts) == 2:
        # Two parts: "first last" and "last first"
        variations.add(f"{parts[0]} {parts[1]}")
        variations.add(f"{parts[1]} {parts[0]}")
    
    elif len(parts) >= 3:
        # Three+ parts: Generate comprehensive variations
        # Full name in original and reversed order
        variations.add(" ".join(parts))  # full name original order
        variations.add(" ".join(reversed(parts)))  # full name reversed
        
        # First + Last (skip middle names)
        first = parts[0]
        last = parts[-1]
        variations.add(f"{first} {last}")
        variations.add(f"{last} {first}")
        
        # First + Middle(s) variations
        for i in range(1, len(parts)):
            variations.add(f"{first} {parts[i]}")
            variations.add(f"{parts[i]} {first}")
        
        # Last + Middle(s) variations  
        for i in range(len(parts) - 1):
            variations.add(f"{parts[i]} {last}")
            variations.add(f"{last} {parts[i]}")
    
    return list(variations)

def normalize_claim(claim: Optional[str]) -> Optional[str]:
    """
    Normalize claim number for matching.
    Returns: uppercase alphanumeric only, or None if invalid/missing.
    """
    if not claim or str(claim).lower() in ["not specified", "unknown", "n/a", "na", ""]:
        return None
    # Remove all non-alphanumeric characters and convert to uppercase
    normalized = "".join(c for c in str(claim).upper() if c.isalnum())
    return normalized if normalized else None

def normalize_dob(dob: Optional[Any]) -> Optional[str]:
    """
    Normalize DOB to YYYY-MM-DD format.
    Handles: datetime objects, ISO strings, common date formats.
    Returns: YYYY-MM-DD string or None if invalid/missing.
    """
    if not dob or str(dob).lower() in ["not specified", "unknown", "n/a", "na", ""]:
        return None
    
    dob_str = str(dob).strip()
    
    # Handle invalid/placeholder dates (00/00/0000, 0000-00-00, etc.)
    invalid_patterns = [
        "00/00/0000", "0000-00-00", "00-00-0000", "0000/00/00",
        "00/00/00", "00-00-00", "0/0/0", "0-0-0",
        "01/01/0001", "0001-01-01", "1/1/1", "1-1-1",
        "01/01/1900", "1900-01-01",  # Common placeholder dates
    ]
    if dob_str in invalid_patterns or dob_str.startswith("00") or dob_str.startswith("0000"):
        logger.debug(f"⚠️ Invalid/placeholder DOB detected: '{dob_str}' - treating as None")
        return None
    
    # If it's already a datetime object
    if isinstance(dob, datetime):
        return dob.strftime("%Y-%m-%d")
    
    # Try ISO format first
    try:
        return datetime.fromisoformat(dob_str).strftime("%Y-%m-%d")
    except:
        pass
    
    # Try common formats
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            parsed = datetime.strptime(dob_str, fmt)
            # Validate the parsed date is reasonable (year > 1900)
            if parsed.year < 1900:
                logger.debug(f"⚠️ DOB year too old: {parsed.year} - treating as None")
                return None
            return parsed.strftime("%Y-%m-%d")
        except:
            pass
    
    # Return as-is if parsing fails (for manual review)
    return dob_str if dob else None

def is_same_patient(name1: str, dob1: Optional[str], claim1: Optional[str],
                   name2: str, dob2: Optional[str], claim2: Optional[str]) -> bool:
    """
    Implements comprehensive patient matching rules with flexible name matching.
    
    🔥 MATCHING RULES:
    Rule 1: BOTH have claims AND they're different → DIFFERENT PATIENT
    Rule 2: BOTH have same claim → SAME PATIENT (highest priority)
    Rule 3: At least ONE claim is None → Use name + DOB logic:
      3A: Names match (any variation) AND DOBs match → SAME
      3B: Names match (any variation) AND both DOBs None → SAME
      3C: Names match (any variation) AND one DOB None → SAME
    
    Name matching tries all possible orderings:
    - "John Smith" matches "Smith John"
    - "Jason Nasr" matches "Nasr Jason"
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Rule 1: If BOTH have claim & different → NOT same
    # KEY FIX: Only reject if BOTH claims exist and are different
    if claim1 and claim2 and claim1 != claim2:
        logger.debug(f"❌ Rule 1: Different claims '{claim1}' vs '{claim2}' - NOT same patient")
        return False
    
    # Rule 2: If BOTH have same claim → same patient (highest priority)
    if claim1 and claim2 and claim1 == claim2:
        logger.debug(f"✅ Rule 2: Same claim '{claim1}' - SAME patient")
        return True
    
    # Rule 3: At least ONE claim is None - rely on name + DOB
    # This includes: (claim1 and not claim2) OR (not claim1 and claim2) OR (not claim1 and not claim2)
    
    # 🆕 ENHANCED: Check if names match using ANY possible variation
    name1_variations = get_name_variations(name1)
    name2_variations = get_name_variations(name2)
    
    # Check if any variation of name1 matches any variation of name2
    names_match = any(v1 == v2 for v1 in name1_variations for v2 in name2_variations)
    
    if names_match:
        logger.info(f"✅ Names match (variations): '{name1}' ~ '{name2}'")
        logger.info(f"   Search variations: {name1_variations}")
        logger.info(f"   DB variations: {name2_variations}")
        
        # 3A: Both DOB provided & match
        if dob1 and dob2 and dob1 == dob2:
            logger.debug(f"✅ Rule 3A: Both DOB match '{dob1}' - SAME patient (one or both claims None)")
            return True
        
        # 3B: Both missing DOB
        if not dob1 and not dob2:
            logger.debug(f"✅ Rule 3B: Both DOB missing - SAME patient (one or both claims None)")
            return True
        
        # 3C: One missing DOB, still same
        if (dob1 and not dob2) or (dob2 and not dob1):
            logger.debug(f"✅ Rule 3C: One DOB missing - SAME patient (one or both claims None)")
            return True
    else:
        logger.debug(f"❌ Names don't match (any variation): '{name1}' vs '{name2}'")
        logger.debug(f"   Search variations: {name1_variations} | DB variations: {name2_variations}")
    
    # Otherwise not same
    logger.debug(f"❌ No matching rules - NOT same patient")
    return False
    

def extract_text_from_summarizer(document, signature_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract text from summarizer response.
    Summarizer returns clean, organized text with natural paragraph structure.
    
    Args:
        document: The Document AI summarizer response
        signature_info: Optional author signature info extracted using extract_author_signature.
                       Used as fallback for author detection if AI summarizer doesn't find it.
    """
    summarized_text = document.text or ""
    
    # Extract page count if available
    page_count = len(document.pages) if document.pages else 0
    
    # Build metadata with signature info if available
    metadata = {
        "has_summary": True,
        "total_chars": len(summarized_text)
    }
    
    # Add signature/author info to metadata if available
    if signature_info:
        metadata["signature_extracted"] = True
        metadata["signature_author"] = signature_info.get("author")
        metadata["signature_confidence"] = signature_info.get("confidence")
        metadata["signature_evidence"] = signature_info.get("evidence")
        metadata["signature_pattern_type"] = signature_info.get("pattern_type")
        logger.info(f"✍️ Signature info added to metadata: {signature_info.get('author')} ({signature_info.get('confidence')} confidence)")
    else:
        metadata["signature_extracted"] = False
    
    # Build simple structure for compatibility with existing code
    return {
        "layout_preserved": summarized_text,
        "raw_text": summarized_text,
        "page_zones": {},  # Summarizer doesn't provide detailed zones
        "structured_document": {
            "document_structure": {
                "total_pages": page_count,
                "summarized": True
            },
            "pages": [],
            "metadata": metadata
        },
        "signature_info": signature_info  # Include full signature info for downstream use
    }


def build_llm_friendly_json(structured_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build LLM-optimized JSON structure from summarizer output.
    """
    return {
        "document_type_hints": {
            "header_text": "",
            "first_page_context": "",
            "has_form_structure": False,
            "is_summarized": True
        },
        "content": {
            "summary": structured_document.get("summarized_text", ""),
            "page_count": structured_document.get("document_structure", {}).get("total_pages", 0)
        },
        "metadata": structured_document.get("metadata", {})
    }