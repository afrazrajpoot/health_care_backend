from difflib import SequenceMatcher
import re
from typing import Optional, Tuple
from utils.logger import logger

class EnhancedPatientLookup:
    """Enhanced patient lookup with fuzzy matching and field normalization"""
    
    def __init__(self, redis_client=None, name_similarity_threshold=0.85, claim_similarity_threshold=0.90):
        self.name_similarity_threshold = name_similarity_threshold
        self.claim_similarity_threshold = claim_similarity_threshold
        self.redis_client = redis_client
    
    def normalize_name(self, name: str) -> str:
        """Normalize patient name for comparison"""
        if not name or str(name).lower() in ["not specified", "unknown", "", "none", "null"]:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = str(name).strip().lower()
        
        # Remove common suffixes/prefixes
        normalized = re.sub(r'\b(mr|mrs|ms|dr|prof|sr|jr|ii|iii|iv)\b\.?', '', normalized)
        
        # Remove special characters but keep spaces and hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Normalize multiple spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def parse_name_components(self, name: str) -> set:
        """Parse name into components for fuzzy matching"""
        if not name:
            return set()
        
        normalized = self.normalize_name(name)
        
        # Split by common delimiters
        parts = re.split(r'[,\s-]+', normalized)
        
        # Remove empty strings and single characters (except initials pattern)
        components = set()
        for part in parts:
            if len(part) > 1 or (len(part) == 1 and part.isalpha()):
                components.add(part)
        
        return components
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using multiple strategies
        Returns score between 0 and 1
        """
        if not name1 or not name2:
            return 0.0
        
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Strategy 1: Direct string similarity
        direct_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Strategy 2: Component-based matching (handles "John Smith" vs "Smith, John")
        components1 = self.parse_name_components(name1)
        components2 = self.parse_name_components(name2)
        
        if components1 and components2:
            intersection = len(components1 & components2)
            union = len(components1 | components2)
            component_similarity = intersection / union if union > 0 else 0.0
        else:
            component_similarity = 0.0
        
        # Strategy 3: Check if one name contains the other (handles "Smith" vs "John Smith")
        containment_score = 0.0
        if norm1 in norm2 or norm2 in norm1:
            shorter_len = min(len(norm1), len(norm2))
            longer_len = max(len(norm1), len(norm2))
            containment_score = shorter_len / longer_len if longer_len > 0 else 0.0
        
        # Take the maximum score from all strategies
        final_score = max(direct_similarity, component_similarity, containment_score)
        
        return final_score
    
    def normalize_claim_number(self, claim: str) -> str:
        """Normalize claim number for comparison"""
        if not claim or str(claim).lower() in ["not specified", "unknown", "", "none", "null"]:
            return ""
        
        # Remove spaces, convert to uppercase
        normalized = str(claim).strip().upper()
        
        # Remove common prefixes
        normalized = re.sub(r'^(CLAIM|CLM|#)\s*', '', normalized)
        
        return normalized
    
    def extract_base_claim(self, claim: str) -> str:
        """Extract base claim number (remove suffixes like -1, -A, etc.)"""
        if not claim:
            return ""
        
        normalized = self.normalize_claim_number(claim)
        
        # Extract base claim (everything before dash/hyphen)
        base = re.split(r'[-_/]', normalized)[0]
        
        return base
    
    def are_claims_related(self, claim1: str, claim2: str) -> Tuple[bool, str]:
        """
        Determine if two claim numbers are related/same patient
        Returns: (are_related, relationship_type)
        
        Relationship types:
        - "exact": Claims are identical
        - "base_match": One is base, other has suffix (012345 vs 012345-6)
        - "contains": One claim contains the other
        - "similar": High string similarity (>90%)
        - "none": Not related
        """
        if not claim1 or not claim2:
            return False, "none"
        
        norm1 = self.normalize_claim_number(claim1)
        norm2 = self.normalize_claim_number(claim2)
        
        if not norm1 or not norm2:
            return False, "none"
        
        # Exact match
        if norm1 == norm2:
            return True, "exact"
        
        # Extract base claim numbers
        base1 = self.extract_base_claim(claim1)
        base2 = self.extract_base_claim(claim2)
        
        # Check if they share the same base (e.g., "012345" vs "012345-6")
        if base1 == base2 and base1:
            return True, "base_match"
        
        # Check if one is contained in the other (with length consideration)
        # Only match if the shorter claim is at least 80% of the longer one
        if base1 and base2:
            shorter = min(len(base1), len(base2))
            longer = max(len(base1), len(base2))
            
            if shorter > 0 and (shorter / longer) >= 0.8:
                if base1 in norm2 or base2 in norm1:
                    return True, "contains"
        
        # High string similarity (for typos or slight variations)
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        if similarity >= 0.90:
            return True, "similar"
        
        return False, "none"
    
    def calculate_claim_similarity(self, claim1: str, claim2: str) -> float:
        """
        Calculate similarity score between two claim numbers
        Returns score between 0 and 1
        """
        are_related, relationship_type = self.are_claims_related(claim1, claim2)
        
        if not are_related:
            # Return raw string similarity if not related
            if claim1 and claim2:
                norm1 = self.normalize_claim_number(claim1)
                norm2 = self.normalize_claim_number(claim2)
                return SequenceMatcher(None, norm1, norm2).ratio()
            return 0.0
        
        # Map relationship types to similarity scores
        relationship_scores = {
            "exact": 1.0,
            "base_match": 0.98,  # Very high - same base claim
            "contains": 0.95,    # High - one contains the other
            "similar": 0.92      # High - very similar strings
        }
        
        return relationship_scores.get(relationship_type, 0.0)
    
    def normalize_dob(self, dob: str) -> str:
        """Normalize date of birth for comparison"""
        if not dob or str(dob).lower() in ["not specified", "unknown", "", "none", "null"]:
            return ""
        
        # Remove all non-alphanumeric characters
        normalized = re.sub(r'[^\w]', '', str(dob))
        
        return normalized.lower()
    
    def is_bad_field(self, value) -> bool:
        """Check if field value is invalid/empty"""
        return not value or str(value).lower() in ["not specified", "unknown", "", "none", "null"]
    
    def standardize_claim_number(self, claim1: str, claim2: str) -> str:
        """
        Given two related claim numbers, return the standardized version
        Prioritizes the more complete/detailed claim number
        
        Examples:
        - "012345" + "012345-6" → "012345-6" (keep suffix)
        - "ABC123" + "abc123" → "ABC123" (keep first format)
        - "12345" + "012345" → "012345" (keep leading zeros)
        """
        if not claim1 and not claim2:
            return ""
        if not claim1:
            return claim2
        if not claim2:
            return claim1
        
        # If they're exactly the same after normalization, keep the first one
        norm1 = self.normalize_claim_number(claim1)
        norm2 = self.normalize_claim_number(claim2)
        
        if norm1 == norm2:
            return claim1  # Keep original format
        
        # Check relationship
        are_related, rel_type = self.are_claims_related(claim1, claim2)
        
        if not are_related:
            # Not related - keep the longer/more detailed one
            return claim1 if len(str(claim1)) >= len(str(claim2)) else claim2
        
        # For base_match (e.g., "012345" vs "012345-6"), keep the one with suffix
        if rel_type == "base_match":
            # The one with a dash/suffix is more specific
            if '-' in str(claim1) or '_' in str(claim1) or '/' in str(claim1):
                return claim1
            if '-' in str(claim2) or '_' in str(claim2) or '/' in str(claim2):
                return claim2
        
        # For contains relationship, keep the longer one
        if rel_type == "contains":
            return claim1 if len(str(claim1)) >= len(str(claim2)) else claim2
        
        # Default: keep the more complete one (longer length)
        return claim1 if len(str(claim1)) >= len(str(claim2)) else claim2
    
    def choose_best_value(self, current_value, fetched_value, field_name: str):
        """
        Choose the best value between current and fetched
        Returns: (best_value, was_updated)
        """
        current_bad = self.is_bad_field(current_value)
        fetched_bad = self.is_bad_field(fetched_value)
        
        # If both bad, keep current
        if current_bad and fetched_bad:
            return current_value, False
        
        # If current is bad but fetched is good, use fetched
        if current_bad and not fetched_bad:
            logger.info(f"✅ Using fetched {field_name}: '{fetched_value}' (current was bad)")
            return fetched_value, True
        
        # If fetched is bad but current is good, use current
        if not current_bad and fetched_bad:
            logger.info(f"✅ Keeping current {field_name}: '{current_value}' (fetched was bad)")
            return current_value, False
        
        # Both are good - choose the more complete/detailed one
        if len(str(fetched_value)) > len(str(current_value)):
            logger.info(f"✅ Using fetched {field_name}: '{fetched_value}' (more complete)")
            return fetched_value, True
        else:
            logger.info(f"✅ Keeping current {field_name}: '{current_value}' (equally good or better)")
            return current_value, False
    
    async def verify_redis_connection(self):
        """Verify Redis connection is working"""
        if not self.redis_client:
            logger.error("❌ Redis client is None - not initialized")
            return False
        
        try:
            await self.redis_client.ping()
            logger.info("✅ Redis connection verified")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def _get_cached_patient_lookup(self, physician_id: str, patient_name: str, claim_number: str, dob: str, db_service):
        """
        Get patient lookup data from cache or database with fuzzy matching support.
        This method uses the webhook service's caching implementation for consistency.
        """
        # For now, directly fetch from database
        # The fuzzy matching logic will handle the comparison
        logger.info("🗄️ Fetching patient lookup data from database...")
        lookup_data = await db_service.get_patient_claim_numbers(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob,
            claim_number=claim_number
        )
        
        # Add search criteria for validation
        if lookup_data:
            lookup_data["_search_criteria"] = {
                "patient_name": patient_name,
                "dob": dob,
                "claim_number": claim_number
            }
        
        return lookup_data
    
    async def perform_patient_lookup(self, db_service, processed_data: dict) -> dict:
        """
        Enhanced patient lookup with fuzzy matching and field normalization
        """
        physician_id = processed_data["physician_id"]
        patient_name = processed_data["patient_name"]
        claim_number = processed_data["claim_number"]
        document_analysis = processed_data["document_analysis"]
        
        logger.info(f"🔍 Performing enhanced patient lookup for physician: {physician_id}")
        
        # 🚨 CRITICAL: Check if both DOB and claim number are not specified
        dob_not_specified = self.is_bad_field(processed_data["dob"])
        claim_not_specified = self.is_bad_field(claim_number)
        
        # If both DOB AND claim number are not specified, skip lookup
        if dob_not_specified and claim_not_specified:
            logger.warning("🚨 SKIPPING PATIENT LOOKUP: Both DOB and claim number are not specified")
            
            return {
                "lookup_data": None,
                "document_status": "failed",
                "pending_reason": "Missing both DOB and claim number - cannot identify patient",
                "patient_name_to_use": patient_name or "Not specified",
                "claim_to_save": claim_number or "Not specified", 
                "document_analysis": document_analysis,
                "field_updates": [],
                "previous_docs_updated": 0,
                "lookup_skipped": True
            }
        
        # ✅ Continue with enhanced patient lookup
        redis_ok = await self.verify_redis_connection()
        if not redis_ok:
            logger.warning("⚠️ Redis not available - proceeding without cache")
        
        # Get patient lookup data
        lookup_data = await self._get_cached_patient_lookup(
            physician_id, patient_name, claim_number, 
            processed_data["dob"], db_service
        )
        
        # Enhanced field matching and normalization
        field_updates = []
        match_info = {
            "name_match": False,
            "dob_match": False,
            "claim_match": False,
            "doi_match": False,
            "name_similarity": 0.0,
            "claim_similarity": 0.0
        }
        
        if lookup_data and lookup_data.get("total_documents", 0) > 0:
            logger.info("🔄 Performing enhanced fuzzy matching...")
            
            # Get fields from lookup data
            fetched_patient_name = lookup_data.get("patient_name")
            fetched_dob = lookup_data.get("dob")
            fetched_claim_number = lookup_data.get("claim_number")
            fetched_doi = lookup_data.get("doi")
            
            # Get current document field values
            current_patient_name = document_analysis.patient_name
            current_dob = getattr(document_analysis, 'dob', None)
            current_claim_number = document_analysis.claim_number
            current_doi = getattr(document_analysis, 'doi', None)
            
            logger.info(f"🔍 CURRENT DOC - Name: '{current_patient_name}', DOB: '{current_dob}', Claim: '{current_claim_number}'")
            logger.info(f"🔍 FETCHED DATA - Name: '{fetched_patient_name}', DOB: '{fetched_dob}', Claim: '{fetched_claim_number}'")
            
            # === NAME MATCHING (Fuzzy) ===
            if not self.is_bad_field(current_patient_name) and not self.is_bad_field(fetched_patient_name):
                name_similarity = self.calculate_name_similarity(current_patient_name, fetched_patient_name)
                match_info["name_similarity"] = name_similarity
                
                if name_similarity >= self.name_similarity_threshold:
                    match_info["name_match"] = True
                    logger.info(f"✅ NAME MATCH (similarity: {name_similarity:.2f}): '{current_patient_name}' ≈ '{fetched_patient_name}'")
                else:
                    logger.info(f"❌ Name similarity too low ({name_similarity:.2f}): '{current_patient_name}' vs '{fetched_patient_name}'")
            
            # === DOB MATCHING (Exact) ===
            if not self.is_bad_field(current_dob) and not self.is_bad_field(fetched_dob):
                norm_current_dob = self.normalize_dob(current_dob)
                norm_fetched_dob = self.normalize_dob(fetched_dob)
                
                if norm_current_dob == norm_fetched_dob:
                    match_info["dob_match"] = True
                    logger.info(f"✅ DOB MATCH: '{current_dob}' == '{fetched_dob}'")
                else:
                    logger.info(f"❌ DOB MISMATCH: '{current_dob}' != '{fetched_dob}'")
            
            # === CLAIM NUMBER MATCHING (Enhanced with relationship detection) ===
            if not self.is_bad_field(current_claim_number) and not self.is_bad_field(fetched_claim_number):
                are_related, rel_type = self.are_claims_related(current_claim_number, fetched_claim_number)
                claim_similarity = self.calculate_claim_similarity(current_claim_number, fetched_claim_number)
                match_info["claim_similarity"] = claim_similarity
                
                if are_related:
                    match_info["claim_match"] = True
                    logger.info(f"✅ CLAIM MATCH (type: {rel_type}, similarity: {claim_similarity:.2f})")
                    logger.info(f"   '{current_claim_number}' ≈ '{fetched_claim_number}'")
                    
                    # Log specific relationship details
                    if rel_type == "base_match":
                        logger.info(f"   → Claims share same base number (likely same patient with document variants)")
                    elif rel_type == "contains":
                        logger.info(f"   → One claim contains the other (likely related claims)")
                    elif rel_type == "similar":
                        logger.info(f"   → High similarity detected (likely same claim with minor variation)")
                else:
                    logger.info(f"❌ Claims NOT related (type: {rel_type}, similarity: {claim_similarity:.2f})")
                    logger.info(f"   '{current_claim_number}' vs '{fetched_claim_number}'")
            elif not self.is_bad_field(current_claim_number) or not self.is_bad_field(fetched_claim_number):
                # One claim is good, consider it a potential match if other fields match strongly
                logger.info(f"⚠️ Claim comparison incomplete (one value missing)")
                match_info["claim_match"] = False
            
            # === DOI MATCHING (Exact) ===
            if not self.is_bad_field(current_doi) and not self.is_bad_field(fetched_doi):
                if self.normalize_dob(current_doi) == self.normalize_dob(fetched_doi):
                    match_info["doi_match"] = True
                    logger.info(f"✅ DOI MATCH: '{current_doi}' == '{fetched_doi}'")
            
            # Count matching fields
            matching_fields = sum([
                match_info["name_match"],
                match_info["dob_match"],
                match_info["claim_match"],
                match_info["doi_match"]
            ])
            
            logger.info(f"🔢 Total matching fields: {matching_fields}/4")
            
            # ℹ️ NOTE: We do NOT update the current document with database values
            # Real-world scenario: A patient may have multiple documents with same DOB/claim
            # (e.g., Left Foot MRI and Right Foot MRI - both for same patient)
            # Each document is UNIQUE and should keep its own extracted data
            # We only use the lookup to verify patient exists and validate matching
            
            logger.info(f"ℹ️ Patient lookup complete - found {lookup_data.get('total_documents', 0)} existing documents")
            logger.info(f"ℹ️ Current document will be saved as a NEW separate document (no field updates)")
        
        # Update processed_data with final values
        processed_data["patient_name"] = document_analysis.patient_name
        processed_data["claim_number"] = document_analysis.claim_number
        processed_data["has_patient_name"] = not self.is_bad_field(document_analysis.patient_name)
        processed_data["has_claim_number"] = not self.is_bad_field(document_analysis.claim_number)
        
        # Determine document status
        base_status = document_analysis.status
        
        if not processed_data["has_patient_name"] and not processed_data["has_claim_number"]:
            document_status = "failed"
            pending_reason = "Missing patient name and claim number"
        elif lookup_data and lookup_data.get("has_conflicting_claims", False):
            document_status = "failed"
            pending_reason = "Conflicting claim numbers found"
        else:
            document_status = base_status
            pending_reason = None
        
        return {
            "lookup_data": lookup_data,
            "document_status": document_status,
            "pending_reason": pending_reason,
            "patient_name_to_use": processed_data["patient_name"] or "Not specified",
            "claim_to_save": processed_data["claim_number"] or "Not specified",
            "document_analysis": document_analysis,
            "field_updates": field_updates,
            "match_info": match_info,
            "lookup_skipped": False
        }