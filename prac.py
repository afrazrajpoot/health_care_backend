async def perform_patient_lookup(self, db_service, processed_data: dict) -> dict:
    """Step 2: Perform patient lookup and update fields bidirectionally with minimum 2-field matching"""
    physician_id = processed_data["physician_id"]
    patient_name = processed_data["patient_name"]
    claim_number = processed_data["claim_number"]
    document_analysis = processed_data["document_analysis"]
    
    logger.info(f"🔍 Performing patient lookup for physician: {physician_id}")
    
    # Helper function to check if field is "bad" (not specified/empty)
    def is_bad_field(value):
        return not value or str(value).lower() in ["not specified", "unknown", "", "none", "null"]
    
    # Helper function to normalize field values for comparison
    def normalize_field(value):
        if not value:
            return ""
        return str(value).strip().lower()
    
    # 🚨 CRITICAL: Check if both DOB and claim number are not specified
    dob_not_specified = is_bad_field(processed_data["dob"])
    claim_not_specified = is_bad_field(claim_number)
    
    # If both DOB AND claim number are not specified, skip lookup and field updates
    if dob_not_specified and claim_not_specified:
        logger.warning("🚨 SKIPPING PATIENT LOOKUP: Both DOB and claim number are not specified - no updates will be performed")
        
        document_status = "failed"
        pending_reason = "Missing both DOB and claim number - cannot identify patient"
        
        return {
            "lookup_data": None,
            "document_status": document_status,
            "pending_reason": pending_reason,
            "patient_name_to_use": patient_name or "Not specified",
            "claim_to_save": claim_number or "Not specified", 
            "document_analysis": document_analysis,
            "field_updates": [],
            "previous_docs_updated": 0,
            "lookup_skipped": True
        }
    
    # ✅ Continue with normal patient lookup
    redis_ok = await self.verify_redis_connection()
    if not redis_ok:
        logger.warning("⚠️ Redis not available - proceeding without cache")
    
    # Get patient lookup data (with Redis caching and 2-field validation)
    lookup_data = await self._get_cached_patient_lookup(physician_id, patient_name, claim_number, processed_data["dob"], db_service)
    
    # Bidirectional field updating logic
    field_updates = []
    updated_previous_docs = 0
    
    if lookup_data and lookup_data.get("total_documents", 0) > 0:
        logger.info("🔄 Checking for bidirectional field updates with minimum 2-field matching...")
        
        # Get fields from lookup data
        fetched_patient_name = lookup_data.get("patient_name")
        fetched_dob = lookup_data.get("dob")
        fetched_claim_number = lookup_data.get("claim_number")
        fetched_doi = lookup_data.get("doi")
        
        # 🚨 CRITICAL FIX: Compare CURRENT DOCUMENT ANALYSIS with FETCHED DATA
        # Count matching fields between CURRENT DOCUMENT and fetched data
        matching_fields = 0
        
        # Get current document field values
        current_patient_name = document_analysis.patient_name
        current_dob = getattr(document_analysis, 'dob', None)
        current_claim_number = document_analysis.claim_number
        current_doi = getattr(document_analysis, 'doi', None)
        
        logger.info(f"🔍 CURRENT DOCUMENT - Patient: '{current_patient_name}', DOB: '{current_dob}', Claim: '{current_claim_number}', DOI: '{current_doi}'")
        logger.info(f"🔍 FETCHED DATA - Patient: '{fetched_patient_name}', DOB: '{fetched_dob}', Claim: '{fetched_claim_number}', DOI: '{fetched_doi}'")
        
        # Check patient name match - CURRENT DOCUMENT vs FETCHED DATA
        current_patient_normalized = normalize_field(current_patient_name)
        fetched_patient_normalized = normalize_field(fetched_patient_name)
        patient_name_matches = (
            not is_bad_field(current_patient_name) and 
            not is_bad_field(fetched_patient_name) and
            current_patient_normalized == fetched_patient_normalized
        )
        if patient_name_matches:
            matching_fields += 1
            logger.info(f"✅ Patient name matches: '{current_patient_name}' == '{fetched_patient_name}'")
        else:
            logger.info(f"❌ Patient name MISMATCH: '{current_patient_name}' != '{fetched_patient_name}'")
        
        # Check DOB match - CURRENT DOCUMENT vs FETCHED DATA
        current_dob_normalized = normalize_field(current_dob)
        fetched_dob_normalized = normalize_field(fetched_dob)
        dob_matches = (
            not is_bad_field(current_dob) and 
            not is_bad_field(fetched_dob) and
            current_dob_normalized == fetched_dob_normalized
        )
        if dob_matches:
            matching_fields += 1
            logger.info(f"✅ DOB matches: '{current_dob}' == '{fetched_dob}'")
        else:
            logger.info(f"❌ DOB MISMATCH: '{current_dob}' != '{fetched_dob}'")
        
        # Check claim number match - CURRENT DOCUMENT vs FETCHED DATA
        current_claim_normalized = normalize_field(current_claim_number)
        fetched_claim_normalized = normalize_field(fetched_claim_number)
        claim_matches = (
            not is_bad_field(current_claim_number) and 
            not is_bad_field(fetched_claim_number) and
            current_claim_normalized == fetched_claim_normalized
        )
        if claim_matches:
            matching_fields += 1
            logger.info(f"✅ Claim number matches: '{current_claim_number}' == '{fetched_claim_number}'")
        else:
            logger.info(f"❌ Claim number MISMATCH: '{current_claim_number}' != '{fetched_claim_number}'")
        
        # Check DOI match - CURRENT DOCUMENT vs FETCHED DATA
        current_doi_normalized = normalize_field(current_doi)
        fetched_doi_normalized = normalize_field(fetched_doi)
        doi_matches = (
            not is_bad_field(current_doi) and 
            not is_bad_field(fetched_doi) and
            current_doi_normalized == fetched_doi_normalized
        )
        if doi_matches:
            matching_fields += 1
            logger.info(f"✅ DOI matches: '{current_doi}' == '{fetched_doi}'")
        else:
            logger.info(f"❌ DOI MISMATCH: '{current_doi}' != '{fetched_doi}'")
        
        logger.info(f"🔢 Field matching summary: {matching_fields} fields match")
        
        # 🚨 CRITICAL: Only proceed with updates if we have AT LEAST 2 matching fields
        if matching_fields >= 2:
            logger.info("✅ Minimum 2-field match satisfied - proceeding with field updates")
            
            # Update document analysis with good values from DB
            # Only update if current document has bad values AND DB has good values
            if is_bad_field(document_analysis.patient_name) and not is_bad_field(fetched_patient_name):
                old_name = document_analysis.patient_name
                document_analysis.patient_name = fetched_patient_name
                field_updates.append(f"patient_name: '{old_name}' → '{fetched_patient_name}'")
                logger.info(f"✅ Updated patient_name from DB: '{old_name}' → '{fetched_patient_name}'")
            
            if hasattr(document_analysis, 'dob') and is_bad_field(document_analysis.dob) and not is_bad_field(fetched_dob):
                old_dob = document_analysis.dob
                document_analysis.dob = fetched_dob
                field_updates.append(f"dob: '{old_dob}' → '{fetched_dob}'")
                logger.info(f"✅ Updated DOB from DB: '{old_dob}' → '{fetched_dob}'")
            
            if is_bad_field(document_analysis.claim_number) and not is_bad_field(fetched_claim_number):
                old_claim = document_analysis.claim_number
                document_analysis.claim_number = fetched_claim_number
                field_updates.append(f"claim_number: '{old_claim}' → '{fetched_claim_number}'")
                logger.info(f"✅ Updated claim_number from DB: '{old_claim}' → '{fetched_claim_number}'")
            
            if (hasattr(document_analysis, 'doi') and 
                is_bad_field(document_analysis.doi) and 
                not is_bad_field(fetched_doi)):
                old_doi = document_analysis.doi
                document_analysis.doi = fetched_doi
                field_updates.append(f"doi: '{old_doi}' → '{fetched_doi}'")
                logger.info(f"✅ Updated DOI from DB: '{old_doi}' → '{fetched_doi}'")
            
            # 🚨 IMPORTANT: Update previous documents ONLY if current document has good identification
            # AND we have the minimum 2-field match
            current_has_good_patient = not is_bad_field(document_analysis.patient_name)
            current_has_good_dob = hasattr(document_analysis, 'dob') and not is_bad_field(document_analysis.dob)
            current_has_good_claim = not is_bad_field(document_analysis.claim_number)
            current_has_good_doi = hasattr(document_analysis, 'doi') and not is_bad_field(document_analysis.doi)
            
            # Only update previous documents if we have at least DOB OR claim number in current document
            current_has_identification = current_has_good_dob or current_has_good_claim
            
            if current_has_identification and (current_has_good_patient or current_has_good_dob or current_has_good_claim or current_has_good_doi):
                try:
                    update_patient = document_analysis.patient_name if current_has_good_patient else None
                    update_dob = document_analysis.dob if current_has_good_dob else None
                    update_claim = document_analysis.claim_number if current_has_good_claim else None
                    update_doi = document_analysis.doi if current_has_good_doi else None
                    
                    if update_patient or update_dob or update_claim:
                        updated_previous_docs = await db_service.update_document_fields(
                            patient_name=update_patient or "Not specified",
                            dob=update_dob or "Not specified",
                            physician_id=physician_id,
                            claim_number=update_claim or "Not specified",
                            doi=update_doi
                        )
                        logger.info(f"🔄 Updated {updated_previous_docs} previous documents with current good fields")
                        
                        # Invalidate cache after updates
                        if updated_previous_docs > 0 and self.redis_client:
                            pattern = f"patient_lookup:{physician_id}:*"
                            keys = await self.redis_client.keys(pattern)
                            if keys:
                                await self.redis_client.delete(*keys)
                                logger.info(f"🗑️ Invalidated {len(keys)} patient lookup cache entries")
                    
                except Exception as update_err:
                    logger.error(f"❌ Error updating previous documents: {update_err}")
            else:
                logger.info("ℹ️ Skipping previous document updates - current document lacks sufficient identification")
            
            logger.info(f"🎯 Bidirectional updates completed: {field_updates}")
        else:
            logger.warning(f"🚨 SKIPPING FIELD UPDATES: Only {matching_fields} field(s) match - minimum 2 fields required")
            logger.info(f"   Required at least 2 matching fields from: patient_name, dob, claim_number, doi")
    
    # Update processed_data with overridden values (only if updates were applied)
    processed_data["patient_name"] = document_analysis.patient_name
    processed_data["claim_number"] = document_analysis.claim_number
    processed_data["has_patient_name"] = not is_bad_field(document_analysis.patient_name)
    processed_data["has_claim_number"] = not is_bad_field(document_analysis.claim_number)
    
    # Determine document status (NO DUPLICATE CHECK)
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
        "previous_docs_updated": updated_previous_docs,
        "lookup_skipped": False
    }