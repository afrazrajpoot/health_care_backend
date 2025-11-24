async def _flexible_duplicate_check(self, physician_id: str, patient_name: str, document_analysis, db_service) -> bool:
    """STRICT duplicate check that requires ALL key fields to match"""
    try:
        # Get available fields
        dob = document_analysis.dob if hasattr(document_analysis, 'dob') else None
        claim_number = document_analysis.claim_number if hasattr(document_analysis, 'claim_number') else None
        doi = document_analysis.doi if hasattr(document_analysis, 'doi') else None
        rd = document_analysis.rd if hasattr(document_analysis, 'rd') else None
        doc_type = document_analysis.document_type if hasattr(document_analysis, 'document_type') else None
        
        logger.info(f"🔍 STRICT DUPLICATE CHECK - Fields available:")
        logger.info(f"  - Patient: {patient_name}")
        logger.info(f"  - DOB: {dob}")
        logger.info(f"  - Claim: {claim_number}")
        logger.info(f"  - DOI: {doi}")
        logger.info(f"  - Report Date: {rd}")
        logger.info(f"  - Document Type: {doc_type}")
        logger.info(f"  - Physician: {physician_id}")
        
        # STRICT CHECK: All key fields must be present and match
        required_fields = [
            patient_name,  # Patient name must be present
            dob,           # DOB must be present  
            claim_number,  # Claim number must be present
            rd,            # Report date must be present
            doi,           # Date of injury must be present
            doc_type       # Document type must be present
        ]
        
        # Check if all required fields are available
        if not all(required_fields):
            missing_fields = []
            if not patient_name: missing_fields.append("patient_name")
            if not dob: missing_fields.append("dob") 
            if not claim_number: missing_fields.append("claim_number")
            if not rd: missing_fields.append("report_date")
            if not doi: missing_fields.append("doi")
            if not doc_type: missing_fields.append("document_type")
            
            logger.info(f"⚠️ Cannot perform strict duplicate check - missing fields: {missing_fields}")
            return False
        
        # Try the strict check first with ALL fields
        try:
            strict_duplicate = await db_service.check_duplicate_document(
                patient_name=patient_name,
                doi=doi,
                report_date=rd,
                document_type=doc_type,
                physician_id=physician_id,
                patient_dob=dob,
                claim_number=claim_number
            )
            
            if strict_duplicate:
                logger.warning(f"🚫 STRICT DUPLICATE FOUND: All fields match for {patient_name}")
                logger.warning(f"🚫 Matching fields: patient={patient_name}, dob={dob}, claim={claim_number}, doi={doi}, rd={rd}, type={doc_type}")
                return True
        except Exception as strict_err:
            logger.warning(f"⚠️ Strict duplicate check failed: {strict_err}")
        
        # FALLBACK: Manual check using patient lookup
        logger.info(f"🔍 Performing manual strict duplicate check for: {patient_name}")
        
        similar_docs = await db_service.get_patient_claim_numbers(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob,
            claim_number=claim_number
        )
        
        if similar_docs and similar_docs.get("total_documents", 0) > 0:
            documents = similar_docs.get("documents", [])
            logger.info(f"🔍 Found {len(documents)} potential documents for strict comparison")
            
            for doc in documents:
                doc_patient = doc.get("patientName")
                doc_dob = doc.get("dob")
                doc_claim = doc.get("claimNumber") 
                doc_doi = doc.get("doi")
                doc_rd = doc.get("reportDate")
                doc_type_db = doc.get("documentType")
                
                # STRICT MATCH: All fields must match exactly
                patient_match = doc_patient and patient_name and doc_patient.lower() == patient_name.lower()
                dob_match = doc_dob and dob and str(doc_dob).lower() == str(dob).lower()
                claim_match = doc_claim and claim_number and doc_claim.lower() == claim_number.lower()
                doi_match = doc_doi and doi and str(doc_doi).lower() == str(doi).lower()
                rd_match = doc_rd and rd and str(doc_rd).lower() == str(rd).lower()
                type_match = doc_type_db and doc_type and doc_type_db.lower() == doc_type.lower()
                
                if patient_match and dob_match and claim_match and doi_match and rd_match and type_match:
                    logger.warning(f"🚫 MANUAL STRICT DUPLICATE FOUND:")
                    logger.warning(f"🚫 Patient: {doc_patient} == {patient_name}")
                    logger.warning(f"🚫 DOB: {doc_dob} == {dob}")
                    logger.warning(f"🚫 Claim: {doc_claim} == {claim_number}")
                    logger.warning(f"🚫 DOI: {doc_doi} == {doi}")
                    logger.warning(f"🚫 Report Date: {doc_rd} == {rd}")
                    logger.warning(f"🚫 Document Type: {doc_type_db} == {doc_type}")
                    return True
                else:
                    logger.info(f"🔍 Document doesn't match strictly:")
                    if not patient_match: logger.info(f"   ❌ Patient mismatch: {doc_patient} vs {patient_name}")
                    if not dob_match: logger.info(f"   ❌ DOB mismatch: {doc_dob} vs {dob}")
                    if not claim_match: logger.info(f"   ❌ Claim mismatch: {doc_claim} vs {claim_number}")
                    if not doi_match: logger.info(f"   ❌ DOI mismatch: {doc_doi} vs {doi}")
                    if not rd_match: logger.info(f"   ❌ Report Date mismatch: {doc_rd} vs {rd}")
                    if not type_match: logger.info(f"   ❌ Document Type mismatch: {doc_type_db} vs {doc_type}")
        
        logger.info("✅ No strict duplicates found - document is unique")
        return False
        
    except Exception as e:
        logger.error(f"❌ Strict duplicate check failed: {e}")
        # In case of error, allow the document to be processed (fail open)
        return False