from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as FastAPIPath  # ✅ Import FastAPI's Path for route parameters
from pathlib import Path as PathLib  # Import pathlib's Path for file operations if needed
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.delete("/fail-docs/{doc_id}")
async def delete_fail_document(
    doc_id: str = FastAPIPath(..., description="ID of the failed document to delete"),
    physicianId: str = Query(..., description="Physician ID to authorize deletion")
):
    """
    Delete a failed document: remove from GCS using blob_path and from DB, scoped to physician.
    """
    try:
        db_service = await get_database_service()
        file_service = FileService()
        
        # Fetch the doc to get blob_path and verify
        fail_doc = await db_service.prisma.faildocs.find_unique(
            where={"id": doc_id}
        )
        
        if not fail_doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        if fail_doc.physicianId != physicianId:
            raise HTTPException(status_code=403, detail="Unauthorized: Access denied")
        
        # Log the blobPath for debugging
        logger.info(f"Debug: blobPath type={type(fail_doc.blobPath)}, value={repr(fail_doc.blobPath)}")
        
        # Safely extract blob_path from GCS
        # Handle None, Ellipsis, or any non-string values
        blob_path = None
        if fail_doc.blobPath and isinstance(fail_doc.blobPath, str) and fail_doc.blobPath != "...":
            blob_path = fail_doc.blobPath
        
        # Delete from GCS (Google Cloud Storage)
        if blob_path:
            try:
                success = file_service.delete_from_gcs(blob_path)
                if success:
                    logger.info(f"✅ Successfully deleted from GCS: {blob_path}")
                else:
                    logger.warning(f"⚠️ GCS delete returned False for {blob_path}, but proceeding with DB delete")
            except Exception as gcs_error:
                logger.error(f"❌ GCS deletion error for {blob_path}: {str(gcs_error)}")
                # Continue with DB deletion even if GCS deletion fails
        else:
            logger.info(f"ℹ️ No valid blob_path found for doc {doc_id}, skipping GCS deletion")
        
        # Delete from Database
        await db_service.delete_fail_doc_by_id(doc_id, physicianId)
        logger.info(f"✅ Successfully deleted from database: {doc_id}")
        
        return {
            "status": "success",
            "message": f"Failed document {doc_id} deleted successfully",
            "doc_id": doc_id,
            "gcs_deleted": bool(blob_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in delete_fail_document for {doc_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# Additional helper for batch deletion if needed
@router.delete("/fail-docs/batch")
async def delete_fail_documents_batch(
    doc_ids: list[str] = Query(..., description="List of document IDs to delete"),
    physicianId: str = Query(..., description="Physician ID to authorize deletion")
):
    """
    Delete multiple failed documents at once.
    """
    try:
        db_service = await get_database_service()
        file_service = FileService()
        
        results = {
            "success": [],
            "failed": [],
            "total": len(doc_ids)
        }
        
        for doc_id in doc_ids:
            try:
                # Fetch document
                fail_doc = await db_service.prisma.faildocs.find_unique(
                    where={"id": doc_id}
                )
                
                if not fail_doc or fail_doc.physicianId != physicianId:
                    results["failed"].append({
                        "doc_id": doc_id,
                        "reason": "Not found or unauthorized"
                    })
                    continue
                
                # Delete from GCS if blob_path exists
                blob_path = fail_doc.blobPath if isinstance(fail_doc.blobPath, str) else None
                if blob_path:
                    try:
                        file_service.delete_from_gcs(blob_path)
                    except Exception as gcs_error:
                        logger.warning(f"GCS deletion failed for {blob_path}: {gcs_error}")
                
                # Delete from DB
                await db_service.delete_fail_doc_by_id(doc_id, physicianId)
                results["success"].append(doc_id)
                
            except Exception as e:
                results["failed"].append({
                    "doc_id": doc_id,
                    "reason": str(e)
                })
        
        return {
            "status": "completed",
            "results": results,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"])
        }
        
    except Exception as e:
        logger.error(f"❌ Batch deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch deletion failed: {str(e)}")