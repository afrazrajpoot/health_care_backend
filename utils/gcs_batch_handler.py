# gcs_batch_handler.py

"""
Google Cloud Storage batch processing utilities for Document AI.
Handles upload, download, and cleanup of files for batch processing.
"""

import os
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from google.cloud import storage
from google.cloud import documentai_v1 as documentai

logger = logging.getLogger("document_ai")


class GCSBatchHandler:
    """Handles GCS operations for Document AI batch processing"""
    
    def __init__(
        self,
        bucket_name: str,
        input_prefix: str = "docai-batch-input/",
        output_prefix: str = "docai-batch-output/",
        timeout_seconds: int = 600
    ):
        self.bucket_name = bucket_name
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.timeout_seconds = timeout_seconds
        self.storage_client: Optional[storage.Client] = None
        self.bucket = None
        self._initialize()
    
    def _initialize(self):
        """Initialize GCS client"""
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logger.info(f"✅ GCS client initialized for batch processing (bucket: {self.bucket_name})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GCS client: {str(e)}")
            logger.warning("⚠️ Batch processing will fall back to chunking method")
            self.storage_client = None
            self.bucket = None
    
    @property
    def is_available(self) -> bool:
        """Check if GCS client is available"""
        return self.storage_client is not None and self.bucket is not None
    
    def upload_for_batch(self, filepath: str) -> str:
        """
        Upload a local file to GCS for batch processing.
        Returns the GCS URI (gs://bucket/path/file.pdf)
        
        NOTE: This is NOT a duplicate upload. The Document AI batch API requires files
        to be in GCS (cannot process local files). This temporary upload is cleaned up
        after processing.
        """
        try:
            # Generate unique filename to avoid collisions
            unique_id = str(uuid.uuid4())[:8]
            original_name = Path(filepath).stem
            extension = Path(filepath).suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_filename = f"{original_name}_{timestamp}_{unique_id}{extension}"
            
            blob_path = f"{self.input_prefix}{gcs_filename}"
            blob = self.bucket.blob(blob_path)
            
            # Upload file
            blob.upload_from_filename(filepath)
            gcs_uri = f"gs://{self.bucket_name}/{blob_path}"
            
            logger.info(f"✅ Uploaded to GCS for batch processing: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to GCS: {str(e)}")
            raise
    
    def get_output_uri(self, input_uri: str) -> str:
        """Generate output GCS URI for batch processing results"""
        input_filename = Path(input_uri).stem
        unique_id = str(uuid.uuid4())[:8]
        output_folder = f"{self.output_prefix}{input_filename}_{unique_id}/"
        return f"gs://{self.bucket_name}/{output_folder}"
    
    def run_batch_process(
        self,
        client: documentai.DocumentProcessorServiceClient,
        processor_path: str,
        gcs_input_uri: str,
        gcs_output_uri: str,
        mime_type: str = "application/pdf"
    ) -> bool:
        """
        Call Document AI batch_process_documents API.
        Returns True if operation completed successfully.
        """
        try:
            logger.info(f"🚀 Starting batch processing...")
            logger.info(f"   Input: {gcs_input_uri}")
            logger.info(f"   Output: {gcs_output_uri}")
            
            # Create batch process request
            gcs_document = documentai.GcsDocument(
                gcs_uri=gcs_input_uri,
                mime_type=mime_type
            )
            
            gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
            
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=gcs_documents
            )
            
            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=gcs_output_uri
                )
            )
            
            request = documentai.BatchProcessRequest(
                name=processor_path,
                input_documents=input_config,
                document_output_config=output_config
            )
            
            # Start the batch operation (returns a long-running operation)
            operation = client.batch_process_documents(request=request)
            
            logger.info(f"⏳ Batch operation started: {operation.operation.name}")
            logger.info(f"⏳ Waiting for completion (timeout: {self.timeout_seconds}s)...")
            
            # Wait for the operation to complete (synchronous blocking)
            result = operation.result(timeout=self.timeout_seconds)
            
            logger.info(f"✅ Batch processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def download_results(self, gcs_output_uri: str) -> str:
        """
        Download and parse results from batch processing output location.
        Returns the extracted summary text.
        """
        try:
            # Extract bucket and prefix from GCS URI
            gcs_output_uri = gcs_output_uri.rstrip('/')
            parts = gcs_output_uri.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            logger.info(f"📥 Downloading batch results from: {gcs_output_uri}")
            logger.info(f"   Bucket: {bucket_name}, Prefix: {prefix}")
            
            # List all blobs in the output folder
            blobs = list(self.storage_client.list_blobs(bucket_name, prefix=prefix))
            
            if not blobs:
                logger.warning(f"⚠️ No output files found at {gcs_output_uri}")
                return ""
            
            logger.info(f"📄 Found {len(blobs)} output file(s)")
            
            # Batch processing outputs JSON files (shards)
            all_text = []
            all_summaries = []
            
            for blob in blobs:
                if blob.name.endswith(".json"):
                    logger.info(f"   Processing: {blob.name}")
                    content = blob.download_as_text()
                    
                    try:
                        # Parse the Document AI output JSON
                        doc_json = json.loads(content)
                        
                        # Extract text from the document
                        if "text" in doc_json:
                            all_text.append(doc_json["text"])
                        
                        # Extract summary from chunkedDocument or entities
                        if "chunkedDocument" in doc_json:
                            chunked_doc = doc_json["chunkedDocument"]
                            if "chunks" in chunked_doc:
                                for chunk in chunked_doc["chunks"]:
                                    if "content" in chunk:
                                        all_summaries.append(chunk["content"])
                        
                        # Also check entities for summary
                        if "entities" in doc_json:
                            for entity in doc_json["entities"]:
                                if entity.get("type") == "summary" or "summary" in entity.get("type", "").lower():
                                    mention_text = entity.get("mentionText", "")
                                    if mention_text:
                                        all_summaries.append(mention_text)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Failed to parse JSON from {blob.name}: {e}")
                        continue
            
            # Combine results - prefer summaries if available, otherwise use text
            if all_summaries:
                result_text = "\n\n".join(all_summaries)
                logger.info(f"✅ Extracted {len(all_summaries)} summary chunk(s), total {len(result_text)} chars")
            elif all_text:
                result_text = "\n\n".join(all_text)
                logger.info(f"✅ Extracted full text, total {len(result_text)} chars")
            else:
                result_text = ""
                logger.warning("⚠️ No text or summary found in batch results")
            
            return result_text
            
        except Exception as e:
            logger.error(f"❌ Failed to download batch results: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ""
    
    def cleanup(self, gcs_input_uri: str, gcs_output_uri: str):
        """Clean up temporary GCS files after batch processing"""
        try:
            # Delete input file
            if gcs_input_uri:
                input_path = gcs_input_uri.replace(f"gs://{self.bucket_name}/", "")
                input_blob = self.bucket.blob(input_path)
                if input_blob.exists():
                    input_blob.delete()
                    logger.debug(f"🧹 Deleted input file: {gcs_input_uri}")
            
            # Delete output folder and contents
            if gcs_output_uri:
                output_prefix = gcs_output_uri.replace(f"gs://{self.bucket_name}/", "").rstrip("/")
                blobs = list(self.storage_client.list_blobs(self.bucket_name, prefix=output_prefix))
                for blob in blobs:
                    blob.delete()
                logger.debug(f"🧹 Deleted {len(blobs)} output file(s) from: {gcs_output_uri}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up GCS files: {str(e)}")
