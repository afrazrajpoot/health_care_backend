from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from models.schemas import ExtractionResult
from services.document_ai_service import get_document_ai_processor
from services.file_service import FileService
from services.document_converter import DocumentConverter
from config.settings import CONFIG
from utils.logger import logger

# New imports for comprehensive analysis
from services.report_analyzer import ReportAnalyzer
from services.database_service import get_database_service

router = APIRouter()

@router.post("/extract-document", response_model=ExtractionResult)
async def extract_document(
    document: UploadFile = File(...),
    processor = Depends(get_document_ai_processor)
):
    """
    Upload and process document with Document AI, then summarize the extracted text using GPT-4o via LangChain.
    
    - **document**: The file to be processed
    """
    start_time = datetime.now()
    file_service = FileService()
    
    try:
        logger.info("\n🔄 === NEW DOCUMENT PROCESSING REQUEST ===")
        logger.info(f"📁 Original filename: {document.filename}")
        
        # Validate and read file
        content = await document.read()
        file_service.validate_file(document, CONFIG["max_file_size"])
             
        logger.info(f"📏 File size: {len(content)} bytes ({len(content)/(1024*1024):.2f} MB)")
        logger.info(f"📋 MIME type: {document.content_type}")
        
        # Save to temporary file
        temp_path = file_service.save_temp_file(content, document.filename)
        converted_path = None
        was_converted = False
        
        try:
            # Check if file needs conversion
            if DocumentConverter.needs_conversion(temp_path):
                logger.info(f"🔄 File requires conversion: {Path(temp_path).suffix}")
                converted_path, was_converted = DocumentConverter.convert_document(temp_path, target_format="pdf")
                processing_path = converted_path
                logger.info(f"✅ File converted successfully: {processing_path}")
            else:
                processing_path = temp_path
                logger.info(f"✅ File format supported directly: {Path(temp_path).suffix}")
            
            # Process document with Document AI
            result = processor.process_document(processing_path)
            
            # Add file info
            result.fileInfo = file_service.get_file_info(document, content)
            
            # Comprehensive analysis with GPT-4o and document type detection
            if result.text:
                logger.info("🤖 Starting comprehensive document analysis...")
                try:
                    analyzer = ReportAnalyzer()
                    
                    # Quick document type detection for early logging
                    detected_type = analyzer.detect_document_type_preview(result.text)
                    logger.info(f"🔍 Detected document type: {detected_type}")
                    
                    comprehensive_analysis = analyzer.analyze_document(result.text)
                    result.comprehensive_analysis = comprehensive_analysis
                    
                    # Enhanced summary with document type context
                    if comprehensive_analysis and comprehensive_analysis.summary:
                        # Create detailed summary including document type
                        summary_parts = comprehensive_analysis.summary
                        result.summary = " | ".join(summary_parts)
                        
                        logger.info(f"📋 Document Analysis Summary:")
                        logger.info(f"   📄 Type: {detected_type}")
                        logger.info(f"   👤 Patient: {comprehensive_analysis.report_json.patient_name or 'Unknown'}")
                        logger.info(f"   📑 Title: {comprehensive_analysis.report_json.report_title or 'Untitled'}")
                        logger.info(f"   📝 Summary: {len(summary_parts)} key points extracted")
                        
                        if comprehensive_analysis.work_status_alert:
                            logger.info(f"   🚨 Alerts: {len(comprehensive_analysis.work_status_alert)} generated")
                    else:
                        result.summary = f"Document Type: {detected_type} - Analysis completed successfully"
                    
                    logger.info("✅ Comprehensive analysis completed")
                    
                except Exception as e:
                    logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
                    # Fallback to basic summary with document type
                    analyzer = ReportAnalyzer()
                    try:
                        detected_type = analyzer.detect_document_type_preview(result.text)
                        result.summary = f"Document Type: {detected_type} - Processing completed with limited analysis due to: {str(e)}"
                        logger.info(f"🔄 Fallback: Document type detected as {detected_type}")
                    except:
                        result.summary = f"Document processed successfully but analysis encountered errors: {str(e)}"
                    result.comprehensive_analysis = None
            else:
                logger.warning("⚠️ No text extracted from document")
                result.summary = "Document processed but no readable text content was extracted"
                result.comprehensive_analysis = None
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"⏱️ Total processing time: {processing_time:.0f}ms")
            
            # Save to database
            try:
                db_service = await get_database_service()
                document_id = await db_service.save_document_analysis(
                    extraction_result=result,
                    file_name=document.filename or "unknown",
                    file_size=len(content),
                    mime_type=document.content_type or "application/octet-stream",
                    processing_time_ms=int(processing_time)
                )
                
                # Add document ID to response
                result.document_id = document_id
                logger.info(f"💾 Document saved to database with ID: {document_id}")
                
            except Exception as db_error:
                logger.error(f"⚠️ Failed to save to database: {str(db_error)}")
                # Continue processing - don't fail the request due to DB issues
                result.database_error = str(db_error)
            
            logger.info("✅ === PROCESSING COMPLETED ===\n")
            
            return result
            
        finally:
            # Clean up temporary files
            file_service.cleanup_temp_file(temp_path)
            if was_converted and converted_path:
                DocumentConverter.cleanup_converted_file(converted_path, was_converted)
    
    except ValueError as ve:
        logger.error(f"❌ Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in document extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")