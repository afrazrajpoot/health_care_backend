from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime

from models.schemas import ExtractionResult
from services.document_ai_service import get_document_ai_processor
from services.file_service import FileService
from config.settings import CONFIG
from utils.logger import logger

# New imports for LangChain and OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
        
        try:
            # Process document with Document AI
            result = processor.process_document(temp_path)
            
            # Add file info
            result.fileInfo = file_service.get_file_info(document, content)
            
            # New: Summarize the extracted text using LangChain and GPT-4o
            if result.text:
                logger.info("📝 Starting summarization with GPT-4o...")
                summary = summarize_text(result.text)
                result.summary = summary  # Assuming ExtractionResult has a 'summary' field added (see note below)
                logger.info(f"✅ Summary generated (length: {len(summary)} characters)")
            else:
                result.summary = ""
                logger.info("⚠️ No text extracted, skipping summarization")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"⏱️ Total processing time: {processing_time:.0f}ms")
            logger.info("✅ === PROCESSING COMPLETED ===\n")
            
            return result
            
        finally:
            # Clean up temporary file
            file_service.cleanup_temp_file(temp_path)
    
    except ValueError as ve:
        logger.error(f"❌ Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in document extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# New function for summarization using LangChain and GPT-4o
def summarize_text(text: str) -> str:
    """
    Use LangChain with GPT-4o to generate an understandable summary of the extracted text.
    """
    # Initialize the LLM (GPT-4o)
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=CONFIG["openai_api_key"],  # Assume this is added to your CONFIG or set as env var: os.getenv("OPENAI_API_KEY")
        temperature=0.2  # Lower temperature for more factual summaries
    )
    
    # Define a prompt template for summarization
    prompt = PromptTemplate.from_template(
        "Summarize the following extracted document text in a clear, concise, and understandable way. "
        "Focus on the main points, key information, and overall meaning:\n\n{extracted_text}"
    )
    
    # Create a simple chain: prompt | LLM | output parser
    chain = (
        {"extracted_text": RunnablePassthrough()}  # Pass the text through
        | prompt
        | llm
        | StrOutputParser()  # Parse the output as a string
    )
    
    # Invoke the chain with the extracted text
    summary = chain.invoke(text)
    
    return summary