#!/usr/bin/env python3
"""
Test Script for Document AI Summarizer
Usage: python test_summarizer.py <file_path>

This script processes a document using the Document AI Summarizer
and displays the results in the terminal.
"""
import sys
import os
import json
import logging

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.short_summary_generator import generate_structured_short_summary
from langchain_openai import AzureChatOpenAI
from config.settings import CONFIG
from services.document_ai_service import DocumentAIProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_summarizer")


def get_llm():
    """Initialize Azure OpenAI LLM."""
    return AzureChatOpenAI(
        azure_deployment=CONFIG.get("azure_openai_deployment"),
        azure_endpoint=CONFIG.get("azure_openai_endpoint"),
        api_key=CONFIG.get("azure_openai_api_key"),
        api_version=CONFIG.get("azure_openai_api_version"),
        temperature=0.1,
        max_tokens=4096,
    )


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def process_document(file_path: str):
    """
    Process a document and display results.
    
    Args:
        file_path: Path to the document file
    """
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Get absolute path
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Determine MIME type
    mime_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }
    
    mime_type = mime_types.get(file_ext)
    if not mime_type:
        print(f"❌ Error: Unsupported file type: {file_ext}")
        print(f"   Supported types: {', '.join(mime_types.keys())}")
        sys.exit(1)
    
    print_separator("DOCUMENT AI SUMMARIZER TEST")
    print(f"📄 File: {file_name}")
    print(f"📁 Path: {file_path}")
    print(f"📋 MIME Type: {mime_type}")
    print(f"📦 Size: {os.path.getsize(file_path):,} bytes")
    
    # Initialize Document AI Service
    print_separator("INITIALIZING DOCUMENT AI SERVICE")
    try:
        doc_ai_service = DocumentAIProcessor()
        print("✅ Document AI Service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Document AI Service: {e}")
        sys.exit(1)
    
    # Process document
    print_separator("PROCESSING DOCUMENT")
    print("⏳ Sending document to Document AI Summarizer...")
    
    try:
        result = doc_ai_service.process_document_with_summarizer(
            filepath=file_path,
            mime_type=mime_type,
            is_first_chunk=True
        )
        
        if not result.success:
            print(f"❌ Document processing failed: {result.error}")
            sys.exit(1)
        
        print("✅ Document processed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display extraction results
    print_separator("EXTRACTION RESULTS")
    print(f"📄 Pages: {result.pages}")
    print(f"📝 Text Length: {len(result.text):,} characters")
    print(f"📋 Raw Text Length: {len(result.raw_text):,} characters")
    
    # Display metadata (patient details)
    if result.metadata:
        print_separator("PATIENT DETAILS (from metadata)")
        print_json(result.metadata)
    
    # Display OCR text (truncated)
    print_separator("OCR SUMMARY TEXT (first 2000 chars)")
    summary_text = result.raw_text[:2000] if len(result.raw_text) > 2000 else result.raw_text
    print(summary_text)
    if len(result.raw_text) > 2000:
        print(f"\n... [truncated, total {len(result.raw_text):,} chars]")
    
    # Generate structured short summary
    print_separator("GENERATING STRUCTURED SHORT SUMMARY")
    print("⏳ Generating UI-ready summary with citations...")
    
    try:
        llm = get_llm()
        
        # Detect document type (simple heuristic)
        doc_type = detect_doc_type(result.raw_text)
        print(f"📋 Detected Document Type: {doc_type}")
        
        # Generate structured summary
        structured_summary = generate_structured_short_summary(
            llm=llm,
            raw_text=result.raw_text,
            doc_type=doc_type,
            long_summary=result.raw_text  # Use raw_text as long_summary for testing
        )
        
        print("✅ Structured summary generated!")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        structured_summary = None
    
    # Display structured summary
    if structured_summary:
        print_separator("STRUCTURED SHORT SUMMARY")
        print_json(structured_summary)
        
        # Display citation statistics
        if "_citation_metadata" in structured_summary:
            print_separator("CITATION STATISTICS")
            print_json(structured_summary["_citation_metadata"])
        
        # Display summary items with citations
        items = structured_summary.get("summary", {}).get("items", [])
        if items:
            print_separator("SUMMARY ITEMS WITH CITATIONS")
            for i, item in enumerate(items, 1):
                print(f"\n--- Item {i}: {item.get('field', 'unknown').upper()} ---")
                print(f"Collapsed: {item.get('collapsed', 'N/A')}")
                print(f"Expanded: {item.get('expanded', 'N/A')[:200]}...")
                
                citations = item.get("citations", [])
                if citations:
                    print(f"\n📚 Citations ({len(citations)}):")
                    for j, citation in enumerate(citations, 1):
                        print(f"  [{j}] Page {citation.get('page_number', '?')}, "
                              f"Para {citation.get('paragraph_index', '?')} | "
                              f"Confidence: {citation.get('confidence', 0):.2%} "
                              f"({citation.get('confidence_level', 'unknown')})")
                        print(f"      Snippet: {citation.get('text_snippet', 'N/A')[:100]}...")
                else:
                    print("  ⚠️ No citations attached")
    
    print_separator("PROCESSING COMPLETE")
    print("✅ All done!")


def detect_doc_type(text: str) -> str:
    """
    Simple document type detection based on text content.
    
    Args:
        text: OCR text from document
        
    Returns:
        Detected document type string
    """
    text_lower = text.lower()
    
    # Check for specific document types
    if "qualified medical evaluator" in text_lower or "qme" in text_lower:
        return "QME"
    elif "agreed medical evaluator" in text_lower or "ame" in text_lower:
        return "AME"
    elif "independent medical evaluation" in text_lower or "ime" in text_lower:
        return "IME"
    elif "utilization review" in text_lower or "ur determination" in text_lower:
        return "UR"
    elif "mri" in text_lower and ("findings" in text_lower or "impression" in text_lower):
        return "MRI"
    elif "ct scan" in text_lower or "computed tomography" in text_lower:
        return "CT"
    elif "x-ray" in text_lower or "xray" in text_lower or "radiograph" in text_lower:
        return "X-RAY"
    elif "emg" in text_lower or "nerve conduction" in text_lower:
        return "EMG"
    elif "physical therapy" in text_lower or "pt evaluation" in text_lower:
        return "PHYSICAL THERAPY"
    elif "progress note" in text_lower or "office visit" in text_lower:
        return "PROGRESS NOTE"
    elif "pr-2" in text_lower or "pr2" in text_lower or "primary treating physician" in text_lower:
        return "PR-2"
    elif "operative report" in text_lower or "surgery" in text_lower:
        return "SURGERY REPORT"
    elif "consultation" in text_lower or "consult" in text_lower:
        return "CONSULT"
    elif "pain management" in text_lower:
        return "PAIN MANAGEMENT"
    else:
        return "MEDICAL REPORT"


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_summarizer.py <file_path>")
        print("\nExamples:")
        print("  python test_summarizer.py document.pdf")
        print("  python test_summarizer.py /path/to/medical_report.pdf")
        print("  python test_summarizer.py scan.png")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_document(file_path)


if __name__ == "__main__":
    main()
