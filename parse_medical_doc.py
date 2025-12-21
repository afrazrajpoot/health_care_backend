import logging
import time
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO)

def parse_medical_doc_optimized(file_path: str):
    path = Path(file_path)
    
    # 1. CONFIGURE THE PIPELINE FOR ACCURACY & LAYOUT
    pipeline_options = PdfPipelineOptions()
    
    # Enable advanced table structure recognition
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE # or FAST
    
    # OCR Settings: If the document is slow, it's usually OCR.
    # If the PDF has text, 'False' makes it lightning fast.
    pipeline_options.do_ocr = True 
    
    # 2. INITIALIZE CONVERTER WITH OPTIONS
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    
    # 3. CONVERT
    result = converter.convert(path)
    
    # 4. EXPORT
    # Using 'JSON' sometimes preserves keys/values better for RFA forms, 
    # but Markdown is best for LLMs.
    output = result.document.export_to_markdown()
    
    end_time = time.time()
    print(f"\nParsing completed in {end_time - start_time:.2f} seconds.")
    print("\n--- MARKDOWN OUTPUT ---\n")
    print(output)

if __name__ == "__main__":
    parse_medical_doc_optimized("doc.pdf")