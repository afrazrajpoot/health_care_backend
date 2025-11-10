# document_ai_service.py
"""
Enhanced Document AI Service with structured JSON output for LLM consumption.
Preserves layout, extracts zones, and generates LLM-optimized JSON structure.
"""

import os
import base64
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient
import logging
from PyPDF2 import PdfReader, PdfWriter
from models.schemas import ExtractionResult, FileInfo
from config.settings import CONFIG

logger = logging.getLogger("document_ai")


class LayoutPreservingTextExtractor:
    """
    Extract text from Document AI results while preserving layout structure.
    Outputs structured JSON for LLM consumption.
    """
    
    @staticmethod
    def extract_text_from_layout(text_anchor, full_text: str) -> str:
        """Extract text from layout text anchor"""
        if not text_anchor or not text_anchor.text_segments:
            return ""
        start_index = text_anchor.text_segments[0].start_index or 0
        end_index = text_anchor.text_segments[0].end_index
        return full_text[start_index:end_index] if end_index else ""
    
    @staticmethod
    def reconstruct_layout_text(document) -> Dict[str, Any]:
        """
        Reconstruct text with layout preservation and structured JSON output.
        
        Returns structured data:
        {
            "layout_preserved": str,  # Text with page markers
            "raw_text": str,  # Original flat text
            "page_zones": {...},  # Per-page zones
            "structured_document": {...}  # LLM-optimized JSON structure
        }
        """
        page_texts = []
        page_zones = {}
        structured_pages = []
        
        for page_idx, page in enumerate(document.pages):
            page_num = page_idx + 1
            page_text_parts = []
            
            # Add page marker
            page_text_parts.append(f"\n{'='*80}\n")
            page_text_parts.append(f"PAGE {page_num}\n")
            page_text_parts.append(f"{'='*80}\n\n")
            
            # Extract zones for this page
            zones = LayoutPreservingTextExtractor._extract_page_zones(
                page, document.text, page_num
            )
            page_zones[str(page_num)] = zones
            
            # Build structured page data
            structured_page = {
                "page_number": page_num,
                "header": zones.get("header", ""),
                "body": zones.get("body", ""),
                "footer": zones.get("footer", ""),
                "signature": zones.get("signature", ""),
                "full_text": ""
            }
            
            # Extract paragraphs
            if page.paragraphs:
                for para in page.paragraphs:
                    para_text = LayoutPreservingTextExtractor.extract_text_from_layout(
                        para.layout.text_anchor, document.text
                    )
                    if para_text.strip():
                        page_text_parts.append(para_text.strip())
                        page_text_parts.append("\n\n")
            elif page.blocks:
                # Fallback: use blocks
                for block in page.blocks:
                    block_text = LayoutPreservingTextExtractor._extract_block_text(
                        block, document.text
                    )
                    if block_text.strip():
                        page_text_parts.append(block_text)
                        page_text_parts.append("\n\n")
            
            page_full_text = "".join(page_text_parts)
            structured_page["full_text"] = page_full_text
            structured_pages.append(structured_page)
            page_texts.append(page_full_text)
        
        layout_preserved = "\n".join(page_texts)
        
        # Build structured document for LLM
        structured_document = {
            "document_structure": {
                "total_pages": len(document.pages),
                "first_page_header": structured_pages[0]["header"] if structured_pages else "",
                "last_page_signature": structured_pages[-1]["signature"] if structured_pages else "",
            },
            "pages": structured_pages,
            "metadata": {
                "has_header": any(p["header"] for p in structured_pages),
                "has_signature": any(p["signature"] for p in structured_pages),
                "total_chars": len(layout_preserved)
            }
        }
        
        return {
            "layout_preserved": layout_preserved,
            "raw_text": document.text or "",
            "page_zones": page_zones,
            "structured_document": structured_document
        }
    
    @staticmethod
    def _extract_block_text(block, full_text: str) -> str:
        """Extract text from a block"""
        if not block.layout or not block.layout.text_anchor:
            return ""
        block_text = LayoutPreservingTextExtractor.extract_text_from_layout(
            block.layout.text_anchor, full_text
        )
        return block_text.strip()
    
    @staticmethod
    def _extract_page_zones(page, full_text: str, page_num: int) -> Dict[str, str]:
        """
        Extract distinct zones from a page: header, body, footer, signature.
        Uses bounding box Y-coordinates to identify zones.
        """
        zones = {
            "header": "",
            "body": "",
            "footer": "",
            "signature": "",
            "page_number": str(page_num),
        }
        
        if not page.paragraphs and not page.blocks:
            return zones
        
        # Collect elements with their Y-coordinates
        elements_with_coords = []
        
        # Try paragraphs first
        if page.paragraphs:
            for para in page.paragraphs:
                if para.layout and para.layout.bounding_poly:
                    vertices = para.layout.bounding_poly.normalized_vertices
                    if vertices:
                        avg_y = sum(v.y for v in vertices) / len(vertices)
                        para_text = LayoutPreservingTextExtractor.extract_text_from_layout(
                            para.layout.text_anchor, full_text
                        )
                        elements_with_coords.append({
                            "text": para_text,
                            "y_pos": avg_y,
                            "confidence": para.layout.confidence if para.layout else 0.0
                        })
        elif page.blocks:
            # Fallback to blocks
            for block in page.blocks:
                if block.layout and block.layout.bounding_poly:
                    vertices = block.layout.bounding_poly.normalized_vertices
                    if vertices:
                        avg_y = sum(v.y for v in vertices) / len(vertices)
                        block_text = LayoutPreservingTextExtractor._extract_block_text(
                            block, full_text
                        )
                        elements_with_coords.append({
                            "text": block_text,
                            "y_pos": avg_y,
                            "confidence": block.layout.confidence if block.layout else 0.0
                        })
        
        if not elements_with_coords:
            return zones
        
        # Sort by Y position (top to bottom)
        elements_with_coords.sort(key=lambda x: x["y_pos"])
        
        # Define zones based on Y-position thresholds
        header_elements = []
        body_elements = []
        footer_elements = []
        
        for element in elements_with_coords:
            y = element["y_pos"]
            text = element["text"]
            
            if y < 0.20:  # Top 20%
                header_elements.append(text)
            elif y > 0.75:  # Bottom 25%
                footer_elements.append(text)
            else:  # Middle section
                body_elements.append(text)
        
        zones["header"] = "\n".join(header_elements)
        zones["body"] = "\n".join(body_elements)
        zones["footer"] = "\n".join(footer_elements)
        
        # Signature is typically in footer, look for signature keywords
        footer_text = zones["footer"].lower()
        if any(keyword in footer_text for keyword in [
            "signature", "signed by", "electronically signed",
            "dr.", "md", "do", "physician"
        ]):
            zones["signature"] = zones["footer"]
        
        return zones


def build_llm_friendly_json(structured_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build LLM-optimized JSON structure with clear annotations.
    This provides structured context that LLMs can parse better than plain text.
    
    Args:
        structured_document: Document structure from layout extraction
    
    Returns:
        JSON object optimized for LLM classification and extraction
    """
    pages = structured_document.get("pages", [])
    
    if not pages:
        return {
            "document_type_hints": {
                "header_text": "",
                "first_page_context": "",
                "has_form_structure": False
            },
            "content": {
                "header": "",
                "body_preview": "",
                "signature": ""
            }
        }
    
    first_page = pages[0]
    last_page = pages[-1]
    
    # Build classification hints
    header_text = first_page.get("header", "").strip()
    
    # Extract potential document type indicators from header
    type_indicators = []
    header_upper = header_text.upper()
    
    # Check for common form identifiers
    form_patterns = {
        "RFA": ["REQUEST FOR AUTHORIZATION", "DWC FORM RFA"],
        "PR2": ["PR-2", "PROGRESS REPORT", "TREATING PHYSICIAN"],
        "DFR": ["DOCTOR'S FIRST REPORT", "DLSR 5021", "FIRST REPORT OF OCCUPATIONAL INJURY"],
        "QME": ["QUALIFIED MEDICAL EVALUATOR", "QME REPORT"],
        "UR": ["UTILIZATION REVIEW", "UR DECISION"],
        "MRI": ["MRI REPORT", "MAGNETIC RESONANCE"],
        "CT": ["CT REPORT", "COMPUTED TOMOGRAPHY"],
    }
    
    for doc_type, patterns in form_patterns.items():
        if any(pattern in header_upper for pattern in patterns):
            type_indicators.append(doc_type)
    
    # Build body preview (first 2000 chars from body zones)
    body_preview_parts = []
    for page in pages[:3]:  # First 3 pages
        body = page.get("body", "").strip()
        if body:
            body_preview_parts.append(body)
    body_preview = "\n\n".join(body_preview_parts)[:2000]
    
    # Signature extraction
    signature_text = last_page.get("signature", "").strip() or last_page.get("footer", "").strip()
    
    llm_text = {
        "document_type_hints": {
            "detected_indicators": type_indicators,
            "header_text": header_text[:500],  # First 500 chars of header
            "first_page_context": first_page.get("full_text", "")[:1000],
            "has_form_structure": bool(type_indicators),
            "page_count": len(pages)
        },
        "content": {
            "header": header_text,
            "body_preview": body_preview,
            "signature": signature_text[:300]
        },
        "zones": {
            "first_page_header": first_page.get("header", ""),
            "first_page_body": first_page.get("body", "")[:1500],
            "last_page_signature": signature_text
        },
        "metadata": structured_document.get("metadata", {})
    }
    
    return llm_text


class PDFSplitter:
    """Utility to split large PDFs into smaller chunks"""
    
    def __init__(self, max_pages_per_chunk: int = 10):
        self.max_pages_per_chunk = max_pages_per_chunk
    
    def split_pdf(self, filepath: str) -> List[str]:
        """Split PDF into multiple chunks"""
        try:
            logger.info(f"🔍 Splitting PDF: {filepath}")
            
            with open(filepath, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
            
            logger.info(f"📄 Total pages: {total_pages}")
            logger.info(f"📦 Max pages per chunk: {self.max_pages_per_chunk}")
            
            if total_pages <= self.max_pages_per_chunk:
                logger.info("✅ No splitting needed")
                return [filepath]
            
            num_chunks = (total_pages + self.max_pages_per_chunk - 1) // self.max_pages_per_chunk
            logger.info(f"✂️ Splitting into {num_chunks} chunks")
            
            chunk_files = []
            for chunk_num in range(num_chunks):
                start_page = chunk_num * self.max_pages_per_chunk
                end_page = min((chunk_num + 1) * self.max_pages_per_chunk, total_pages)
                
                chunk_file = self._create_chunk(filepath, start_page, end_page, chunk_num)
                chunk_files.append(chunk_file)
                logger.info(f"✅ Created chunk {chunk_num + 1}: pages {start_page + 1}-{end_page}")
            
            return chunk_files
        
        except Exception as e:
            logger.error(f"❌ Error splitting PDF: {str(e)}")
            raise
    
    def _create_chunk(self, original_path: str, start: int, end: int, chunk_num: int) -> str:
        """Create a single PDF chunk"""
        with open(original_path, "rb") as file:
            pdf_reader = PdfReader(file)
            pdf_writer = PdfWriter()
            
            for page_num in range(start, end):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            original_stem = Path(original_path).stem
            timestamp = datetime.now().strftime("%H%M%S%f")
            output_filename = f"{original_stem}_chunk{chunk_num + 1}_{timestamp}.pdf"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            with open(output_path, "wb") as output_file:
                pdf_writer.write(output_file)
            
            return output_path
    
    def cleanup_chunks(self, chunk_files: List[str]):
        """Clean up temporary chunk files"""
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file) and "chunk" in chunk_file:
                    os.remove(chunk_file)
                    logger.debug(f"🧹 Cleaned up {chunk_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up {chunk_file}: {str(e)}")


class DocumentAIProcessor:
    """Service for Document AI processing with structured JSON output"""
    
    def __init__(self):
        self.client: Optional[DocumentProcessorServiceClient] = None
        self.processor_path: Optional[str] = None
        self.pdf_splitter = PDFSplitter(max_pages_per_chunk=10)
        self.layout_extractor = LayoutPreservingTextExtractor()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Document AI client"""
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CONFIG["credentials_path"]
            logger.info(f"🔑 Using credentials: {CONFIG['credentials_path']}")
            logger.info(f"🆔 Project ID: {CONFIG['project_id']}")
            
            api_endpoint = f"{CONFIG['location']}-documentai.googleapis.com"
            logger.info(f"🌐 API Endpoint: {api_endpoint}")
            
            self.client = DocumentProcessorServiceClient(
                client_options={"api_endpoint": api_endpoint}
            )
            
            self.processor_path = self.client.processor_path(
                CONFIG["project_id"], CONFIG["location"], CONFIG["processor_id"]
            )
            
            logger.info("✅ Document AI Client initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document AI Client: {str(e)}")
            raise
    
    def get_mime_type(self, filepath: str) -> str:
        """Get MIME type based on file extension"""
        mime_mapping = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        
        file_ext = Path(filepath).suffix.lower()
        return mime_mapping.get(file_ext, "application/octet-stream")
    
    def extract_entities(self, document) -> List[Dict[str, Any]]:
        """Extract entities from document"""
        entities = []
        if document.entities:
            for entity in document.entities:
                entities.append({
                    "type": entity.type_,
                    "mentionText": entity.mention_text,
                    "confidence": float(entity.confidence),
                    "id": entity.id,
                })
        return entities
    
    def extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        tables = []
        if not document.pages:
            return tables
        
        for page_index, page in enumerate(document.pages):
            if page.tables:
                for table_index, table in enumerate(page.tables):
                    table_data = {
                        "pageNumber": page_index + 1,
                        "tableIndex": table_index + 1,
                        "headerRows": [],
                        "bodyRows": [],
                    }
                    
                    if table.header_rows:
                        for row in table.header_rows:
                            row_data = []
                            for cell in row.cells:
                                cell_text = (
                                    self.layout_extractor.extract_text_from_layout(
                                        cell.layout.text_anchor, document.text
                                    )
                                    if cell.layout and cell.layout.text_anchor
                                    else ""
                                )
                                row_data.append({
                                    "text": cell_text.strip(),
                                    "confidence": float(cell.layout.confidence) if cell.layout else 0.0,
                                })
                            table_data["headerRows"].append(row_data)
                    
                    if table.body_rows:
                        for row in table.body_rows:
                            row_data = []
                            for cell in row.cells:
                                cell_text = (
                                    self.layout_extractor.extract_text_from_layout(
                                        cell.layout.text_anchor, document.text
                                    )
                                    if cell.layout and cell.layout.text_anchor
                                    else ""
                                )
                                row_data.append({
                                    "text": cell_text.strip(),
                                    "confidence": float(cell.layout.confidence) if cell.layout else 0.0,
                                })
                            table_data["bodyRows"].append(row_data)
                    
                    tables.append(table_data)
        
        return tables
    
    def extract_form_fields(self, document) -> List[Dict[str, Any]]:
        """Extract form fields from document"""
        form_fields = []
        if not document.pages:
            return form_fields
        
        for page in document.pages:
            if page.form_fields:
                for field in page.form_fields:
                    field_name = ""
                    field_value = ""
                    
                    if field.field_name and field.field_name.text_anchor:
                        field_name = self.layout_extractor.extract_text_from_layout(
                            field.field_name.text_anchor, document.text
                        )
                    
                    if field.field_value and field.field_value.text_anchor:
                        field_value = self.layout_extractor.extract_text_from_layout(
                            field.field_value.text_anchor, document.text
                        )
                    
                    form_fields.append({
                        "name": field_name.strip(),
                        "value": field_value.strip(),
                        "nameConfidence": float(field.field_name.confidence) if field.field_name else 0.0,
                        "valueConfidence": float(field.field_value.confidence) if field.field_value else 0.0,
                    })
        
        return form_fields
    
    def calculate_overall_confidence(self, document) -> float:
        """Calculate overall confidence score"""
        if not document.pages:
            return 0.0
        
        total_confidence = 0
        count = 0
        
        for page in document.pages:
            if page.layout and page.layout.confidence:
                total_confidence += float(page.layout.confidence)
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def _process_document_direct(self, filepath: str) -> ExtractionResult:
        """
        Direct document processing with structured JSON output.
        """
        try:
            mime_type = self.get_mime_type(filepath)
            logger.info(f"📄 Processing document: {filepath}")
            logger.info(f"📋 MIME type: {mime_type}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"📦 File size: {file_size} bytes")
            
            with open(filepath, "rb") as file:
                file_content = file.read()
            
            encoded_content = base64.b64encode(file_content).decode("utf-8")
            
            request = {
                "name": self.processor_path,
                "raw_document": {
                    "content": encoded_content,
                    "mime_type": mime_type,
                },
            }
            
            logger.info("📤 Sending request to Document AI...")
            response = self.client.process_document(request=request)
            result = response.document
            
            logger.info("✅ Document processed successfully!")
            logger.info(f"📝 Extracted text length: {len(result.text) if result.text else 0} characters")
            logger.info(f"📄 Pages found: {len(result.pages) if result.pages else 0}")
            
            # **KEY ENHANCEMENT: Extract layout + build structured JSON**
            layout_data = self.layout_extractor.reconstruct_layout_text(result)
            
            logger.info(f"🔍 Layout preservation complete:")
            logger.info(f"  - Raw text: {len(layout_data['raw_text'])} chars")
            logger.info(f"  - Layout text: {len(layout_data['layout_preserved'])} chars")
            logger.info(f"  - Page zones: {len(layout_data['page_zones'])} pages")
            
            # **NEW: Build LLM-friendly JSON**
            llm_json = build_llm_friendly_json(layout_data['structured_document'])
            
            logger.info(f"🤖 LLM JSON structure built:")
            logger.info(f"  - Type indicators: {llm_json['document_type_hints']['detected_indicators']}")
            logger.info(f"  - Has form structure: {llm_json['document_type_hints']['has_form_structure']}")
            
            # Convert JSON to formatted string for LLM
            llm_text = json.dumps(llm_json, indent=2)
            
            # Preview JSON (first 500 chars)
            json_preview = llm_text[:500] + "..."
            logger.info(f"📖 LLM JSON preview:\n{json_preview}")
            
            processed_result = ExtractionResult(
                # Layout-preserved text
                text=layout_data["layout_preserved"],
                
                # Raw text for backward compatibility
                raw_text=layout_data["raw_text"],
                
                # **NEW: Structured JSON string for LLM**
                llm_text=llm_text,
                
                # Page zones
                page_zones=layout_data["page_zones"],
                
                # Standard fields
                pages=len(result.pages) if result.pages else 0,
                entities=self.extract_entities(result),
                tables=self.extract_tables(result),
                formFields=self.extract_form_fields(result),
                confidence=self.calculate_overall_confidence(result),
                success=True,
            )
            
            logger.info("📊 Extraction summary:")
            logger.info(f"  - Text characters: {len(processed_result.text)}")
            logger.info(f"  - Pages: {processed_result.pages}")
            logger.info(f"  - Entities: {len(processed_result.entities)}")
            logger.info(f"  - Tables: {len(processed_result.tables)}")
            logger.info(f"  - Overall confidence: {processed_result.confidence * 100:.2f}%")
            
            # Debug: Verify page_zones is in the result
            logger.info(f"🔍 Result object check:")
            logger.info(f"  - Has page_zones: {processed_result.page_zones is not None}")
            logger.info(f"  - page_zones keys: {list(processed_result.page_zones.keys()) if processed_result.page_zones else 'None'}")
            logger.info(f"  - Has llm_text: {processed_result.llm_text is not None}")
            logger.info(f"  - llm_text length: {len(processed_result.llm_text) if processed_result.llm_text else 0}")
            
            return processed_result
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error processing document: {error_msg}")
            return ExtractionResult(success=False, error=error_msg)
    
    def process_large_document(self, filepath: str) -> ExtractionResult:
        """Process large documents by splitting them first"""
        try:
            chunk_files = self.pdf_splitter.split_pdf(filepath)
            
            if len(chunk_files) == 1:
                return self._process_document_direct(chunk_files[0])
            
            logger.info(f"📦 Processing {len(chunk_files)} chunks")
            
            all_results = []
            for i, chunk_file in enumerate(chunk_files):
                logger.info(f"🔄 Processing chunk {i + 1}/{len(chunk_files)}")
                
                try:
                    chunk_result = self._process_document_direct(chunk_file)
                    if chunk_result.success:
                        all_results.append(chunk_result)
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i + 1}: {str(e)}")
            
            self.pdf_splitter.cleanup_chunks(chunk_files)
            
            if not all_results:
                return ExtractionResult(success=False, error="All chunks failed")
            
            return self._merge_results(all_results, filepath)
        
        except Exception as e:
            logger.error(f"❌ Error processing large document: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")
    
    def _merge_results(self, results: List[ExtractionResult], original_file: str) -> ExtractionResult:
        """Merge results from multiple chunks"""
        if not results:
            return ExtractionResult(success=False, error="No successful chunks")
        
        merged_text = ""
        merged_raw_text = ""
        merged_page_zones = {}
        merged_llm_texts = []
        
        logger.info(f"🔗 Starting merge of {len(results)} chunks...")
        
        for i, result in enumerate(results):
            chunk_num = i + 1
            logger.info(f"📦 Processing chunk {chunk_num}:")
            logger.info(f"   - Has text: {bool(result.text)}")
            logger.info(f"   - Has page_zones: {bool(result.page_zones)}")
            logger.info(f"   - Pages in chunk: {result.pages}")
            
            if result.text:
                if i > 0:
                    merged_text += f"\n\n{'='*80}\nCHUNK {i + 1}\n{'='*80}\n\n"
                merged_text += result.text
            
            if hasattr(result, 'raw_text') and result.raw_text:
                merged_raw_text += result.raw_text + "\n\n"
            
            if hasattr(result, 'page_zones') and result.page_zones:
                page_offset = sum(r.pages for r in results[:i])
                logger.info(f"   - Page offset for chunk {chunk_num}: {page_offset}")
                logger.info(f"   - Page_zones keys in chunk: {list(result.page_zones.keys())}")
                
                for page_num_str, zones in result.page_zones.items():
                    page_num = int(page_num_str) if isinstance(page_num_str, str) else page_num_str
                    new_page_num = page_num + page_offset
                    merged_page_zones[str(new_page_num)] = zones
                    logger.info(f"   - Mapped page {page_num_str} → {new_page_num}")
            else:
                logger.warning(f"   ⚠️ Chunk {chunk_num} has NO page_zones!")
            
            # Merge LLM text strings
            if hasattr(result, 'llm_text') and result.llm_text:
                merged_llm_texts.append(f"CHUNK {i + 1}:\n{result.llm_text}")
        
        # Combine all LLM texts with chunk markers
        merged_llm_text = "\n\n".join(merged_llm_texts) if merged_llm_texts else None
        
        merged_entities = []
        merged_tables = []
        merged_form_fields = []
        total_pages = 0
        total_confidence = 0.0
        
        for result in results:
            merged_entities.extend(result.entities)
            merged_tables.extend(result.tables)
            merged_form_fields.extend(result.formFields)
            total_pages += result.pages
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        logger.info(f"🔗 Merge complete:")
        logger.info(f"  - Total pages: {total_pages}")
        logger.info(f"  - Total text: {len(merged_text)} chars")
        logger.info(f"  - Merged page_zones: {len(merged_page_zones)} pages")
        logger.info(f"  - Page_zones keys: {sorted([int(k) for k in merged_page_zones.keys()])}")
        
        merged_result = ExtractionResult(
            text=merged_text,
            raw_text=merged_raw_text,
            llm_text=merged_llm_text,
            page_zones=merged_page_zones,
            pages=total_pages,
            entities=merged_entities,
            tables=merged_tables,
            formFields=merged_form_fields,
            confidence=avg_confidence,
            success=True,
        )
        
        # Debug: Verify merged result has page_zones
        logger.info(f"🔍 Merged result check:")
        logger.info(f"  - Has page_zones: {merged_result.page_zones is not None}")
        logger.info(f"  - page_zones keys: {list(merged_result.page_zones.keys()) if merged_result.page_zones else 'None'}")
        logger.info(f"  - Has llm_text: {merged_result.llm_text is not None}")
        
        return merged_result
    
    def process_document(self, filepath: str) -> ExtractionResult:
        """Main document processing method"""
        try:
            mime_type = self.get_mime_type(filepath)
            
            if mime_type == "application/pdf":
                try:
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        page_count = len(pdf_reader.pages)
                    
                    logger.info(f"📄 Document has {page_count} pages")
                    
                    if page_count > 10:
                        logger.info("📦 Using chunked processing")
                        return self.process_large_document(filepath)
                    else:
                        return self._process_document_direct(filepath)
                
                except Exception as e:
                    logger.warning(f"⚠️ Could not check page count: {str(e)}")
                    return self._process_document_direct(filepath)
            else:
                return self._process_document_direct(filepath)
        
        except Exception as e:
            logger.error(f"❌ Error in main processing: {str(e)}")
            return ExtractionResult(success=False, error=f"Failed: {str(e)}")


# Global processor instance
processor_instance = None


def get_document_ai_processor() -> DocumentAIProcessor:
    """Get singleton DocumentAIProcessor instance"""
    global processor_instance
    if processor_instance is None:
        try:
            logger.info("🚀 Initializing Document AI processor...")
            processor_instance = DocumentAIProcessor()
            logger.info("✅ Document AI processor ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            raise
    return processor_instance


def process_document_smart(filepath: str) -> ExtractionResult:
    """Smart document processing with structured JSON output"""
    processor = get_document_ai_processor()
    return processor.process_document(filepath)
