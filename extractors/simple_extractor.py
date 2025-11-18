"""
Enhanced Simple Extractor with Type-Specific Sections from PromptManager
"""

import re
import logging
import time
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from typing import List, Tuple
from utils.doctor_detector import DoctorDetector
from extractors.prompt_manager import PromptManager

logger = logging.getLogger("document_ai")


class SimpleExtractor:
    """Enhanced extractor with type-specific sections and better error handling"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.doctor_detector = DoctorDetector(llm)
        self.prompt_manager = PromptManager()  # Now instantiated as class instance
        
        logger.info("✅ SimpleExtractor initialized with type-specific sections")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> Dict:
        """
        Enhanced extraction with automatic type-specific sections and comprehensive error handling.
        """
        logger.info("=" * 80)
        logger.info(f"🚀 EXTRACTING: {doc_type}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Empty document text provided")
            
            # Get prompt and section info for logging
            prompt_info = self.prompt_manager.get_prompt_info(doc_type)
            section_info = self.prompt_manager.get_section_info(doc_type)
            logger.info(f"🎯 Prompt routing: {prompt_info}")
            logger.info(f"📋 Section mapping: {section_info}")
            
            # Stage 1: Extract with type-specific prompt
            raw_data = self._extract_with_type_prompt(text, doc_type, fallback_date)
            
            # Stage 2: Always detect and include physician
            physician_name = self._detect_physician(text, page_zones)
            if physician_name:
                raw_data["physician_name"] = physician_name
            
            # Stage 3: Build long summary with type-specific sections
            long_summary = self._build_long_summary(raw_data, doc_type, fallback_date, physician_name)
            
            # Stage 4: Generate short summary
            short_summary = self._generate_short_summary(long_summary, doc_type)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Extraction completed in {elapsed_time:.2f}s")
            logger.info(f"📊 Results: {len(long_summary)} chars, {len(short_summary.split())} words")
            
            return {
                "long_summary": long_summary,
                "short_summary": short_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_type}: {str(e)}")
            return self._create_error_response(doc_type, str(e))

    def _extract_with_type_prompt(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Extract data using universal prompt"""
        prompt = self.prompt_manager.get_extraction_prompt(doc_type)
        
        try:
            # Use appropriate context length based on document type
            context_length = self._get_context_length(doc_type)
            context_text = text[:context_length]
            
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": context_text,
                "doc_type": doc_type,  # Pass doc_type for context
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Validate and clean extracted data
            result = self._clean_extracted_data(result, fallback_date)
            
            logger.info(f"✅ Universal extraction complete for {doc_type} - extracted {len(result)} fields")
            return result
            
        except Exception as e:
            logger.error(f"❌ Extraction failed for {doc_type}: {e}")
            return self._create_fallback_data(fallback_date, doc_type)
    def _get_context_length(self, doc_type: str) -> int:
        """Determine appropriate context length based on document type"""
        # Longer context for complex documents, shorter for simple ones
        complex_docs = ["QME", "AME", "IME", "SURGERY_REPORT", "DISCHARGE"]
        if doc_type in complex_docs:
            return 16000  # Longer context for complex documents
        return 12000  # Standard context for most documents

    def _clean_extracted_data(self, data: Dict, fallback_date: str) -> Dict:
        """Clean and validate extracted data"""
        if not isinstance(data, dict):
            return {"date": fallback_date, "key_findings": "Invalid data format"}
        
        # Ensure date field exists
        if not data.get("date"):
            data["date"] = fallback_date
        
        # Remove empty or invalid fields
        cleaned_data = {}
        for key, value in data.items():
            if self._is_valid_content(value):
                cleaned_data[key] = value
        
        return cleaned_data

    def _detect_physician(self, text: str, page_zones: Optional[Dict]) -> str:
        """Always detect physician from document with enhanced fallback"""
        try:
            detection_result = self.doctor_detector.detect_doctor(
                text=text,
                page_zones=page_zones
            )
            doctor_name = detection_result.get("doctor_name", "")
            
            # Validate doctor name
            if doctor_name and len(doctor_name.strip()) > 2:  # Basic validation
                logger.info(f"👨‍⚕️ Physician detected: {doctor_name}")
                return doctor_name.strip()
            else:
                logger.warning("⚠️ No valid physician name detected")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Physician detection failed: {e}")
            return ""
    


    def _generate_short_summary(self, long_summary: str, doc_type: str) -> str:
        """Generate short summary from the comprehensive long summary"""
        prompt = self.prompt_manager.get_short_summary_prompt(doc_type)
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "long_summary": long_summary,  # Use the full comprehensive summary
                "doc_type": doc_type
            })
            
            short_summary = response.content.strip()
            cleaned_summary = self._clean_and_validate_short_summary(short_summary)
            
            word_count = len(cleaned_summary.split())
            logger.info(f"✅ Short summary generated from comprehensive source: {word_count} words")
            return cleaned_summary
            
        except Exception as e:
            logger.error(f"❌ Short summary generation failed: {e}")
            return self._create_intelligent_fallback_summary(long_summary, doc_type)
    
    def _build_enhanced_structured_summary(self, raw_data: Dict, doc_type: str, fallback_date: str, physician_name: str = "") -> str:
        """Build enhanced structured summary when LLM fails"""
        sections = [f"📄 COMPREHENSIVE {doc_type} REPORT", "=" * 60]
        
        # Enhanced header with more context
        info_lines = [
            f"Report Date: {raw_data.get('date', fallback_date)}",
            f"Document Type: {doc_type}",
            f"Comprehensive Medical Summary"
        ]
        if physician_name:
            info_lines.append(f"Evaluating Physician: {physician_name}")
        sections.append("\n".join(info_lines))
        
        # Get type-specific section mapping
        section_mapping = self.prompt_manager.get_section_mapping(doc_type)
        
        # Enhanced section content with more detail
        added_sections = 0
        for section_name, data_key in section_mapping:
            content = raw_data.get(data_key)
            if content and self._is_valid_content(content):
                sections.append(f"\n{section_name}")
                sections.append("-" * 40)
                
                # Enhanced content formatting
                formatted_content = self._format_section_content(content, section_name)
                sections.append(formatted_content)
                added_sections += 1
        
        # Add clinical context section if minimal content
        if added_sections < 3:
            sections.append("\nCLINICAL CONTEXT")
            sections.append("-" * 40)
            key_findings = raw_data.get('key_findings', 'Comprehensive clinical assessment performed.')
            sections.append(f"This {doc_type.lower()} report documents a medical evaluation with the following key aspects: {key_findings}")
            
            # Add any available additional context
            if raw_data.get('diagnosis'):
                sections.append(f"Primary diagnosis includes: {raw_data.get('diagnosis')}")
            if raw_data.get('treatment'):
                sections.append(f"Treatment considerations: {raw_data.get('treatment')}")
        
        summary = "\n\n".join(sections)
        word_count = len(summary.split())
        logger.info(f"📊 Enhanced structured summary: {word_count} words")
        
        return summary

 
    def _build_structured_summary(self, raw_data: Dict, doc_type: str, fallback_date: str, physician_name: str = "") -> str:
        """Build structured summary with type-specific sections"""
        sections = [f"📄 {doc_type} REPORT", "=" * 50]
        
        # Basic info (always include physician if available)
        info_lines = [f"Date: {raw_data.get('date', fallback_date)}"]
        if physician_name:
            info_lines.append(f"Physician: {physician_name}")
        sections.append("\n".join(info_lines))
        
        # Get type-specific section mapping
        section_mapping = self.prompt_manager.get_section_mapping(doc_type)
        
        # Track added sections for logging
        added_sections = 0
        
        # Add sections based on available data
        for section_name, data_key in section_mapping:
            content = raw_data.get(data_key)
            if content and self._is_valid_content(content):
                sections.append(f"\n{section_name}")
                sections.append("-" * 30)
                sections.append(str(content))
                added_sections += 1
        
        summary = "\n\n".join(sections)
        
        # If no content sections were added, create a basic summary
        if added_sections == 0:
            key_findings = raw_data.get('key_findings', 'No specific findings extracted')
            summary += f"\n\nSUMMARY: {key_findings}"
            logger.warning(f"⚠️ No detailed sections extracted for {doc_type}, using basic summary")
        else:
            logger.info(f"✅ Structured summary built with {added_sections} sections")
        
        return summary

    def _is_valid_content(self, content) -> bool:
        """Enhanced content validation"""
        if not content:
            return False
        if isinstance(content, str):
            clean_content = content.strip()
            if not clean_content or clean_content in ['', 'None', 'Not specified', 'N/A', 'null']:
                return False
            # Check if content has meaningful length
            if len(clean_content) < 10:  # Very short content might not be useful
                return False
        elif isinstance(content, list):
            if len(content) == 0:
                return False
            # Check if list contains meaningful items
            meaningful_items = [item for item in content if item and str(item).strip()]
            return len(meaningful_items) > 0
        elif isinstance(content, dict):
            return len(content) > 0
        return True

  
    def _clean_and_validate_short_summary(self, summary: str) -> str:
        """Enhanced cleaning and validation for short summary"""
        # Clean the summary
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = re.sub(r'[\*\#\-]', '', summary)  # Remove markdown
        summary = re.sub(r'^(60-word summary:|summary:|medical summary:)\s*', '', summary, flags=re.IGNORECASE)
        
        words = summary.split()
        word_count = len(words)
        
        # Strict word count enforcement with intelligent adjustment
        if word_count != 60:
            logger.warning(f"⚠️ Short summary word count: {word_count} (target: 60)")
            
            if word_count > 60:
                # Remove less critical words from the end while preserving meaning
                summary = self._truncate_intelligently(words, 60)
            elif word_count < 55:
                # Add meaningful context if too short
                summary = self._enhance_short_summary(summary, word_count)
        
        return ' '.join(summary.split()[:60])  # Final enforcement

    def _truncate_intelligently(self, words: list, target_length: int) -> str:
        """Intelligently truncate summary while preserving meaning"""
        # Keep the most important parts (usually the beginning)
        return ' '.join(words[:target_length])

    def _enhance_short_summary(self, summary: str, current_word_count: int) -> str:
        """Enhance short summary with meaningful context"""
        if current_word_count < 50:
            summary = f"{summary} Additional clinical details documented in full report."
        elif current_word_count < 55:
            summary = f"{summary} Comprehensive evaluation completed per medical standards."
        
        return summary

    def _create_intelligent_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create intelligent fallback summary from long summary"""
        try:
            # Extract key sentences with priority for clinical information
            sentences = re.split(r'[.!?]+', long_summary)
            key_sentences = []
            
            # Priority keywords for different document types
            clinical_keywords = ['diagnosis', 'findings', 'treatment', 'pain', 'symptoms', 'results']
            evaluation_keywords = ['mmi', 'disability', 'work', 'restrictions', 'rating']
            authorization_keywords = ['approved', 'denied', 'authorized', 'appeal', 'decision']
            
            all_keywords = clinical_keywords + evaluation_keywords + authorization_keywords
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 20:  # Skip very short sentences
                    continue
                    
                # Prioritize sentences with key information
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in all_keywords):
                    key_sentences.append(sentence)
                
                if len(key_sentences) >= 4:  # Limit to 4 key sentences
                    break
            
            # If no key sentences found, use first meaningful sentences
            if not key_sentences:
                key_sentences = [s.strip() for s in sentences[:3] if s.strip() and len(s.strip()) > 30]
            
            summary = ' '.join(key_sentences[:4])  # Use up to 4 sentences
            
            # Ensure 60 words
            words = summary.split()
            if len(words) > 60:
                summary = ' '.join(words[:60])
            elif len(words) < 50:
                summary = f"{summary} Additional details in comprehensive report."
            
            logger.info(f"🔄 Using fallback summary: {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Fallback summary creation failed: {e}")
            return f"{doc_type} report summary - detailed extraction unavailable."

    
    def _create_fallback_data(self, fallback_date: str, doc_type: str) -> Dict:
        """Create comprehensive fallback data structure"""
        return {
            "date": fallback_date,
            "key_findings": f"Extraction incomplete for {doc_type}. Manual review recommended.",
            "diagnosis": "Not extracted",
            "treatment": "Not extracted",
            "physician_name": "Not detected"
        }
    def _build_enhanced_section_summary(self, raw_data: Dict, doc_type: str, fallback_date: str, physician_name: str, section_mapping: List[Tuple[str, str]]) -> str:
        """Build enhanced section-based summary with type-specific headings"""
        sections = []
        
        # Document header
        sections.append(f"📄 COMPREHENSIVE {doc_type} REPORT")
        sections.append("=" * 60)
        
        # Document info
        info_lines = [f"Report Date: {raw_data.get('date', fallback_date)}"]
        if physician_name:
            info_lines.append(f"Physician: {physician_name}")
        sections.append("\n".join(info_lines))
        sections.append("")  # Empty line for spacing
        
        # Add each section with content
        for section_name, data_key in section_mapping:
            content = raw_data.get(data_key)
            if content and self._is_valid_content(content):
                sections.append(section_name)
                sections.append("-" * 40)
                
                formatted_content = self._format_section_content(content, section_name)
                sections.append(formatted_content)
                sections.append("")  # Empty line between sections
        
        # Add summary if minimal sections
        if len(sections) <= 5:  # Only header and 1-2 sections
            sections.append("CLINICAL SUMMARY")
            sections.append("-" * 40)
            key_info = []
            if raw_data.get('key_findings'):
                key_info.append(f"Key Findings: {raw_data.get('key_findings')}")
            if raw_data.get('diagnosis'):
                key_info.append(f"Diagnosis: {raw_data.get('diagnosis')}")
            if raw_data.get('treatment'):
                key_info.append(f"Treatment: {raw_data.get('treatment')}")
            if raw_data.get('recommendations'):
                key_info.append(f"Recommendations: {raw_data.get('recommendations')}")
            
            if key_info:
                sections.append("\n".join(key_info))
            else:
                sections.append("Comprehensive medical evaluation completed. Detailed clinical assessment documented in full report.")
        
        summary = "\n".join(sections)
        word_count = len(summary.split())
        logger.info(f"📊 Enhanced section summary: {word_count} words, {len([s for s in sections if '=' in s or '---' in s])} sections")
        
        return summary

    def _build_long_summary(self, raw_data: Dict, doc_type: str, fallback_date: str, physician_name: str = "") -> str:
        """Build section-based long summary using type-specific headings"""
        try:
            # Get type-specific section mapping
            section_mapping = self.prompt_manager.get_section_mapping(doc_type)
            
            # Prepare section headings for the prompt
            section_headings = [section[0] for section in section_mapping]
            headings_text = "\n".join([f"- {heading}" for heading in section_headings])
            
            # Prepare data for LLM with physician info
            summary_data = raw_data.copy()
            if physician_name and "physician_name" not in summary_data:
                summary_data["physician_name"] = physician_name
            
            # Get the section-based long summary prompt
            prompt = self.prompt_manager.get_long_summary_prompt(doc_type)
            chain = prompt | self.llm
            response = chain.invoke({
                "raw_data": summary_data,
                "doc_type": doc_type,
                "section_headings": headings_text
            })
            
            long_summary = response.content.strip()
            word_count = len(long_summary.split())
            
            # Validate word count and structure
            if word_count >= 350 and any(heading in long_summary for heading in section_headings[:3]):
                logger.info(f"✅ Section-based long summary generated: {word_count} words, {len(section_headings)} sections")
                return long_summary
            else:
                logger.warning(f"⚠️ LLM summary incomplete, using enhanced structured format")
                return self._build_enhanced_section_summary(raw_data, doc_type, fallback_date, physician_name, section_mapping)
                
        except Exception as e:
            logger.error(f"❌ Section-based long summary failed: {e}")
            section_mapping = self.prompt_manager.get_section_mapping(doc_type)
            return self._build_enhanced_section_summary(raw_data, doc_type, fallback_date, physician_name, section_mapping)


    def _format_section_content(self, content, section_name: str) -> str:
        """Format section content appropriately"""
        if isinstance(content, list):
            # For lists, create bullet points
            items = [f"• {str(item).strip()}" for item in content if item and str(item).strip()]
            return "\n".join(items) if items else "No specific information available."
        elif isinstance(content, dict):
            # For dictionaries, create key-value pairs
            items = [f"• {key}: {value}" for key, value in content.items() if value]
            return "\n".join(items) if items else "No specific information available."
        else:
            # For strings, ensure proper formatting
            content_str = str(content).strip()
            if not content_str or content_str in ['', 'None', 'Not specified']:
                return "No specific information available."
            
            # Add basic formatting for long text
            if len(content_str) > 120:
                # Simple paragraph formatting
                sentences = content_str.split('. ')
                formatted = []
                current_para = []
                for sentence in sentences:
                    if sentence.strip():
                        current_para.append(sentence.strip())
                        if len('. '.join(current_para)) > 80:
                            formatted.append('. '.join(current_para) + '.')
                            current_para = []
                if current_para:
                    formatted.append('. '.join(current_para) + '.')
                return '\n\n'.join(formatted) if len(formatted) > 1 else content_str
            return content_str
    def _create_error_response(self, doc_type: str, error_msg: str) -> Dict:
        """Create consistent error response"""
        return {
            "long_summary": f"Extraction failed for {doc_type}: {error_msg}",
            "short_summary": f"{doc_type} summary unavailable due to processing error."
        }