"""
Long Summary Formatter

Transforms bullet-point summaries produced by the document summarizer into
well-structured, professionally formatted output with contextual headings,
paragraphs, and bullet points.

This is a FORMATTING task only — not summarization or generation.

CRITICAL RULES:
- Do NOT generate any new information
- Do NOT skip or omit any bullet point from the original summarizer output
- No hallucination or fabrication under any circumstances
- Preserve all original meaning and content, only improve structure and readability
"""
import logging
import json
import re
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from config.settings import CONFIG

logger = logging.getLogger("document_ai")


# ============================================================================
# PYDANTIC MODELS FOR FORMATTED LONG SUMMARY OUTPUT
# ============================================================================

class ContentBlock(BaseModel):
    """A content block that can be either a paragraph or bullet list"""
    type: Literal["paragraph", "bullets"] = Field(description="Type of content block")
    content: str = Field(default="", description="Paragraph text (for type='paragraph')")
    items: List[str] = Field(default_factory=list, description="Bullet items (for type='bullets')")


class FormattedSection(BaseModel):
    """A formatted section with contextual heading and mixed content"""
    heading: str = Field(description="Contextual section heading derived from content")
    content_blocks: List[ContentBlock] = Field(
        default_factory=list, 
        description="Mixed content: paragraphs and bullet lists in logical order"
    )


class FormattedLongSummary(BaseModel):
    """
    Complete formatted long summary with professional structure.
    
    Sections have contextual headings derived from the content,
    with a mix of paragraphs (for narrative) and bullets (for lists).
    """
    author_signed_by: str = Field(
        default="",
        description="Author or physician who signed/created the document, with credentials (e.g., 'John Smith, MD')"
    )
    sections: List[FormattedSection] = Field(
        default_factory=list,
        description="List of formatted sections with contextual headings"
    )
    content_type: Literal["medical", "administrative", "pr2", "imaging", "legal", "clinical", "unknown"] = Field(
        default="unknown",
        description="Type of content detected"
    )


# ============================================================================
# ENHANCED HELPER FUNCTIONS FOR PHYSICIAN-CENTRIC FORMATTING
# ============================================================================

def detect_content_type(bullet_summary: str, doc_type_hint: str = "") -> str:
    """Detect the content type from the bullet summary with document type hint."""
    summary_lower = bullet_summary.lower()
    doc_type_lower = doc_type_hint.lower() if doc_type_hint else ""
    
    # Use document type hint first if available
    if doc_type_hint:
        if any(x in doc_type_lower for x in ['pr-2', 'pr2', 'progress report']):
            return 'pr2'
        if any(x in doc_type_lower for x in ['qme', 'ame', 'ime', 'independent medical']):
            return 'medical'
        if any(x in doc_type_lower for x in ['mri', 'ct', 'x-ray', 'xray', 'imaging']):
            return 'imaging'
        if any(x in doc_type_lower for x in ['rfa', 'ur', 'utilization', 'authorization']):
            return 'administrative'
        if any(x in doc_type_lower for x in ['legal', 'attorney', 'deposition']):
            return 'legal'
    
    # Fallback to content detection
    # Progress notes / Clinical notes
    if any(x in summary_lower for x in ['progress note', 'session', 'telehealth', 'mental status', 'psychiatric']):
        return 'clinical'
    
    # PR-2 indicators
    if 'pr-2' in summary_lower or 'pr2' in summary_lower:
        return 'pr2'
    
    # Imaging indicators
    imaging_keywords = ['mri', 'ct scan', 'x-ray', 'xray', 'ultrasound', 'imaging', 'radiolog', 'findings:', 'impression:']
    if any(keyword in summary_lower for keyword in imaging_keywords):
        return 'imaging'
    
    # Legal indicators
    if any(x in summary_lower for x in ['attorney', 'legal', 'deposition', 'litigation', 'causation', 'apportionment']):
        return 'legal'
    
    # Administrative indicators
    admin_keywords = ['claim', 'adjuster', 'administrative', 'correspondence', 'authorization', 'rfa', 'utilization review']
    if any(keyword in summary_lower for keyword in admin_keywords):
        return 'administrative'
    
    # Medical indicators (default for clinical content)
    medical_keywords = ['diagnosis', 'treatment', 'medication', 'patient', 'physician', 'clinical', 'exam', 'findings']
    if any(keyword in summary_lower for keyword in medical_keywords):
        return 'medical'
    
    return 'unknown'


def extract_author_from_summary(bullet_summary: str) -> str:
    """
    Extract author information from bullet summary.
    Looks for patterns indicating author/signature information.
    """
    lines = [line.strip() for line in bullet_summary.split('\n') if line.strip()]
    
    author_patterns = [
        r'Signed by:\s*([^\n]+)',
        r'Author:\s*([^\n]+)',
        r'Physician:\s*([^\n]+)',
        r'Prepared by:\s*([^\n]+)',
        r'By:\s*([^\n,]+)',
        r'Provider:\s*([^\n]+)',
        r'Dr\.\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s*(?:MD|DO|Ph\.D|PhD|PsyD|NP|PA)',
    ]
    
    for line in lines:
        for pattern in author_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                # Clean up the author string
                author = re.sub(r'^Dr\.?\s*', '', author, flags=re.IGNORECASE)
                # Remove any trailing parentheses or extra info
                author = re.sub(r'\s*\([^)]*\)', '', author)
                return author
    
    return ""


def identify_physician_priority_sections(content_type: str) -> List[str]:
    """
    Identify priority sections based on content type for physician review.
    """
    section_priority_map = {
        'clinical': [
            "Provider and Patient Information",
            "Visit Overview", 
            "Presenting Complaints and Symptoms",
            "Mental Status Examination",
            "Safety and Risk Assessment",
            "Diagnoses",
            "Medications",
            "Treatment Plan",
            "Follow-Up Instructions"
        ],
        'medical': [
            "Report Overview",
            "Patient Information",
            "Clinical History",
            "Examination Findings",
            "Diagnoses",
            "Treatment Recommendations",
            "Follow-Up Plan"
        ],
        'imaging': [
            "Exam Information",
            "Clinical Indication",
            "Technique",
            "Findings",
            "Impression/Conclusion",
            "Recommendations"
        ],
        'pr2': [
            "Progress Update",
            "Treatment Response",
            "Current Symptoms",
            "Objective Findings",
            "Work Status",
            "Treatment Plan",
            "Next Follow-Up"
        ],
        'administrative': [
            "Document Overview",
            "Request Details",
            "Medical Necessity",
            "Supporting Documentation",
            "Decision and Rationale",
            "Next Steps"
        ],
        'legal': [
            "Document Purpose",
            "Case Overview",
            "Medical Review",
            "Causation Analysis",
            "Impairment Assessment",
            "Conclusions",
            "Recommendations"
        ]
    }
    
    return section_priority_map.get(content_type, [
        "Document Summary",
        "Key Information",
        "Findings and Recommendations"
    ])


def create_fallback_formatted_summary(bullet_summary: str, error_message: str = "", doc_type: str = "") -> FormattedLongSummary:
    """
    Create a fallback formatted summary when LLM processing fails.
    Intelligently groups content into sections based on context and document type.
    """
    lines = [line.strip() for line in bullet_summary.split('\n') if line.strip()]
    
    # Extract author first
    author = extract_author_from_summary(bullet_summary)
    
    # Determine content type
    content_type = detect_content_type(bullet_summary, doc_type)
    
    # Get priority sections for this content type
    priority_sections = identify_physician_priority_sections(content_type)
    
    # Try to identify natural section breaks with intelligent grouping
    sections = []
    current_section_heading = priority_sections[0] if priority_sections else "Document Summary"
    current_content = []
    
    for line in lines:
        # Skip if line is just author info we already extracted
        if author and author.lower() in line.lower():
            continue
            
        # Check if this looks like a header (emoji headers, ALL CAPS, ends with colon)
        is_header = False
        header_text = ""
        
        # Pattern 1: Emoji headers
        if re.match(r'^[📋🎯👤💬🔬🏥💊📈✅💼📅🚨📞📄⚖️👥✉️]+\s*(.+)', line):
            is_header = True
            header_text = re.sub(r'^[📋🎯👤💬🔬🏥💊📈✅💼📅🚨📞📄⚖️👥✉️]+\s*', '', line).strip()
        # Pattern 2: Section headers in ALL CAPS
        elif re.match(r'^[A-Z][A-Z\s&]+\s*$', line) and 10 < len(line) < 60:
            is_header = True
            header_text = line.strip()
        # Pattern 3: Ends with colon (common in bullet summaries)
        elif line.endswith(':') and len(line) < 60 and not line.startswith('•'):
            is_header = True
            header_text = line.rstrip(':').strip()
        # Pattern 4: Common medical section headers
        elif any(header in line.lower() for header in ['findings:', 'impression:', 'diagnosis:', 'treatment:', 'medications:', 'recommendations:']):
            is_header = True
            header_text = line.rstrip(':').strip().title()
        # Pattern 5: Skip divider lines
        elif re.match(r'^[-=]{3,}$', line):
            continue
        
        if is_header and header_text:
            # Save previous section
            if current_content:
                sections.append(FormattedSection(
                    heading=current_section_heading,
                    content_blocks=[ContentBlock(
                        type="bullets",
                        items=[re.sub(r'^[•\-\*]\s*', '', c).strip() for c in current_content if c.strip()]
                    )]
                ))
            
            # Try to match with priority sections
            matched_priority = None
            for priority in priority_sections:
                if priority.lower() in header_text.lower() or header_text.lower() in priority.lower():
                    matched_priority = priority
                    break
            
            current_section_heading = matched_priority or header_text
            current_content = []
        else:
            if line and not line.startswith('===') and not line.startswith('---'):
                current_content.append(line)
    
    # Don't forget the last section
    if current_content:
        sections.append(FormattedSection(
            heading=current_section_heading,
            content_blocks=[ContentBlock(
                type="bullets",
                items=[re.sub(r'^[•\-\*]\s*', '', c).strip() for c in current_content if c.strip()]
            )]
        ))
    
    # If we have no sections but have content, create one
    if not sections and lines:
        sections.append(FormattedSection(
            heading=current_section_heading,
            content_blocks=[ContentBlock(
                type="bullets",
                items=[re.sub(r'^[•\-\*]\s*', '', line).strip() for line in lines if line.strip()]
            )]
        ))
    
    return FormattedLongSummary(
        author_signed_by=author,
        sections=sections,
        content_type=content_type
    )


# ============================================================================
# ENHANCED MAIN FORMATTER FUNCTION WITH PHYSICIAN-CENTRIC FOCUS
# ============================================================================

def format_bullet_summary_to_json(
    bullet_summary: str,
    llm: Optional[AzureChatOpenAI] = None,
    document_type: str = ""
) -> Dict[str, Any]:
    """
    Transform a bullet-point summary into a well-structured, professionally formatted output.
    
    Creates contextual section headings based on content, uses paragraphs for narrative
    information, and bullet points only where lists make sense.
    
    CRITICAL RULES:
    - Do NOT generate any new information
    - Do NOT skip or omit any content from the original summarizer output
    - No hallucination or fabrication under any circumstances
    - Preserve all original meaning and content, only improve structure and readability
    
    Args:
        bullet_summary: The bullet-point summary from the document summarizer
        llm: Optional Azure OpenAI LLM instance (will create one if not provided)
        document_type: Optional document type hint
        
    Returns:
        Dict containing the formatted long summary as structured JSON
    """
    logger.info(f"📝 Starting Long Summary Formatter for document type: {document_type}")
    
    if not bullet_summary or not bullet_summary.strip():
        logger.warning("⚠️ Empty bullet summary provided")
        return FormattedLongSummary().model_dump()
    
    # Create LLM if not provided
    if llm is None:
        try:
            llm = AzureChatOpenAI(
                azure_deployment=CONFIG.get("azure_openai_deployment"),
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.0,  # Zero temperature for pure formatting
                max_tokens=12000,
                timeout=180,
                request_timeout=180,
            )
        except Exception as e:
            logger.error(f"❌ Failed to create LLM: {e}")
            return create_fallback_formatted_summary(bullet_summary, str(e), document_type).model_dump()
    
    # Create Pydantic parser for structured output
    pydantic_parser = PydanticOutputParser(pydantic_object=FormattedLongSummary)
    
    # Determine content type for section guidance
    content_type = detect_content_type(bullet_summary, document_type)
    priority_sections = identify_physician_priority_sections(content_type)
    
    # Enhanced system prompt with physician-centric focus
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a PHYSICIAN-OPTIMIZED DOCUMENT FORMATTER specializing in medical and legal documentation.

Your SOLE TASK is to transform raw bullet-point medical summaries into beautifully structured,
professional documents optimized for PHYSICIAN REVIEW.

🎯 **PHYSICIAN WORKFLOW PRIORITY:**
Structure information so physicians can quickly find:
1. Critical clinical findings
2. Treatment recommendations
3. Medication changes
4. Work status and restrictions
5. Follow-up instructions
6. Urgent/emergent issues

🔴 **ABSOLUTE RULES — ZERO TOLERANCE FOR VIOLATION:**

1. **NO NEW INFORMATION GENERATION**
   - Every word MUST come directly from the input
   - No inferences, assumptions, or additions
   - If it's not in the input, DO NOT include it

2. **NO CONTENT OMISSION**
   - Every single bullet point MUST appear in your output
   - Missing clinical information = CRITICAL FAILURE

3. **NO HALLUCINATION**
   - Do not invent dates, names, numbers, or clinical facts
   - Do not "fill in" missing information

4. **PRESERVE CLINICAL DETAILS**
   - Keep exact medical terminology
   - Preserve measurements, dosages, and frequencies
   - Maintain temporal relationships
   - Keep severity assessments and qualifiers

📋 **PHYSICIAN-CENTRIC FORMATTING REQUIREMENTS:**

**DOCUMENT TYPE: {document_type}**
**CONTENT TYPE: {content_type}**

**PRIORITY SECTIONS FOR THIS DOCUMENT TYPE:**
{priority_sections_list}

**1. EXTRACT AUTHOR INFORMATION FIRST**
   - Look for: "Signed by:", "Author:", "Physician:", "Prepared by:", "Provider:"
   - Include credentials: "John Smith, MD", "Jane Doe, Ph.D."
   - Remove "Dr." prefix if present
   - Place in "author_signed_by" field

**2. CREATE CONTEXTUAL SECTION HEADINGS**
   - Derive headings from CONTENT, not generic templates
   - Match to priority sections when content aligns
   - Examples of physician-friendly headings:
     * For Clinical Notes: "Provider and Patient Information", "Presenting Complaints", "Mental Status Exam", "Safety Assessment", "Treatment Plan"
     * For Imaging: "Exam Details", "Clinical Indication", "Technical Factors", "Findings with Measurements", "Impression", "Recommendations"
     * For Progress Reports: "Progress Update", "Treatment Response", "Current Symptoms", "Objective Findings", "Work Status Update"
     * For Administrative: "Request Overview", "Medical Necessity", "Supporting Evidence", "Decision and Rationale"

**3. USE PARAGRAPHS FOR NARRATIVE CONTENT**
   - Opening/introductory information → PARAGRAPH
   - Context-setting statements → PARAGRAPH
   - Related sentences about same topic → COMBINE into flowing paragraph
   
   **GOOD Example (paragraph for narrative):**
   "Dr. Michael Johnson, Orthopedic Surgeon at City Medical Center, evaluated John Doe (DOB: 05/15/1978) 
   for persistent low back pain following a work-related injury on 2024-01-15. The patient reported 
   gradual onset of symptoms over the past 6 weeks with radiation to the right lower extremity."

   **BAD Example (unnecessary bullets for narrative):**
   • Dr. Michael Johnson evaluated John Doe
   • Patient has low back pain
   • Symptoms started 6 weeks ago

**4. USE BULLETS ONLY FOR TRUE LISTS**
   - Lists of diagnoses with ICD codes
   - Medication lists with dosages
   - Specific exam findings
   - Treatment recommendations
   - Action items / follow-up tasks
   
   **GOOD Example (bullets for list):**
   The patient's current medications include:
   • Hydrocodone/Acetaminophen 5/325mg, 1-2 tablets every 6 hours as needed for pain
   • Gabapentin 300mg, three times daily
   • Cyclobenzaprine 10mg, at bedtime as needed for muscle spasms

**5. GROUP RELATED CLINICAL INFORMATION**
   - Keep all medication information together
   - Group exam findings by body system
   - Keep treatment plan components together
   - Separate patient-reported vs provider-documented

**6. HIGHLIGHT CRITICAL INFORMATION**
   - Flag abnormal lab values
   - Highlight urgent recommendations
   - Emphasize work restrictions
   - Note medication changes

**7. PRESERVE DECISION TERMS EXACTLY**
   - "authorized", "approved", "denied", "deferred"
   - "recommended", "not recommended", "contraindicated"
   - "at MMI", "not at MMI", "permanent and stationary"

📊 **OUTPUT STRUCTURE REQUIREMENTS:**

{{
  "author_signed_by": "Author Name, MD (extract from document)",
  "sections": [
    {{
      "heading": "Contextual Heading Based on Content",
      "content_blocks": [
        {{
          "type": "paragraph",
          "content": "Narrative text that flows naturally...",
          "items": []
        }},
        {{
          "type": "bullets",
          "content": "",
          "items": ["Item 1", "Item 2", "Item 3"]
        }}
      ]
    }}
  ],
  "content_type": "{content_type}"
}}

🚨 **VALIDATION CHECKLIST (MUST PASS ALL):**
□ All input content preserved
□ No new information added
□ Author information extracted if present
□ Paragraphs used for narrative content
□ Bullets used only for true lists
□ Clinical details preserved exactly
□ Section headings are contextual
□ Critical information highlighted
□ Decision terms preserved exactly

{format_instructions}

Output valid JSON only. No markdown code blocks. No explanatory text.
""")

    user_prompt = HumanMessagePromptTemplate.from_template("""
**DOCUMENT TYPE:** {document_type}
**CONTENT TYPE:** {content_type}

**PRIORITY SECTIONS FOR PHYSICIAN REVIEW:**
{priority_sections_list}

**RAW BULLET-POINT SUMMARY TO FORMAT:**
---
{bullet_summary}
---

**TASK:** Transform this into a professionally structured document OPTIMIZED FOR PHYSICIAN REVIEW.

**CRITICAL REQUIREMENTS:**

1. **EXTRACT AUTHOR INFORMATION:** Look for and include author/signature information

2. **USE PRIORITY SECTIONS:** Structure content using the priority sections above when content matches

3. **PARAGRAPHS FOR NARRATIVE:**
   - Combine related bullet points into flowing paragraphs
   - Use paragraphs for: introductions, context, summaries, explanations
   - Example: Convert multiple related symptoms into a symptom description paragraph

4. **BULLETS FOR LISTS:**
   - Use bullets only for: medication lists, diagnosis lists, exam findings, action items
   - Keep dosages, frequencies, and measurements in bullet format

5. **PRESERVE ALL CONTENT:**
   - Every bullet point MUST appear somewhere in the output
   - No information can be omitted
   - Clinical details must be preserved exactly

6. **PHYSICIAN-FRIENDLY ORGANIZATION:**
   - Group related clinical findings
   - Keep medications together
   - Separate subjective vs objective
   - Highlight urgent/important information

7. **MAINTAIN CLINICAL ACCURACY:**
   - Preserve exact medical terminology
   - Keep measurements and units
   - Maintain temporal relationships
   - Preserve severity assessments

**OUTPUT:** Valid JSON matching the required structure.
""")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    
    # Retry mechanism
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Formatting attempt {attempt + 1}/{max_retries}")
            
            chain = chat_prompt | llm
            response = chain.invoke({
                "bullet_summary": bullet_summary,
                "document_type": document_type,
                "content_type": content_type,
                "priority_sections_list": "\n".join([f"- {section}" for section in priority_sections]),
                "format_instructions": pydantic_parser.get_format_instructions()
            })
            
            # Extract response content
            response_content = response.content.strip()
            
            # Clean JSON extraction
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                response_content = response_content[start_idx:end_idx + 1]
            
            # Clean control characters
            response_content = "".join(
                ch for ch in response_content 
                if ch >= ' ' or ch in '\n\r\t'
            )
            
            # Parse JSON
            result = json.loads(response_content, strict=False)
            
            # Validate structure
            formatted_summary = FormattedLongSummary(**result)
            
            # Post-process validation
            formatted_summary = validate_and_enhance_formatted_summary(formatted_summary, bullet_summary)
            
            logger.info(f"✅ Successfully formatted summary with {len(formatted_summary.sections)} sections")
            return formatted_summary.model_dump()
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {e}"
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
            if attempt < max_retries - 1:
                logger.info("🔄 Retrying with simplified parsing...")
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
            if attempt < max_retries - 1:
                logger.info("🔄 Retrying...")
    
    # All retries failed - use enhanced fallback
    logger.error(f"❌ All formatting attempts failed. Using enhanced fallback. Last error: {last_error}")
    return create_fallback_formatted_summary(bullet_summary, last_error or "Unknown error", document_type).model_dump()


def validate_and_enhance_formatted_summary(
    formatted_summary: FormattedLongSummary,
    original_bullet_summary: str
) -> FormattedLongSummary:
    """
    Validate that the formatted summary preserves all content and enhance for physician review.
    """
    # Count original bullet points
    original_lines = [line.strip() for line in original_bullet_summary.split('\n') if line.strip()]
    original_bullets = [line for line in original_lines if line.startswith(('•', '-', '*'))]
    
    # Count content in formatted summary
    formatted_content_count = 0
    for section in formatted_summary.sections:
        for block in section.content_blocks:
            if block.type == "paragraph":
                formatted_content_count += len(block.content.split())
            elif block.type == "bullets":
                formatted_content_count += len(block.items)
    
    # If content seems significantly reduced, log warning
    if formatted_content_count < len(original_bullets) * 0.5:  # Less than 50% of original
        logger.warning(f"⚠️ Possible content loss: {formatted_content_count} items vs {len(original_bullets)} original bullets")
    
    # Ensure author is cleaned
    if formatted_summary.author_signed_by:
        author = formatted_summary.author_signed_by
        author = re.sub(r'^Dr\.?\s*', '', author, flags=re.IGNORECASE)
        formatted_summary.author_signed_by = author.strip()
    
    # Ensure sections have content
    valid_sections = []
    for section in formatted_summary.sections:
        # Check if section has any content
        has_content = False
        for block in section.content_blocks:
            if block.type == "paragraph" and block.content.strip():
                has_content = True
                break
            elif block.type == "bullets" and block.items:
                has_content = True
                break
        
        if has_content:
            valid_sections.append(section)
        else:
            logger.warning(f"⚠️ Removing empty section: {section.heading}")
    
    formatted_summary.sections = valid_sections
    
    return formatted_summary

def format_long_summary_to_text(formatted_summary: Dict[str, Any]) -> str:
    """
    Convert a formatted JSON summary into beautifully formatted text.
    
    Creates professional output with:
    - Author/Signed By information at the top
    - Clear section headings
    - Flowing paragraphs for narrative content
    - Bullet points only for true lists
    
    Args:
        formatted_summary: The JSON dict from format_bullet_summary_to_json
        
    Returns:
        Professionally formatted text string
    """
    lines = []
    
    # Add Author/Signed By at the top if present
    author_signed_by = formatted_summary.get('author_signed_by', '')
    if author_signed_by:
        lines.append(f"Author/Signed By: {author_signed_by}")
        lines.append("")
    
    # Process each section
    for section in formatted_summary.get('sections', []):
        heading = section.get('heading', '')
        
        if heading:
            # Add section heading with visual separator
            lines.append("")
            lines.append(heading)
            lines.append("-" * len(heading))
            lines.append("")
        
        # Process content blocks
        for block in section.get('content_blocks', []):
            block_type = block.get('type', 'paragraph')
            
            if block_type == 'paragraph':
                content = block.get('content', '')
                if content:
                    lines.append(content)
                    lines.append("")
            
            elif block_type == 'bullets':
                items = block.get('items', [])
                if items:
                    # Add a blank line before bullet list if not already
                    if lines and lines[-1] != "":
                        lines.append("")
                    for item in items:
                        if item:
                            lines.append(f"• {item}")
                    lines.append("")
    
    # Clean up multiple blank lines
    result = '\n'.join(lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()