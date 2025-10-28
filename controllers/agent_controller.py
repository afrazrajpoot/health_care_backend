from fastapi import APIRouter, HTTPException, Body
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict
from config.settings import CONFIG
import os

# 🔑 Load Tavily key
os.environ["TAVILY_API_KEY"] = CONFIG["tavily_api_key"]

router = APIRouter()

# 🧠 Azure OpenAI setup
llm = AzureChatOpenAI(
    azure_endpoint=CONFIG["azure_openai_endpoint"],
    api_key=CONFIG["azure_openai_api_key"],
    deployment_name=CONFIG["azure_openai_deployment"],
    api_version=CONFIG["azure_openai_api_version"],
    temperature=0.1,  # Low temperature for consistent, factual output
)

# Define the allowed domains for targeted searches
ALLOWED_DOMAINS = [
    "www.aetna.com",
    "www.uhcprovider.com",
    "www.blueshieldca.com"
]

# State definition for the workflow
class AgentState(TypedDict):
    input: dict
    search_results: str
    rebuttal: str

async def search_node(state: AgentState) -> AgentState:
    """
    Node 1: Perform targeted web search using Tavily on specific insurance policy domains.
    Constructs a query from input fields and retrieves summarized results.
    """
    input_data = state["input"]
    
    # Construct a precise search query based on input
    body_part = input_data.get("body_part", "")
    modality = input_data.get("modality", "")
    diagnosis = input_data.get("diagnosis", "")
    
    if not all([body_part, modality, diagnosis]):
        raise ValueError("Missing required input fields: body_part, modality, diagnosis")
    
    search_query = f"{modality} guidelines for {diagnosis} {body_part}"
    
    # Initialize Tavily tool with domain restrictions
    tool = TavilySearchResults(
        max_results=5,
        search_kwargs={
            "include_domains": ALLOWED_DOMAINS,
            "search_depth": "advanced",  # For more detailed summaries
            "include_answer": True,  # Include direct answer if available
        }
    )
    
    # Execute async search
    search_output = await tool.ainvoke({"query": search_query})
    
    # Store the raw search results string (summarized by Tavily)
    state["search_results"] = search_output
    return state

async def generate_rebuttal_node(state: AgentState) -> AgentState:
    """
    Node 2: Generate original rebuttal using LLM.
    Feeds input + search results into a structured prompt.
    Ensures output is transformative: cites sources briefly, no full-text copying.
    """
    input_data = state["input"]
    search_results = state["search_results"]
    
    # Extract input fields
    guideline_source = input_data.get("guideline_source", "insurance policies (Aetna, UHC, Blue Shield CA)")
    reason_for_denial = input_data.get("reason_for_denial", "")
    previous_response = input_data.get("previous_response", "")
    body_part = input_data.get("body_part", "")
    modality = input_data.get("modality", "")
    diagnosis = input_data.get("diagnosis", "")
    
    # Structured prompt based on sample framework
    prompt = f"""Based on {guideline_source} and/or current clinical guidelines from the provided search results, write a concise justification for using {modality} to treat {diagnosis} in the {body_part}.

Key details from request:
- Reason for denial: {reason_for_denial}
- Previous response/evidence: {previous_response}

Relevant guideline summaries (do not copy verbatim; use to inform reasoning):
{search_results}

Instructions:
- Generate an original, transformative response. Do not redistribute full text from sources.
- Include references to physical exam findings, treatment duration, response to prior therapies, and impact on ADLs where applicable and supported by summaries.
- Cite sources briefly (e.g., "Per Aetna Clinical Policy Bulletin, 2024" or "Per UHC Policies, Section X") without quoting blocks of text.
- Format as a professional rebuttal letter with markdown headings for structure:
  - ## Clinical Summary (brief overview of diagnosis and requested modality)
  - ## Justification for Medical Necessity (explain why it's needed, addressing denial reason)
  - ## Supporting Evidence from Guidelines (key points from searches with citations)
  - ## Recommended Authorization (conclusion and request)
- Use markdown for headings (e.g., ## Heading) and keep the entire response under 300 words for brevity."""

    # Invoke LLM asynchronously
    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    
    # Store the rebuttal
    state["rebuttal"] = response.content.strip()
    return state

# Build the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("generate", generate_rebuttal_node)

# Define edges: sequential flow (search -> generate -> end)
workflow.set_entry_point("search")
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

@router.post("/rebuttal")
async def generate_rebuttal(data: dict = Body(...)):
    """
    Endpoint to generate a rebuttal justification using LangGraph workflow.
    Input format:
    {
      "body_part": "shoulder",
      "modality": "acupuncture",
      "diagnosis": "chronic shoulder pain",
      "guideline_source": "MTUS",
      "reason_for_denial": "lack of physical exam findings",
      "previous_response": "Patient reports decreased pain and improved ADLs"
    }
    """
    try:
        # Validate required fields
        required_fields = ["body_part", "modality", "diagnosis"]
        if not all(field in data for field in required_fields):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(required_fields)}"
            )

        # Run the workflow
        result = await app.ainvoke({"input": data})

        return {"rebuttal": result["rebuttal"]}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")