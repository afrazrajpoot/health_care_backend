from langchain_openai import ChatOpenAI
from langchain.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
import os

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-key"

# Initialize Tavily tool
tools = [TavilySearchResults(max_results=3)]

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create ReAct agent
agent = create_react_agent(llm, tools)

async def run_agent(query: str):
    """Run the LangGraph agent with the user's query"""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1]["content"]
