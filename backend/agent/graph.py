# backend/agent/graph.py
from typing import Literal
from langchain_core.messages import HumanMessage
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from backend.agent.state import AgentState
from backend.tools.arxiv_tool import search_arxiv
from backend.tools.web_search_tool import search_web
from backend.config import get_settings

# Initialize settings
settings = get_settings()

# Initialize tools
tools = [search_arxiv, search_web]

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    google_api_key=settings.google_api_key,
    convert_system_message_to_human=True  # Gemini doesn't support system messages
)


llm_with_tools = llm.bind_tools(tools)

# Create tool node
tool_node = ToolNode(tools)


def llm_node(state: AgentState) -> AgentState:
    """
    LLM Node: The agent's reasoning center.
    
    This node:
    1. Receives the current conversation state
    2. Analyzes user query and tool results
    3. Decides whether to call tools or provide final answer
    4. Increments iteration counter
    
    Args:
        state: Current agent state with messages and iterations
    
    Returns:
        Updated state with LLM response and incremented iterations
    """
    messages = state["messages"]
    
    # Invoke LLM with conversation history
    response = llm_with_tools.invoke(messages)
    
    # Increment iteration counter
    iterations = state.get("iterations", 0) + 1
    
    return {
        "messages": [response],
        "iterations": iterations
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: Determines next step in the agent flow.
    
    This function decides whether to:
    - Route to tools node (if LLM wants to call tools)
    - Route to END (if LLM has final answer or max iterations reached)
    
    Args:
        state: Current agent state
    
    Returns:
        "tools": Route to tool execution
        "end": Terminate the agent and return response
    """
    messages = state["messages"]
    last_message = messages[-1]
    iterations = state.get("iterations", 0)
    
    # Safety check: Prevent infinite loops
    if iterations > settings.max_iterations:
        print(f"⚠️  Max iterations ({settings.max_iterations}) reached. Ending.")
        return "end"
    
    # Check if LLM wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        print(f"🔧 LLM calling tools: {', '.join(tool_names)}")
        return "tools"
    
    # No tools needed - we have final answer
    print("✅ LLM has final answer")
    return "end"


def create_research_agent():
    """
    Creates and compiles the research assistant agent graph.
    
    Graph structure:
        START → LLM Node → (conditional) → [Tool Node → LLM Node] or END
    
    The agent follows the ReAct pattern:
        Reason (LLM) → Act (Tools) → Observe (Tool Results) → Repeat or Respond
    
    Returns:
        Compiled LangGraph agent ready to invoke
    """
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("llm")
    
    # Add conditional edge from LLM
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",  # Go to tools if needed
            "end": END         # End if done
        }
    )
    
    # Add edge from tools back to LLM (the crucial loop!)
    workflow.add_edge("tools", "llm")
    
    # Compile the graph
    return workflow.compile()


# Create the agent instance
research_agent = create_research_agent()


def run_research_query(query: str) -> dict:
    """
    Run a research query through the agent.
    
    Args:
        query: User's research question
    
    Returns:
        dict with:
            - response: Final agent response
            - iterations: Number of iterations taken
            - success: Whether query was successful
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {query}")
        print(f"{'='*60}\n")
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "iterations": 0
        }
        
        # Run the agent
        result = research_agent.invoke(initial_state)
        
        # Extract response
        final_message = result["messages"][-1]
        
        print(f"\n{'='*60}")
        print(f"✅ Query complete!")
        print(f"📊 Iterations: {result['iterations']}")
        print(f"{'='*60}\n")
        
        return {
            "response": final_message.content,
            "iterations": result["iterations"],
            "success": True
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        return {
            "response": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            "iterations": 0,
            "success": False
        }


# Test the agent (run this file directly to test)
if __name__ == "__main__":
    # Test query
    test_query = "Find me the latest developments in AI and based on the news give the best papers published on the topics"
    
    result = run_research_query(test_query)
    
    print("\nFINAL RESPONSE:")
    print("-" * 60)
    print(result["response"][0]['text'])
