from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State of the Research Agent

    Attributes:
        messages: Conversation history with automatic message aggregation through reducers (add_messages)
        iterations: Number of tools call iterations to prevent infinite loops
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int