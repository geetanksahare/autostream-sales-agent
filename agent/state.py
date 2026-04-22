"""
State Management for AutoStream LangGraph Agent
Defines the AgentState TypedDict and all possible lead collection stages.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from enum import Enum

from langgraph.graph.message import add_messages


class LeadStage(str, Enum):
    """Tracks which stage of lead collection we're at."""
    NONE = "none"               # No lead flow started
    ASKING_NAME = "asking_name"         # Asked for name, waiting
    ASKING_EMAIL = "asking_email"        # Have name, asked for email
    ASKING_PLATFORM = "asking_platform"  # Have name+email, asked for platform
    COMPLETE = "complete"               # All collected, tool triggered


class AgentState(TypedDict):
    """
    Full conversation state for the AutoStream agent.
    Retained across all conversation turns via LangGraph.
    """
    # Full conversation history (LangGraph reducer appends automatically)
    messages: Annotated[list, add_messages]

    # Detected intent for the latest user message
    current_intent: Optional[str]

    # Lead collection stage tracker
    lead_stage: LeadStage

    # Collected lead fields
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Whether lead has been successfully captured
    lead_captured: bool

    # Turn counter for debugging / evaluation
    turn_count: int
