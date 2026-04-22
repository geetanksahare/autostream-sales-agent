"""
AutoStream LangGraph Agent — Core
Implements the full conversational AI agent using LangGraph StateGraph.

Architecture:
  User Message → classify_intent → route_response →
    ├─ greeting_node
    ├─ rag_response_node
    └─ lead_collection_node → [check_lead_complete] → lead_capture_node
"""

import os
import re
import sys

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState, LeadStage
from agent.prompts import build_system_prompt
from tools.lead_capture import mock_lead_capture
from utils.intent_classifier import Intent, keyword_classify

# ─────────────────────────────────────────────
# LLM Initialization
# ─────────────────────────────────────────────

def get_llm():
    """Initialize Google Gemini LLM."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Please set it in your environment or .env file."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────
# Helper: Extract lead field from user message
# ─────────────────────────────────────────────

def extract_email(text: str) -> str | None:
    """Extract email address from user message."""
    match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return match.group(0) if match else None


# ─────────────────────────────────────────────
# Node: Intent Classification
# ─────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> AgentState:
    """
    Classifies the latest user message intent.
    Uses keyword classifier as fast pre-filter; LLM handles nuanced cases.
    """
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {**state, "current_intent": Intent.CASUAL_GREETING}

    user_text = last_human.content.strip()

    # Keyword-based classification (fast path)
    intent = keyword_classify(user_text)

    # If we're mid-lead-collection, keep the lead flow going
    lead_stage = state.get("lead_stage", LeadStage.NONE)
    if lead_stage not in (LeadStage.NONE, LeadStage.COMPLETE):
        intent = Intent.HIGH_INTENT_LEAD

    return {**state, "current_intent": intent, "turn_count": state.get("turn_count", 0) + 1}


# ─────────────────────────────────────────────
# Node: RAG-powered LLM Response
# ─────────────────────────────────────────────

def rag_response_node(state: AgentState) -> AgentState:
    """
    Generates an LLM response grounded in the RAG knowledge base.
    Used for greetings and product/pricing inquiries.
    """
    llm = get_llm()
    system_prompt = build_system_prompt()

    # Gemini doesn't support SystemMessage directly — merge into first human message
    history = list(state["messages"])
    if history and isinstance(history[0], HumanMessage):
        first_msg = history[0]
        history[0] = HumanMessage(content=f"{system_prompt}\n\n{first_msg.content}")
        chat_messages = history
    else:
        chat_messages = [HumanMessage(content=system_prompt)] + history

    response = llm.invoke(chat_messages)

    updated_messages = list(state["messages"]) + [AIMessage(content=response.content)]
    return {**state, "messages": updated_messages}


# ─────────────────────────────────────────────
# Node: Lead Collection (Step-by-step)
# ─────────────────────────────────────────────

def lead_collection_node(state: AgentState) -> AgentState:
    """
    Manages step-by-step lead data collection.
    Extracts one field at a time from the conversation.
    """
    messages = state["messages"]
    lead_stage = state.get("lead_stage", LeadStage.NONE)

    # Get the latest user message
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content.strip() if last_human else ""

    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")

    ai_response = ""

    # ── Stage: NONE → just detected high intent, start collection ──
    if lead_stage == LeadStage.NONE:
        ai_response = (
            "🚀 Awesome! I'd love to get you set up with AutoStream. "
            "Let's start — could you share your **full name**?"
        )
        lead_stage = LeadStage.ASKING_NAME

    # ── Stage: Waiting for Name ──
    elif lead_stage == LeadStage.ASKING_NAME:
        if len(user_text.split()) >= 1 and len(user_text) > 1:
            lead_name = user_text.title()
            ai_response = (
                f"Nice to meet you, {lead_name}! 😊 "
                f"Now, could you share your **email address**?"
            )
            lead_stage = LeadStage.ASKING_EMAIL
        else:
            ai_response = "I didn't catch that — could you share your **full name** please?"

    # ── Stage: Waiting for Email ──
    elif lead_stage == LeadStage.ASKING_EMAIL:
        email = extract_email(user_text)
        if email:
            lead_email = email
            ai_response = (
                f"Got it! ✅ And finally, which **creator platform** are you primarily on? "
                f"(e.g., YouTube, Instagram, TikTok, Twitter/X, etc.)"
            )
            lead_stage = LeadStage.ASKING_PLATFORM
        else:
            ai_response = (
                "Hmm, that doesn't look like a valid email address. "
                "Could you double-check and share it again?"
            )

    # ── Stage: Waiting for Platform ──
    elif lead_stage == LeadStage.ASKING_PLATFORM:
        if len(user_text) > 1:
            lead_platform = user_text.strip()
            ai_response = (
                f"Perfect! Let me confirm your details:\n\n"
                f"- **Name**: {lead_name}\n"
                f"- **Email**: {lead_email}\n"
                f"- **Platform**: {lead_platform}\n\n"
                f"Registering you now... 🎬"
            )
            lead_stage = LeadStage.COMPLETE
        else:
            ai_response = (
                "Could you let me know which platform you create content on? "
                "(e.g., YouTube, Instagram, TikTok)"
            )

    updated_messages = list(messages) + [AIMessage(content=ai_response)]

    return {
        **state,
        "messages": updated_messages,
        "lead_stage": lead_stage,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
    }


# ─────────────────────────────────────────────
# Node: Lead Capture Tool Execution
# ─────────────────────────────────────────────

def lead_capture_node(state: AgentState) -> AgentState:
    """
    Fires the mock_lead_capture tool once all 3 fields are confirmed.
    Only triggered when lead_stage == COMPLETE.
    """
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"],
    )

    if result["success"]:
        confirmation = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"Your lead has been successfully registered (ID: `{result['lead_id']}`).\n\n"
            f"Our team will reach out to **{state['lead_email']}** shortly to get you started on AutoStream. "
            f"Welcome aboard! 🚀"
        )
    else:
        confirmation = (
            f"Hmm, something went wrong registering your details: {result.get('error')}. "
            "Please try again or contact support@autostream.io."
        )

    updated_messages = list(state["messages"]) + [AIMessage(content=confirmation)]

    return {
        **state,
        "messages": updated_messages,
        "lead_captured": result["success"],
    }


# ─────────────────────────────────────────────
# Router: Decide next node after intent classification
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> Literal["rag_response", "lead_collection"]:
    """Routes to lead_collection if high intent detected, else RAG response."""
    intent = state.get("current_intent", Intent.PRODUCT_INQUIRY)
    lead_stage = state.get("lead_stage", LeadStage.NONE)

    # Always use RAG if lead is already captured
    if state.get("lead_captured", False):
        return "rag_response"

    if intent == Intent.HIGH_INTENT_LEAD or lead_stage not in (LeadStage.NONE, LeadStage.COMPLETE):
        return "lead_collection"

    return "rag_response"


def route_after_lead_collection(state: AgentState) -> Literal["lead_capture", "end"]:
    """After collection node, check if we should fire the capture tool."""
    if state.get("lead_stage") == LeadStage.COMPLETE:
        return "lead_capture"
    return "end"


# ─────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────

def build_agent() -> StateGraph:
    """
    Assembles and compiles the LangGraph StateGraph for the AutoStream agent.

    Graph Flow:
        START → classify_intent
                    ↓
            [route_after_intent]
           /                    \\
    rag_response         lead_collection
         |                      |
        END           [route_after_lead_collection]
                       /                  \\
               lead_capture              END
                    |
                   END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("rag_response", rag_response_node)
    graph.add_node("lead_collection", lead_collection_node)
    graph.add_node("lead_capture", lead_capture_node)

    # Add edges
    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "rag_response": "rag_response",
            "lead_collection": "lead_collection",
        }
    )

    graph.add_edge("rag_response", END)

    graph.add_conditional_edges(
        "lead_collection",
        route_after_lead_collection,
        {
            "lead_capture": "lead_capture",
            "end": END,
        }
    )

    graph.add_edge("lead_capture", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Convenience: get initial state
# ─────────────────────────────────────────────

def get_initial_state() -> AgentState:
    return {
        "messages": [],
        "current_intent": None,
        "lead_stage": LeadStage.NONE,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "turn_count": 0,
    }