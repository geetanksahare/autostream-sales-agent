"""
AutoStream Agent — Main CLI Entry Point
Run this file to start an interactive conversation with the AutoStream agent.

Usage:
    python main.py

Environment Variables:
    GOOGLE_API_KEY   — Required. Your Google API key.
"""

import os
import sys

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import HumanMessage

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import build_agent, get_initial_state

BANNER = """
╔══════════════════════════════════════════════════════╗
║       🎬 AutoStream AI Sales Assistant               ║
║        Powered by Google Gemini + LangGraph          ║
╠══════════════════════════════════════════════════════╣
║  Type your message and press Enter to chat.          ║
║  Type 'quit' or 'exit' to end the conversation.      ║
╚══════════════════════════════════════════════════════╝
"""


def print_agent_response(content: str) -> None:
    """Pretty-print the agent's response."""
    print(f"\n🤖 Alex (AutoStream):\n{content}\n")
    print("─" * 54)


def run_chat() -> None:
    """Main interactive chat loop."""
    print(BANNER)

    # Build agent graph
    try:
        agent = build_agent()
    except EnvironmentError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Initialize state
    state = get_initial_state()

    print("💬 You: ", end="", flush=True)

    while True:
        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Thanks for chatting! Goodbye.")
            break

        if not user_input:
            print("💬 You: ", end="", flush=True)
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\n🤖 Alex: Thanks for stopping by! Feel free to come back anytime. 👋\n")
            break

        # Append user message to state
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # Run agent graph
        try:
            result = agent.invoke(state)
            state = result
        except Exception as e:
            print(f"\n❌ Agent error: {e}\n")
            print("💬 You: ", end="", flush=True)
            continue

        # Extract and print the latest AI message
        from langchain_core.messages import AIMessage
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print_agent_response(ai_messages[-1].content)

        # If lead just captured, show a small separator
        if state.get("lead_captured"):
            print("✅ Lead registration complete. Continuing conversation...\n")

        print("💬 You: ", end="", flush=True)


if __name__ == "__main__":
    run_chat()