"""
RAG Pipeline for AutoStream Agent
Handles knowledge base loading and retrieval using simple semantic matching.
"""

import json
import os
from typing import Optional


def load_knowledge_base(path: str = None) -> dict:
    """Load the AutoStream knowledge base from JSON file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "autostream_kb.json")
    with open(path, "r") as f:
        return json.load(f)


def format_knowledge_as_context(kb: dict) -> str:
    """
    Convert the full knowledge base into a readable context string
    for the LLM to reference during RAG-based answering.
    """
    lines = []

    # Company info
    company = kb.get("company", {})
    lines.append(f"=== COMPANY OVERVIEW ===")
    lines.append(f"Name: {company.get('name')}")
    lines.append(f"Tagline: {company.get('tagline')}")
    lines.append(f"Description: {company.get('description')}")
    lines.append("")

    # Pricing Plans
    lines.append("=== PRICING PLANS ===")
    for plan in kb.get("pricing_plans", []):
        lines.append(f"\n{plan['name']} — {plan['price_label']}")
        lines.append(f"Best for: {plan['best_for']}")
        lines.append("Features:")
        for feature in plan["features"]:
            lines.append(f"  • {feature}")

    lines.append("")

    # Policies
    lines.append("=== COMPANY POLICIES ===")
    for policy in kb.get("company_policies", []):
        lines.append(f"\n{policy['policy']}:")
        lines.append(f"  {policy['details']}")

    lines.append("")

    # FAQs
    lines.append("=== FAQs ===")
    for faq in kb.get("faqs", []):
        lines.append(f"\nQ: {faq['question']}")
        lines.append(f"A: {faq['answer']}")

    return "\n".join(lines)


# Singleton load
_kb_cache: Optional[str] = None


def get_rag_context() -> str:
    """Returns the formatted RAG context string (cached after first load)."""
    global _kb_cache
    if _kb_cache is None:
        kb = load_knowledge_base()
        _kb_cache = format_knowledge_as_context(kb)
    return _kb_cache


if __name__ == "__main__":
    print(get_rag_context())
