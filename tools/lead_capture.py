"""
Tool Execution: Lead Capture
Simulates a CRM API call that registers a qualified lead for AutoStream.
"""

import re
from datetime import datetime
from typing import Optional


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email.strip()))


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock CRM API function that captures a qualified lead.

    Args:
        name (str): Full name of the lead.
        email (str): Email address of the lead.
        platform (str): Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict: Result containing success status and lead ID.
    """
    # Validate inputs
    if not name or not name.strip():
        return {"success": False, "error": "Name cannot be empty."}

    if not email or not validate_email(email):
        return {"success": False, "error": f"Invalid email address: {email}"}

    if not platform or not platform.strip():
        return {"success": False, "error": "Platform cannot be empty."}

    # Simulate lead ID generation
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    lead_id = f"LEAD-AS-{timestamp}"

    # Mock API call output
    print("\n" + "=" * 55)
    print("🎯  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Lead ID   : {lead_id}")
    print(f"  Name      : {name.strip()}")
    print(f"  Email     : {email.strip()}")
    print(f"  Platform  : {platform.strip()}")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print("  ✅ Lead has been queued for sales team follow-up.")
    print("=" * 55 + "\n")

    return {
        "success": True,
        "lead_id": lead_id,
        "name": name.strip(),
        "email": email.strip(),
        "platform": platform.strip(),
        "message": f"Lead captured successfully: {name}, {email}, {platform}"
    }


if __name__ == "__main__":
    # Quick test
    result = mock_lead_capture("Jane Doe", "jane@example.com", "YouTube")
    print(result)
