# 🎬 AutoStream AI Sales Agent

> **Social-to-Lead Agentic Workflow** — Built for ServiceHive / Inflx ML Intern Assignment

A production-grade Conversational AI Agent that converts social media conversations into qualified business leads — powered by **Google Gemini 2.5 Flash**, **LangGraph**, and a **RAG knowledge pipeline**.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [How to Run Locally](#how-to-run-locally)
5. [Project Structure](#project-structure)
6. [Conversation Flow Demo](#conversation-flow-demo)
7. [WhatsApp Webhook Deployment](#whatsapp-webhook-deployment)
8. [Design Decisions](#design-decisions)
9. [Evaluation Checklist](#evaluation-checklist)

---

## Project Overview

AutoStream's AI Sales Agent (`Alex`) is a multi-turn conversational agent that:

- **Understands** user intent across 3 categories (greeting, inquiry, high-intent)
- **Retrieves** accurate product information via a local RAG knowledge base
- **Qualifies** high-intent leads through structured step-by-step collection
- **Captures** leads by calling a mock CRM API function only when all data is confirmed

---

## Features

| Feature | Implementation |
|---|---|
| 🧠 Intent Detection | Keyword pre-filter + LLM-based classification |
| 📚 RAG Knowledge Retrieval | Local JSON KB → formatted context injected into LLM |
| 🎯 Lead Qualification | Step-by-step collection (name → email → platform) |
| 🔧 Tool Execution | `mock_lead_capture()` fires only after all 3 fields collected |
| 💾 State Management | LangGraph `StateGraph` with typed `AgentState` |
| 🔄 Multi-turn Memory | Full conversation history retained across all turns |
| ✅ Email Validation | Regex-based validation before accepting email field |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Message                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              classify_intent_node                        │
│  • Keyword classifier (fast path)                        │
│  • Detects: CASUAL_GREETING / PRODUCT_INQUIRY /          │
│             HIGH_INTENT_LEAD                             │
│  • Preserves lead_stage if mid-collection                │
└──────────────┬──────────────────┬───────────────────────┘
               │                  │
    intent=RAG │                  │ intent=HIGH_INTENT
               ▼                  ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  rag_response    │   │      lead_collection_node         │
│  _node           │   │                                   │
│                  │   │  Stage Machine:                   │
│  • Injects full  │   │  NONE → ask name                  │
│    RAG context   │   │  ASKING_NAME → capture name       │
│  • Gemini 2.5    │   │  ASKING_EMAIL → validate + capture│
│    Flash LLM     │   │  ASKING_PLATFORM → capture        │
│    generates     │   │  COMPLETE → trigger tool          │
│    response      │   └──────────────┬───────────────────┘
└────────┬─────────┘                  │
         │                            ▼ (if COMPLETE)
         ▼               ┌────────────────────────────────┐
        END              │      lead_capture_node          │
                         │                                 │
                         │  • Calls mock_lead_capture()    │
                         │  • Generates lead ID            │
                         │  • Sends confirmation to user   │
                         └────────────────┬───────────────┘
                                          │
                                         END
```

### Why LangGraph?

LangGraph was chosen over vanilla LangChain or AutoGen for three key reasons:

1. **Explicit State Machine** — LangGraph's `StateGraph` makes the agent's behavior deterministic and auditable. Each node has a single responsibility, and conditional edges enforce rules like "don't trigger lead capture until all fields are collected." This is crucial for production safety.

2. **Typed State with Persistence** — The `AgentState` TypedDict retains the full conversation history, lead collection stage, and all captured fields across every turn. There's no ambiguity about what the agent "knows" — it's all in one place, inspectable at any point.

3. **Clean Separation of Concerns** — Intent classification, RAG retrieval, lead collection, and tool execution are separate nodes. This makes the codebase easy to test, extend, and debug — compared to AutoGen's multi-agent model which adds unnecessary complexity for a single-agent workflow.

### State Management

State flows through the graph as an immutable `AgentState` dict. LangGraph uses an `add_messages` reducer on the `messages` field so each node appends (never overwrites) conversation history. The `lead_stage` enum (`NONE → ASKING_NAME → ASKING_EMAIL → ASKING_PLATFORM → COMPLETE`) ensures the tool fires exactly once, only after all three lead fields are confirmed.

---

## How to Run Locally

### Prerequisites

- Python 3.9+
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
# Option A: Create a .env file (recommended)
echo "GOOGLE_API_KEY=your_key_here" > .env

# Option B: Export directly
export GOOGLE_API_KEY=your_key_here   # macOS / Linux
set GOOGLE_API_KEY=your_key_here      # Windows CMD
```

### 5. Run the agent

```bash
python main.py
```

You should see:

```
╔══════════════════════════════════════════════════════╗
║       🎬 AutoStream AI Sales Assistant               ║
║        Powered by Google Gemini + LangGraph          ║
╚══════════════════════════════════════════════════════╝

💬 You: _
```

---

## Project Structure

```
autostream_agent/
│
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not committed)
│
├── knowledge_base/
│   ├── autostream_kb.json           # RAG knowledge base (pricing, policies, FAQs)
│   └── rag_pipeline.py              # Loads + formats KB as LLM context
│
├── agent/
│   ├── graph.py                     # LangGraph StateGraph (core agent logic)
│   ├── state.py                     # AgentState TypedDict + LeadStage enum
│   └── prompts.py                   # System prompt builder with RAG injection
│
├── tools/
│   └── lead_capture.py              # mock_lead_capture() tool + email validation
│
└── utils/
    └── intent_classifier.py         # Keyword-based intent pre-classifier
```

---

## Conversation Flow Demo

```
💬 You: Hi there!

🤖 Alex: Hey there! 👋 Welcome to AutoStream — your AI-powered video editing
platform for content creators! Whether you're on YouTube, Instagram, or TikTok,
we've got plans to help you create stunning videos effortlessly. 🎬
Feel free to ask me about our plans, features, or anything else!

──────────────────────────────────────────────────────

💬 You: Tell me about your pricing plans

🤖 Alex: Great question! Here's what AutoStream offers:

**Basic Plan — $29/month**
- 10 videos/month
- 720p resolution
- Email support

**Pro Plan — $79/month**
- Unlimited videos
- 4K resolution
- AI captions
- 24/7 support
- Custom branding + analytics

Both plans come with a 7-day free trial. Would you like to get started? 🚀

──────────────────────────────────────────────────────

💬 You: That sounds great, I want to try the Pro plan for my YouTube channel

🤖 Alex: 🚀 Awesome! I'd love to get you set up with AutoStream.
Let's start — could you share your **full name**?

──────────────────────────────────────────────────────

💬 You: John Smith

🤖 Alex: Nice to meet you, John Smith! 😊
Now, could you share your **email address**?

──────────────────────────────────────────────────────

💬 You: john.smith@gmail.com

🤖 Alex: Got it! ✅ And finally, which **creator platform** are you primarily on?
(e.g., YouTube, Instagram, TikTok, Twitter/X, etc.)

──────────────────────────────────────────────────────

💬 You: YouTube

🤖 Alex: Perfect! Let me confirm your details:
- **Name**: John Smith
- **Email**: john.smith@gmail.com
- **Platform**: YouTube

Registering you now... 🎬

═══════════════════════════════════════════════════════
🎯  LEAD CAPTURED SUCCESSFULLY
═══════════════════════════════════════════════════════
  Lead ID   : LEAD-AS-20250421183045
  Name      : John Smith
  Email     : john.smith@gmail.com
  Platform  : YouTube
  Timestamp : 2025-04-21 18:30:45
═══════════════════════════════════════════════════════
  ✅ Lead has been queued for sales team follow-up.
═══════════════════════════════════════════════════════

🤖 Alex: 🎉 You're all set, John Smith! Your lead has been successfully
registered (ID: `LEAD-AS-20250421183045`). Our team will reach out to
john.smith@gmail.com shortly to get you started on AutoStream. Welcome aboard! 🚀
```

---

## WhatsApp Webhook Deployment

### Overview

To deploy this agent on WhatsApp, we use the **WhatsApp Business API (Cloud API)** via Meta, combined with a webhook server that bridges incoming WhatsApp messages to our LangGraph agent.

### Architecture

```
WhatsApp User
     │
     │  (sends message)
     ▼
Meta WhatsApp Cloud API
     │
     │  HTTP POST (webhook event)
     ▼
┌─────────────────────────────────────────┐
│         FastAPI Webhook Server           │
│                                         │
│  POST /webhook                          │
│  ├─ Verify webhook (GET challenge)      │
│  ├─ Parse incoming message              │
│  ├─ Retrieve session state from Redis   │
│  ├─ Run LangGraph agent.invoke(state)   │
│  ├─ Save updated state to Redis         │
│  └─ POST reply to WhatsApp API          │
└─────────────────────────────────────────┘
     │
     │  (sends reply)
     ▼
WhatsApp User
```

### Implementation Steps

#### Step 1 — Create a FastAPI Webhook Server

```python
# webhook_server.py
from fastapi import FastAPI, Request
import httpx, redis, json, os
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_agent, get_initial_state

app = FastAPI()
agent = build_agent()
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
WA_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]
PHONE_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params["hub.challenge"])
    return {"error": "Invalid token"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    entry = body["entry"][0]["changes"][0]["value"]

    if "messages" not in entry:
        return {"status": "no_message"}

    msg = entry["messages"][0]
    user_phone = msg["from"]
    user_text = msg["text"]["body"]

    # Load or initialize session state per user
    raw_state = r.get(f"session:{user_phone}")
    state = json.loads(raw_state) if raw_state else get_initial_state()

    # Reconstruct LangChain messages from stored dicts
    state["messages"] = deserialize_messages(state["messages"])
    state["messages"].append(HumanMessage(content=user_text))

    # Run agent
    result = agent.invoke(state)

    # Save updated state
    r.set(f"session:{user_phone}", json.dumps(serialize_state(result)), ex=86400)

    # Extract latest AI response
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply_text = ai_msgs[-1].content if ai_msgs else "Sorry, something went wrong."

    # Send reply via WhatsApp API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": user_phone,
                "type": "text",
                "text": {"body": reply_text}
            }
        )

    return {"status": "ok"}
```

#### Step 2 — Session State with Redis

Each WhatsApp user (identified by phone number) gets their own persistent session stored in Redis with a 24-hour TTL. This ensures:
- Multi-turn context is maintained per user
- The lead stage is preserved across messages
- Concurrent users don't interfere with each other

#### Step 3 — Deploy & Register Webhook

1. Deploy the FastAPI server (e.g., Railway, Render, or AWS EC2)
2. Expose it via HTTPS (required by Meta)
3. In Meta Developer Console → WhatsApp → Configuration → Webhook URL, enter your endpoint
4. Subscribe to the `messages` webhook field

#### Step 4 — Environment Variables

```env
GOOGLE_API_KEY=your_google_ai_studio_key
WHATSAPP_VERIFY_TOKEN=your_random_verify_token
WHATSAPP_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
REDIS_URL=redis://localhost:6379
```

### Key Considerations

| Challenge | Solution |
|---|---|
| Stateless HTTP requests | Redis stores full `AgentState` per user session |
| Message serialization | Convert LangChain messages to/from JSON for Redis |
| Concurrent users | Each phone number = isolated session key |
| Webhook verification | GET endpoint returns `hub.challenge` for Meta handshake |
| Rate limits | Queue messages with Celery for high-volume deployments |

---

## Design Decisions

### Why Google Gemini 2.5 Flash?
Fast response times (ideal for chat UX), cost-efficient, and strong at instruction-following for intent classification and guided FAQ responses. The system prompt + RAG context effectively constrains the model to only answer from the knowledge base.

### Why Local JSON over Vector DB?
The knowledge base is small and static. A vector database (Pinecone, Chroma) would add latency and infrastructure complexity with no benefit at this scale. The full KB fits comfortably within Gemini's context window, making retrieval instant and free.

### Why Email Validation Before Accepting?
The lead tool requires a valid email. Validating with regex before recording prevents junk data from entering the CRM — a real-world concern for any sales pipeline. The agent re-prompts the user politely if the email format is invalid.

### Why a Keyword Pre-Filter for Intent?
Using a fast keyword classifier before routing to the LLM saves a full LLM call for obvious cases (e.g., "Hi", "signup"). It also acts as a safety net: if we're mid-lead-collection, the classifier overrides to `HIGH_INTENT_LEAD` regardless of the user's latest message wording.

---

## Evaluation Checklist

| Criterion | Status |
|---|---|
| Agent reasoning & intent detection | ✅ Keyword pre-filter + LLM-grounded classification |
| Correct use of RAG | ✅ JSON KB injected via structured system prompt |
| Clean state management | ✅ LangGraph `StateGraph` + `LeadStage` enum |
| Proper tool calling logic | ✅ Tool fires only after all 3 fields confirmed |
| Code clarity & structure | ✅ Modular packages, typed, fully documented |
| Real-world deployability | ✅ WhatsApp webhook + Redis architecture provided |

---

*Built for ServiceHive / Inflx ML Intern Assignment. Candidate submission.*