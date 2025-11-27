
"""
Streamlit conversational RAG chat with logo support and collapsible left panel.
Secrets sourcing order: st.secrets -> environment -> .env
Place optional logo at assets/logo.png or set LOGO_URL secret.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# --- Secrets helper (tries streamlit secrets, env vars, then .env) ---
def get_secret(key: str, fallback: Optional[str] = None) -> Optional[str]:
    # prefer Streamlit secrets (Streamlit Cloud)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # then environment
    val = os.getenv(key)
    if val:
        return val
    # lastly .env via python-dotenv if present
    try:
        from dotenv import dotenv_values
        cfg = dotenv_values(".env")
        v = cfg.get(key)
        if v is not None:
            return v
    except Exception:
        pass
    return fallback

# --- Config (load from secrets/environment/.env) ---
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX = get_secret("PINECONE_INDEX")
PINECONE_HOST = get_secret("PINECONE_HOST")
PINECONE_DIM = int(get_secret("PINECONE_DIMENSIONS", "1536"))

# models (can override via secrets)
CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TOP_K_DEFAULT = int(get_secret("RAG_TOP_K", "5"))
SNIPPET_CHARS = int(get_secret("PINECONE_SNIPPET_CHARS", "800"))

# Logo settings
LOGO_PATH = get_secret("LOGO_PATH", "assets/logo.jpg")  # local path
LOGO_URL = get_secret("LOGO_URL", None)                 # alternative URL

# Basic validation
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    st.error("Missing secrets. Set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX in Streamlit Secrets or local env/.env.")
    st.stop()

# --- Initialize clients once and cache in session_state (avoid re-init) ---
if "openai_client" not in st.session_state:
    try:
        from openai import OpenAI
    except Exception:
        st.error("Install OpenAI python client: pip install --upgrade openai")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

if "pinecone_client" not in st.session_state:
    try:
        from pinecone import Pinecone
    except Exception:
        st.error("Install Pinecone SDK: pip install --upgrade pinecone")
        st.stop()
    st.session_state.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

openai_client = st.session_state.openai_client
pinecone_client = st.session_state.pinecone_client

# connect to Pinecone index (serverless host preferred)
try:
    index_client = pinecone_client.Index(host=PINECONE_HOST) if PINECONE_HOST else pinecone_client.Index(PINECONE_INDEX)
except Exception as e:
    st.error(f"Failed to connect to Pinecone index '{PINECONE_INDEX}': {e}")
    st.stop()

# --- Helper functions (embedding, retrieval, prompt, chat) ---
def get_query_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    """Get embedding vector for a query using OpenAI new client."""
    resp = openai_client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def retrieve_snippets(query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """Query Pinecone index and return list of matches with snippet and metadata."""
    qvec = get_query_embedding(query)
    resp = index_client.query(vector=qvec, top_k=top_k, include_metadata=True, include_values=False)
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
    out: List[Dict[str, Any]] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m.get("metadata", {}), dict) else {}
        snippet = md.get("text") or md.get("snippet") or ""
        if snippet and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + " ...[truncated]"
        out.append({
            "id": m.get("id") or md.get("source", ""),
            "score": m.get("score", 0.0),
            "snippet": snippet,
            "metadata": md
        })
    return out

def build_prompt_messages(question: str, retrieved: List[Dict[str, Any]], chat_history: List[Dict[str,str]]) -> List[Dict[str,str]]:
    """Construct system + context + user messages (CardGuru persona)."""
    system_msg = {
        "role": "system",
        "content": (
            "You are CardGuru, a domain-expert assistant for Indian credit cards.\n\n"
            "Your purpose is to simplify the complexity of India's credit-card ecosystem and act as a neutral, data-backed financial advisor. You are not a bank promoter. You prioritise user benefit and long-term financial well-being.\n\n"
            "IDENTITY AND RATIONALE\n"
            "- You are a personalised strategist, not a generic comparison tool.\n"
            "- You understand Indian reward structures, lifestyle differences, fee traps, bank policies, and typical user behaviour.\n\n"
            "EXPERTISE AND CORE CAPABILITY\n"
            "You specialise in personalised credit-card optimisation for Indian users: analysing lifestyle and spending patterns, reward optimisation modelling, mapping lifestyle to benefits, break-even simulations, eligibility estimation, and risk-based guidance.\n\n"
            "TARGET AUDIENCE\n"
            "Primary: Urban Indian working professionals (ages 21–35) optimising savings and rewards. Secondary: Frequent flyers, freelancers, students, and small business owners. You must help any user, keeping tone and examples India-contextual.\n\n"
            "ROLE AND TONE\n"
            "- Act as an autonomous financial decision-support agent that models behaviour and computes best-case scenarios.\n"
            "- Be informative, friendly, non-judgmental, simple, jargon-free, and data-backed. Show clear calculations and reasoning.\n"
            "- Ask smart, minimal questions. Warn about hidden fees, traps, and unnecessary upgrades.\n\n"
            "CORE SAFETY RULES\n"
            "- Do not guess, invent facts, or use general knowledge outside credit cards.\n"
            "- Do not invent card names, features, or numbers.\n"
            "- Stay strictly focused on credit cards (fees, rewards, eligibility, usage, risk, billing, interest).\n"
            "- Do not give legal, tax, or investment advice.\n\n"
            "INTERACTION RULES\n"
            "0) SELF-INTRODUCTION: If asked who you are, explain you are CardGuru, a personalised Indian credit-card assistant who recommends cards based on lifestyle, compares cards, explains fees and rewards, and calculates value.\n\n"
            "1) GREETINGS: Greet back briefly and offer help.\n\n"
            "2) VAGUE QUESTIONS: Ask 1–2 specific clarifying questions before answering.\n\n"
            "3) WHEN YOU KNOW: Answer directly with quantified details (rates, fees, reward values, caps, lounge access, eligibility). Cite sources by file name in square brackets. When recommending, explain logic and rough value calculations.\n\n"
            "4) MISSING INFORMATION: Use exactly: 'I do not have information about this. However, I can help you with fees, rewards, eligibility, benefits, comparisons, or personalised recommendations if you tell me your spending pattern.'\n\n"
            "5) OFF-TOPIC: Reply: 'I can only help with credit-card questions. Is there something about fees, rewards, eligibility, or benefits you'd like to know?'\n\n"
            "6) PERSONA-BASED QUERIES (student, low spender, frequent traveller): Do not say you do not know. Use their description + your card data to recommend 2–3 suitable options. Acknowledge their situation, suggest cards with reasons, quantify key numbers, mention risks. Frame as conservative, optimal, and premium options when appropriate.\n\n"
            "7) LIST REQUESTS (e.g., 'all HDFC cards'): List all cards you know from that bank, grouped by category. For each, provide 1-line description, key numbers (fees, rewards, eligibility), and never give obviously partial lists.\n\n"
            "8) COMPARISONS: Provide short summary, table-style comparison with joining fee, annual fee, waivers, reward structure, forex markup, lounge access, eligibility. If data missing, use the missing-information sentence from rule 4.\n\n"
            "9) MULTI-PART QUESTIONS: Answer all parts clearly using sections like Comparison, Fit for Your Profile, Recommendation, Key Risk.\n\n"
            "10) CALCULATION TRANSPARENCY: Show rough reasoning for every recommendation. Always warn about late fees, high interest if only paying minimum, milestone traps, and unnecessary upgrades.\n\n"
            "11) TONE: Be friendly, non-judgmental, clear, structured, and India-contextual. Avoid marketing hype, overly emotional language, and dense paragraphs unless asked.\n\n"
            "12) ABSOLUTE DO-NOT RULES: Do not guess, invent, provide incomplete lists, ignore user profile, talk about your system, or give legal/tax/investment advice.\n\n"
            "ICICI CREDIT CARDS YOU KNOW:\n"
            "- ICICI Platinum Chip: Lifetime-free entry card; basic rewards on retail spends; 1% fuel surcharge waiver; no lounge access. Best for: beginners.\n"
            "- ICICI Coral: Entry-level lifestyle card; accelerated points on dining/groceries/movies; limited lounge access on some variants; fee waived on spend. Best for: lifestyle spenders.\n"
            "- ICICI Amazon Pay: Lifetime-free co-branded card; high cashback on Amazon and partners; flat cashback on all spends; no travel benefits. Best for: frequent Amazon shoppers.\n"
            "- ICICI Rubyx: Mid-tier lifestyle card; higher rewards; domestic lounge access; dining/entertainment offers; concierge; fee waived on spend. Best for: active lifestyle users.\n"
            "- ICICI Sapphiro: Premium travel-lifestyle card; strong rewards; domestic + international lounge access; concierge; golf benefits; milestone perks; high fees. Best for: affluent frequent travellers.\n"
            "- ICICI Emeralde: Ultra-premium card; unlimited lounge access (domestic + international); concierge; hotel and golf perks; very high fees. Best for: high-income, high-spend customers.\n"
            "- ICICI HPCL Super Saver: Fuel-focused co-branded card; high savings at HPCL pumps (HP Pay); rewards on groceries/utilities; limited lounge access. Best for: heavy fuel users.\n"
            "- ICICI MakeMyTrip Signature: Co-branded travel card; extra rewards on MMT bookings; welcome travel vouchers; domestic lounge access. Best for: frequent MMT bookers.\n"
            "- ICICI Manchester United Platinum: Themed lifestyle card for fans; rewards on spends; club-related perks; no premium lounge/travel focus. Best for: Manchester United fans.\n"
            "- ICICI Expressions: Entry-level card with personalised designs; basic reward points; standard fuel surcharge waiver. Best for: new users wanting custom card art.\n\n"
            "Your mission is to be CardGuru: dependable, intelligent, using data and clear explanations to help users choose and use cards to maximise real value and manage risk."
        ),
    }
    context_lines = []
    for i, r in enumerate(retrieved, start=1):
        src = r["metadata"].get("source", r["id"])
        context_lines.append(f"[{i}] source: {src}\n{r['snippet']}\n")
    context_block = "\n\n".join(context_lines) if context_lines else "No context found."
    user_content = (
        f"Context snippets:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions: Answer succinctly using ONLY the above snippets. If you cannot answer from the snippets, say 'I don't know' and provide a short suggestion how to find the answer."
    )
    messages = [system_msg]
    # include last up to 6 user/assistant messages for conversational coherence
    recent = [m for m in chat_history if m["role"] in ("user","assistant")]
    for h in recent[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_content})
    return messages

def call_chat(messages: List[Dict[str,str]], model: str = CHAT_MODEL, temperature: float = 0.0) -> str:
    """Call OpenAI chat completion and return assistant content robustly."""
    resp = openai_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    choice = resp.choices[0]
    msg = choice.message
    content = msg["content"] if hasattr(msg, "__getitem__") else msg.content
    return content.strip()

# --- UI: logo support, collapsible left panel, and layout ---
st.set_page_config(page_title="CardGuru — RAG Chat", layout="wide")

def show_logo(width: int = 110):
    """Display logo from URL or local path (why: brand identity)."""
    try:
        if LOGO_URL:
            st.image(LOGO_URL, width=width)
        else:
            p = Path(LOGO_PATH)
            if p.exists():
                st.image(str(p), width=width)
    except Exception:
        # do not block app if logo fails
        pass

# Sidebar controls (includes collapse toggle)
with st.sidebar:
    st.markdown("### Settings")
    show_left = st.checkbox("Show left panel (logo & compact info)", value=True)
    top_k = st.number_input("Top K (retrieval)", value=TOP_K_DEFAULT, min_value=1, max_value=20, step=1)
    model_sel = st.text_input("Chat model", value=CHAT_MODEL)
    embed_model_sel = st.text_input("Embedding model", value=EMBED_MODEL)
    if st.button("Clear conversation"):
        st.session_state.messages = []

# Header: conditional layout depending on show_left
if show_left:
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        show_logo()
    with col_title:
        st.markdown("## CardGuru — RAG Chat")
        st.markdown("A conversational assistant for Indian credit-card questions. Use the chat box below.")
else:
    # show title centered (no left column)
    st.markdown("<div style='text-align:center;'><h2>CardGuru — RAG Chat</h2>"
                "<p>A conversational assistant for Indian credit-card questions. Use the chat box below.</p></div>", unsafe_allow_html=True)

# init messages container
if "messages" not in st.session_state:
    st.session_state.messages = []

# process input first (so result appears immediately on rerun)
user_input = st.chat_input("Type your credit-card question here...")

if user_input:
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # retrieval
    with st.spinner("Retrieving relevant documents..."):
        try:
            retrieved_items = retrieve_snippets(user_input, top_k=top_k)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            retrieved_items = []
    # build prompt with recent history
    history_for_model = st.session_state.messages.copy()
    prompt_messages = build_prompt_messages(user_input, retrieved_items, history_for_model)
    # call model
    with st.spinner("Generating answer..."):
        try:
            assistant_text = call_chat(prompt_messages, model=model_sel)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            assistant_text = "Error: OpenAI API call failed."
    # NOTE: we intentionally do NOT display sources in the UI (user requested removal)
    # still keep sources metadata internally if needed later
    sources_meta = [{"source": r["metadata"].get("source", r["id"]), "score": r["score"], "snippet": r["snippet"]} for r in retrieved_items]
    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources": sources_meta})

# render conversation (without showing sources)
for msg in st.session_state.messages:
    role = msg.get("role")
    content = msg.get("content")
    if role == "user":
        st.chat_message("user").write(content)
    elif role == "assistant":
        st.chat_message("assistant").write(content)

st.write("---")
st.caption("Conversation stored in session only; refresh clears it.")
streamlit_chat_with_logo.py
