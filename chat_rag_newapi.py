# streamlit_chat_conversation_fixed.py
"""
Streamlit conversational RAG chat (reads secrets from Streamlit Cloud st.secrets or .env fallback).
Run locally: pip install -r requirements.txt and set OPENAI_API_KEY / PINECONE_API_KEY in .env or env vars.
Deploy to Streamlit Cloud: add secrets via the UI (see instructions).
"""

from __future__ import annotations
import streamlit as st
from typing import List, Dict, Any
from pathlib import Path
import os

# try to get secrets from Streamlit Cloud; fall back to environment or .env
def get_secret(key: str, fallback: str | None = None) -> str | None:
    # 1) st.secrets (Streamlit Cloud)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # 2) environment variable
    val = os.getenv(key)
    if val:
        return val
    # 3) provided fallback
    return fallback

# External clients
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("Install new OpenAI client: pip install --upgrade openai")

try:
    from pinecone import Pinecone
except Exception:
    raise SystemExit("Install pinecone: pip install --upgrade pinecone")

# ---- CONFIG ------------------------------------------------
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX = get_secret("PINECONE_INDEX")
PINECONE_HOST = get_secret("PINECONE_HOST")
PINECONE_DIM = int(get_secret("PINECONE_DIMENSIONS", "1536"))
CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TOP_K = int(get_secret("RAG_TOP_K", "5"))
SNIPPET_CHARS = int(get_secret("PINECONE_SNIPPET_CHARS", "800"))

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    st.error("Missing secrets. Set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX in Streamlit Secrets or local env.")
    st.stop()

# ---------- init clients (store in session_state to avoid re-init) ----------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

openai_client = st.session_state.openai_client
pinecone_client = st.session_state.pinecone_client

# connect to index (serverless host preferred)
try:
    index_client = pinecone_client.Index(host=PINECONE_HOST) if PINECONE_HOST else pinecone_client.Index(PINECONE_INDEX)
except Exception as e:
    st.error(f"Failed to connect to Pinecone index: {e}")
    st.stop()

# ---------- helper functions ----------
def get_query_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    resp = openai_client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def retrieve_snippets(query: str, top_k: int = TOP_K):
    qvec = get_query_embedding(query)
    resp = index_client.query(vector=qvec, top_k=top_k, include_metadata=True, include_values=False)
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
    out = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m.get("metadata", {}), dict) else {}
        snippet = md.get("text") or md.get("snippet") or ""
        if snippet and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + " ...[truncated]"
        out.append({
            "id": m.get("id") or md.get("source", ""),
            "score": m.get("score", 0.0),
            "snippet": snippet,
            "metadata": md,
        })
    return out

def build_prompt_messages(question: str, retrieved: List[Dict[str, Any]], chat_history: List[Dict[str,str]]) -> List[Dict[str,str]]:
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that answers questions using only the provided context snippets. "
            "Cite sources by file name in square brackets when relevant. If the answer is not in the snippets, say you don't know."
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
    # include last few turns
    history_slice = [m for m in chat_history if m["role"] in ("user", "assistant")]
    if history_slice:
        for h in history_slice[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_content})
    return messages

def call_chat(messages: List[Dict[str,str]], model: str = CHAT_MODEL, temperature: float = 0.0) -> str:
    resp = openai_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    choice = resp.choices[0]
    msg = choice.message
    content = msg["content"] if hasattr(msg, "__getitem__") else msg.content
    return content.strip()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Chat — Conversation", layout="wide")
st.title("RAG Chat — Conversation")

# session messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# sidebar
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.number_input("Top K (retrieval)", value=TOP_K, min_value=1, max_value=20, step=1)
    model_sel = st.text_input("Chat model", value=CHAT_MODEL)
    embed_model_sel = st.text_input("Embedding model", value=EMBED_MODEL)
    if st.button("Clear conversation"):
        st.session_state.messages = []

# process input first
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Retrieving relevant documents..."):
        retrieved_items = retrieve_snippets(user_input, top_k=top_k)
    history_for_model = st.session_state.messages.copy()
    prompt_messages = build_prompt_messages(user_input, retrieved_items, history_for_model)
    with st.spinner("Generating answer..."):
        try:
            assistant_text = call_chat(prompt_messages, model=model_sel)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            assistant_text = "Error: OpenAI API call failed."
    sources_meta = [{"source": r["metadata"].get("source", r["id"]), "score": r["score"], "snippet": r["snippet"]} for r in retrieved_items]
    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources": sources_meta})

# render conversation
for msg in st.session_state.messages:
    role = msg.get("role")
    content = msg.get("content")
    sources = msg.get("sources", None)
    if role == "user":
        st.chat_message("user").write(content)
    elif role == "assistant":
        m = st.chat_message("assistant")
        m.write(content)
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, start=1):
                    src = s.get("source", f"doc{i}")
                    score = s.get("score")
                    snippet = s.get("snippet", "")
                    st.markdown(f"**{i}. {src}** — score: {score:.4f}")
                    st.write(snippet)

st.write("---")
st.caption("Conversation stored in session only; refresh clears it.")
