# streamlit_chat_conversation_fixed.py
"""
Streamlit conversational RAG chat (fixed: no experimental_rerun).
Run:
  streamlit run streamlit_chat_conversation_fixed.py
Requires:
  pip install streamlit python-dotenv openai pinecone
"""

from __future__ import annotations
import streamlit as st
from dotenv import dotenv_values
from typing import List, Dict, Any
from pathlib import Path

# OpenAI v1+ client
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("Install the new OpenAI client: pip install --upgrade openai")

# Pinecone new SDK
try:
    from pinecone import Pinecone
except Exception:
    raise SystemExit("Install pinecone: pip install --upgrade pinecone")

# ---------- load config ----------
cfg = dotenv_values(".env")
OPENAI_API_KEY = cfg.get("OPENAI_API_KEY")
PINECONE_API_KEY = cfg.get("PINECONE_API_KEY")
PINECONE_INDEX = cfg.get("PINECONE_INDEX")
PINECONE_HOST = cfg.get("PINECONE_HOST")
PINECONE_DIM = int(cfg.get("PINECONE_DIMENSIONS", "1536"))
TOP_K = int(cfg.get("RAG_TOP_K", "5"))
SNIPPET_CHARS = int(cfg.get("PINECONE_SNIPPET_CHARS", "800"))
CHAT_MODEL = cfg.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = cfg.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    st.error("Please set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_INDEX in your .env")
    st.stop()

# ---------- init clients (store in session) ----------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

openai_client: OpenAI = st.session_state.openai_client
pinecone_client: Pinecone = st.session_state.pinecone_client

# connect to index (serverless host preferred)
try:
    index_client = pinecone_client.Index(host=PINECONE_HOST) if PINECONE_HOST else pinecone_client.Index(PINECONE_INDEX)
except Exception as e:
    st.error(f"Failed to connect to Pinecone index: {e}")
    st.stop()

# initialize messages container
if "messages" not in st.session_state:
    # messages: list of {"role": "user"/"assistant"/"system", "content": str, "sources": Optional[list]}
    st.session_state.messages = []

# ---------- helpers ----------
def get_query_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    resp = openai_client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def retrieve_snippets(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
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
    # include last few turns from history (only user/assistant)
    history_slice = [m for m in chat_history if m["role"] in ("user", "assistant")]
    if history_slice:
        # include last up to 6 entries
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

# --------- Streamlit layout ---------
st.set_page_config(page_title="RAG Chat — Conversation", layout="wide")
st.title("RAG Chat — Conversation")

# sidebar controls
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.number_input("Top K (retrieval)", value=TOP_K, min_value=1, max_value=20, step=1)
    model_sel = st.text_input("Chat model", value=CHAT_MODEL)
    embed_model_sel = st.text_input("Embedding model", value=EMBED_MODEL)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else None

# ---- process input FIRST (so updated messages render below) ----
user_input = st.chat_input("Type your question here...")

if user_input:
    # record user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # retrieve
    with st.spinner("Retrieving relevant documents..."):
        retrieved_items = retrieve_snippets(user_input, top_k=top_k)
    # build model prompt (includes recent history)
    history_for_model = st.session_state.messages.copy()
    prompt_messages = build_prompt_messages(user_input, retrieved_items, history_for_model)
    # call model
    with st.spinner("Generating answer..."):
        try:
            assistant_text = call_chat(prompt_messages, model=model_sel)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            assistant_text = "Error: OpenAI API call failed."
    # attach assistant response + sources
    sources_meta = [{"source": r["metadata"].get("source", r["id"]), "score": r["score"], "snippet": r["snippet"]} for r in retrieved_items]
    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources": sources_meta})
    # no rerun needed — continue to render below

# ---- render conversation (after processing possible input) ----
chat_box = st.container()
with chat_box:
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

# small footer
st.write("---")
st.caption("Conversation stored in session only. Refresh clears it.")

