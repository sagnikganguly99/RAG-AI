# chat_rag.py
"""
Minimal RAG chat assistant (CLI)
- Loads config from .env
- Uses Pinecone (serverless host) for retrieval
- Uses OpenAI chat model for answer generation
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import dotenv_values

# external deps: pip install python-dotenv openai pinecone
try:
    import openai
except Exception:
    raise SystemExit("Install openai: pip install openai")

try:
    from pinecone import Pinecone
except Exception:
    raise SystemExit("Install pinecone (new SDK): pip install pinecone")

# ---------- config ----------
cfg = dotenv_values(".env")
OPENAI_API_KEY = cfg.get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = cfg.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # set in .env if you prefer another
PINECONE_API_KEY = cfg.get("PINECONE_API_KEY")
PINECONE_INDEX = cfg.get("PINECONE_INDEX")
PINECONE_HOST = cfg.get("PINECONE_HOST")
PINECONE_DIM = int(cfg.get("PINECONE_DIMENSIONS", "1536"))
TOP_K = int(cfg.get("RAG_TOP_K", "5"))
SNIPPET_CHARS = int(cfg.get("PINECONE_SNIPPET_CHARS", "800"))

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    raise SystemExit("Set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX in .env")

openai.api_key = OPENAI_API_KEY

# ---------- Pinecone init ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
# For serverless indices prefer using host endpoint
index_client = pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(PINECONE_INDEX)

# ---------- helpers ----------
def retrieve_snippets(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Query pinecone and return a list of retrieved items:
    [{'id': 'chunk_0', 'score': 0.9, 'metadata': {...}}...]
    """
    # use vector query; if you're using vector embeddings saved as values in metadata, ensure index supports query by text
    # Here we use the vector-based search: send text to OpenAI to create a query embedding (if you don't have a query embedding encoder, we'll call OpenAI's text-embedding-3-small)
    # Simpler approach: use openai embeddings for the query on-the-fly
    emb_resp = openai.Embedding.create(model="text-embedding-3-small", input=query)
    qvec = emb_resp["data"][0]["embedding"]
    resp = index_client.query(vector=qvec, top_k=top_k, include_metadata=True, include_values=False)
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
    results = []
    for m in matches:
        mid = m.get("id") or m.get("metadata", {}).get("source", "")
        score = m.get("score", 0.0)
        md = m.get("metadata", {}) if isinstance(m.get("metadata", {}), dict) else {}
        # ensure snippet is short
        snippet = md.get("text") or md.get("snippet") or ""
        if snippet and len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + " ...[truncated]"
        results.append({"id": mid, "score": score, "snippet": snippet, "metadata": md})
    return results

def build_prompt(question: str, retrieved: List[Dict[str, Any]], history: List[Dict[str,str]]) -> List[Dict[str,str]]:
    """
    Build chat messages structure for OpenAI chat completion.
    We'll use a system instruction, the retrieved documents as context, and a short conversation history.
    """
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context snippets to answer the user's question. "
            "Cite sources by file name when relevant. If the answer is not in the context, say you don't know and suggest searching."
        )
    }

    # format retrieved context (short)
    context_lines = []
    for i, r in enumerate(retrieved, start=1):
        meta = r.get("metadata", {})
        source = meta.get("source", r.get("id", f"doc{i}"))
        snippet = r.get("snippet", "")
        context_lines.append(f"[{i}] source: {source}\n{snippet}\n")

    context_block = "\n\n".join(context_lines) or "No context found."

    # build user content: include the retrieved context and the question
    user_content = (
        f"Context snippets:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions: Answer succinctly using ONLY the above snippets. "
        "If you cannot answer from the snippets, say 'I don't know' and provide a short suggestion how to find the answer."
    )

    # include limited conversation history (if any)
    messages = [system]
    # append last up to 6 turns from history
    if history:
        # history is list of {"role":"user"/"assistant","content":...}
        for h in history[-6:]:
            messages.append(h)
    # then put the user content with context as a final user message
    messages.append({"role": "user", "content": user_content})
    return messages

def call_openai_chat(messages: List[Dict[str,str]], model: str = OPENAI_CHAT_MODEL, temperature: float = 0.0) -> str:
    """
    Call OpenAI chat completions and return assistant response text.
    """
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    text = resp["choices"][0]["message"]["content"]
    return text.strip()

# ---------- CLI chat loop ----------
def chat_loop():
    history: List[Dict[str,str]] = []
    print("RAG chat ready. Type your question (or 'exit'):")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("bye")
            break

        retrieved = retrieve_snippets(q, top_k=TOP_K)
        print(f"[retrieval] found {len(retrieved)} snippets (top {TOP_K})")
        # show brief citations
        for i, r in enumerate(retrieved, start=1):
            src = r.get("metadata", {}).get("source", r.get("id", ""))
            print(f"  [{i}] {src} (score={r.get('score'):.3f})")

        messages = build_prompt(q, retrieved, history)
        print("[assistant] generating answer...")
        try:
            ans = call_openai_chat(messages)
        except Exception as e:
            print("OpenAI error:", e)
            continue

        # append to history (store compact)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})

        print("\nAssistant:")
        print(ans)
        # print source list for transparency
        if retrieved:
            srcs = [r.get("metadata", {}).get("source", r.get("id", "")) for r in retrieved]
            print("\nSources:", ", ".join(srcs))

if __name__ == "__main__":
    chat_loop()
