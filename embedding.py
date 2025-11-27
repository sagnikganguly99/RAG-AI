# import os
# from pathlib import Path
# from langchain_openai import OpenAIEmbeddings
# import json

# # ------------------------------------------------------
# #  KEEP YOUR API KEY EXACTLY AS PROVIDED
# # ------------------------------------------------------
# os.environ["OPENAI_API_KEY"] = "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"

# # ------------------------------------------------------
# #  OPTION B — USE A 1024-DIM EMBEDDING MODEL
# #  (text-embedding-3-small supports 1024 dimensions)
# # ------------------------------------------------------
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

# # ------------------------------------------------------
# # LOAD CHUNKS
# # ------------------------------------------------------
# def load_chunks_from_folder(chunks_folder: str = "chunks") -> list:
#     """
#     Load chunk files and return list of (filename, content)
#     """
#     chunks_path = Path(chunks_folder)
#     if not chunks_path.exists():
#         raise FileNotFoundError(f"Chunks folder not found: {chunks_folder}")

#     chunk_list = []
#     for chunk_file in sorted(chunks_path.glob("chunk_*.txt")):
#         content = chunk_file.read_text(encoding="utf-8").strip()
#         chunk_list.append((chunk_file.name, content))

#     print(f"Loaded {len(chunk_list)} chunks from {chunks_folder}")
#     return chunk_list

# # ------------------------------------------------------
# # BATCHED EMBEDDING GENERATION (Option B)
# # ------------------------------------------------------
# def create_embeddings_for_chunks_batched(chunk_list, batch_size=64):
#     """
#     Create 1024-dim embeddings in batches using embed_documents.
#     """
#     embeddings_dict = {}
#     total = len(chunk_list)

#     print(f"Creating embeddings for {total} chunks (batch size = {batch_size})")

#     # Extract just the text list (keep filenames in parallel)
#     filenames = [fn for fn, _ in chunk_list]
#     contents  = [txt for _, txt in chunk_list]

#     # batched loop
#     for i in range(0, total, batch_size):
#         batch_texts = contents[i:i+batch_size]
#         batch_files = filenames[i:i+batch_size]

#         print(f"  Embedding batch {i//batch_size + 1}: items {i}–{i + len(batch_texts) - 1}")

#         # embed in batch
#         vecs = embeddings.embed_documents(batch_texts)

#         # store result into dict
#         for fname, vec in zip(batch_files, vecs):
#             embeddings_dict[fname] = vec

#     print(f"Created {len(embeddings_dict)} embeddings.")
#     return embeddings_dict

# # ------------------------------------------------------
# # SAVE EMBEDDINGS
# # ------------------------------------------------------
# def save_embeddings(embeddings_dict, output_file="embeddings_1024.json"):
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(embeddings_dict, f, indent=2)
#     print(f"Embeddings saved to {output_file}")

# # ------------------------------------------------------
# # MAIN
# # ------------------------------------------------------
# def main():
#     try:
#         # Load chunks
#         chunk_list = load_chunks_from_folder("chunks")

#         # Create embeddings (1024-dim)
#         embeddings_dict = create_embeddings_for_chunks_batched(chunk_list)

#         # Save to JSON
#         save_embeddings(embeddings_dict)

#         print("\nEmbedding process completed successfully!")

#     except Exception as e:
#         print(f"Error during embedding process: {e}")
#         raise

# if __name__ == "__main__":
#     main()
# embed_chunks_streamlit.py
"""
Create 1024-d embeddings for chunk files using OpenAI embeddings.
Secrets sourcing order:
  1) streamlit.st.secrets
  2) environment variables
  3) .env file (python-dotenv)
Outputs: embeddings_1024.json
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os

# secret helpers
def get_secret(key: str, fallback: Optional[str] = None) -> Optional[str]:
    # 1) streamlit secrets
    try:
        import streamlit as _st
        if key in _st.secrets:
            return _st.secrets[key]
    except Exception:
        pass
    # 2) environment
    v = os.getenv(key)
    if v:
        return v
    # 3) .env via python-dotenv
    try:
        from dotenv import dotenv_values
        cfg = dotenv_values(".env")
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    except Exception:
        pass
    return fallback

# Attempt to get API key (Streamlit cloud or local .env)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not found. Set Streamlit secret, env var, or .env")

# Ensure LangChain/OpenAI wrapper sees the key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embeddings client
try:
    from langchain_openai import OpenAIEmbeddings
except Exception as e:
    raise SystemExit("Please install langchain-openai (or supply another embeddings client). Error: " + str(e))

# choose 1024-d model
EMBED_MODEL = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

emb_client = OpenAIEmbeddings(model=EMBED_MODEL)

# chunk loader
def load_chunks_from_folder(chunks_folder: str = "chunks") -> List[Tuple[str, str]]:
    p = Path(chunks_folder)
    if not p.exists():
        raise FileNotFoundError(f"Chunks folder not found: {chunks_folder}")
    out: List[Tuple[str, str]] = []
    for f in sorted(p.glob("chunk_*.txt")):
        txt = f.read_text(encoding="utf-8").strip()
        if txt:
            out.append((f.name, txt))
    print(f"Loaded {len(out)} chunks from {chunks_folder}")
    return out

def create_embeddings_for_chunks_batched(chunk_list: List[Tuple[str, str]], batch_size: int = 64) -> Dict[str, List[float]]:
    embeddings_dict: Dict[str, List[float]] = {}
    total = len(chunk_list)
    print(f"Creating embeddings for {total} chunks (batch size={batch_size})")
    filenames = [fn for fn, _ in chunk_list]
    texts = [txt for _, txt in chunk_list]
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_files = filenames[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}: items {i}..{i + len(batch_texts) - 1}")
        # embed_documents returns list of vectors
        vecs = emb_client.embed_documents(batch_texts)
        if not isinstance(vecs, list) or len(vecs) != len(batch_texts):
            raise RuntimeError("Unexpected response from embed_documents")
        for fname, vec in zip(batch_files, vecs):
            embeddings_dict[fname] = [float(x) for x in vec]
        time.sleep(0.1)
    print(f"Created {len(embeddings_dict)} embeddings.")
    return embeddings_dict

def save_embeddings(embeddings_dict: Dict[str, List[float]], output_file: str = "embeddings_1024.json") -> None:
    Path(output_file).write_text(json.dumps(embeddings_dict, indent=2), encoding="utf-8")
    print(f"Embeddings saved to {output_file}")

def main():
    chunks = load_chunks_from_folder("chunks")
    if not chunks:
        print("No chunks found. Exiting.")
        return
    batch_size = int(get_secret("EMBEDDING_BATCH_SIZE", "64"))
    embeddings_map = create_embeddings_for_chunks_batched(chunks, batch_size=batch_size)
    save_embeddings(embeddings_map, get_secret("EMBEDDINGS_FILE", "embeddings_1024.json"))
    # verify dims
    from collections import Counter
    dims = Counter(len(v) for v in embeddings_map.values())
    print("Dimension counts:", dims)
    if len(dims) == 1:
        print("All embeddings same-dim:", dims)
    else:
        print("Warning: multiple embedding dimensions present:", dims)

if __name__ == "__main__":
    main()
