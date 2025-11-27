# # upsert_from_embeddings_env.py
# """
# Upsert embeddings.json -> Pinecone, loading configuration from a .env file
# using python-dotenv (no direct os.getenv calls).

# .env keys used (examples):
#   PINECONE_API_KEY=pcsk_...
#   PINECONE_INDEX=my-ai
#   PINECONE_HOST=https://your-pinecone-host-url
#   PINECONE_DIMENSIONS=1024
#   EMBEDDINGS_FILE=embeddings.json
#   CHUNKS_DIR=chunks
#   PINECONE_BATCH_SIZE=200
# """

# from __future__ import annotations
# import json
# import time
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple, Any

# # dotenv
# from dotenv import dotenv_values

# # Pinecone new SDK
# try:
#     from pinecone import Pinecone
# except Exception as exc:
#     raise ImportError(
#         "pinecone new SDK not available. Install with: pip install --upgrade pinecone"
#     ) from exc


# # ---------- Load config from .env ----------
# config: Dict[str, str] = dotenv_values(".env")  # returns mapping of keys->values (strings)

# PINE_API_KEY: Optional[str] = config.get("PINECONE_API_KEY")
# PINE_INDEX: Optional[str] = config.get("PINECONE_INDEX")
# PINE_HOST: Optional[str] = config.get("PINECONE_HOST")
# EXPECTED_DIM: int = int(config.get("PINECONE_DIMENSIONS", "1024"))
# EMBED_PATH: str = config.get("EMBEDDINGS_FILE", "embeddings_1024.json")
# CHUNKS_DIR: Path = Path(config.get("CHUNKS_DIR", "chunks"))
# BATCH_SIZE: int = int(config.get("PINECONE_BATCH_SIZE", "200"))
# SLEEP_BETWEEN_BATCHES: float = float(config.get("PINECONE_SLEEP", "0.05"))

# # Basic validation
# if not PINE_API_KEY:
#     raise SystemExit("Missing PINECONE_API_KEY in .env")
# if not PINE_INDEX:
#     raise SystemExit("Missing PINECONE_INDEX in .env")


# # ---------- Helpers ----------
# def load_json(path: str) -> Dict[str, Any]:
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"{p} not found")
#     return json.loads(p.read_text(encoding="utf-8"))


# def read_chunk_text(chunks_dir: Path, filename: str) -> Optional[str]:
#     p = chunks_dir / filename
#     if not p.exists():
#         return None
#     try:
#         return p.read_text(encoding="utf-8")
#     except Exception:
#         return None


# def batch_iterable(iterable: Iterable, batch_size: int) -> Iterable[List]:
#     batch: List = []
#     for item in iterable:
#         batch.append(item)
#         if len(batch) >= batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch


# def _is_numeric_vector(x: Any) -> bool:
#     if not isinstance(x, (list, tuple)):
#         return False
#     return all(isinstance(v, (int, float)) for v in x)


# # ---------- Main flow ----------
# def main() -> None:
#     print(f"Loading embeddings from {EMBED_PATH}...")
#     data = load_json(EMBED_PATH)
#     if not isinstance(data, dict):
#         raise SystemExit("Embeddings JSON must be an object mapping filename -> vector")

#     items: List[Tuple[str, List[float], Dict[str, Any]]] = []
#     skipped = 0
#     for fname, vec in data.items():
#         if not isinstance(fname, str) or not _is_numeric_vector(vec):
#             skipped += 1
#             continue
#         if len(vec) != EXPECTED_DIM:
#             print(f"Skipping {fname}: dim {len(vec)} != {EXPECTED_DIM}")
#             skipped += 1
#             continue
#         vid = Path(fname).stem
#         metadata: Dict[str, Any] = {"source": fname}
#         text = read_chunk_text(CHUNKS_DIR, fname)
#         if text:
#             metadata["text"] = text
#         items.append((vid, list(vec), metadata))

#     print(f"Prepared {len(items)} items, skipped {skipped}")

#     if not items:
#         print("No valid embeddings to upsert. Exiting.")
#         return

#     # Initialize Pinecone (new SDK)
#     pc = Pinecone(api_key=PINE_API_KEY)
#     # Choose index connection: host may be required for serverless indexes
#     index = pc.Index(PINE_INDEX) if not PINE_HOST else pc.Index(host=PINE_HOST)

#     # Upsert in batches
#     batch_no = 0
#     for batch in batch_iterable(items, BATCH_SIZE):
#         batch_no += 1
#         print(f"Upserting batch {batch_no} (size={len(batch)})...")
#         try:
#             index.upsert(vectors=batch)
#         except Exception as e:
#             print(f"Error upserting batch {batch_no}: {e}")
#             # optional: one retry
#             try:
#                 time.sleep(1.0)
#                 index.upsert(vectors=batch)
#             except Exception as e2:
#                 print(f"Retry failed for batch {batch_no}: {e2}")
#                 raise
#         time.sleep(SLEEP_BETWEEN_BATCHES)

#     print("Upsert complete.")
#     print(f"Summary: upserted={len(items)} skipped={skipped} batches={batch_no}")


# if __name__ == "__main__":
#     main()
# vectordb_multi_dim_upsert.py
# vectordb_final_single_index.py
# upsert_to_my_ai.py
# upsert_to_vectorindex_1536.py
"""
Upsert embeddings into your serverless index vectorindex-1536.
Reads config from .env (python-dotenv). Filters for vectors matching PINECONE_DIMENSIONS (1536)
and upserts them using dynamic batching (avoids Pinecone 2MB request limit).
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values

# Pinecone new SDK
try:
    from pinecone import Pinecone
    from pinecone.exceptions.exceptions import PineconeApiException
except Exception as exc:
    raise SystemExit("Install Pinecone SDK in your venv: pip install --upgrade pinecone") from exc

# ---------------- load .env ----------------
cfg = dotenv_values(".env")
PINE_API_KEY: Optional[str] = cfg.get("PINECONE_API_KEY")
PINE_INDEX: str = cfg.get("PINECONE_INDEX", "vectorindex-1536")
PINE_HOST: Optional[str] = cfg.get("PINECONE_HOST") or None
PINE_DIM: int = int(cfg.get("PINECONE_DIMENSIONS", "1536"))
PINE_METRIC: str = cfg.get("PINECONE_METRIC", "cosine")

EMBEDDINGS_FILE: str = cfg.get("EMBEDDINGS_FILE", "embeddings_1024.json")
CHUNKS_DIR: Path = Path(cfg.get("CHUNKS_DIR", "chunks"))
BATCH_SIZE: int = int(cfg.get("PINECONE_BATCH_SIZE", "200"))
SLEEP_BETWEEN_BATCHES: float = float(cfg.get("PINECONE_SLEEP", "0.05"))
SNIPPET_CHARS: int = int(cfg.get("PINECONE_SNIPPET_CHARS", "1000"))

if not PINE_API_KEY:
    raise SystemExit("PINECONE_API_KEY missing in .env")

# ---------------- helpers ----------------
def load_embeddings(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Embeddings file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def detect_dims(emb_map: Dict[str, Any]) -> Counter:
    cnt = Counter()
    for k, v in emb_map.items():
        if isinstance(v, (list, tuple)):
            cnt[len(v)] += 1
        else:
            cnt["invalid"] += 1
    return cnt

def is_numeric_vector(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(v, (int, float)) for v in x)

def read_snippet(chunks_dir: Path, filename: str, max_chars: int = SNIPPET_CHARS) -> Optional[str]:
    p = chunks_dir / filename
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    if not txt:
        return None
    return txt[:max_chars] + ("...[truncated]" if len(txt) > max_chars else "")

# dynamic payload sizing (avoid Pinecone 2MB limit)
import json as _json
MAX_REQUEST_BYTES = 2 * 1024 * 1024
SAFETY_MARGIN = 200 * 1024
TARGET_BYTES = MAX_REQUEST_BYTES - SAFETY_MARGIN

def estimate_batch_bytes(batch: List[Tuple[str, List[float], Dict[str, Any]]]) -> int:
    try:
        vectors = [{"id": id_, "values": vec, "metadata": md} for id_, vec, md in batch]
        return len(_json.dumps({"vectors": vectors}, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return sum(len(_json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in batch)

# ---------------- main ----------------
def main():
    print("Loading embeddings:", EMBEDDINGS_FILE)
    emb_map = load_embeddings(EMBEDDINGS_FILE)
    dims = detect_dims(emb_map)
    print("Embedding dimension counts:", dims)

    # filter vectors that match index dim
    matched_items: List[Tuple[str, List[float], Dict[str, Any]]] = []
    skipped = 0
    for fname, vec in emb_map.items():
        if not is_numeric_vector(vec):
            skipped += 1
            continue
        if len(vec) != PINE_DIM:
            skipped += 1
            continue
        item_id = Path(fname).stem
        md = {"source": fname}
        snippet = read_snippet(CHUNKS_DIR, fname)
        if snippet:
            md["snippet"] = snippet
        matched_items.append((item_id, list(vec), md))

    print(f"Vectors matching index dimension {PINE_DIM}: {len(matched_items)} (skipped {skipped})")

    if not matched_items:
        print("\nNo vectors match the target index dimension.")
        print("Your .env is configured for index dimension:", PINE_DIM)
        print("Your embeddings file dimensions:", dict(dims))
        print("Either regenerate embeddings to 1536-d or set EMBEDDINGS_FILE to the file that has 1536-d vectors.")
        return

    # init pinecone
    pc = Pinecone(api_key=PINE_API_KEY)

    # connect to index (serverless host preferred)
    try:
        # For serverless indexes use the provided host URL
        if PINE_HOST:
            index_client = pc.Index(host=PINE_HOST)
        else:
            index_client = pc.Index(PINE_INDEX)
        print(f"Connected to index {PINE_INDEX} (host={PINE_HOST})")
    except Exception as e:
        raise SystemExit(f"Failed to connect to Pinecone index '{PINE_INDEX}': {e}")

    # dynamic batching and upsert
    batch: List[Tuple[str, List[float], Dict[str, Any]]] = []
    batches_sent = 0
    for item in matched_items:
        candidate = batch + [item]
        if estimate_batch_bytes(candidate) > TARGET_BYTES:
            # flush
            if batch:
                batches_sent += 1
                print(f"Upserting batch {batches_sent} size={len(batch)} (approx {estimate_batch_bytes(batch)} bytes)")
                try:
                    index_client.upsert(vectors=batch)
                except PineconeApiException as e:
                    raise SystemExit(f"Pinecone upsert error: {e}")
                time.sleep(SLEEP_BETWEEN_BATCHES)
                batch = []
            # if single item itself too big, truncate snippet and upsert alone
            if estimate_batch_bytes([item]) > TARGET_BYTES:
                id_, vec, md = item
                if "snippet" in md:
                    md["snippet"] = md["snippet"][:200] + "...[truncated]"
                print(f"Upserting oversized single item {id_}")
                index_client.upsert(vectors=[(id_, vec, md)])
                time.sleep(SLEEP_BETWEEN_BATCHES)
                continue
        batch.append(item)

    # flush remainder
    if batch:
        batches_sent += 1
        print(f"Upserting final batch {batches_sent} size={len(batch)}")
        index_client.upsert(vectors=batch)

    print("\nUpsert complete.")
    print(f"Summary: index={PINE_INDEX} dimension={PINE_DIM} upserted={len(matched_items)} batches={batches_sent} skipped={skipped}")

if __name__ == "__main__":
    main()

