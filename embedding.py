# import getpass
# import os
# from pathlib import Path
# from langchain_openai import OpenAIEmbeddings
# import json

# # Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = "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"

# # Initialize embeddings model
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     # With the `text-embedding-3` class
#     # of models, you can specify the size
#     # of the embeddings you want returned.
#     # dimensions=1024
# )

# def load_chunks_from_folder(chunks_folder: str = "chunks") -> dict:
#     """
#     Load all chunk files from the chunks folder.
    
#     Args:
#         chunks_folder: Path to the chunks folder
        
#     Returns:
#         Dictionary with chunk filenames as keys and content as values
#     """
#     chunks = {}
#     chunks_path = Path(chunks_folder)
    
#     if not chunks_path.exists():
#         raise FileNotFoundError(f"Chunks folder not found: {chunks_folder}")
    
#     # Load all .txt files from chunks folder
#     for chunk_file in sorted(chunks_path.glob("chunk_*.txt")):
#         with open(chunk_file, 'r', encoding='utf-8') as f:
#             chunks[chunk_file.name] = f.read().strip()
    
#     print(f"Loaded {len(chunks)} chunks from {chunks_folder}")
#     return chunks

# def create_embeddings_for_chunks(chunks: dict) -> dict:
#     """
#     Create embeddings for all chunks.
    
#     Args:
#         chunks: Dictionary with chunk filenames as keys and content as values
        
#     Returns:
#         Dictionary with chunk filenames as keys and embeddings as values
#     """
#     embeddings_dict = {}
    
#     for filename, content in chunks.items():
#         print(f"Creating embedding for {filename}...")
#         embedding = embeddings.embed_query(content)
#         embeddings_dict[filename] = embedding
    
#     print(f"Created embeddings for {len(embeddings_dict)} chunks")
#     return embeddings_dict

# def save_embeddings(embeddings_dict: dict, output_file: str = "embeddings.json") -> None:
#     """
#     Save embeddings to a JSON file.
    
#     Args:
#         embeddings_dict: Dictionary with chunk filenames and their embeddings
#         output_file: Path to save the embeddings JSON file
#     """
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(embeddings_dict, f, indent=2)
#     print(f"Embeddings saved to {output_file}")

# def main():
#     """Main function to load chunks and create embeddings."""
#     try:
#         # Load chunks from folder
#         chunks = load_chunks_from_folder("chunks")
        
#         # Create embeddings
#         embeddings_dict = create_embeddings_for_chunks(chunks)
        
#         # Save embeddings to file
#         save_embeddings(embeddings_dict)
        
#         print("\nEmbedding process completed successfully!")
        
#     except Exception as e:
#         print(f"Error during embedding process: {e}")
#         raise

# if __name__ == "__main__":
#     main()
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
import json

# ------------------------------------------------------
#  KEEP YOUR API KEY EXACTLY AS PROVIDED
# ------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "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"

# ------------------------------------------------------
#  OPTION B — USE A 1024-DIM EMBEDDING MODEL
#  (text-embedding-3-small supports 1024 dimensions)
# ------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ------------------------------------------------------
# LOAD CHUNKS
# ------------------------------------------------------
def load_chunks_from_folder(chunks_folder: str = "chunks") -> list:
    """
    Load chunk files and return list of (filename, content)
    """
    chunks_path = Path(chunks_folder)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks folder not found: {chunks_folder}")

    chunk_list = []
    for chunk_file in sorted(chunks_path.glob("chunk_*.txt")):
        content = chunk_file.read_text(encoding="utf-8").strip()
        chunk_list.append((chunk_file.name, content))

    print(f"Loaded {len(chunk_list)} chunks from {chunks_folder}")
    return chunk_list

# ------------------------------------------------------
# BATCHED EMBEDDING GENERATION (Option B)
# ------------------------------------------------------
def create_embeddings_for_chunks_batched(chunk_list, batch_size=64):
    """
    Create 1024-dim embeddings in batches using embed_documents.
    """
    embeddings_dict = {}
    total = len(chunk_list)

    print(f"Creating embeddings for {total} chunks (batch size = {batch_size})")

    # Extract just the text list (keep filenames in parallel)
    filenames = [fn for fn, _ in chunk_list]
    contents  = [txt for _, txt in chunk_list]

    # batched loop
    for i in range(0, total, batch_size):
        batch_texts = contents[i:i+batch_size]
        batch_files = filenames[i:i+batch_size]

        print(f"  Embedding batch {i//batch_size + 1}: items {i}–{i + len(batch_texts) - 1}")

        # embed in batch
        vecs = embeddings.embed_documents(batch_texts)

        # store result into dict
        for fname, vec in zip(batch_files, vecs):
            embeddings_dict[fname] = vec

    print(f"Created {len(embeddings_dict)} embeddings.")
    return embeddings_dict

# ------------------------------------------------------
# SAVE EMBEDDINGS
# ------------------------------------------------------
def save_embeddings(embeddings_dict, output_file="embeddings_1024.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"Embeddings saved to {output_file}")

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    try:
        # Load chunks
        chunk_list = load_chunks_from_folder("chunks")

        # Create embeddings (1024-dim)
        embeddings_dict = create_embeddings_for_chunks_batched(chunk_list)

        # Save to JSON
        save_embeddings(embeddings_dict)

        print("\nEmbedding process completed successfully!")

    except Exception as e:
        print(f"Error during embedding process: {e}")
        raise

if __name__ == "__main__":
    main()
