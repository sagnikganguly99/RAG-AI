
import os
import glob
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Configuration
DATA_DIR = "data"
CHUNKS_DIR = "chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure chunks directory exists
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Load all CSV files
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
all_text = ""
for file in csv_files:
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            all_text += " ".join(str(cell) for cell in row if cell) + "\n"

# Load all PDF files
pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    for doc in documents:
        all_text += doc.page_content + "\n"
splitter = RecursiveCharacterTextSplitter(
	chunk_size=CHUNK_SIZE,
	chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_text(all_text)

# Save chunks to folder
for i, chunk in enumerate(chunks):
	with open(os.path.join(CHUNKS_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
		f.write(chunk)

print(f"Chunking complete! {len(chunks)} chunks saved to {CHUNKS_DIR}/")
