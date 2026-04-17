import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(path="../chroma_db")
try:
    client.delete_collection(name="department_docs")
except:
    pass
collection = client.get_or_create_collection(name="department_docs")

DATA_PATH = "../data/raw/general"

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_md(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def get_source_type(file_path):
    path_lower = file_path.lower()
    if "faculty" in path_lower: return "faculty"
    elif "research" in path_lower: return "research"
    elif "placement" in path_lower: return "placements"
    elif "student" in path_lower: return "students"
    elif "pages" in path_lower: return "pages"
    elif "pdfs" in path_lower: return "pdfs"
    elif "events" in path_lower: return "events"
    elif "general" in path_lower: return "general"
    else: return "general"

MAX_CHUNK_SIZE = 900

def chunk_text(text, source, source_type):
    chunks = []
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")
    if not paragraphs or all(len(p.strip()) == 0 for p in paragraphs):
        if text.strip():
            return [{"text": text.strip(), "source": source, "source_type": source_type}]
        return []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append({"text": current_chunk.strip(), "source": source, "source_type": source_type})
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "source": source, "source_type": source_type})
    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source, "source_type": source_type})
    return chunks

documents = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        file_path = os.path.join(root, file)
        text = None
        if file.endswith(".txt"):
            text = read_txt(file_path)
        elif file.endswith(".pdf"):
            text = read_pdf(file_path)
        elif file.endswith(".md"):
            text = read_md(file_path)
        else:
            continue
        if text and text.strip():
            documents.append({
                "text": text,
                "source": file,
                "source_type": get_source_type(file_path)
            })

print(f"Found {len(documents)} documents")

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc["text"], doc["source"], doc["source_type"]))

print(f"Total chunks: {len(all_chunks)}")

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True).tolist()

for i, chunk in enumerate(all_chunks):
    collection.add(
        documents=[chunk["text"]],
        embeddings=[embeddings[i]],
        ids=[f"doc_{i}"],
        metadatas=[{"source": chunk["source"], "source_type": chunk["source_type"]}]
    )

print(f"\nAll documents successfully indexed.")
print(f"Total chunks stored: {len(all_chunks)}")

from collections import Counter
for stype, count in Counter(c["source_type"] for c in all_chunks).items():
    print(f"  [{stype}]: {count} chunks")