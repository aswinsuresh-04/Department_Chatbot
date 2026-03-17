import os
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# -----------------------------
# Connect to ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path="../chroma_db")

# Reset collection
try:
    client.delete_collection(name="department_docs")
except:
    pass

collection = client.get_or_create_collection(name="department_docs")

DATA_PATH = "../data/raw"

# -----------------------------
# Functions to read documents
# -----------------------------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Scan folders recursively
# -----------------------------
documents = []

for root, dirs, files in os.walk(DATA_PATH):

    for file in files:

        file_path = os.path.join(root, file)

        if file.endswith(".txt"):
            text = read_txt(file_path)

        elif file.endswith(".pdf"):
            text = read_pdf(file_path)

        else:
            continue

        documents.append({
            "text": text,
            "source": file
        })


# -----------------------------
# Chunking
# -----------------------------
chunks = []

MAX_CHUNK_SIZE = 500

for doc in documents:

    paragraphs = doc["text"].split("\n\n")

    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:

            current_chunk += para + "\n"

        else:

            chunks.append({
                "text": current_chunk.strip(),
                "source": doc["source"]
            })

            current_chunk = para

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "source": doc["source"]
        })


# -----------------------------
# Create embeddings
# -----------------------------
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
).tolist()


# -----------------------------
# Store in ChromaDB
# -----------------------------
for i, chunk in enumerate(chunks):

    collection.add(
        documents=[chunk["text"]],
        embeddings=[embeddings[i]],
        ids=[f"doc_{i}"],
        metadatas=[{
            "source": chunk["source"]
        }]
    )


print("All documents successfully indexed.")
print("Total chunks stored:", len(chunks))