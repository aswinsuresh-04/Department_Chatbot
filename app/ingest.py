import os
import re
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
# Detect source type from folder path
# -----------------------------
def get_source_type(file_path):
    path_lower = file_path.lower()
    if "faculty" in path_lower:
        return "faculty"
    elif "research" in path_lower:
        return "research"
    elif "placement" in path_lower:
        return "placements"
    elif "student" in path_lower:
        return "students"
    elif "general" in path_lower:
        return "general"
    else:
        return "general"


# -----------------------------
# Extract section title from text
# Detects lines like "6. Faculty" or "## Faculty"
# -----------------------------
def extract_section_title(text):
    lines = text.strip().split("\n")
    for line in lines[:3]:
        line = line.strip()
        if re.match(r"^(\d+\.|\#{1,3})\s+\w+", line):
            return line
    return ""


# -----------------------------
# Smart chunking:
# - Keeps numbered sections together
# - Keeps ALL faculty lines in one chunk per section
# - Avoids oversized chunks
# -----------------------------
MAX_CHUNK_SIZE = 900

def smart_chunk(text, source, source_type):
    chunks = []

    # Split by numbered section headers like "1. Basic Info" or "## Faculty"
    section_pattern = re.compile(r'(?=\n(?:\d+\.|#{1,3})\s+\w+)', re.MULTILINE)
    sections = section_pattern.split(text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_title = extract_section_title(section)
        lines = section.split("\n")

        # Detect if this section contains faculty entries (lines with Dr. or titles)
        faculty_lines = [l for l in lines if re.search(r'\bDr\.\b|\bProf\.\b', l)]
        is_faculty_section = len(faculty_lines) >= 2

        if is_faculty_section:
            # Keep ALL faculty lines together in one chunk
            faculty_block = "\n".join(
                l for l in lines if l.strip()
            )
            # If still too big, split every N faculty entries
            if len(faculty_block) > MAX_CHUNK_SIZE * 2:
                group = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    group.append(line)
                    if len("\n".join(group)) >= MAX_CHUNK_SIZE:
                        chunks.append({
                            "text": "\n".join(group),
                            "source": source,
                            "source_type": source_type,
                            "section": section_title
                        })
                        group = []
                if group:
                    chunks.append({
                        "text": "\n".join(group),
                        "source": source,
                        "source_type": source_type,
                        "section": section_title
                    })
            else:
                chunks.append({
                    "text": faculty_block,
                    "source": source,
                    "source_type": source_type,
                    "section": section_title
                })

        else:
            # Normal chunking: accumulate paragraphs up to MAX_CHUNK_SIZE
            paragraphs = section.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(current_chunk) + len(para) + 1 < MAX_CHUNK_SIZE:
                    current_chunk += para + "\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": source,
                            "source_type": source_type,
                            "section": section_title
                        })
                    current_chunk = para + "\n"

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "source_type": source_type,
                    "section": section_title
                })

    return chunks


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
            "source": file,
            "source_type": get_source_type(file_path)
        })


# -----------------------------
# Chunk all documents
# -----------------------------
all_chunks = []

for doc in documents:
    doc_chunks = smart_chunk(doc["text"], doc["source"], doc["source_type"])
    all_chunks.extend(doc_chunks)


# -----------------------------
# Create embeddings
# BGE requires "query: " prefix for queries,
# but plain text (no prefix) for documents during indexing.
# -----------------------------
texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True
).tolist()


# -----------------------------
# Store in ChromaDB with richer metadata
# -----------------------------
for i, chunk in enumerate(all_chunks):
    collection.add(
        documents=[chunk["text"]],
        embeddings=[embeddings[i]],
        ids=[f"doc_{i}"],
        metadatas=[{
            "source": chunk["source"],
            "source_type": chunk["source_type"],
            "section": chunk.get("section", "")
        }]
    )

print("\nAll documents successfully indexed.")
print(f"Total chunks stored: {len(all_chunks)}")

# Print a summary of chunks per source type
from collections import Counter
type_counts = Counter(c["source_type"] for c in all_chunks)
for stype, count in type_counts.items():
    print(f"  [{stype}]: {count} chunks")