import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from collections import Counter

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(path="../chroma_db")
try:
    client.delete_collection(name="department_docs")
except:
    pass
collection = client.get_or_create_collection(name="department_docs")

DATA_PATH = "../data/raw/general"


def read_pdf(file_path):
    filename = os.path.basename(file_path)
    text = ""
    try:
        reader = PdfReader(file_path, strict=False)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except:
                continue
        if text.strip():
            return text
    except Exception as e:
        print(f"   Failed to read {filename}: {e}")
    return ""


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_source_type(file_path):
    path_lower = file_path.lower()
    if "faculty" in path_lower: return "faculty"
    elif "research" in path_lower: return "research"
    elif "placement" in path_lower: return "placements"
    elif "student" in path_lower: return "students"
    else: return "general"


MAX_CHUNK_SIZE = 800
OVERLAP_SIZE = 150


def merge_short_lines(text):
    """Merge short lines that belong together (table rows, names, dates)"""
    lines = text.split("\n")
    merged = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")
            continue
        if len(stripped) < 60 and buffer and len(buffer) < 400:
            buffer += " | " + stripped
        elif buffer and len(buffer) + len(stripped) < 400 and not any(
            stripped.startswith(h) for h in ["Professor", "Senior Professor", "Associate", "Assistant", "Head"]
        ):
            buffer += " | " + stripped
        else:
            if buffer:
                merged.append(buffer)
            buffer = stripped
    if buffer:
        merged.append(buffer)
    return "\n".join(merged)


def chunk_text(text, source, source_type):
    """Chunk text with overlap for better retrieval"""
    chunks = []
    text = merge_short_lines(text)

    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        if text.strip():
            return [{"text": text.strip(), "source": source, "source_type": source_type}]
        return []

    current_chunk = ""
    prev_chunk_tail = ""

    for para in paragraphs:
        if not para:
            continue
        if len(current_chunk) + len(para) >= MAX_CHUNK_SIZE and current_chunk.strip():
            chunk_text_final = current_chunk.strip()
            chunks.append({"text": chunk_text_final, "source": source, "source_type": source_type})
            prev_chunk_tail = chunk_text_final[-OVERLAP_SIZE:] if len(chunk_text_final) > OVERLAP_SIZE else chunk_text_final
            current_chunk = prev_chunk_tail + "\n" + para + "\n"
        else:
            current_chunk += para + "\n"

    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "source": source, "source_type": source_type})

    # Add section summary chunks for key topics
    section_chunks = create_section_summaries(text, source, source_type)
    chunks.extend(section_chunks)

    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source, "source_type": source_type})

    return chunks


def create_section_summaries(text, source, source_type):
    """Create dedicated chunks for important sections"""
    summary_chunks = []
    lower = text.lower()

    sections_to_extract = [
        ("professor & head", "Faculty Members of the Department", 2000),
        ("ph.d", "PhD Scholars Produced by the Department", 3000),
        ("core areas of research", "Research Areas of the Department", 1500),
        ("placement", "Placement Information", 2000),
        ("non-teaching", "Non-Teaching Staff of the Department", 1500),
        ("fee structure", "Fee Structure Information", 3000),
        ("tuition fee", "Fee Structure Information", 3000),
    ]

    for keyword, prefix, length in sections_to_extract:
        idx = lower.find(keyword)
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(text), idx + length)
            block = text[start:end].strip()
            if block:
                summary_chunks.append({
                    "text": prefix + ":\n" + block,
                    "source": source,
                    "source_type": source_type
                })

    return summary_chunks


# === MAIN ===
print(f"Scanning: {DATA_PATH}\n")

documents = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in sorted(files):
        file_path = os.path.join(root, file)
        text = None
        if file.endswith(".pdf"):
            text = read_pdf(file_path)
        elif file.endswith(".txt") or file.endswith(".md"):
            text = read_txt(file_path)
        else:
            continue
        if text and text.strip():
            documents.append({
                "text": text,
                "source": file,
                "source_type": get_source_type(file_path)
            })
            print(f"  + {file} ({len(text)} chars)")
        else:
            print(f"  - {file} (no text extracted)")

print(f"\nFound {len(documents)} documents")

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

print(f"\nIndexed {len(all_chunks)} chunks successfully.")
for stype, count in Counter(c["source_type"] for c in all_chunks).items():
    print(f"  [{stype}]: {count} chunks")