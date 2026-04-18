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


# ============================================================
# IMPROVED CHUNKING
# - Merges short lines (names, dates, titles) into coherent blocks
# - Preserves table rows together (Name + Date + Title + Guide)
# - Adds overlap between chunks so context isn't lost at boundaries
# - Keeps section headers with their content
# ============================================================

MAX_CHUNK_SIZE = 800
OVERLAP_SIZE = 150


def merge_short_lines(text):
    """Merge very short lines that belong together (table rows, lists)"""
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


def detect_section_header(line):
    """Check if a line is a section header"""
    headers = [
        "professor & head", "senior professor", "professor", "associate professor",
        "assistant professor", "faculty", "people", "research", "programmes",
        "placement", "alumni", "facilities", "labs", "library", "computer center",
        "admission", "ph.d", "phd", "lecture series", "e-learning", "about",
        "department overview", "leadership", "contact", "core areas",
        "vision", "mission", "news", "events", "non-teaching"
    ]
    lower = line.strip().lower()
    return any(lower.startswith(h) or lower == h for h in headers)


def chunk_text_improved(text, source, source_type):
    """Smart chunking with overlap and context preservation"""
    chunks = []

    # Step 1: Merge short lines
    text = merge_short_lines(text)

    # Step 2: Split into paragraphs
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")

    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        if text.strip():
            return [{"text": text.strip(), "source": source, "source_type": source_type}]
        return []

    # Step 3: Build chunks with overlap
    current_chunk = ""
    prev_chunk_tail = ""

    for para in paragraphs:
        if not para:
            continue

        if len(current_chunk) + len(para) >= MAX_CHUNK_SIZE and current_chunk.strip():
            chunk_text_final = current_chunk.strip()
            chunks.append({
                "text": chunk_text_final,
                "source": source,
                "source_type": source_type
            })

            prev_chunk_tail = chunk_text_final[-OVERLAP_SIZE:] if len(chunk_text_final) > OVERLAP_SIZE else chunk_text_final
            current_chunk = prev_chunk_tail + "\n" + para + "\n"
        else:
            current_chunk += para + "\n"

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "source": source,
            "source_type": source_type
        })

    # Step 4: Add section-aware summary chunks for key sections
    section_chunks = create_section_summaries(text, source, source_type)
    chunks.extend(section_chunks)

    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source, "source_type": source_type})

    return chunks


def create_section_summaries(text, source, source_type):
    """Create dedicated chunks for important sections so retrieval always finds them"""
    summary_chunks = []
    lower_text = text.lower()

    # Faculty section
    if "professor & head" in lower_text or "professor &amp; head" in lower_text:
        start = lower_text.find("professor")
        if start >= 0:
            end = start + 2000
            faculty_block = text[start:min(end, len(text))]
            summary_chunks.append({
                "text": "Faculty Members of the Department of Computer Science:\n" + faculty_block.strip(),
                "source": source,
                "source_type": source_type
            })

    # PhD produced section
    if "ph.d" in lower_text and "produced" in lower_text:
        start = lower_text.find("ph.d")
        while start >= 0:
            nearby = lower_text[start:start+100]
            if "produced" in nearby:
                end = start + 3000
                phd_block = text[start:min(end, len(text))]
                summary_chunks.append({
                    "text": "PhD Scholars Produced by the Department:\n" + phd_block.strip(),
                    "source": source,
                    "source_type": source_type
                })
                break
            start = lower_text.find("ph.d", start + 1)

    # Research areas
    if "core areas of research" in lower_text:
        start = lower_text.find("core areas of research")
        end = start + 1500
        research_block = text[start:min(end, len(text))]
        summary_chunks.append({
            "text": "Research Areas of the Department:\n" + research_block.strip(),
            "source": source,
            "source_type": source_type
        })

    # Placements
    if "placement" in lower_text:
        start = lower_text.find("placement")
        end = start + 2000
        placement_block = text[start:min(end, len(text))]
        summary_chunks.append({
            "text": "Placement Information:\n" + placement_block.strip(),
            "source": source,
            "source_type": source_type
        })

    # Labs/Facilities
    if "lab" in lower_text and ("dcs has" in lower_text or "facilities" in lower_text):
        for keyword in ["dcs has", "facilities"]:
            idx = lower_text.find(keyword)
            if idx >= 0:
                end = idx + 2000
                lab_block = text[idx:min(end, len(text))]
                summary_chunks.append({
                    "text": "Department Labs and Facilities:\n" + lab_block.strip(),
                    "source": source,
                    "source_type": source_type
                })
                break

    # Non-teaching staff
    if "non-teaching" in lower_text or "section officer" in lower_text:
        for keyword in ["non-teaching", "section officer"]:
            idx = lower_text.find(keyword)
            if idx >= 0:
                start = max(0, idx - 50)
                end = idx + 1500
                staff_block = text[start:min(end, len(text))]
                summary_chunks.append({
                    "text": "Non-Teaching Staff of the Department:\n" + staff_block.strip(),
                    "source": source,
                    "source_type": source_type
                })
                break

    return summary_chunks


# ============================================================
# MAIN INGESTION
# ============================================================

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
    all_chunks.extend(chunk_text_improved(doc["text"], doc["source"], doc["source_type"]))

print(f"Total chunks: {len(all_chunks)}")

sizes = [len(c["text"]) for c in all_chunks]
print(f"Chunk sizes - min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")

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