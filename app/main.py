import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM", "groq")
if LLM_PROVIDER == "groq":
    from llm.groq import call_llm
    print("Using LLM: Groq (llama-4-scout)")
else:
    from llm.ollama import call_llm
    print("Using LLM: Ollama (llama3)")

UPLOAD_PASSWORD = os.environ.get("UPLOAD_PASSWORD", "cusat@cs2024")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
chroma_client = chromadb.PersistentClient(path="../chroma_db")
collection = chroma_client.get_or_create_collection(name="department_docs")

DATA_PATH = "../data/raw/general"


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t: text += t + "\n"
    return text


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# IMPROVED CHUNKING (same as ingest.py for uploaded files)
# ============================================================

MAX_CHUNK_SIZE = 800
OVERLAP_SIZE = 150


def merge_short_lines(text):
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


def chunk_text(text, source):
    chunks = []
    text = merge_short_lines(text)
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        if text.strip():
            return [{"text": text.strip(), "source": source}]
        return []

    current_chunk = ""
    prev_chunk_tail = ""

    for para in paragraphs:
        if not para:
            continue
        if len(current_chunk) + len(para) >= MAX_CHUNK_SIZE and current_chunk.strip():
            chunk_text_final = current_chunk.strip()
            chunks.append({"text": chunk_text_final, "source": source})
            prev_chunk_tail = chunk_text_final[-OVERLAP_SIZE:] if len(chunk_text_final) > OVERLAP_SIZE else chunk_text_final
            current_chunk = prev_chunk_tail + "\n" + para + "\n"
        else:
            current_chunk += para + "\n"

    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "source": source})

    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source})
    return chunks


def ingest_file(file_path, source_name):
    if file_path.endswith(".pdf"): text = read_pdf(file_path)
    elif file_path.endswith(".txt"): text = read_txt(file_path)
    else: return 0
    chunks = chunk_text(text, source_name)
    if not chunks: return 0
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
    existing = collection.count()
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embeddings[i]],
            ids=[f"doc_{existing + i}"],
            metadatas=[{"source": chunk["source"], "source_type": "uploaded"}]
        )
    return len(chunks)


# ============================================================
# IMPROVED RETRIEVAL
# - Hybrid search: semantic + keyword matching
# - Multiple query variations for better coverage
# - Deduplication of results
# - Smarter ranking
# ============================================================

def retrieve(query, n_results=10):
    total_docs = collection.count()
    if total_docs == 0:
        return {"documents": [[]], "metadatas": [[]]}

    # --- Semantic search ---
    qe = embed_model.encode("query: " + query, normalize_embeddings=True).tolist()
    semantic_results = collection.query(
        query_embeddings=[qe],
        n_results=min(n_results * 3, total_docs),
        include=["documents", "metadatas", "distances"]
    )

    # --- Keyword search ---
    # Extract important keywords from query
    keywords = extract_keywords(query)
    keyword_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    if keywords:
        try:
            # Try keyword-based search using ChromaDB where filter
            keyword_results = collection.query(
                query_embeddings=[qe],
                n_results=min(n_results * 2, total_docs),
                where_document={"$contains": keywords[0]},
                include=["documents", "metadatas", "distances"]
            )
        except:
            pass

    # --- Merge and deduplicate ---
    seen_docs = set()
    final_docs = []
    final_metas = []

    # Add keyword matches first (higher priority for exact matches)
    for doc, meta in zip(keyword_results["documents"][0], keyword_results["metadatas"][0]):
        doc_hash = hash(doc[:200])
        if doc_hash not in seen_docs:
            seen_docs.add(doc_hash)
            final_docs.append(doc)
            final_metas.append(meta)

    # Then add semantic matches
    for doc, meta in zip(semantic_results["documents"][0], semantic_results["metadatas"][0]):
        doc_hash = hash(doc[:200])
        if doc_hash not in seen_docs:
            seen_docs.add(doc_hash)
            final_docs.append(doc)
            final_metas.append(meta)

    return {
        "documents": [final_docs[:n_results]],
        "metadatas": [final_metas[:n_results]]
    }


def extract_keywords(query):
    """Extract important keywords from query for keyword search"""
    # Remove common words
    stop_words = {
        "what", "who", "how", "when", "where", "is", "are", "the", "a", "an",
        "in", "of", "for", "to", "and", "or", "can", "do", "does", "did",
        "tell", "me", "about", "give", "list", "show", "details", "information",
        "info", "any", "some", "please", "could", "would", "should", "has",
        "have", "had", "be", "been", "being", "was", "were", "will", "shall",
        "this", "that", "these", "those", "it", "its", "i", "my", "your",
        "on", "at", "by", "with", "from", "up", "out", "there", "here"
    }

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Also add common synonyms/expansions
    synonym_map = {
        "hod": "head",
        "faculty": "professor",
        "staff": "officer",
        "courses": "programmes",
        "programs": "programmes",
        "labs": "lab",
        "phd": "ph.d",
        "founded": "established",
        "started": "established",
        "contact": "email",
        "phone": "email",
        "placed": "placement",
        "jobs": "placement",
    }

    expanded = []
    for k in keywords:
        expanded.append(k)
        if k in synonym_map:
            expanded.append(synonym_map[k])

    return expanded


def expand_query(query):
    """Expand query for better semantic search"""
    try:
        from groq import Groq
        gc = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        res = gc.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a search query optimizer. Return ONLY the expanded query, nothing else."},
                {"role": "user", "content": f"Expand for a university department database. Expand abbreviations (HoD=Head of Department, PhD=Doctor of Philosophy, CS=Computer Science). Keep under 25 words. Return ONLY the query.\n\nQuery: {query}"}
            ],
            max_tokens=60
        )
        expanded = res.choices[0].message.content.strip().strip('"')
        return expanded if expanded and len(expanded) <= 200 else query
    except:
        return query


# ============================================================
# STAFF ENDPOINTS
# ============================================================

class PasswordRequest(BaseModel):
    password: str

@app.post("/verify-password")
def verify_password(req: PasswordRequest):
    if req.password == UPLOAD_PASSWORD:
        return {"success": True}
    raise HTTPException(status_code=401, detail="Incorrect password.")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), password: str = Form(...)):
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password.")
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        return {"success": False, "message": "Only PDF and TXT files are supported."}
    os.makedirs(DATA_PATH, exist_ok=True)
    save_path = os.path.join(DATA_PATH, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks_added = ingest_file(save_path, file.filename)
    return {"success": True, "message": f"'{file.filename}' uploaded and indexed successfully. {chunks_added} chunks added.", "filename": file.filename, "chunks": chunks_added}

class DeleteRequest(BaseModel):
    filename: str
    password: str

@app.post("/delete")
def delete_file(req: DeleteRequest):
    if req.password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password.")
    file_path = os.path.join(DATA_PATH, req.filename)
    if not os.path.exists(file_path):
        return {"success": False, "message": "File not found."}
    os.remove(file_path)
    results = collection.get(where={"source": req.filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
    return {"success": True, "message": f"'{req.filename}' deleted successfully."}

@app.get("/documents")
def list_documents():
    docs = []
    if os.path.exists(DATA_PATH):
        for f in os.listdir(DATA_PATH):
            if f.endswith(".pdf") or f.endswith(".txt"):
                docs.append(f)
    return {"documents": docs}


# ============================================================
# CHAT ENDPOINT
# ============================================================

class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []

@app.post("/chat")
def chat(req: QueryRequest):
    query = req.question.strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": []}

    # Step 1: Expand query
    expanded = expand_query(query)

    # Step 2: Retrieve with both original and expanded queries
    results1 = retrieve(expanded, n_results=10)
    results2 = retrieve(query, n_results=5)

    # Merge results, deduplicate
    seen = set()
    all_docs = []
    all_metas = []

    for doc, meta in zip(results1["documents"][0], results1["metadatas"][0]):
        h = hash(doc[:200])
        if h not in seen:
            seen.add(h)
            all_docs.append(doc)
            all_metas.append(meta)

    for doc, meta in zip(results2["documents"][0], results2["metadatas"][0]):
        h = hash(doc[:200])
        if h not in seen:
            seen.add(h)
            all_docs.append(doc)
            all_metas.append(meta)

    # Take top 12 results
    retrieved_docs = all_docs[:12]
    retrieved_metas = all_metas[:12]

    context = "\n".join(retrieved_docs)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

    print(f"\n--- Query: {query}")
    print(f"--- Expanded: {expanded}")
    print(f"--- Retrieved {len(retrieved_docs)} chunks from {len(sources)} sources")

    history_str = ""
    if req.history:
        lines = []
        for msg in req.history[-4:]:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {msg.text}")
        history_str = "\n".join(lines)

    prompt = f"""You are a helpful assistant for the Department of Computer Science at CUSAT.

Context:
{context}
{"Previous conversation:" if history_str else ""}
{history_str}

Question: {query}

Rules:
- Answer only exactly what was asked. Nothing more.
- If the answer is not in the context, say "I don't have that information." and stop. Do not add anything else.
- Use previous conversation to understand follow-up questions like "his", "her", "they".
- Include full name, title and designation when answering about a person.
- Only mention contact details if the user asks how to contact the department.
- Never mention the context, documents, or knowledge base.
- For lists use a clean numbered format, one item per line.

Answer:"""

    answer = call_llm(prompt)
    return {"answer": answer, "sources": sources}