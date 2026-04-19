import os
import re
import shutil
import uuid
import sqlite3
from typing import Optional
from datetime import datetime
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


# ============================================================
# SQLite CHAT HISTORY
# ============================================================
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            title TEXT DEFAULT 'New Conversation'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT DEFAULT '',
            timestamp TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_conversation_history(conversation_id: str):
    if not conversation_id:
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conversation_id,)
    )
    history = [{"role": row[0], "text": row[1]} for row in cursor.fetchall()]
    conn.close()
    return history

def save_message(conversation_id: str, role: str, content: str, sources: str = ""):
    if not conversation_id:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, sources, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def create_new_conversation(title: str = "New Conversation"):
    conversation_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
        (conversation_id, datetime.now().isoformat(), title)
    )
    conn.commit()
    conn.close()
    return conversation_id

def update_conversation_title(conversation_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    conn.commit()
    conn.close()

def get_all_conversations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC LIMIT 20"
    )
    convos = [{"id": row[0], "title": row[1], "created_at": row[2]} for row in cursor.fetchall()]
    conn.close()
    return convos

def get_conversation_messages(conversation_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conversation_id,)
    )
    msgs = []
    for row in cursor.fetchall():
        sources = row[2].split(",") if row[2] else []
        msgs.append({"role": row[0], "text": row[1], "sources": sources})
    conn.close()
    return msgs

def delete_conversation(conversation_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


# ============================================================
# FILE READING
# ============================================================

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path, strict=False)
    except:
        reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                text += t + "\n"
        except:
            continue
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# CHUNKING
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
    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = read_txt(file_path)
    else:
        return 0
    chunks = chunk_text(text, source_name)
    if not chunks:
        return 0
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
# RETRIEVAL
# ============================================================

def extract_keywords(query):
    stop_words = {
        "what", "who", "how", "when", "where", "is", "are", "the", "a", "an",
        "in", "of", "for", "to", "and", "or", "can", "do", "does", "did",
        "tell", "me", "about", "give", "list", "show", "details", "information",
        "info", "any", "some", "please", "could", "would", "should", "has",
        "have", "had", "be", "been", "being", "was", "were", "will", "shall",
        "this", "that", "these", "those", "it", "its", "i", "my", "your",
        "on", "at", "by", "with", "from", "up", "out", "there", "here",
        "more", "also", "know", "you", "she", "he", "her", "his",
        "they", "them", "got", "get", "only", "thing", "many", "much"
    }
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    synonym_map = {
        "hod": "head", "faculty": "professor", "faculties": "professor",
        "staff": "officer", "courses": "programmes", "programs": "programmes",
        "labs": "lab", "phd": "ph.d", "phds": "ph.d",
        "founded": "established", "started": "established",
        "contact": "email", "phone": "email",
        "placed": "placement", "jobs": "placement",
        "fee": "tuition", "fees": "fee", "produced": "ph.d",
    }
    expanded = []
    for k in keywords:
        expanded.append(k)
        if k in synonym_map:
            expanded.append(synonym_map[k])
    return expanded

def retrieve(query, n_results=10):
    total_docs = collection.count()
    if total_docs == 0:
        return {"documents": [[]], "metadatas": [[]]}

    qe = embed_model.encode("query: " + query, normalize_embeddings=True).tolist()

    semantic_results = collection.query(
        query_embeddings=[qe],
        n_results=min(n_results * 3, total_docs),
        include=["documents", "metadatas", "distances"]
    )

    keywords = extract_keywords(query)
    keyword_results = {"documents": [[]], "metadatas": [[]]}
    if keywords:
        for kw in keywords[:3]:
            try:
                kw_results = collection.query(
                    query_embeddings=[qe],
                    n_results=min(n_results, total_docs),
                    where_document={"$contains": kw},
                    include=["documents", "metadatas", "distances"]
                )
                keyword_results["documents"][0].extend(kw_results["documents"][0])
                keyword_results["metadatas"][0].extend(kw_results["metadatas"][0])
            except:
                pass

    seen = set()
    final_docs = []
    final_metas = []

    for doc, meta in zip(keyword_results["documents"][0], keyword_results["metadatas"][0]):
        h = hash(doc[:200])
        if h not in seen:
            seen.add(h)
            final_docs.append(doc)
            final_metas.append(meta)

    for doc, meta in zip(semantic_results["documents"][0], semantic_results["metadatas"][0]):
        h = hash(doc[:200])
        if h not in seen:
            seen.add(h)
            final_docs.append(doc)
            final_metas.append(meta)

    return {
        "documents": [final_docs[:n_results]],
        "metadatas": [final_metas[:n_results]]
    }

def expand_query(query):
    try:
        from groq import Groq
        gc = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        res = gc.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a search query optimizer. Return ONLY the expanded query, nothing else."},
                {"role": "user", "content": f"Expand for a university department database. Expand abbreviations (HoD=Head of Department, PhD=Doctor of Philosophy, CS=Computer Science, faculty/faculties=department faculty members/professors). When someone says 'faculties' they mean teaching staff/professors of the department, NOT university faculties like Engineering or Science. Keep under 25 words. Return ONLY the query.\n\nQuery: {query}"}
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
# CONVERSATION HISTORY ENDPOINTS
# ============================================================

@app.get("/conversations")
def list_conversations():
    return {"conversations": get_all_conversations()}

class LoadConversationRequest(BaseModel):
    conversation_id: Optional[str] = None

@app.post("/conversation/load")
def load_conversation(req: LoadConversationRequest):
    if not req.conversation_id:
        return {"messages": [], "conversation_id": None}
    msgs = get_conversation_messages(req.conversation_id)
    return {"messages": msgs, "conversation_id": req.conversation_id}

@app.post("/conversation/delete")
def delete_conversation_endpoint(req: LoadConversationRequest):
    if req.conversation_id:
        delete_conversation(req.conversation_id)
    return {"success": True}


# ============================================================
# CHAT
# ============================================================

class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    conversation_id: Optional[str] = None

@app.post("/chat")
def chat(req: QueryRequest):
    query = req.question.strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": [], "conversation_id": req.conversation_id}

    # Create conversation if needed
    if not req.conversation_id:
        title = query[:50] + ("..." if len(query) > 50 else "")
        req.conversation_id = create_new_conversation(title)

    # Load history from SQLite
    db_history = get_conversation_history(req.conversation_id)

    history_str = ""
    if db_history:
        lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['text']}" for m in db_history[-6:]]
        history_str = "\n".join(lines)

    # Expand and retrieve
    expanded = expand_query(query)
    results1 = retrieve(expanded, n_results=10)
    results2 = retrieve(query, n_results=5)

    # Merge results
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

    retrieved_docs = all_docs[:12]
    retrieved_metas = all_metas[:12]
    context = "\n".join(retrieved_docs)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

    prompt = f"""You work at the Department of Computer Science, CUSAT. A visitor asks you a question. Answer naturally, like a knowledgeable staff member would — clearly, concisely, and helpfully.

Here is what you know:
---
{context}
---
{"Conversation so far:" if history_str else ""}
{history_str}

Visitor: {query}

Your answer (only use the information above, never make things up):"""

    answer = call_llm(prompt)

    # Save to SQLite
    save_message(req.conversation_id, "user", query)
    save_message(req.conversation_id, "assistant", answer, ",".join(sources))

    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": req.conversation_id
    }