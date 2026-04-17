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

def chunk_text(text, source):
    chunks = []
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")
    if not paragraphs or all(len(p.strip()) == 0 for p in paragraphs):
        if text.strip():
            return [{"text": text.strip(), "source": source}]
        return []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(current_chunk) + len(para) < 900:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append({"text": current_chunk.strip(), "source": source})
            current_chunk = para + "\n"
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

def retrieve(query, n_results=7):
    qe = embed_model.encode("query: " + query, normalize_embeddings=True).tolist()

    # Get more results to ensure general/faculty info is included
    results = collection.query(
        query_embeddings=[qe],
        n_results=min(n_results * 2, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Separate general and other chunks
    general_docs, general_metas = [], []
    other_docs, other_metas = [], []

    for doc, meta, dist in zip(docs, metas, distances):
        if meta.get("source_type") == "general" or meta.get("source_type") == "uploaded":
            general_docs.append(doc)
            general_metas.append(meta)
        else:
            other_docs.append(doc)
            other_metas.append(meta)

    # Always include general chunks first, then fill with others up to n_results
    final_docs = general_docs + other_docs
    final_metas = general_metas + other_metas

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
                {"role": "user", "content": f"Expand for a university department database. Expand abbreviations. Keep under 25 words. Return ONLY the query.\n\nQuery: {query}"}
            ],
            max_tokens=60
        )
        expanded = res.choices[0].message.content.strip().strip('"')
        return expanded if expanded and len(expanded) <= 200 else query
    except:
        return query

# Staff endpoints
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

# Chat
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

    expanded = expand_query(query)
    results = retrieve(expanded)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join(retrieved_docs)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

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