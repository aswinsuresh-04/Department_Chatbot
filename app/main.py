import os
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
    print("Using LLM: Groq (llama-3.3-70b)")
else:
    from llm.ollama import call_llm
    print("Using LLM: Ollama (llama3)")

UPLOAD_PASSWORD = os.environ.get("UPLOAD_PASSWORD", "cusat@cs2024")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
chroma_client = chromadb.PersistentClient(path="../chroma_db")
collection = chroma_client.get_or_create_collection(name="department_docs")

DATA_PATH = "../data/raw/general"

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, source: str):
    chunks = []
    MAX_CHUNK_SIZE = 900
    paragraphs = text.split("\n\n")
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append({"text": current_chunk.strip(), "source": source})
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "source": source})
    return chunks

def ingest_file(file_path: str, source_name: str):
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

def retrieve(query: str, n_results: int = 20):
    query_embedding = embed_model.encode("query: " + query, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

# -----------------------------
# Prompt expansion
# Rewrites the user query into a richer search query
# for better ChromaDB retrieval
# -----------------------------
def expand_query(query: str) -> str:
    # Use Groq directly with a separate minimal call for expansion
    # so it doesn't go through the detailed system message
    try:
        from groq import Groq
        import os
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a search query optimizer. Return ONLY the expanded query, nothing else. No explanations."
                },
                {
                    "role": "user",
                    "content": f"""Expand this short query into a detailed search query for a university department database.
Expand abbreviations (HoD = Head of Department, MSc = Master of Science, etc).
Keep under 25 words. Return ONLY the expanded query.

Query: {query}"""
                }
            ],
            max_tokens=60
        )
        expanded = response.choices[0].message.content.strip().strip('"')
        if len(expanded) > 200 or not expanded:
            return query
        return expanded
    except:
        return query

# -----------------------------
# Verify staff password
# -----------------------------
class PasswordRequest(BaseModel):
    password: str

@app.post("/verify-password")
def verify_password(req: PasswordRequest):
    if req.password == UPLOAD_PASSWORD:
        return {"success": True}
    raise HTTPException(status_code=401, detail="Incorrect password.")

# -----------------------------
# Upload — password protected
# -----------------------------
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
    return {
        "success": True,
        "message": f"'{file.filename}' uploaded and indexed successfully. {chunks_added} chunks added.",
        "filename": file.filename,
        "chunks": chunks_added
    }

# -----------------------------
# Delete — password protected
# -----------------------------
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
    # Remove from disk
    os.remove(file_path)
    # Remove from ChromaDB
    results = collection.get(where={"source": req.filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
    return {"success": True, "message": f"'{req.filename}' deleted successfully."}

# -----------------------------
# List documents
# -----------------------------
@app.get("/documents")
def list_documents():
    docs = []
    if os.path.exists(DATA_PATH):
        for f in os.listdir(DATA_PATH):
            if f.endswith(".pdf") or f.endswith(".txt"):
                docs.append(f)
    return {"documents": docs}

# -----------------------------
# Chat
# -----------------------------
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: QueryRequest):
    query = req.question.strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": []}

    # Expand query for better retrieval
    expanded = expand_query(query)
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")

    results = retrieve(expanded)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join(retrieved_docs)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))
    prompt = f"""You are a knowledgeable and friendly assistant for the Department of Computer Science at CUSAT (Cochin University of Science and Technology). You help students, parents, and visitors learn about the department.

Use the context below to answer the question. Respond like a helpful person — natural, warm, and informative.

Rules:
- Give complete, expressive and detailed answers using ALL relevant information from the context. Never give a one-line answer when more detail is available — always elaborate with role, responsibilities, contact info, or any other relevant details present in the context.
- Never mention the context, documents, file system, or knowledge base in your answer. Just answer naturally.
- Never say things like "in the context" or "based on the provided information" or "there is no information in the context".
- Never start your answer by introducing the department. Get straight to the point.
- When listing, use a clean numbered list with one item per line.
- Do not invent facts not present in the context.
- If something is genuinely not available, just say: "I don't have that specific detail right now. For accurate information, please reach out to the department at csdir@cusat.ac.in or +91 484 2862301."

Context:
{context}

Question:
{query}

Important: Answer naturally and proportionally — short questions deserve concise answers, detailed questions deserve detailed answers. Never invent or assume information not present in the context.

Answer:"""
    answer = call_llm(prompt)
    return {"answer": answer, "sources": sources}