import re
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# Load model and DB
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

PEOPLE_KEYWORDS = {"professor", "professors", "faculty", "faculties", "staff", "hod", "head", "lecturer", "associate", "assistant"}

# -----------------------------
# Retrieve
# -----------------------------
def retrieve(query: str, n_results: int = 20):
    query_embedding = model.encode("query: " + query, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

# -----------------------------
# Call Ollama
# -----------------------------
def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# -----------------------------
# Extract list
# -----------------------------
def extract_list_from_context(context: str, topic_keywords: list) -> list:
    items = []
    seen = set()
    section_header = re.compile(r'^(\d{1,2}\.|#{1,3})\s+[A-Z][a-zA-Z\s]{2,40}$')
    is_people_query = any(kw.lower() in PEOPLE_KEYWORDS for kw in topic_keywords)

    for line in context.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        if section_header.match(line):
            continue
        if line.endswith(":"):
            continue

        cleaned = re.sub(r'^[\-\*\•\d\.\)\s]+', '', line).strip()
        if not cleaned or len(cleaned) < 5:
            continue

        if is_people_query:
            if not re.search(r'\b(Dr\.|Prof\.)\s+\w+', cleaned):
                continue
            rank_keywords = [kw for kw in topic_keywords if kw.lower() not in {"faculty", "faculties", "staff"}]
            if rank_keywords:
                if not all(kw.lower() in cleaned.lower() for kw in rank_keywords):
                    continue
        else:
            if not any(kw.lower() in cleaned.lower() for kw in topic_keywords):
                continue

        if cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)

    return items

# -----------------------------
# Detect intent
# -----------------------------
def detect_intent(query: str):
    q = query.lower()
    count_triggers = ["how many", "count", "number of", "total"]
    list_triggers = ["list", "all", "what are", "who are", "name", "there", "including"]
    is_count = any(t in q for t in count_triggers)
    is_list = any(t in q for t in list_triggers)

    stop_words = {
        "how", "many", "are", "there", "what", "who", "is", "the", "a", "an",
        "all", "of", "in", "and", "or", "list", "name", "give", "me", "tell",
        "including", "number", "total", "count", "please", "department", "its"
    }
    topic_keywords = [w for w in re.findall(r'\b\w+\b', q) if w not in stop_words and len(w) > 2]

    if is_count:
        return "count", topic_keywords
    if is_list:
        return "list", topic_keywords
    return "general", topic_keywords

# -----------------------------
# API endpoint
# -----------------------------
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: QueryRequest):
    query = req.question.strip()
    if not query:
        return {"answer": "Please ask a question.", "sources": []}

    results = retrieve(query)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join(retrieved_docs)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

    intent, topic_keywords = detect_intent(query)

    if intent in ("list", "count"):
        extracted = extract_list_from_context(context, topic_keywords)
        if extracted:
            numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(extracted, 1))
            topic_label = " ".join(topic_keywords).title()
            answer = f"Total {topic_label} found: {len(extracted)}\n\n{numbered}"
            return {"answer": answer, "sources": sources}
        intent = "general"

    prompt = f"""You are the official AI assistant for the Department of Computer Science at CUSAT.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Base your answer strictly on the context.
- Do not invent information not present in the context.
- If information is not available, say: "I could not find that information in the department records."
- Keep answers clear and professional.

Context:
{context}

Question:
{query}

Answer:"""

    answer = call_llm(prompt)
    return {"answer": answer, "sources": sources}