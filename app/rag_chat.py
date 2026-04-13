import re
import requests
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

DEBUG = False


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve(query: str, n_results: int = 20):
    query_embedding = model.encode(
        "query: " + query,
        normalize_embeddings=True
    ).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )


# -----------------------------
# Call Ollama LLM
# -----------------------------
def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# -----------------------------
# Extract items from context that are relevant to the query topic.
# Strategy:
# - Find lines that contain ALL or MOST of the topic keywords
# - Only accept lines that look like actual data entries (not section headers)
# - Deduplicate across chunks/sources
# -----------------------------
def extract_list_from_context(context: str, topic_keywords: list[str]) -> list[str]:
    items = []
    seen = set()

    # Section header: short line like "6. Faculty" or "## Research Areas" — always skip
    section_header = re.compile(r'^(\d{1,2}\.|#{1,3})\s+[A-Z][a-zA-Z\s]{2,40}$')

    for line in context.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue

        if section_header.match(line):
            continue

        # Clean leading bullets/numbers to get the actual content
        cleaned = re.sub(r'^[\-\*\•\d\.\)\s]+', '', line).strip()
        if not cleaned or len(cleaned) < 5:
            continue

        # Count how many topic keywords appear in this line
        matches = sum(1 for kw in topic_keywords if kw.lower() in cleaned.lower())

        # Accept line only if at least ONE keyword matches
        # For multi-keyword queries (e.g. "assistant professor"), require ALL keywords to match
        required = len(topic_keywords) if len(topic_keywords) <= 3 else max(2, len(topic_keywords) // 2)
        if matches < min(required, len(topic_keywords)):
            continue

        if cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)

    return items


# -----------------------------
# Detect query intent: count, list, or general
# Also extracts topic keywords for list extraction
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
        "including", "number", "total", "count", "please", "department"
    }
    topic_keywords = [
        w for w in re.findall(r'\b\w+\b', q)
        if w not in stop_words and len(w) > 2
    ]

    if is_count:
        return "count", topic_keywords
    if is_list:
        return "list", topic_keywords
    return "general", topic_keywords


# -----------------------------
# Main chat loop
# -----------------------------
print("Department of Computer Science – CUSAT AI Assistant")
print("Type 'exit' to quit.\n")

while True:

    query = input("\nAsk a question: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    # Step 1: Retrieve
    results = retrieve(query, n_results=20)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join(retrieved_docs)

    if DEBUG:
        print("\n--- Retrieved Chunks ---")
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), 1):
            print(f"[{i}] {meta.get('source')} | {meta.get('section','')}")
            print(doc[:300])
            print()

    # Step 2: Detect intent
    intent, topic_keywords = detect_intent(query)

    # Step 3: For list/count queries — Python extracts, counts, and formats the answer.
    # The LLM is NOT involved in counting or listing. This eliminates counting errors entirely.
    if intent in ("list", "count"):
        extracted = extract_list_from_context(context, topic_keywords)

        if extracted:
            numbered = "\n".join(f"  {i}. {item}" for i, item in enumerate(extracted, 1))
            topic_label = " ".join(topic_keywords).title()
            print("\nAnswer:\n")
            print(f"Total {topic_label} found: {len(extracted)}\n")
            print(numbered)
            sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))
            print(f"\n[Sources: {', '.join(sources)}]")
            continue  # skip LLM entirely

        # Nothing extracted — fall through to LLM
        intent = "general"

    # Step 4: General queries — use LLM with RAG prompt
    prompt = f"""You are the official AI assistant for the Department of Computer Science at CUSAT (Cochin University of Science and Technology).

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

    # Step 5: Call LLM
    answer = call_llm(prompt)

    # Step 6: Print answer and sources
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

    print("\nAnswer:\n")
    print(answer)
    print(f"\n[Sources: {', '.join(sources)}]")