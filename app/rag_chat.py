import os
from sentence_transformers import SentenceTransformer
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

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

DEBUG = False

def retrieve(query: str, n_results: int = 20):
    query_embedding = model.encode("query: " + query, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

print("Department of Computer Science – CUSAT AI Assistant")
print("Type 'exit' to quit.\n")

while True:
    query = input("\nAsk a question: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    results = retrieve(query)
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join(retrieved_docs)

    if DEBUG:
        print("\n--- Retrieved Chunks ---")
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), 1):
            print(f"[{i}] {meta.get('source')} | {meta.get('section','')}")
            print(doc[:300])
            print()

    prompt = f"""You are a friendly and helpful AI assistant for the Department of Computer Science at CUSAT (Cochin University of Science and Technology).

Answer the user's question using ONLY the information provided in the context below.

Guidelines:
- Answer in a natural, conversational and helpful tone — like a knowledgeable person, not a robot.
- Be concise but complete.
- When listing items, present them as a clean numbered list with one item per line.
- When counting, give the exact number first, then list the items.
- Do not invent or assume any information not present in the context.
- If the information is not available, say: "I'm sorry, I couldn't find that information in the department records."

Context:
{context}

Question:
{query}

Answer:"""

    answer = call_llm(prompt)
    sources = list(dict.fromkeys(m.get("source", "unknown") for m in retrieved_metas))

    print("\nAnswer:\n")
    print(answer)
    print(f"\n[Sources: {', '.join(sources)}]")