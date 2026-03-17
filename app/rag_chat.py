from sentence_transformers import SentenceTransformer
import chromadb
import requests

# Load embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

# Turn this to True if you want to see retrieved chunks
DEBUG = False


while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Convert query to embedding
    query_embedding = model.encode(
        "query: " + query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    retrieved_docs = results["documents"][0]
    context = "\n".join(retrieved_docs)

    # Debug mode to inspect retrieved chunks
    if DEBUG:
        print("\n--- Retrieved Context ---\n")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"[Chunk {i}]")
            print(doc)
            print()

    # General RAG prompt
    prompt = f"""
You are the official AI assistant for the Department of Computer Science at CUSAT.

Answer the user's question using ONLY the information provided in the context.

Guidelines:
- Base your answer strictly on the context.
- Do not invent information.
- If the information is not present in the context, say:
"I could not find that information in the department records."
- Provide clear and professional answers suitable for a university website.
- When listing items such as faculty members, laboratories, programs, or facilities,
  present them in a clean numbered or bullet list.
- Avoid unnecessary repetition.

Context:
{context}

Question:
{query}

Answer:
"""

    # Send prompt to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    print("\nAnswer:\n")
    print(answer)