from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

query = input("Ask a question: ")

query_embedding = model.encode(
    "query: " + query,
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

print("\nRelevant Documents:\n")

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
), 1):
    print(f"[{i}] Source: {meta['source']} | Type: {meta.get('source_type', 'N/A')} | Score: {1 - dist:.3f}")
    print(doc[:200])
    print()