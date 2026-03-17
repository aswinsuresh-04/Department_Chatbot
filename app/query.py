from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

query = input("Ask a question: ")

# BGE models work better with "query:" prefix
query_embedding = model.encode(
    "query: " + query,
    normalize_embeddings=True
).tolist()

# Retrieve relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nRelevant Documents:\n")

for doc in results["documents"][0]:
    print("-", doc)