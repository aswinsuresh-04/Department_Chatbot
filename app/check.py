import chromadb

client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

print(f"Total chunks in database: {collection.count()}")

results = collection.get(include=["metadatas"])
sources = {}
for meta in results["metadatas"]:
    src = meta.get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1

print("\nChunks per file:")
for src, count in sources.items():
    print(f"  {src}: {count} chunks")