import chromadb

client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="department_docs")

print("🔍 CHATBOT DATABASE CHECK (Upload Diagnosis)\n")
print(f"Total chunks in database : {collection.count():,}\n")

# Get all metadata
results = collection.get(include=["metadatas"])

# Group by source_type and source filename
from collections import defaultdict
by_type = defaultdict(lambda: defaultdict(int))
all_sources = set()

for meta in results["metadatas"]:
    source = meta.get("source", "unknown")
    source_type = meta.get("source_type", "unknown")
    by_type[source_type][source] += 1
    all_sources.add(source)

# === SUMMARY ===
print("📊 FILES BY TYPE:")
for stype in sorted(by_type.keys()):
    files = by_type[stype]
    total = sum(files.values())
    print(f"  • {stype.upper():12} → {total:3} chunks from {len(files)} file(s)")

print("\n📋 ALL FILES CURRENTLY IN DATABASE:")
for src in sorted(all_sources):
    # Find which type it belongs to
    for stype, files in by_type.items():
        if src in files:
            print(f"     {src:40} → {files[src]:3} chunks  [{stype}]")
            break

# === SPECIFIC CHECK FOR UPLOADED FILES ===
print("\n🔥 UPLOADED FILES (via Staff Panel):")
uploaded = by_type.get("uploaded", {})
if uploaded:
    for file, count in sorted(uploaded.items()):
        print(f"   ✅ FOUND : {file} → {count} chunks")
else:
    print("   ❌ No uploaded files found yet")

# === Quick check for a specific file (change the name if you want) ===
specific_file = "fee-structure.pdf"          # ← Change this to your file name
results_specific = collection.get(where={"source": specific_file})
print(f"\n🔍 Specific check for '{specific_file}': {len(results_specific['ids'])} chunks")