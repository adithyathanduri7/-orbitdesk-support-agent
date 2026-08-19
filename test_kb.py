from knowledge_base import KnowledgeBase

print("Loading Knowledge Base...")

kb = KnowledgeBase()

print("\n✅ Knowledge Base Loaded Successfully!")
print(f"Total Chunks: {len(kb.chunks)}")

query = "login issue"

results = kb.retrieve(query)

print(f"\nResults for query: '{query}'")

if len(results) == 0:
    print("No matching documents found.")
else:
    for i, result in enumerate(results, start=1):
        print("\n" + "=" * 50)
        print(f"Result {i}")
        print(f"Score : {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Type  : {result['type']}")
        print("Text:")
        print(result["text"][:300])