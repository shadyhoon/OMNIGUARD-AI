from utils.rag import rag_cross_reference, rag_health
print("Health:", rag_health())
print()
hits = rag_cross_reference(
    "The sun is a star at the centre of our solar system. Vaccines do not cause autism.",
    max_claims=2,
)
for h in hits:
    print("  - claim:  ", h["claim"][:60])
    print("    status: ", h["status"])
    print("    engine: ", h["engine"])
    print("    relev.: ", h["relevance"])
    print("    source: ", h["source"][:80])
    print()
