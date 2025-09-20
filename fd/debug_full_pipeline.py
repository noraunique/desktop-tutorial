from retrieval import RetrievalPipeline

# Test the full pipeline
pipeline = RetrievalPipeline()

# Test the exact query that's failing
query = "What Can PHP Do"
course = "DSA"

print(f"Testing query: '{query}' for course: '{course}'")
print("="*50)

try:
    result = pipeline.query(course, query)
    
    print(f"Query: {result.query}")
    print(f"Found in notes: {result.found_in_notes}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Citations: {result.citations}")
    print(f"Source chunks count: {len(result.source_chunks)}")
    
    if result.source_chunks:
        print("\nTop source chunks:")
        for i, (chunk, score) in enumerate(result.source_chunks[:3]):
            print(f"  {i+1}. {chunk.filename} (Score: {score:.3f})")
            print(f"     Text: {chunk.text[:150]}...")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
