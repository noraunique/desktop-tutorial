from retrieval import AnswerComposer
from chunker import ProcessedChunk

# Create test chunk with PHP content
test_chunk = ProcessedChunk(
    text="What Can PHP Do PHP can create, open, read, write, delete, and close files on the server PHP can collect form data PHP can send and receive cookies PHP can add, delete, modify data in your database",
    filename="HND WAD W11 PHP Part I.pdf",
    page=1,
    section=None,
    course="DSA"
)

# Test the definition detection
composer = AnswerComposer()

# Test different queries
queries = [
    "What Can PHP Do",
    "PHP",
    "php programming",
    "what php can do"
]

for query in queries:
    topic = composer._extract_topic(query)
    chunks_with_scores = [(test_chunk, 0.903)]
    
    has_definition = composer._find_definition_chunk(chunks_with_scores, topic)
    
    print(f"Query: '{query}'")
    print(f"  Extracted topic: '{topic}'")
    print(f"  Has definition: {has_definition}")
    print()
