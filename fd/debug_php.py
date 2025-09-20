import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the index and metadata
metadata_file = os.path.join('index', 'DSA_metadata.pkl')
with open(metadata_file, 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']

# Load embedding model
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Test query
query = 'What Can PHP Do'
query_embedding = model.encode([query], convert_to_numpy=True)

# Find PHP chunks and calculate similarity
php_chunks = []
for i, chunk in enumerate(chunks):
    if 'php' in chunk.text.lower():
        chunk_embedding = model.encode([chunk.text], convert_to_numpy=True)
        similarity = np.dot(query_embedding[0], chunk_embedding[0]) / (np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_embedding[0]))
        php_chunks.append((i, chunk.filename, similarity, chunk.text[:200]))

print(f'PHP chunks and their similarity to "{query}": (threshold=0.25)')
for i, filename, sim, text in sorted(php_chunks, key=lambda x: x[2], reverse=True):
    print(f'{i}: {filename} - Similarity: {sim:.3f} - {"ABOVE" if sim >= 0.25 else "BELOW"} threshold')
    print(f'   Text: {text}...')
    print()
