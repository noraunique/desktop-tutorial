"""Embedding and vector index management for the Notes Q&A system."""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
try:
    from rank_bm25 import BM25Okapi
except Exception:  # Package not installed or import failed
    BM25Okapi = None

from chunker import ProcessedChunk
from config import (
    EMBEDDING_MODEL, INDEX_DIR, TOP_K, SIMILARITY_THRESHOLD,
    INDEX_TYPE, NPROBE,
    HYBRID_RETRIEVAL,
    EMBED_WEIGHT_DEFAULT, LEXICAL_WEIGHT_DEFAULT,
    EMBED_WEIGHT_SHORT_OR_HEADING, LEXICAL_WEIGHT_SHORT_OR_HEADING,
    FUSION_POOL_CAP
)


class EmbeddingManager:
    """Manages text embeddings and FAISS vector index."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_metadata = []
        self.dimension = None
        self.bm25 = None
        self._bm25_corpus_tokens: List[List[str]] = []
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_chunks(self, chunks: List[ProcessedChunk]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        if not chunks:
            return np.array([])
        
        self.load_model()
        
        texts = [chunk.text for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches for memory efficiency
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def build_index(self, chunks: List[ProcessedChunk], course_name: str) -> str:
        """Build FAISS index from processed chunks."""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Generate embeddings
        embeddings = self.embed_chunks(chunks)
        
        # Create FAISS index
        if INDEX_TYPE == "flat":
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif INDEX_TYPE == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, len(chunks) // 10))
            index.train(embeddings)
            index.nprobe = NPROBE
        else:
            raise ValueError(f"Unsupported index type: {INDEX_TYPE}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Store metadata
        self.index = index
        self.chunk_metadata = chunks
        # Prepare BM25 corpus tokens for lexical retrieval
        self._bm25_corpus_tokens = [self._tokenize(c.text) for c in chunks]
        # Initialize BM25 model (optional dependency)
        try:
            if HYBRID_RETRIEVAL and BM25Okapi is not None:
                self.bm25 = BM25Okapi(self._bm25_corpus_tokens)
            else:
                if HYBRID_RETRIEVAL and BM25Okapi is None:
                    print("Warning: HYBRID_RETRIEVAL enabled but rank_bm25 is not installed; proceeding without lexical channel.")
                self.bm25 = None
        except Exception as e:
            print(f"Warning: Failed to initialize BM25 model: {e}")
            self.bm25 = None
        
        # Save index and metadata
        index_path = self.save_index(course_name)
        
        print(f"Index built successfully:")
        print(f"  - {len(chunks)} chunks indexed")
        print(f"  - Index type: {INDEX_TYPE}")
        print(f"  - Saved to: {index_path}")
        
        return index_path
    
    def save_index(self, course_name: str) -> str:
        """Save FAISS index and metadata to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(INDEX_DIR, f"{course_name}_index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(INDEX_DIR, f"{course_name}_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunk_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension,
                'index_type': INDEX_TYPE,
                'bm25_corpus_tokens': self._bm25_corpus_tokens,
            }, f)
        
        return index_file
    
    def load_index(self, course_name: str) -> bool:
        """Load FAISS index and metadata from disk."""
        index_file = os.path.join(INDEX_DIR, f"{course_name}_index.faiss")
        metadata_file = os.path.join(INDEX_DIR, f"{course_name}_metadata.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunk_metadata = metadata['chunks']
            saved_model_name = metadata['model_name']
            self.dimension = metadata['dimension']
            
            # Verify model compatibility
            if saved_model_name != self.model_name:
                print(f"Warning: Index was built with {saved_model_name}, "
                      f"but current model is {self.model_name}")
            
            # Rebuild BM25 if tokens are available and hybrid retrieval is enabled
            self._bm25_corpus_tokens = metadata.get('bm25_corpus_tokens', [])
            if HYBRID_RETRIEVAL and self._bm25_corpus_tokens and BM25Okapi is not None:
                try:
                    self.bm25 = BM25Okapi(self._bm25_corpus_tokens)
                except Exception as e:
                    print(f"Warning: Failed to reconstruct BM25 model: {e}")
                    self.bm25 = None
            else:
                self.bm25 = None

            print(f"Index loaded: {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = TOP_K) -> List[Tuple[ProcessedChunk, float]]:
        """Search for similar chunks given a query with optional hybrid fusion (embedding + lexical)."""
        if self.index is None or not self.chunk_metadata:
            raise ValueError("No index loaded. Build or load an index first.")
        
        self.load_model()
        
        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search embeddings for a broader pool (use pool cap)
        pool_n = max(k, min(FUSION_POOL_CAP, len(self.chunk_metadata)))
        emb_scores, emb_indices = self.index.search(query_embedding, pool_n)
        emb_scores = emb_scores[0]
        emb_indices = emb_indices[0]

        # Normalize embedding scores to [0,1]
        emb_scores_norm = self._min_max_normalize(emb_scores)

        # Lexical BM25 scores over the same pool (or entire corpus if bm25 exists)
        lex_scores_norm = None
        if HYBRID_RETRIEVAL and self.bm25 is not None and BM25Okapi is not None:
            query_tokens = self._tokenize(query)
            try:
                lex_scores_all = np.array(self.bm25.get_scores(query_tokens), dtype=float)
            except Exception as e:
                print(f"Warning: BM25 scoring failed: {e}")
                lex_scores_all = np.zeros(len(self.chunk_metadata), dtype=float)
            # Gather lexical scores for the embedding pool indices
            lex_scores_pool = lex_scores_all[emb_indices.clip(min=0)]
            lex_scores_norm = self._min_max_normalize(lex_scores_pool)
        else:
            lex_scores_norm = np.zeros_like(emb_scores_norm)

        # Determine fusion weights based on query type
        tokens = [t for t in self._tokenize(query) if t]
        is_short = len(tokens) <= 2
        # Exact heading match heuristic: any chunk.section equals the query (case-insensitive)
        query_lower = query.strip().lower()
        has_exact_heading = any((c.section or '').strip().lower() == query_lower for c in self.chunk_metadata)
        if is_short or has_exact_heading:
            w_e = EMBED_WEIGHT_SHORT_OR_HEADING
            w_l = LEXICAL_WEIGHT_SHORT_OR_HEADING
        else:
            w_e = EMBED_WEIGHT_DEFAULT
            w_l = LEXICAL_WEIGHT_DEFAULT

        fused = w_e * emb_scores_norm + w_l * lex_scores_norm

        # Rank by fused score and filter by similarity threshold where applicable
        order = np.argsort(-fused)
        results = []
        for idx_in_pool in order[:k*2]:  # inspect a bit more, then filter
            faiss_idx = int(emb_indices[idx_in_pool])
            if faiss_idx < 0:
                continue
            score = float(fused[idx_in_pool])
            if score < SIMILARITY_THRESHOLD:
                continue
            chunk = self.chunk_metadata[faiss_idx]
            results.append((chunk, score))
            if len(results) >= k:
                break
        return results

    # ---------------------
    # Helpers
    # ---------------------
    def _tokenize(self, text: str) -> List[str]:
        """Lightweight tokenizer for BM25 (lowercase, split on non-alphanum)."""
        return [t for t in re.split(r"[^\w]+", text.lower()) if t]

    def _min_max_normalize(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        a_min = np.min(arr)
        a_max = np.max(arr)
        if a_max - a_min == 0:
            return np.zeros_like(arr)
        return (arr - a_min) / (a_max - a_min)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.index is None:
            return {}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': INDEX_TYPE,
            'model_name': self.model_name,
            'chunks_count': len(self.chunk_metadata)
        }
        
        if hasattr(self.index, 'nprobe'):
            stats['nprobe'] = self.index.nprobe
        
        return stats


def build_course_index(course_name: str, course_dir: str) -> str:
    """Build complete index for a course directory."""
    from chunker import chunk_course_documents
    
    print(f"Building index for course: {course_name}")
    print(f"Processing directory: {course_dir}")
    
    # Extract and chunk documents
    chunks = chunk_course_documents(course_dir, course_name)
    
    if not chunks:
        raise ValueError(f"No chunks extracted from {course_dir}")
    
    # Build embeddings and index
    embedding_manager = EmbeddingManager()
    index_path = embedding_manager.build_index(chunks, course_name)
    
    # Print final statistics
    stats = embedding_manager.get_index_stats()
    print(f"\nFinal Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return index_path


def search_course(course_name: str, query: str, k: int = TOP_K) -> List[Tuple[ProcessedChunk, float]]:
    """Search a specific course index."""
    embedding_manager = EmbeddingManager()
    
    # Force reload index to ensure we have the latest version
    embedding_manager.index = None
    embedding_manager.chunk_metadata = []
    
    if not embedding_manager.load_index(course_name):
        raise ValueError(f"No index found for course: {course_name}")
    
    return embedding_manager.search(query, k)
