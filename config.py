"""Configuration settings for the Notes Q&A Retrieval Bot."""

import os

# Data paths
NOTES_DIR = "notes"
AUX_DIR = "aux"
INDEX_DIR = "index"
GLOSSARY_FILE = os.path.join(AUX_DIR, "glossary.md")
EXCLUSIONS_FILE = os.path.join(AUX_DIR, "index_exclusions.txt")
TRANSLATION_TA_FILE = os.path.join(AUX_DIR, "translation_ta.json")
TRANSLATION_SI_FILE = os.path.join(AUX_DIR, "translation_si.json")

# Chunking parameters
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 120  # character overlap between chunks
MIN_CHUNK_SIZE = 100  # minimum chunk size to keep

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Lightweight but effective
# Alternative: "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval parameters
TOP_K = 5  # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.25  # Minimum similarity for relevant chunks (lowered further to capture more content)
MAX_ANSWER_LENGTH = 500  # maximum characters in generated answer

# FAISS index settings
INDEX_TYPE = "flat"  # "flat" for small datasets, "ivf" for larger ones
NPROBE = 10  # for IVF index only

# File processing
SUPPORTED_EXTENSIONS = [".pdf", ".md", ".txt", ".docx", ".pptx"]
OCR_ENABLED = False  # Set to True if you have pytesseract installed

# Citation format
CITATION_FORMAT = "({filename} · p. {page})"  # for PDFs
SECTION_CITATION_FORMAT = "({filename} · \"{section}\")"  # for sections
UNIT_CITATION_FORMAT = "(Unit {unit} → {section})"  # for unit-based sections

# Evaluation
GOLD_QUESTIONS_FILE = os.path.join(AUX_DIR, "gold_questions.json")
EVALUATION_METRICS = ["answerable_rate", "precision_at_k", "citation_coverage", "hallucination_rate"]

# UI settings
CLI_COLORS = True
SHOW_SOURCE_SNIPPETS = True
MAX_SNIPPET_LENGTH = 200

# Locale and hybrid retrieval settings
# Locale policy: normalize clean text to UK spellings while preserving original text in citations
LOCALE_POLICY = "UK"

# Enable hybrid retrieval channel (embedding + lexical BM25/TF-IDF)
HYBRID_RETRIEVAL = True

# Fusion weights
# Default: more weight on embeddings; for short queries (<=2 tokens) or exact-heading matches, favor lexical
EMBED_WEIGHT_DEFAULT = 0.7
LEXICAL_WEIGHT_DEFAULT = 0.3
EMBED_WEIGHT_SHORT_OR_HEADING = 0.4
LEXICAL_WEIGHT_SHORT_OR_HEADING = 0.6

# Candidate pool caps and quotas
FUSION_POOL_CAP = 100  # max candidates considered before composing answer
RERANK_MIN_HEADING_QUOTA = 10  # ensure at least N candidates from a heading if query matches a known heading

