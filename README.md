# Offline Notes Q&A Retrieval Bot

A privacy-first, fully offline question-answering system for your study notes and documents.

## Features

- **100% Offline**: All processing happens locally - no data leaves your machine
- **Multi-format Support**: PDFs, Markdown, and text files
- **Smart Citations**: Every answer includes source file and page/section references
- **Semantic Search**: Uses advanced embeddings for accurate content retrieval
- **Course Organization**: Organize notes by subject/course for focused searches

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize Your Notes**
   ```
   notes/
     DSA/
       01-intro.pdf
       02-sorting.pdf
     OS/
       01-processes.pdf
   aux/
     glossary.md
     index_exclusions.txt
   ```

3. **Build Index**
   ```bash
   python build_index.py --course DSA
   ```

4. **Ask Questions**
   ```bash
   python query.py "What is the time complexity of quicksort?"
   ```

## Architecture

- **Text Extraction**: Preserves page numbers and section headings
- **Chunking**: 800 characters per chunk with 120 character overlap
- **Embeddings**: BGE-small-en model for semantic understanding
- **Index**: FAISS for fast similarity search
- **Retrieval**: Top-k=5 with confidence thresholds

## Success Criteria

- ≥90% of answers include proper citations
- ≥85% of questions retrieve relevant content
- 0% hallucination rate (only answers from your notes)
- <1 second retrieval latency

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Similarity thresholds
- Top-k retrieval count
- Embedding model selection
