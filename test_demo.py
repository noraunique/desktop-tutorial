"""Test the Notes Q&A system with sample questions."""

import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SimpleChunk:
    text: str
    filename: str
    page: int = None
    section: str = None


class SimpleRetrieval:
    """Simple text-based retrieval for demo purposes."""
    
    def __init__(self):
        self.chunks = []
    
    def load_notes(self, notes_dir: str):
        """Load and chunk markdown files from notes directory."""
        dsa_dir = os.path.join(notes_dir, "DSA")
        
        if not os.path.exists(dsa_dir):
            print(f"Directory not found: {dsa_dir}")
            return
        
        for filename in os.listdir(dsa_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(dsa_dir, filename)
                self._process_markdown_file(filepath, filename)
        
        print(f"Loaded {len(self.chunks)} chunks from {dsa_dir}")
    
    def _process_markdown_file(self, filepath: str, filename: str):
        """Process a markdown file into chunks."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        current_section = None
        current_text = []
        
        lines = content.split('\n')
        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section if exists
                if current_text:
                    text = '\n'.join(current_text).strip()
                    if len(text) > 50:  # Only keep substantial chunks
                        self.chunks.append(SimpleChunk(
                            text=text,
                            filename=filename,
                            section=current_section
                        ))
                    current_text = []
                
                current_section = header_match.group(2)
                current_text.append(line)
            else:
                current_text.append(line)
        
        # Add final section
        if current_text:
            text = '\n'.join(current_text).strip()
            if len(text) > 50:
                self.chunks.append(SimpleChunk(
                    text=text,
                    filename=filename,
                    section=current_section
                ))
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[SimpleChunk, float]]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\w+', chunk.text.lower()))
            
            # Calculate simple overlap score
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(query_words)
                results.append((chunk, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def format_answer(self, query: str, results: List[Tuple[SimpleChunk, float]]) -> str:
        """Format search results into an answer."""
        if not results:
            return "Not found in your notes."
        
        output = [f"Question: {query}", ""]
        
        # Combine relevant text from top results
        answer_parts = []
        citations = []
        
        for chunk, score in results:
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', chunk.text)
            relevant_sentences = []
            
            query_words = set(query.lower().split())
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    sentence_words = set(re.findall(r'\w+', sentence.lower()))
                    if query_words.intersection(sentence_words):
                        relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer_parts.extend(relevant_sentences[:2])  # Top 2 sentences per chunk
            
            # Add citation
            if chunk.section:
                citation = f"({chunk.filename} · \"{chunk.section}\")"
            else:
                citation = f"({chunk.filename})"
            
            if citation not in citations:
                citations.append(citation)
        
        # Format answer
        if answer_parts:
            output.append("Answer:")
            output.append(" ".join(answer_parts[:3]))  # Limit answer length
            output.append("")
        
        # Add citations
        if citations:
            output.append("Sources:")
            for citation in citations:
                output.append(f"  • {citation}")
            output.append("")
        
        return "\n".join(output)


def main():
    """Test the retrieval system with sample questions."""
    print("Notes Q&A System Test")
    print("=" * 40)
    
    # Initialize retrieval system
    retrieval = SimpleRetrieval()
    retrieval.load_notes("notes")
    
    if not retrieval.chunks:
        print("No notes found. Make sure you have markdown files in notes/DSA/")
        return
    
    # Test questions
    test_questions = [
        "What is quicksort?",
        "What is the time complexity of quicksort?",
        "What are arrays?",
        "What is a binary tree?",
        "How does merge sort work?"
    ]
    
    print(f"\nTesting {len(test_questions)} questions:")
    print("-" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}:")
        results = retrieval.search(question)
        answer = retrieval.format_answer(question, results)
        print(answer)
        print("-" * 40)
    
    print("\nTest completed! The system is working.")
    print("You can now:")
    print("1. Add your own PDF/markdown files to notes/DSA/")
    print("2. Install full dependencies: pip install sentence-transformers faiss-cpu")
    print("3. Run: python build_index.py --course DSA")
    print("4. Run: python query.py --course DSA --interactive")


if __name__ == '__main__':
    main()
