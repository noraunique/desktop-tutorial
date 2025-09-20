"""Text chunking utilities for the Notes Q&A system."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, replace

from text_extractor import ExtractedChunk
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class ProcessedChunk:
    """Represents a processed chunk ready for embedding."""
    text: str
    filename: str
    page: int = None
    section: str = None
    char_offset: int = 0
    course: str = None
    chunk_id: str = None


class TextChunker:
    """Handles intelligent text chunking with overlap and metadata preservation."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = MIN_CHUNK_SIZE
    
    def chunk_documents(self, extracted_chunks: List[ExtractedChunk]) -> List[ProcessedChunk]:
        """Process extracted chunks into smaller, overlapping chunks."""
        processed_chunks = []
        
        for chunk in extracted_chunks:
            chunks = self._chunk_text(chunk)
            processed_chunks.extend(chunks)
        
        # Generate unique IDs for each chunk
        for i, chunk in enumerate(processed_chunks):
            chunk.chunk_id = f"{chunk.filename}_{i:04d}"
        
        return processed_chunks
    
    def _chunk_text(self, extracted_chunk: ExtractedChunk) -> List[ProcessedChunk]:
        """Split a single extracted chunk into smaller overlapping chunks."""
        text = extracted_chunk.text
        
        # First, try to split on natural boundaries
        natural_chunks = self._split_on_boundaries(text)
        
        processed_chunks = []
        for natural_chunk in natural_chunks:
            if len(natural_chunk) <= self.chunk_size:
                # Chunk is small enough, use as-is
                if len(natural_chunk.strip()) >= self.min_chunk_size:
                    processed_chunks.append(self._create_processed_chunk(
                        natural_chunk, extracted_chunk, len(processed_chunks)
                    ))
            else:
                # Chunk is too large, split with overlap
                sub_chunks = self._split_with_overlap(natural_chunk)
                for i, sub_chunk in enumerate(sub_chunks):
                    if len(sub_chunk.strip()) >= self.min_chunk_size:
                        processed_chunks.append(self._create_processed_chunk(
                            sub_chunk, extracted_chunk, len(processed_chunks)
                        ))
        
        return processed_chunks
    
    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split text on natural boundaries like paragraphs and sentences."""
        # First split on double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk + "\n\n" + paragraph) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks of specified size."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            
            # Ensure we don't get stuck in an infinite loop
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the given range."""
        # Look for sentence endings (., !, ?) followed by space or newline
        sentence_pattern = r'[.!?]\s+'
        
        # Search backwards from end to start
        for match in re.finditer(sentence_pattern, text[start:end]):
            boundary = start + match.end()
            if boundary > start + 50:  # Ensure minimum chunk size
                return boundary
        
        # If no sentence boundary found, look for other boundaries
        other_boundaries = ['\n', '. ', '! ', '? ']
        for boundary_char in other_boundaries:
            pos = text.rfind(boundary_char, start, end)
            if pos > start + 50:
                return pos + len(boundary_char)
        
        return end
    
    def _create_processed_chunk(self, text: str, original_chunk: ExtractedChunk, 
                              chunk_index: int) -> ProcessedChunk:
        """Create a ProcessedChunk from text and original metadata."""
        return ProcessedChunk(
            text=text.strip(),
            filename=original_chunk.filename,
            page=original_chunk.page,
            section=original_chunk.section,
            course=original_chunk.course,
            char_offset=chunk_index * self.chunk_size  # Approximate offset
        )
    
    def get_chunk_statistics(self, chunks: List[ProcessedChunk]) -> Dict:
        """Get statistics about the chunked data."""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths),
            'files_processed': len(set(chunk.filename for chunk in chunks)),
            'courses': list(set(chunk.course for chunk in chunks if chunk.course))
        }
        
        return stats


def chunk_course_documents(course_dir: str, course_name: str) -> List[ProcessedChunk]:
    """Extract and chunk all documents from a course directory."""
    from text_extractor import extract_text_from_directory
    
    # Extract text from all files in the course directory
    extracted_chunks = extract_text_from_directory(course_dir, course_name)
    
    if not extracted_chunks:
        print(f"No text extracted from {course_dir}")
        return []
    
    # Chunk the extracted text
    chunker = TextChunker()
    processed_chunks = chunker.chunk_documents(extracted_chunks)
    
    # Print statistics
    stats = chunker.get_chunk_statistics(processed_chunks)
    print(f"\nChunking Statistics for {course_name}:")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  Average chunk length: {stats.get('avg_chunk_length', 0):.1f} characters")
    print(f"  Files processed: {stats.get('files_processed', 0)}")
    print(f"  Total characters: {stats.get('total_characters', 0):,}")
    
    return processed_chunks
