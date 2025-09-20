"""Retrieval pipeline and answer composition for the Notes Q&A system."""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from embeddings import search_course, EmbeddingManager
from chunker import ProcessedChunk
from config import (
    TOP_K, SIMILARITY_THRESHOLD, MAX_ANSWER_LENGTH,
    CITATION_FORMAT, SECTION_CITATION_FORMAT, SHOW_SOURCE_SNIPPETS,
    MAX_SNIPPET_LENGTH,
    RERANK_MIN_HEADING_QUOTA, FUSION_POOL_CAP,
    TRANSLATION_TA_FILE, TRANSLATION_SI_FILE
)
try:
    from deep_translator import GoogleTranslator  # Optional translation layer
except Exception:
    GoogleTranslator = None

# Optional run-on word splitter for display cleanup
try:
    import wordninja  # lightweight, offline-friendly
except Exception:
    wordninja = None


@dataclass
class SearchResult:
    """Represents a search result with answer and citations."""
    query: str
    answer: str
    citations: List[str]
    source_chunks: List[Tuple[ProcessedChunk, float]]
    confidence: float
    found_in_notes: bool


class AnswerComposer:
    """Composes answers from retrieved chunks with proper citations."""
    
    def __init__(self):
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.max_answer_length = MAX_ANSWER_LENGTH
    
    def compose_answer(self, query: str, retrieved_chunks: List[Tuple[ProcessedChunk, float]]) -> SearchResult:
        """Compose an answer from retrieved chunks with citations."""
        if not retrieved_chunks:
            return SearchResult(
                query=query,
                answer="Not found in your notes.",
                citations=[],
                source_chunks=[],
                confidence=0.0,
                found_in_notes=False
            )
        
        # Filter chunks by similarity threshold
        relevant_chunks = [
            (chunk, score) for chunk, score in retrieved_chunks 
            if score >= self.similarity_threshold
        ]
        
        if not relevant_chunks:
            return SearchResult(
                query=query,
                answer="Not found in your notes.",
                citations=[],
                source_chunks=retrieved_chunks,
                confidence=0.0,
                found_in_notes=False
            )
        
        # Check if we have a proper definition chunk for the query topic
        query_topic = self._extract_topic(query)
        definition_chunk = self._find_definition_chunk(relevant_chunks, query_topic)
        
        if not definition_chunk and query_topic:
            return SearchResult(
                query=query,
                answer="Not found in your notes.",
                citations=[],
                source_chunks=relevant_chunks,
                confidence=0.0,
                found_in_notes=False
            )
        
        # Extract and combine relevant text
        answer_text = self._extract_answer_text(relevant_chunks)
        citations = self._generate_citations(relevant_chunks)
        confidence = self._calculate_confidence(relevant_chunks)
        
        return SearchResult(
            query=query,
            answer=answer_text,
            citations=citations,
            source_chunks=relevant_chunks,
            confidence=confidence,
            found_in_notes=True
        )
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        # Simple topic extraction for common patterns
        topic_patterns = [
            r'what can (\w+(?:\s+\w+)*) do',
            r'what is (\w+(?:\s+\w+)*)',
            r'define (\w+(?:\s+\w+)*)',
            r'explain (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*) is'
        ]
        
        query_lower = query.lower()
        for pattern in topic_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: extract key nouns
        words = query_lower.split()
        important_words = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'why', 'when', 'where', 'can']]
        return ' '.join(important_words[:2]) if important_words else ''
    
    def _find_definition_chunk(self, chunks: List[Tuple[ProcessedChunk, float]], topic: str) -> bool:
        """Check if any chunk contains a proper definition for the topic."""
        if not topic:
            return True  # Allow queries without specific topics
        
        # For very short or generic queries, be more lenient
        if len(topic.split()) <= 2 or any(word in topic.lower() for word in ["how", "who", "what", "why", "when", "where"]):
            return True
        
        for chunk, score in chunks:
            chunk_text = chunk.text.lower()
            
            # For System Maintenance, look for relevant content
            if "system maintenance" in topic.lower():
                # Block System Monitoring sections
                if chunk.section and "monitoring" in chunk.section.lower():
                    continue
                
                # Accept if chunk contains maintenance-related content
                maintenance_indicators = [
                    "maintenance",
                    "regular maintenance",
                    "system maintenance",
                    "maintenance schedule"
                ]
                
                if any(indicator in chunk_text for indicator in maintenance_indicators):
                    return True
            
            # For CPD queries, look for CPD-related content
            if "cpd" in topic.lower() or "continuous professional development" in topic.lower():
                cpd_indicators = [
                    "cpd",
                    "continuous professional development",
                    "professional development",
                    "importance of cpd"
                ]
                
                if any(indicator in chunk_text for indicator in cpd_indicators):
                    return True
            
            # For PHP queries, look for PHP-related content
            if "php" in topic.lower():
                php_indicators = [
                    "php",
                    "what can php do",
                    "php programming",
                    "php language",
                    "hypertext preprocessor"
                ]
                
                if any(indicator in chunk_text for indicator in php_indicators):
                    return True
            
            # For document/file format queries, look for relevant content
            if any(word in topic.lower() for word in ["document", "file", "text", "format", "processing"]):
                format_indicators = [
                    "text file",
                    "document",
                    "file format",
                    "processing",
                    "plain text",
                    "word document",
                    "powerpoint",
                    "presentation"
                ]
                
                if any(indicator in chunk_text for indicator in format_indicators):
                    return True
            
            # For SDLC queries, look for software development content
            if any(word in topic.lower() for word in ["sdlc", "software", "development", "lifecycle"]):
                sdlc_indicators = [
                    "sdlc",
                    "software development",
                    "software lifecycle",
                    "development life cycle",
                    "software engineering",
                    "learning outcomes",
                    "unit 7"
                ]
                
                if any(indicator in chunk_text for indicator in sdlc_indicators):
                    return True
            
            # For stakeholder queries, look for business stakeholder content
            if any(word in topic.lower() for word in ["stakeholder", "internal", "external", "business"]):
                stakeholder_indicators = [
                    "stakeholder",
                    "internal stakeholder",
                    "external stakeholder",
                    "employees",
                    "owners",
                    "managers",
                    "customers",
                    "shareholders"
                ]
                
                if any(indicator in chunk_text for indicator in stakeholder_indicators):
                    return True
            
            # For sorting/algorithm queries, look for algorithm content
            if any(word in topic.lower() for word in ["sort", "algorithm", "quicksort", "merge", "bubble"]):
                algorithm_indicators = [
                    "sort",
                    "algorithm",
                    "quicksort",
                    "merge sort",
                    "bubble sort",
                    "time complexity",
                    "space complexity",
                    "divide-and-conquer",
                    "pivot",
                    "partition"
                ]
                
                if any(indicator in chunk_text for indicator in algorithm_indicators):
                    return True
            
            # For goal/objective queries, look for relevant content
            if any(word in topic.lower() for word in ["goal", "objective", "vs", "difference"]):
                goal_indicators = [
                    "goal",
                    "objective",
                    "goals vs objectives",
                    "difference between",
                    "goal is",
                    "objective is"
                ]
                
                if any(indicator in chunk_text for indicator in goal_indicators):
                    return True
            
            # For data protection queries, look for relevant content
            if any(word in topic.lower() for word in ["data", "protection", "act", "developed"]):
                data_protection_indicators = [
                    "data protection",
                    "data protection act",
                    "privacy",
                    "gdpr",
                    "personal data",
                    "developed",
                    "why"
                ]
                
                if any(indicator in chunk_text for indicator in data_protection_indicators):
                    return True
            
            # For computer/act/legal queries, look for relevant content
            if any(word in topic.lower() for word in ["computer", "act", "misuse", "law", "legal", "involved", "works"]):
                legal_indicators = [
                    "act",
                    "computer misuse",
                    "misuse act",
                    "1990",
                    "law",
                    "legal",
                    "involved",
                    "works",
                    "computer",
                    "legislation"
                ]
                
                if any(indicator in chunk_text for indicator in legal_indicators):
                    return True
            
            # General definition patterns for other topics
            definition_patterns = [
                rf'{re.escape(topic.lower())}\s+is\s+',
                rf'{re.escape(topic.lower())}\s+refers\s+to\s+',
                rf'{re.escape(topic.lower())}\s+means\s+',
                r'\bis\s+the\s+',
                r'\bis\s+a\s+'
            ]
            
            if any(re.search(pattern, chunk_text) for pattern in definition_patterns):
                # Block clearly unrelated starts
                bad_starts = [
                    r'^(network monitoring|monitoring)',
                    r'^(the current version|version \d+)'
                ]
                
                first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text
                if not any(re.search(pattern, first_sentence.strip()) for pattern in bad_starts):
                    return True
        
        return False
    
    def _extract_answer_text(self, chunks: List[Tuple[ProcessedChunk, float]]) -> str:
        """Extract and combine text from relevant chunks."""
        # Sort chunks by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Filter out System Monitoring chunks for System Maintenance queries
        filtered_chunks = []
        query_lower = getattr(self, '_current_query', '').lower()
        
        for chunk, score in sorted_chunks:
            # Block System Monitoring sections when asking about System Maintenance
            if "system maintenance" in query_lower:
                if chunk.section and "monitoring" in chunk.section.lower():
                    continue
            
            filtered_chunks.append((chunk, score))
        
        if not filtered_chunks:
            return ""
        
        # Prioritize chunks with definitions or from relevant sections
        priority_chunks = []
        other_chunks = []
        
        for chunk, score in filtered_chunks:
            chunk_text_lower = chunk.text.lower()
            
            # Check for definition patterns
            definition_patterns = [
                r'\bis\s+the\s+',
                r'\bis\s+a\s+',
                r'\brefers\s+to\s+',
                r'\bmeans\s+',
                r'definition\s*:'
            ]
            
            has_def = any(re.search(pattern, chunk_text_lower) for pattern in definition_patterns)
            
            # Check if chunk is from relevant section (exact match for System Maintenance)
            relevant_section = False
            if chunk.section and query_lower:
                if "system maintenance" in query_lower:
                    relevant_section = "system maintenance" in chunk.section.lower()
                else:
                    relevant_section = any(word in chunk.section.lower() for word in query_lower.split() if len(word) > 3)
            
            if has_def or relevant_section:
                priority_chunks.append((chunk, score))
            else:
                other_chunks.append((chunk, score))
        
        # Use priority chunks first, then others
        ordered_chunks = priority_chunks + other_chunks
        
        combined_text = []
        total_length = 0
        seen_content = set()
        
        # Only use chunks from the same section to avoid mixing topics
        if priority_chunks:
            # Use only priority chunks from the same section
            main_section = priority_chunks[0][0].section
            same_section_chunks = [(c, s) for c, s in priority_chunks if c.section == main_section]
            chunks_to_process = same_section_chunks[:2]  # Limit to 2 chunks from same section
        else:
            # Use top chunk only if no priority chunks
            chunks_to_process = other_chunks[:1]
        
        # Process selected chunks
        for chunk, score in chunks_to_process:
            # Avoid duplicate content
            chunk_key = self._normalize_text(chunk.text[:100])
            if chunk_key in seen_content:
                continue
            seen_content.add(chunk_key)
            
            # Extract key sentences that might answer the query
            relevant_sentences = self._extract_relevant_sentences(chunk.text)
            
            for sentence in relevant_sentences:
                if total_length + len(sentence) > self.max_answer_length:
                    break
                combined_text.append(sentence.strip())
                total_length += len(sentence)
            
            if total_length >= self.max_answer_length:
                break
        
        if not combined_text:
            # Fallback: use first chunk's text but clean it
            if chunks_to_process:
                first_chunk = chunks_to_process[0][0]
                text = first_chunk.text[:self.max_answer_length]
                # Clean up fragmented text
                text = re.sub(r'\s+', ' ', text.strip())
                return text + "..." if len(first_chunk.text) > self.max_answer_length else text
            else:
                return "No relevant content found."
        
        # Join sentences properly
        result = " ".join(combined_text)
        # Clean up any remaining artifacts
        result = re.sub(r'\s+', ' ', result.strip())
        return result
    
    def _extract_relevant_sentences(self, text: str) -> List[str]:
        """Extract sentences that are likely to contain relevant information."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 300:  # Reasonable sentence length
                relevant_sentences.append(sentence)
        
        return relevant_sentences[:3]  # Limit to top 3 sentences per chunk
    
    def _generate_citations(self, chunks: List[Tuple[ProcessedChunk, float]]) -> List[str]:
        """Generate properly formatted citations."""
        from config import CITATION_FORMAT, SECTION_CITATION_FORMAT, UNIT_CITATION_FORMAT
        
        citations = []
        seen_sources = set()
        
        for chunk, score in chunks:
            # Create citation based on available metadata
            if chunk.page is not None:
                citation = CITATION_FORMAT.format(
                    filename=chunk.filename,
                    page=chunk.page
                )
            elif chunk.section:
                # Check if section follows "Unit X: Topic" pattern
                unit_match = re.search(r'Unit (\d+)', chunk.section)
                if unit_match:
                    unit_num = unit_match.group(1)
                    section_name = re.sub(r'Unit \d+:\s*', '', chunk.section)
                    citation = UNIT_CITATION_FORMAT.format(
                        unit=unit_num,
                        section=section_name
                    )
                else:
                    citation = SECTION_CITATION_FORMAT.format(
                        filename=chunk.filename,
                        section=chunk.section
                    )
            else:
                citation = f"({chunk.filename})"
            
            # Avoid duplicate citations
            if citation not in seen_sources:
                citations.append(citation)
                seen_sources.add(citation)
        
        return citations

    def _format_citation(self, chunk: ProcessedChunk) -> str:
        if chunk.page is not None:
            return CITATION_FORMAT.format(filename=chunk.filename, page=chunk.page)
        elif chunk.section:
            return SECTION_CITATION_FORMAT.format(filename=chunk.filename, section=chunk.section)
        else:
            return f"({chunk.filename})"
    
    def _calculate_confidence(self, chunks: List[Tuple[ProcessedChunk, float]]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not chunks:
            return 0.0
        
        scores = [score for _, score in chunks]
        
        # Weighted average with emphasis on top results
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for duplicate detection."""
        return re.sub(r'\s+', ' ', text.lower().strip())


class RetrievalPipeline:
    """Main retrieval pipeline that orchestrates search and answer composition."""
    
    def __init__(self):
        self.answer_composer = AnswerComposer()
        self.embedding_manager = EmbeddingManager()
        # Offline translation caches (loaded lazily)
        self._ta_map = None
        self._si_map = None
    
    def query(self, course_name: str, question: str, k: int = TOP_K) -> SearchResult:
        """Process a query and return a complete search result."""
        # Normalize the query
        normalized_query = self._normalize_query(question)
        
        # Store query for answer composition
        self.answer_composer._current_query = normalized_query
        
        try:
            # Search with a larger pool to allow heading-aware selection
            pool_k = max(k, FUSION_POOL_CAP)
            pool_chunks = search_course(course_name, normalized_query, pool_k)

            # Apply heading-aware quota if query matches a known heading (exact or soft match)
            def _norm(s: str) -> str:
                return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()
            query_lower = _norm(normalized_query)
            heading_matches = [(c, s) for (c, s) in pool_chunks if _norm(c.section) == query_lower]
            # Soft match: section contains query or vice versa
            if not heading_matches:
                soft = [(c, s) for (c, s) in pool_chunks if _norm(c.section) and (query_lower in _norm(c.section) or _norm(c.section) in query_lower)]
                # Keep top-N soft matches to seed selection
                heading_matches = soft[:RERANK_MIN_HEADING_QUOTA] if soft else []

            selected: List[Tuple[ProcessedChunk, float]] = []
            if heading_matches:
                # Ensure at least N candidates from this heading
                selected.extend(heading_matches[:RERANK_MIN_HEADING_QUOTA])
                # Fill the rest, preferring same filename as top-1 to resolve heading collisions
                top_file = heading_matches[0][0].filename
                # First, add more from same file
                for c, s in pool_chunks:
                    if len(selected) >= k:
                        break
                    if (c, s) in selected:
                        continue
                    if c.filename == top_file:
                        selected.append((c, s))
                # Then, add remaining highest scored from pool
                for c, s in pool_chunks:
                    if len(selected) >= k:
                        break
                    if (c, s) in selected:
                        continue
                    selected.append((c, s))
            else:
                # No exact heading match: prefer same-file candidates around the very top result to avoid cross-topic bleed
                if pool_chunks:
                    # Special intent handling: if capability-style question, re-rank entire pool by capability signal
                    ql = query_lower
                    def capability_signal(chunk: ProcessedChunk) -> float:
                        # Boost if section resembles the query and if lines indicate capability bullets
                        score = 0.0
                        sec = (chunk.section or '').lower()
                        txt = (chunk.text or '')
                        if 'what can php do' in ql:
                            if 'what can php do' in sec:
                                score += 0.5
                            # Count lines like bullet + "PHP can ..."
                            lines = txt.splitlines()
                            pattern = re.compile(r"^\s*(?:[\u2022\-\*\u25AA\u25CF\u25E6\u2219]\s+)?php\s+can\b", re.IGNORECASE)
                            php_can = sum(1 for ln in lines if pattern.search(ln))
                            if php_can:
                                score += min(0.5, 0.1 * php_can)
                        return score

                    if 'what can php do' in ql:
                        # Re-rank entire pool by capability signal combined with base score
                        pool_ranked = sorted(
                            pool_chunks,
                            key=lambda cs: (cs[1] + capability_signal(cs[0])),
                            reverse=True
                        )
                        selected = pool_ranked[:k]
                    else:
                        # Default behavior (same-file first)
                        top_file = pool_chunks[0][0].filename
                        same_file = [(c, s) for (c, s) in pool_chunks if c.filename == top_file]
                        other_file = [(c, s) for (c, s) in pool_chunks if c.filename != top_file]
                        selected = (same_file + other_file)[:k]
                else:
                    selected = []

            # Compose answer with citations
            result = self.answer_composer.compose_answer(normalized_query, selected)
            
            return result
            
        except Exception as e:
            return SearchResult(
                query=question,
                answer=f"Error processing query: {str(e)}",
                citations=[],
                source_chunks=[],
                confidence=0.0,
                found_in_notes=False
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize and clean the input query."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations (could be enhanced with glossary)
        abbreviations = {
            'algo': 'algorithm',
            'ds': 'data structure',
            'os': 'operating system',
            'db': 'database'
        }
        
        words = query.lower().split()
        expanded_words = [abbreviations.get(word, word) for word in words]
        
        return ' '.join(expanded_words)
    
    def get_source_snippets(self, result: SearchResult) -> List[Dict]:
        """Get formatted source snippets for display."""
        snippets = []
        
        for chunk, score in result.source_chunks:
            snippet_text = chunk.text
            if len(snippet_text) > MAX_SNIPPET_LENGTH:
                snippet_text = snippet_text[:MAX_SNIPPET_LENGTH] + "..."
            
            snippet = {
                'text': snippet_text,
                'filename': chunk.filename,
                'page': chunk.page,
                'section': chunk.section,
                'score': score,
                'course': chunk.course
            }
            snippets.append(snippet)
        
        return snippets

    # --------------------- Offline translation helpers ---------------------
    def _get_offline_map(self, lang: str) -> Optional[Dict[str, str]]:
        """Load and cache offline translation maps for Tamil/Sinhala."""
        try:
            if lang == 'ta':
                if self._ta_map is None:
                    self._ta_map = self._load_translation_file(TRANSLATION_TA_FILE)
                return self._ta_map
            if lang == 'si':
                if self._si_map is None:
                    self._si_map = self._load_translation_file(TRANSLATION_SI_FILE)
                return self._si_map
        except Exception:
            return None
        return None

    def _load_translation_file(self, path: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not path or not os.path.exists(path):
            return mapping
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data.get('replacements', []):
                src = str(item.get('src', '')).strip()
                tgt = str(item.get('tgt', '')).strip()
                if src and tgt:
                    mapping[src.lower()] = tgt
        except Exception:
            return {}
        return mapping

    def _apply_glossary(self, text: str, mapping: Dict[str, str]) -> str:
        """Apply simple case-insensitive phrase replacements based on mapping.
        This is conservative and only replaces exact substring matches of known phrases.
        """
        if not mapping:
            return text
        # Replace longer phrases first to avoid partial overlaps
        for src in sorted(mapping.keys(), key=lambda s: -len(s)):
            tgt = mapping[src]
            # Case-insensitive replace: use regex with word boundaries where applicable
            pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
            text = pattern.sub(tgt, text)
        return text

    # --------------------- StudyBot Formatting ---------------------
    def _detect_lang(self, text: str) -> str:
        """Very lightweight language detection for Tamil/Sinhala; default English."""
        # Tamil unicode block: 0B80–0BFF, Sinhala: 0D80–0DFF
        for ch in text:
            code = ord(ch)
            if 0x0B80 <= code <= 0x0BFF:
                return 'ta'
            if 0x0D80 <= code <= 0x0DFF:
                return 'si'
        return 'en'

    def _confidence_label(self, score: float) -> str:
        if score >= 0.85:
            return 'High'
        if score >= 0.75:
            return 'Medium'
        return 'Low'

    def _classify_intent(self, question: str) -> str:
        q = question.strip().lower()
        if any(p in q for p in ['what is', 'define', 'meaning of']):
            return 'definition'
        if any(p in q for p in ['what can', 'capabilities', 'features', 'uses']):
            return 'capabilities'
        if any(p in q for p in ['steps', 'procedure', 'algorithm', 'process', 'how to']):
            return 'steps'
        if any(p in q for p in ['importance', 'benefits', 'why']):
            return 'importance'
        if ' vs ' in q or ' versus ' in q:
            return 'compare'
        # default fallback
        return 'definition'

    def format_studybot(self, question: str, result: SearchResult, target_lang: Optional[str]) -> str:
        """Format output to StudyBot spec with multilingual labels and strict structure."""
        # Determine language
        lang = (target_lang or '').strip().lower()
        if lang not in {'en', 'ta', 'si'}:
            lang = self._detect_lang(question)

        labels = {
            'en': {'answer': 'Answer', 'sources': 'Sources', 'confidence': 'Confidence'},
            'ta': {'answer': 'பதில்', 'sources': 'ஆதாரங்கள்', 'confidence': 'நம்பிக்கை'},
            'si': {'answer': 'පිළිතුර', 'sources': 'මූලාශ්‍ර', 'confidence': 'විශ්වාසය'},
        }[lang]

        # NOT_FOUND path
        if not result.found_in_notes or not result.citations:
            # Per global guard: return the exact NOT_FOUND line (no extra UI hints)
            return "NOT_FOUND: This isn’t stated in your notes."

        # Build body (translate answer text only; never translate citations)
        lines: List[str] = []
        lines.append(f"{labels['answer']}:")
        # Global guard: compute top score from source chunks when available
        top_score = result.source_chunks[0][1] if result.source_chunks else 0.0
        # Intent routing
        intent = self._classify_intent(question)
        # Prefer using ONLY the top source chunk for composing the answer, as requested
        top_only_chunks = result.source_chunks[:1] if result.source_chunks else []
        # Table detection from top chunk (render table if present)
        table = self._extract_table(top_only_chunks) if top_only_chunks else None
        # Prefer bullet/numbered formatting if detected in top chunk or implied by intent
        lead_in, bullets, ordered, bullet_char = self._extract_list_items(top_only_chunks if top_only_chunks else result.source_chunks)
        # Prefer bullets that align with the query intent (e.g., "What can PHP do" → bullets starting with "PHP can ...")
        bullets = self._filter_bullets_by_intent(bullets, question)
        use_list = (
            len(bullets) >= 3 or
            any(w in question.lower() for w in ["importance", "benefits", "advantages", "steps", "process", "operators", "types"])
        )

        # Strict thresholds for intents A–E: require >= 0.80 else NOT_FOUND
        if intent in {'definition', 'capabilities', 'steps', 'importance', 'compare'} and top_score < 0.80:
            return "NOT_FOUND: This isn’t stated in your notes."

        # Prefer table rendering when a clear table is present
        if table is not None:
            headers, rows = table
            # Translate cells if needed (but keep tech tokens safe)
            def tr(text: str) -> str:
                t = self._clean_text_for_display(text)
                if lang in {'ta','si'}:
                    translated = None
                    protected_text, placeholders = self._protect_terms(t)
                    if GoogleTranslator is not None:
                        try:
                            translated = GoogleTranslator(target=lang).translate(protected_text)
                        except Exception:
                            translated = None
                    if translated is None:
                        mapping = self._get_offline_map(lang)
                        translated = self._apply_glossary(protected_text, mapping) if mapping else t
                    t = self._restore_terms(translated, placeholders)
                return t
            headers_t = [tr(h) for h in headers] if headers else []
            rows_t = [[tr(c) for c in r] for r in rows]
            table_text = self._render_text_table(headers_t, rows_t)
            lines.append(table_text)
        elif use_list and bullets and intent in {'capabilities', 'importance'}:
            # Translate each bullet if needed
            out_items: List[str] = []
            # Optional lead-in sentence above the list
            if lead_in:
                lead_text = lead_in.strip()
                # Cleanup spacing and minor run-ons conservatively
                lead_text = self._clean_text_for_display(lead_text)
                if lang in {'ta', 'si'}:
                    translated = None
                    protected_text, placeholders = self._protect_terms(lead_text)
                    if GoogleTranslator is not None:
                        try:
                            translated = GoogleTranslator(target=lang).translate(protected_text)
                        except Exception:
                            translated = None
                    if translated is None:
                        mapping = self._get_offline_map(lang)
                        translated = self._apply_glossary(protected_text, mapping) if mapping else lead_text
                    lead_text = self._restore_terms(translated, placeholders)
                lines.append(lead_text)

            for idx, item in enumerate(bullets[:8]):  # keep concise up to 8 items
                text_item = item.strip()
                # Cleanup spacing and minor run-ons conservatively
                text_item = self._clean_text_for_display(text_item)
                if lang in {'ta', 'si'}:
                    translated = None
                    protected_text, placeholders = self._protect_terms(text_item)
                    if GoogleTranslator is not None:
                        try:
                            translated = GoogleTranslator(target=lang).translate(protected_text)
                        except Exception:
                            translated = None
                    if translated is None:
                        mapping = self._get_offline_map(lang)
                        translated = self._apply_glossary(protected_text, mapping) if mapping else text_item
                    text_item = self._restore_terms(translated, placeholders)
                # Prefix bullets or numbers based on detected style
                if ordered:
                    prefix = f"{idx+1}. "
                else:
                    # Use source bullet char if available, else fallback to •
                    prefix = f"{(bullet_char or '•')} "
                out_items.append(f"{prefix}{text_item}")
            # Join as lines
            lines.extend(out_items)
        elif intent == 'steps' and top_only_chunks:
            # Numbered steps from the top chunk only
            steps = self._extract_numbered_items(top_only_chunks)
            if not steps:
                # Fallback to bullets if numbers absent
                steps = bullets
            out_items: List[str] = []
            for idx, item in enumerate(steps[:8], 1):
                text_item = self._clean_text_for_display(item.strip())
                if lang in {'ta', 'si'}:
                    translated = None
                    protected_text, placeholders = self._protect_terms(text_item)
                    if GoogleTranslator is not None:
                        try:
                            translated = GoogleTranslator(target=lang).translate(protected_text)
                        except Exception:
                            translated = None
                    if translated is None:
                        mapping = self._get_offline_map(lang)
                        translated = self._apply_glossary(protected_text, mapping) if mapping else text_item
                    text_item = self._restore_terms(translated, placeholders)
                out_items.append(f"{idx}. {text_item}")
            lines.extend(out_items)
        else:
            # Compose a concise paragraph ONLY from the top source chunk
            if top_only_chunks:
                raw = (top_only_chunks[0][0].text or '').strip()
            else:
                raw = result.answer.strip()
            # Cleanup spacing before sentence selection
            raw = self._clean_text_for_display(raw)
            # Take 2-3 reasonable sentences from the start
            sentences = [s.strip() for s in re.split(r'[.!?]+', raw) if 20 <= len(s.strip()) <= 300]
            answer_text = " ".join(sentences[:3]) if sentences else raw[:300]
            if lang in {'ta', 'si'}:
                translated = None
                protected_text, placeholders = self._protect_terms(answer_text)
                # Try online translator first if available
                if GoogleTranslator is not None:
                    try:
                        translated = GoogleTranslator(target=lang).translate(protected_text)
                    except Exception:
                        translated = None
                # Fallback to offline glossary-based translation
                if translated is None:
                    mapping = self._get_offline_map(lang)
                    translated = self._apply_glossary(protected_text, mapping) if mapping else answer_text
                answer_text = self._restore_terms(translated, placeholders)
            lines.append(answer_text)
        lines.append("")
        lines.append(f"{labels['sources']}:")
        # Confine citations to same section as the top chunk and limit to 1–2
        same_section_cits = []
        if result.source_chunks:
            top_chunk = result.source_chunks[0][0]
            top_section = (top_chunk.section or '')
            for c, s in result.source_chunks:
                if (c.section or '') == top_section:
                    same_section_cits.append(self._format_citation(c))
                if len(same_section_cits) >= 2:
                    break
        for cit in same_section_cits if same_section_cits else [self._format_citation(result.source_chunks[0][0])] if result.source_chunks else []:
            lines.append(f"• {cit}")
        lines.append("")
        lines.append(f"{labels['confidence']}: {self._confidence_label(top_score)}")
        return "\n".join(lines)

    def _extract_list_items(self, source_chunks: List[Tuple[ProcessedChunk, float]]) -> Tuple[Optional[str], List[str], bool, Optional[str]]:
        """Scan top source chunks for bullet/numbered list items.
        Returns: (lead_in_sentence, items, ordered_detected, bullet_char_if_any)
        """
        items: List[str] = []
        ordered_detected = False
        bullet_char: Optional[str] = None
        lead_in: Optional[str] = None
        # Look through top 3 chunks for list patterns and capture the line before the first list as lead-in
        for i, (chunk, score) in enumerate(source_chunks[:3]):
            lines = chunk.text.splitlines()
            for li, line in enumerate(lines):
                # Match bullet char (•, -, *, or other Unicode bullets like \u25AA) or ordered like 1. / 1)
                m = re.match(r"^\s*((?P<bullet>[\u2022\-\*\u25AA\u25CF\u25E6\u2219])\s+|(?P<num>\d+)[\.)]\s+)(?P<content>.+)$", line)
                if m:
                    content = m.group('content').strip()
                    if content:
                        items.append(content)
                        if m.group('num'):
                            ordered_detected = True
                        if m.group('bullet') and bullet_char is None:
                            bullet_char = m.group('bullet')
                        # Capture lead-in as previous non-empty line once
                        if lead_in is None and li > 0:
                            # search backwards for the nearest non-empty line
                            for back in range(li-1, max(li-5, -1), -1):
                                prev = lines[back].strip()
                                if prev:
                                    lead_in = prev
                                    break
        return lead_in, items, ordered_detected, bullet_char

    def _extract_numbered_items(self, source_chunks: List[Tuple[ProcessedChunk, float]]) -> List[str]:
        items: List[str] = []
        for chunk, _ in source_chunks:
            for line in chunk.text.splitlines():
                m = re.match(r"^\s*\d+[\.)]\s+(.+)$", line)
                if m:
                    items.append(m.group(1).strip())
        return items

    def _extract_table(self, source_chunks: List[Tuple[ProcessedChunk, float]]):
        """Detect a simple table from the top chunk text. Supports pipe ('|') or tab-separated rows.
        Returns (headers, rows) or None if no clear table found.
        """
        if not source_chunks:
            return None
        text = source_chunks[0][0].text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Pipe tables
        pipe_rows = [ln for ln in lines if '|' in ln and len(ln.split('|')) >= 3]
        if len(pipe_rows) >= 2:
            # Remove markdown separator lines (---|---)
            clean = [ln for ln in pipe_rows if not re.match(r"^\|?\s*:?[-]{2,}.*\|.*", ln)]
            cells = [ [c.strip() for c in ln.strip('|').split('|')] for ln in clean ]
            if len(cells) >= 2:
                headers = cells[0]
                rows = cells[1:]
                return headers, rows
        # Tab-separated fallback
        tab_rows = [ln for ln in lines if '\t' in ln]
        if len(tab_rows) >= 2:
            cells = [ [c.strip() for c in ln.split('\t') if c.strip()] for ln in tab_rows ]
            if len(cells) >= 2 and max(len(r) for r in cells) >= 2:
                headers = cells[0]
                rows = cells[1:]
                return headers, rows
        # Multi-space column heuristic: lines with 2+ groups separated by 3+ spaces
        spaced = [re.split(r"\s{3,}", ln) for ln in lines if re.search(r"\s{3,}", ln)]
        if len(spaced) >= 2 and max(len(r) for r in spaced) >= 2:
            headers = [c.strip() for c in spaced[0]]
            rows = [[c.strip() for c in r] for r in spaced[1:]]
            return headers, rows
        return None

    def _render_text_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render a simple monospaced text table suitable for the chat area.
        Example:
        Col A | Col B
        ----- | -----
        a1    | b1
        """
        cols = max([len(headers) if headers else 0] + [len(r) for r in rows] + [1])
        widths = [0]*cols
        if headers:
            for i, h in enumerate(headers):
                widths[i] = max(widths[i], len(h))
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        def fmt_row(r: List[str]) -> str:
            cells = []
            for i in range(cols):
                val = r[i] if i < len(r) else ''
                cells.append(val.ljust(widths[i]))
            return ' | '.join(cells).rstrip()
        out = []
        if headers:
            out.append(fmt_row(headers))
            out.append(' | '.join(['-'*w for w in widths]).rstrip())
        for r in rows:
            out.append(fmt_row(r))
        return "\n".join(out)
    def _format_citation(self, chunk: ProcessedChunk) -> str:
        """Format a single citation string consistent with global guard rules."""
        if chunk.page is not None:
            return CITATION_FORMAT.format(filename=chunk.filename, page=chunk.page)
        elif chunk.section:
            return SECTION_CITATION_FORMAT.format(filename=chunk.filename, section=chunk.section)
        else:
            return f"({chunk.filename})"

    def _filter_bullets_by_intent(self, bullets: List[str], question: str) -> List[str]:
        """Filter or re-rank bullets to better match intent. For example,
        for queries like 'What can PHP do', prefer bullets starting with 'PHP can ...'.
        """
        if not bullets:
            return bullets
        q = question.lower().strip()
        # If asking capabilities
        if 'what can php do' in q or ('php' in q and 'can' in q):
            preferred = [b for b in bullets if re.match(r"^\s*php\s+can\b", b, flags=re.IGNORECASE)]
            others = [b for b in bullets if b not in preferred]
            return preferred + others
        # If asking 'what is php', prefer definition-like bullets
        if 'what is php' in q:
            preferred = [b for b in bullets if re.search(r"\b(acronym|open source|executed on the server|free to download)\b", b, flags=re.IGNORECASE)]
            others = [b for b in bullets if b not in preferred]
            return preferred + others
        return bullets

    def _protect_terms(self, text: str, extra: Optional[List[str]] = None) -> Tuple[str, Dict[str, str]]:
        """Protect proper nouns, technical tokens, law titles, URLs, versions, and optional extras from translation.
        Returns text with placeholders and a map to restore.
        """
        if not text:
            return text, {}
        out = text
        placeholders: Dict[str, str] = {}

        # Robust placeholder generator using unlikely sentinel tokens that translators won't alter
        counter = { 'n': 0 }
        def next_key(prefix: str) -> str:
            counter['n'] += 1
            # Use section-sign wrapped tokens to avoid accidental translation or casing changes
            return f"§{prefix}{counter['n']}§"

        # 1) Static tech/org tokens
        static_tokens = [
            'PHP', 'HTML', 'CSS', 'JavaScript', 'JSON', 'XML', 'REST', 'CRUD',
            'SQL', 'MySQL', 'PostgreSQL', 'MariaDB', 'IPv4', 'IPv6', 'TCP/IP',
            'HTTP', 'HTTPS', '.php', '.html', '.css', '.js', 'HyperText Preprocessor',
            'GDPR', 'ICO', 'BTEC', 'Pearson', 'UK', 'Sri Lanka'
        ]
        if extra:
            static_tokens.extend(extra)

        def put(term: str, key: Optional[str] = None):
            nonlocal out
            if not term:
                return
            if key is None:
                key = next_key('T')
            if term in out and key not in placeholders:
                placeholders[key] = term
                out = out.replace(term, key)

        # Replace static tokens first
        for term in static_tokens:
            put(term)

        # 2) URLs/emails
        for m in re.finditer(r"https?://\S+|www\.\S+|\S+@\S+", out):
            term = m.group(0)
            put(term, next_key('U'))

        # 3) Version numbers and years
        for m in re.finditer(r"\b\d+\.\d+(?:\.\d+)*\b|\b(19|20)\d{2}\b", out):
            term = m.group(0)
            put(term, next_key('V'))

        # 4) All-caps or camel-case acronyms with slashes/dashes/underscores
        for m in re.finditer(r"\b[A-Z]{2,}(?:/[A-Z]{2,})*\b|\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b", out):
            term = m.group(0)
            put(term, next_key('A'))

        # 5) Law/Act titles (e.g., Data Protection Act 1998/2018)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act(?:\s+of)?\s+(19|20)\d{2}\b", out):
            term = m.group(0)
            put(term, next_key('L'))

        # 6) Proper noun phrases (2+ capitalized words), conservative
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", out):
            term = m.group(0)
            # Skip sentence starts followed by common words
            if term.split()[0] in {"The", "A", "An"}:
                continue
            put(term, next_key('P'))

        return out, placeholders

    def _restore_terms(self, text: str, placeholders: Dict[str, str]) -> str:
        if not placeholders or not text:
            return text
        out = text
        # Replace longer keys first to avoid partial collisions
        for key in sorted(placeholders.keys(), key=len, reverse=True):
            term = placeholders[key]
            # Case-insensitive restoration to handle translators that change key casing (e.g., §P12§ -> §p12§)
            pattern = re.compile(re.escape(key), flags=re.IGNORECASE)
            out = pattern.sub(term, out)
        return out

    def _clean_text_for_display(self, text: str) -> str:
        """Conservative cleanup to fix spacing and common OCR run-ons without altering technical tokens.
        - Normalize whitespace
        - Ensure a space after punctuation
        - Optionally split very long lowercase tokens using wordninja
        """
        if not text:
            return text
        # Normalize weird spaces
        t = re.sub(r"\s+", " ", text)
        # Space after punctuation if missing
        t = re.sub(r"([.,;:!?])(\S)", r"\1 \2", t)
        # Split candidate run-on tokens
        def split_token(tok: str) -> str:
            if wordninja is None:
                return tok
            # Only split long, all-lowercase alphabetic tokens (avoid breaking acronyms/code)
            if len(tok) >= 12 and tok.isalpha() and tok.islower():
                parts = wordninja.split(tok)
                if len(parts) >= 2:
                    return " ".join(parts)
            return tok
        tokens = t.split(' ')
        tokens = [split_token(tok) for tok in tokens]
        t = ' '.join(tokens)
        return t.strip()


def query_notes(course_name: str, question: str) -> SearchResult:
    """Convenience function to query notes for a specific course."""
    pipeline = RetrievalPipeline()
    return pipeline.query(course_name, question)


def studybot_query(course_name: str, question: str, target_lang: Optional[str] = None) -> str:
    """High-level StudyBot entrypoint returning fully formatted output.

    TARGET_LANG may be one of {'en','ta','si'} or None/empty to auto-detect based on the QUESTION.
    """
    pipeline = RetrievalPipeline()
    result = pipeline.query(course_name, question)
    return pipeline.format_studybot(question, result, target_lang)


def studybot_query_with_meta(course_name: str, question: str, target_lang: Optional[str] = None) -> Tuple[str, float]:
    """Return formatted StudyBot output and the top rerank score for debugging/telemetry."""
    pipeline = RetrievalPipeline()
    result = pipeline.query(course_name, question)
    top_score = result.source_chunks[0][1] if result.source_chunks else 0.0
    formatted = pipeline.format_studybot(question, result, target_lang)
    return formatted, float(top_score)


def format_search_result(result: SearchResult, show_snippets: bool = SHOW_SOURCE_SNIPPETS) -> str:
    """Format a search result for display."""
    output = []
    
    # Query
    output.append(f"Question: {result.query}")
    output.append("")
    
    # Answer
    if result.found_in_notes:
        output.append("Answer:")
        output.append(result.answer)
        output.append("")
        
        # Citations
        if result.citations:
            output.append("Sources:")
            for citation in result.citations:
                output.append(f"  • {citation}")
            output.append("")
        
        # Confidence
        output.append(f"Confidence: {result.confidence:.2f}")
        
        # Source snippets (optional)
        if show_snippets and result.source_chunks:
            output.append("")
            output.append("Source Snippets:")
            pipeline = RetrievalPipeline()
            snippets = pipeline.get_source_snippets(result)
            
            for i, snippet in enumerate(snippets[:3], 1):  # Show top 3
                output.append(f"  {i}. {snippet['filename']} (Score: {snippet['score']:.3f})")
                output.append(f"     {snippet['text']}")
                output.append("")
    else:
        output.append("Answer: Not found in your notes.")
        output.append("")
        output.append("This question cannot be answered based on the content in your notes.")
    
    return "\n".join(output)
