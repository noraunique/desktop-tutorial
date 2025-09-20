"""Evaluation framework for the Notes Q&A system."""

import json
import os
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from retrieval import query_notes, SearchResult
from config import AUX_DIR, GOLD_QUESTIONS_FILE, EVALUATION_METRICS


@dataclass
class EvaluationResult:
    """Results of evaluating the Q&A system."""
    answerable_rate: float
    precision_at_k: float
    citation_coverage: float
    hallucination_rate: float
    avg_confidence: float
    avg_latency: float
    total_questions: int
    detailed_results: List[Dict]


class Evaluator:
    """Evaluates the performance of the Notes Q&A system."""
    
    def __init__(self):
        self.gold_questions = []
    
    def load_gold_questions(self, filepath: str = GOLD_QUESTIONS_FILE) -> bool:
        """Load gold standard questions from JSON file."""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.gold_questions = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading gold questions: {str(e)}")
            return False
    
    def create_sample_gold_questions(self, course_name: str) -> str:
        """Create sample gold questions file."""
        sample_questions = [
            {
                "id": 1,
                "question": "What is the time complexity of quicksort?",
                "course": course_name,
                "expected_topics": ["quicksort", "time complexity", "O(n log n)", "O(n²)"],
                "should_find_answer": True,
                "difficulty": "easy"
            },
            {
                "id": 2,
                "question": "How does merge sort work?",
                "course": course_name,
                "expected_topics": ["merge sort", "divide and conquer", "merging"],
                "should_find_answer": True,
                "difficulty": "medium"
            },
            {
                "id": 3,
                "question": "What is the difference between arrays and linked lists?",
                "course": course_name,
                "expected_topics": ["arrays", "linked lists", "memory", "access time"],
                "should_find_answer": True,
                "difficulty": "easy"
            },
            {
                "id": 4,
                "question": "Explain binary tree traversal methods",
                "course": course_name,
                "expected_topics": ["binary tree", "traversal", "inorder", "preorder", "postorder"],
                "should_find_answer": True,
                "difficulty": "medium"
            },
            {
                "id": 5,
                "question": "What is quantum computing?",
                "course": course_name,
                "expected_topics": [],
                "should_find_answer": False,
                "difficulty": "hard"
            }
        ]
        
        os.makedirs(AUX_DIR, exist_ok=True)
        with open(GOLD_QUESTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_questions, f, indent=2)
        
        return GOLD_QUESTIONS_FILE
    
    def evaluate_course(self, course_name: str) -> EvaluationResult:
        """Evaluate the Q&A system on a specific course."""
        if not self.gold_questions:
            if not self.load_gold_questions():
                # Create sample questions if none exist
                self.create_sample_gold_questions(course_name)
                self.load_gold_questions()
        
        # Filter questions for this course
        course_questions = [q for q in self.gold_questions if q.get('course') == course_name]
        
        if not course_questions:
            raise ValueError(f"No gold questions found for course: {course_name}")
        
        detailed_results = []
        total_latency = 0
        
        print(f"Evaluating {len(course_questions)} questions for course: {course_name}")
        
        for question_data in course_questions:
            start_time = time.time()
            
            try:
                result = query_notes(course_name, question_data['question'])
                latency = time.time() - start_time
                total_latency += latency
                
                # Evaluate this result
                evaluation = self._evaluate_single_result(question_data, result, latency)
                detailed_results.append(evaluation)
                
                print(f"  ✓ Q{question_data['id']}: {evaluation['answerable']}")
                
            except Exception as e:
                latency = time.time() - start_time
                total_latency += latency
                
                evaluation = {
                    'question_id': question_data['id'],
                    'question': question_data['question'],
                    'error': str(e),
                    'answerable': False,
                    'has_citations': False,
                    'confidence': 0.0,
                    'latency': latency,
                    'hallucination': False
                }
                detailed_results.append(evaluation)
                print(f"  ✗ Q{question_data['id']}: Error - {str(e)}")
        
        # Calculate aggregate metrics
        return self._calculate_metrics(detailed_results, total_latency)
    
    def _evaluate_single_result(self, question_data: Dict, result: SearchResult, latency: float) -> Dict:
        """Evaluate a single Q&A result."""
        expected_answerable = question_data.get('should_find_answer', True)
        expected_topics = question_data.get('expected_topics', [])
        
        # Check if answer was found
        answerable = result.found_in_notes
        
        # Check citation coverage
        has_citations = len(result.citations) > 0
        
        # Check for topic coverage (simple keyword matching)
        topic_coverage = 0
        if expected_topics and result.answer:
            answer_lower = result.answer.lower()
            matched_topics = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
            topic_coverage = matched_topics / len(expected_topics)
        
        # Check for hallucination (answered when shouldn't have)
        hallucination = answerable and not expected_answerable
        
        return {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'expected_answerable': expected_answerable,
            'answerable': answerable,
            'has_citations': has_citations,
            'topic_coverage': topic_coverage,
            'confidence': result.confidence,
            'latency': latency,
            'hallucination': hallucination,
            'answer_length': len(result.answer),
            'num_sources': len(result.source_chunks)
        }
    
    def _calculate_metrics(self, detailed_results: List[Dict], total_latency: float) -> EvaluationResult:
        """Calculate aggregate evaluation metrics."""
        total_questions = len(detailed_results)
        
        if total_questions == 0:
            return EvaluationResult(0, 0, 0, 0, 0, 0, 0, [])
        
        # Answerable rate: % of questions that should be answerable and were answered
        expected_answerable = [r for r in detailed_results if r.get('expected_answerable', True)]
        actually_answered = [r for r in expected_answerable if r.get('answerable', False)]
        answerable_rate = len(actually_answered) / len(expected_answerable) if expected_answerable else 0
        
        # Citation coverage: % of answered questions that have citations
        answered_questions = [r for r in detailed_results if r.get('answerable', False)]
        with_citations = [r for r in answered_questions if r.get('has_citations', False)]
        citation_coverage = len(with_citations) / len(answered_questions) if answered_questions else 0
        
        # Hallucination rate: % of questions answered when they shouldn't be
        should_not_answer = [r for r in detailed_results if not r.get('expected_answerable', True)]
        hallucinations = [r for r in should_not_answer if r.get('answerable', False)]
        hallucination_rate = len(hallucinations) / len(should_not_answer) if should_not_answer else 0
        
        # Precision@k: Average topic coverage for answered questions
        topic_coverages = [r.get('topic_coverage', 0) for r in answered_questions]
        precision_at_k = sum(topic_coverages) / len(topic_coverages) if topic_coverages else 0
        
        # Average confidence and latency
        confidences = [r.get('confidence', 0) for r in detailed_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_latency = total_latency / total_questions
        
        return EvaluationResult(
            answerable_rate=answerable_rate,
            precision_at_k=precision_at_k,
            citation_coverage=citation_coverage,
            hallucination_rate=hallucination_rate,
            avg_confidence=avg_confidence,
            avg_latency=avg_latency,
            total_questions=total_questions,
            detailed_results=detailed_results
        )
    
    def print_evaluation_report(self, result: EvaluationResult):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"Total Questions: {result.total_questions}")
        print(f"Average Latency: {result.avg_latency:.3f}s")
        print()
        
        print("METRICS:")
        print(f"  Answerable Rate:    {result.answerable_rate:.1%}")
        print(f"  Citation Coverage:  {result.citation_coverage:.1%}")
        print(f"  Precision@K:        {result.precision_at_k:.1%}")
        print(f"  Hallucination Rate: {result.hallucination_rate:.1%}")
        print(f"  Average Confidence: {result.avg_confidence:.3f}")
        
        # Success criteria check
        print("\nSUCCESS CRITERIA:")
        print(f"  ≥90% Citation Coverage: {'✅' if result.citation_coverage >= 0.9 else '❌'}")
        print(f"  ≥85% Answerable Rate:   {'✅' if result.answerable_rate >= 0.85 else '❌'}")
        print(f"  0% Hallucination Rate:  {'✅' if result.hallucination_rate == 0 else '❌'}")
        print(f"  <1s Average Latency:    {'✅' if result.avg_latency < 1.0 else '❌'}")


def evaluate_course(course_name: str) -> EvaluationResult:
    """Convenience function to evaluate a course."""
    evaluator = Evaluator()
    return evaluator.evaluate_course(course_name)
