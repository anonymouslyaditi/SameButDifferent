"""Evaluation metrics for paraphrase generation."""

from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import evaluate library with alias to avoid conflicts
import evaluate as hf_evaluate

# Lazy imports to avoid loading all at once
_bleu_scorer = None
_rouge_scorer = None


def get_bleu_scorer():
    """Get or create BLEU scorer."""
    global _bleu_scorer
    if _bleu_scorer is None:
        _bleu_scorer = hf_evaluate.load("sacrebleu")
    return _bleu_scorer


def get_rouge_scorer():
    """Get or create ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = hf_evaluate.load("rouge")
    return _rouge_scorer


class MetricsCalculator:
    """Calculate evaluation metrics for paraphrase generation."""

    def __init__(
        self,
        metrics: List[str] = ["bleu", "rouge", "semantic_similarity"],
    ):
        self.metrics = metrics
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU score."""
        scorer = get_bleu_scorer()
        # BLEU expects references as list of lists
        refs = [[ref] for ref in references]
        result = scorer.compute(predictions=predictions, references=refs)
        return {
            "bleu": result["score"],
            "bleu_precisions": result["precisions"]
        }
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scorer = get_rouge_scorer()
        result = scorer.compute(predictions=predictions, references=references)
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"]
        }
    
    def calculate_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate semantic similarity using TF-IDF cosine similarity (no model download required)."""
        similarities = []
        for pred, ref in zip(predictions, references):
            # Use TF-IDF vectorization for semantic similarity
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([ref, pred])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities.append(similarity)
            except ValueError:
                # Handle empty strings
                similarities.append(0.0)

        return {
            "semantic_similarity": np.mean(similarities)
        }
    
    def calculate_length_ratio(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate length preservation ratio."""
        ratios = []
        for pred, ref in zip(predictions, references):
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            if ref_len > 0:
                ratios.append(pred_len / ref_len)
        
        return {
            "avg_length_ratio": np.mean(ratios),
            "min_length_ratio": np.min(ratios),
            "max_length_ratio": np.max(ratios),
            "length_preservation_rate": sum(1 for r in ratios if r >= 0.8) / len(ratios)
        }
    
    def calculate_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        results = {}
        
        if "bleu" in self.metrics:
            results.update(self.calculate_bleu(predictions, references))
        
        if "rouge" in self.metrics:
            results.update(self.calculate_rouge(predictions, references))
        
        if "semantic_similarity" in self.metrics:
            results.update(self.calculate_semantic_similarity(predictions, references))
        
        # Always calculate length metrics
        results.update(self.calculate_length_ratio(predictions, references))
        
        return results

