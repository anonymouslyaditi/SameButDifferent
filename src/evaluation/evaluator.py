"""Comprehensive evaluator for comparing paraphrase generators."""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .metrics import MetricsCalculator
from ..inference.generator import ParaphraseGenerator


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    latency: Dict[str, float]
    samples: List[Dict]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "latency": self.latency,
            "samples": self.samples,
            "timestamp": self.timestamp
        }
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ParaphraseEvaluator:
    """Evaluate and compare paraphrase generators."""
    
    def __init__(
        self,
        metrics: List[str] = ["bleu", "rouge", "semantic_similarity"],
        results_dir: str = "./outputs/evaluation"
    ):
        self.metrics_calculator = MetricsCalculator(metrics=metrics)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate(
        self,
        generator: ParaphraseGenerator,
        test_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        model_name: str = "model",
        measure_latency: bool = True,
        warmup_runs: int = 3,
        num_runs: int = 10,
        **generation_kwargs
    ) -> EvaluationResult:
        """Evaluate a single generator."""
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        # Generate paraphrases with latency measurement
        if measure_latency:
            result = generator.generate(
                test_texts[0],  # Use first text for latency
                measure_latency=True,
                warmup_runs=warmup_runs,
                num_runs=num_runs,
                **generation_kwargs
            )
            latency = result["latency"]
        else:
            latency = {}
        
        # Generate all paraphrases
        print("Generating paraphrases for all test samples...")
        predictions = generator.generate_batch(test_texts, **generation_kwargs)
        
        # Use original texts as references if not provided
        references = reference_texts if reference_texts else test_texts
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.metrics_calculator.calculate_all(predictions, references)
        
        # Create samples
        samples = []
        for i, (src, pred) in enumerate(zip(test_texts[:5], predictions[:5])):
            samples.append({
                "source": src[:200] + "..." if len(src) > 200 else src,
                "prediction": pred[:200] + "..." if len(pred) > 200 else pred
            })
        
        result = EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            latency=latency,
            samples=samples,
            timestamp=datetime.now().isoformat()
        )
        
        # Print results
        self._print_results(result)
        
        return result
    
    def compare(
        self,
        results: List[EvaluationResult]
    ) -> pd.DataFrame:
        """Compare multiple evaluation results."""
        comparison_data = []
        
        for result in results:
            row = {"Model": result.model_name}
            row.update(result.metrics)
            if result.latency:
                row["Latency (s)"] = result.latency.get("total_time_seconds", 0)
                row["Tokens/s"] = result.latency.get("tokens_per_second", 0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = os.path.join(self.results_dir, "comparison.csv")
        df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to {comparison_path}")
        
        return df
    
    def _print_results(self, result: EvaluationResult):
        """Print evaluation results."""
        print(f"\n--- Results for {result.model_name} ---")
        print("\nMetrics:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        if result.latency:
            print("\nLatency:")
            for key, value in result.latency.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\nSample outputs:")
        for i, sample in enumerate(result.samples[:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Source: {sample['source'][:100]}...")
            print(f"    Output: {sample['prediction'][:100]}...")

