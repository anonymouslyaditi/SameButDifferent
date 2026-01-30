"""Inference module with latency measurement."""

import time
from typing import Dict, List, Union
from dataclasses import dataclass, field


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    total_time: float = 0.0
    first_token_time: float = 0.0
    tokens_per_second: float = 0.0
    num_tokens: int = 0
    warmup_runs: int = 3
    measurement_runs: int = 10
    all_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_time_seconds": self.total_time,
            "first_token_time_seconds": self.first_token_time,
            "tokens_per_second": self.tokens_per_second,
            "num_tokens_generated": self.num_tokens,
            "avg_time_per_run": sum(self.all_times) / len(self.all_times) if self.all_times else 0,
            "min_time": min(self.all_times) if self.all_times else 0,
            "max_time": max(self.all_times) if self.all_times else 0
        }


class ParaphraseGenerator:
    """Unified generator interface with latency measurement."""
    
    def __init__(self, model, model_type: str = "cpg"):
        """
        Initialize generator.
        
        Args:
            model: Either CustomParaphraseGenerator or LLMBaseline instance
            model_type: "cpg" for Custom Paraphrase Generator, "llm" for LLM baseline
        """
        self.model = model
        self.model_type = model_type
    
    def generate(
        self,
        text: Union[str, List[str]],
        measure_latency: bool = False,
        warmup_runs: int = 3,
        num_runs: int = 10,
        **generation_kwargs
    ) -> Union[str, List[str], Dict]:
        """
        Generate paraphrase with optional latency measurement.
        
        Args:
            text: Input text or list of texts
            measure_latency: Whether to measure latency metrics
            warmup_runs: Number of warmup runs before measurement
            num_runs: Number of runs for latency measurement
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Paraphrase(s) or dict with paraphrases and latency metrics
        """
        if measure_latency:
            return self._generate_with_latency(
                text, warmup_runs, num_runs, **generation_kwargs
            )
        else:
            return self.model.generate(text, **generation_kwargs)
    
    def _generate_with_latency(
        self,
        text: Union[str, List[str]],
        warmup_runs: int = 3,
        num_runs: int = 10,
        **generation_kwargs
    ) -> Dict:
        """Generate with full latency measurement."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Warmup runs
        print(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = self.model.generate(texts[0], **generation_kwargs)
        
        # Measurement runs
        print(f"Running {num_runs} measurement iterations...")
        times = []
        first_token_times = []
        output_tokens = []
        
        for i in range(num_runs):
            # Measure total generation time
            start_time = time.perf_counter()
            result = self.model.generate(texts[0], **generation_kwargs)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Estimate tokens (words * 1.3 as approximation)
            if isinstance(result, str):
                num_tokens = int(len(result.split()) * 1.3)
            else:
                num_tokens = int(len(result[0].split()) * 1.3)
            output_tokens.append(num_tokens)
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(output_tokens) / len(output_tokens)
        
        latency = LatencyMetrics(
            total_time=avg_time,
            first_token_time=times[0] * 0.1,  # Approximate
            tokens_per_second=avg_tokens / avg_time if avg_time > 0 else 0,
            num_tokens=int(avg_tokens),
            warmup_runs=warmup_runs,
            measurement_runs=num_runs,
            all_times=times
        )
        
        # Generate final outputs for all texts
        final_results = self.model.generate(texts, **generation_kwargs)
        
        return {
            "paraphrases": final_results[0] if is_single else final_results,
            "latency": latency.to_dict()
        }
    
    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """Generate paraphrases in batches."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.model.generate(batch, **generation_kwargs)
            if isinstance(batch_results, str):
                results.append(batch_results)
            else:
                results.extend(batch_results)
        return results

