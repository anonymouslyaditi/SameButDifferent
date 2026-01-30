#!/usr/bin/env python3
"""Unified CLI for Paraphrase Generation System.

Usage:
    paraphrase compare <text> [options]
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="paraphrase",
        description="Paraphrase Generation System - Generate high-quality paraphrases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paraphrase compare --file input.txt --show_metrics --include_llm
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare CPG vs LLM baseline")
    _add_compare_args(compare_parser)

    return parser


def _add_compare_args(parser):
    """Add comparison arguments."""
    parser.add_argument("text", nargs="?", help="Text to compare")
    parser.add_argument("--file", "-f", help="Read text from file")
    parser.add_argument("--cpg_model", default="google/flan-t5-base")
    parser.add_argument("--pretrained_path", default="./outputs/pretrained/flan-t5-base",
                       help="Path to local pretrained model")
    parser.add_argument("--finetuned_path", default="./outputs/checkpoints/final_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--show_metrics", action="store_true")
    parser.add_argument("--include_llm", action="store_true",
                       help="Include LLM baseline comparison (requires more memory)")
    parser.add_argument("--llm_model", default="google/flan-t5-large",
                       help="LLM model for comparison (default: google/flan-t5-large)")


def cmd_compare(args):
    """Execute compare command - compare CPG vs LLM with quality and latency metrics."""
    from src.models.cpg_model import CustomParaphraseGenerator
    import os
    import time

    text = args.text
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read().strip()

    if not text:
        print("Error: Provide text or --file")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PARAPHRASE GENERATOR COMPARISON")
    print("=" * 70)
    print(f"\nOriginal Text ({len(text.split())} words):")
    print("-" * 70)
    print(text)

    results = {}
    latencies = {}

    # Load and run CPG (Fine-tuned)
    print("\n" + "=" * 70)
    print("CPG (Fine-tuned FLAN-T5 with LoRA)")
    print("=" * 70)
    if os.path.exists(args.finetuned_path):
        cpg = CustomParaphraseGenerator.load(args.finetuned_path)
        print(f"Loaded from: {args.finetuned_path}")

        # Warmup run
        _ = cpg.generate(text[:100])

        # Measure latency
        start_time = time.perf_counter()
        cpg_result = cpg.generate(text)
        cpg_latency = time.perf_counter() - start_time

        results["CPG"] = cpg_result
        latencies["CPG"] = cpg_latency

        print(f"\nOutput ({len(cpg_result.split())} words):")
        print("-" * 70)
        print(cpg_result)
        print(f"\nLatency: {cpg_latency:.3f}s")
    else:
        print(f"ERROR: Fine-tuned model not found at {args.finetuned_path}")
        print("Run 'paraphrase train' first to create a fine-tuned model.")
        sys.exit(1)

    # Load and run LLM baseline if requested
    if args.include_llm:
        print("\n" + "=" * 70)
        print(f"LLM Baseline ({args.llm_model})")
        print("=" * 70)
        try:
            from src.models.llm_baseline import LLMBaseline
            llm = LLMBaseline(model_name=args.llm_model)
            print(f"Loaded: {args.llm_model}")

            # Warmup run
            _ = llm.generate(text[:100])

            # Measure latency
            start_time = time.perf_counter()
            llm_result = llm.generate(text)
            llm_latency = time.perf_counter() - start_time

            results["LLM"] = llm_result
            latencies["LLM"] = llm_latency

            print(f"\nOutput ({len(llm_result.split())} words):")
            print("-" * 70)
            print(llm_result)
            print(f"\nLatency: {llm_latency:.3f}s")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Skipping LLM comparison.")

    # Show comparison metrics
    if args.show_metrics and results:
        print("\n" + "=" * 70)
        print("COMPARISON: TEXT QUALITY METRICS")
        print("=" * 70)
        try:
            from src.evaluation.metrics import MetricsCalculator
            calculator = MetricsCalculator()

            all_metrics = {}
            for model_name, result in results.items():
                all_metrics[model_name] = calculator.calculate_all([result], [text])

            # Determine columns
            models = list(results.keys())
            header = f"{'Metric':<25}" + "".join(f"{m:>15}" for m in models)
            print(f"\n{header}")
            print("-" * (25 + 15 * len(models)))

            metric_names = [
                ("BLEU Score", "bleu"),
                ("ROUGE-1", "rouge1"),
                ("ROUGE-2", "rouge2"),
                ("ROUGE-L", "rougeL"),
                ("Semantic Similarity", "semantic_similarity"),
                ("Length Ratio", "avg_length_ratio"),
            ]

            for display_name, key in metric_names:
                row = f"{display_name:<25}"
                for model in models:
                    val = all_metrics[model].get(key, 0)
                    row += f"{val:>15.4f}"
                print(row)

            # Latency comparison
            print("\n" + "=" * 70)
            print("COMPARISON: SYSTEM LATENCY")
            print("=" * 70)
            header = f"{'Metric':<25}" + "".join(f"{m:>15}" for m in models)
            print(f"\n{header}")
            print("-" * (25 + 15 * len(models)))

            # Total latency
            row = f"{'Latency (seconds)':<25}"
            for model in models:
                row += f"{latencies[model]:>15.3f}"
            print(row)

            # Tokens per second (approximate)
            row = f"{'Tokens/second (approx)':<25}"
            for model in models:
                output_tokens = int(len(results[model].split()) * 1.3)
                tps = output_tokens / latencies[model] if latencies[model] > 0 else 0
                row += f"{tps:>15.1f}"
            print(row)

            # Speedup (if LLM included)
            if "LLM" in models and "CPG" in models:
                speedup = latencies["LLM"] / latencies["CPG"] if latencies["CPG"] > 0 else 0
                print(f"\n{'CPG Speedup vs LLM':<25}{speedup:>15.2f}x")

        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

