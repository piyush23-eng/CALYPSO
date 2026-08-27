#!/usr/bin/env python3
"""CLI Script to execute the full GATE-CS-Bench Benchmark Suite."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backend.engine import GGUFInferenceEngine
from src.eval.gate_bench import GATECSBench
from src.utils.logger import setup_logger

logger = setup_logger("run_benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run standardized GATE-CS-Bench evaluation.")
    parser.add_argument("--test-split", type=str, default="data/splits/test.jsonl", help="Test split path.")
    parser.add_argument("--sample", type=int, default=None, help="Optional limit on number of test samples.")
    parser.add_argument("--model-path", type=str, default="models/gguf/gate-qwen-1.5b-q4_k_m.gguf", help="GGUF model path.")
    parser.add_argument("--output-json", type=str, default="eval_results/benchmark_summary_v2.json", help="Path to save output JSON.")
    parser.add_argument("--output-report", type=str, default="eval_results/BENCHMARK_REPORT.md", help="Path to save markdown report.")
    args = parser.parse_args()

    logger.info("Initializing GATE-CS-Bench Runner...")
    bench = GATECSBench(data_path=args.test_split)
    engine = GGUFInferenceEngine(model_path=args.model_path)

    def inference_fn(item):
        q = item.get("question_text") or item.get("question", "")
        options = item.get("options", {})
        sub = item.get("subject", "Algorithms")
        top = item.get("topic", "General")
        q_type = item.get("question_type", "MCQ")
        marks = item.get("marks", 2)
        return engine.solve_sync(
            subject=sub,
            topic=top,
            question_type=q_type,
            marks=marks,
            question=q,
            options=options,
        )

    summary = bench.evaluate_model(
        inference_fn=inference_fn,
        limit=args.sample,
        model_name="CALYPSO-GATE-Qwen-1.5B (GGUF-Q4_K_M)",
    )

    # Save JSON summary
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {args.output_json}")

    # Generate and save markdown report
    report_md = bench.generate_markdown_report(summary)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved benchmark Markdown report to {args.output_report}")

    print("\n" + report_md + "\n")


if __name__ == "__main__":
    main()
