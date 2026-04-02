"""
Predictive Analytics Agent — CLI entry point.

Runs the full ML pipeline on any CSV/Excel dataset:
profiling, preprocessing, feature selection, model training,
evaluation, and report generation. No API key required.

Usage:
    python main.py data/sample_customers.csv
    python main.py your_data.xlsx --output reports/
"""

import argparse
import os
import sys

from agents.orchestrator import MLPipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Automated ML pipeline — train, evaluate, and report"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to a CSV or Excel file to analyze",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for reports (default: output/)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with GridSearchCV",
    )

    args = parser.parse_args()

    # Default to sample data if no file provided
    if not args.filepath:
        sample = os.path.join(os.path.dirname(__file__), "data", "sample_customers.csv")
        if os.path.exists(sample):
            args.filepath = sample
            print("No file provided — using sample dataset.\n")
        else:
            print("Error: Please provide a data file path.")
            print("Usage: python main.py data.csv")
            sys.exit(1)

    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)

    print("=" * 60)
    print("  PREDICTIVE ANALYTICS PIPELINE")
    print(f"  Data: {os.path.basename(args.filepath)}")
    print(f"  No API key required — runs entirely locally")
    print("=" * 60)

    orch = MLPipelineOrchestrator(output_dir=args.output)
    report_path = orch.run(args.filepath, tune=args.tune)

    print("\n" + "=" * 60)
    print(f"  Report ready: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
