"""
Entrypoint script to analyze a YOLO dataset and generate visualizations.
"""
import argparse
from pathlib import Path
from sentry_ai.analysis.dataset_analysis import analyze_dataset

def main():
    """
    Parse arguments and run the dataset analyzer.
    """
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset and generate plots.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Path to save analysis plots and summary")
    
    args = parser.parse_args()
    
    print(f"Starting analysis for dataset: {args.dataset_dir}")
    analyze_dataset(args.dataset_dir, args.output_dir)
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
