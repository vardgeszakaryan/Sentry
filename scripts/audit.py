"""
Entrypoint script to audit a YOLO dataset and print statistics.
"""
import argparse
import json
from sentry_ai.dataset.audit import audit_yolo_dataset

def main():
    """
    Parse arguments and print audit statistics in JSON format.
    """
    parser = argparse.ArgumentParser(description="Audit YOLO dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root")
    
    args = parser.parse_args()
    
    stats = audit_yolo_dataset(args.dataset_dir)
    print("Dataset Audit Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
