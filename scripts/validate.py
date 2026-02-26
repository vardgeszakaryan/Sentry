"""
Entrypoint script to validate a YOLO dataset for strict format adherence.
"""
import argparse
import sys
from sentry_ai.dataset.validate import validate_yolo_dataset

def main():
    """
    Parse arguments and run YOLO dataset validation.
    
    Exits with code 1 if errors are found, printing up to 50 errors.
    Exits with code 0 if the dataset is perfectly valid.
    """
    parser = argparse.ArgumentParser(description="Validate YOLO dataset strictness.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--max_class_id", type=int, default=1000, help="Maximum allowed class ID")
    
    args = parser.parse_args()
    
    errors = validate_yolo_dataset(args.dataset_dir, args.max_class_id)
    if errors:
        print(f"Dataset Validation FAILED with {len(errors)} errors:")
        for err in errors[:50]:
            print(f" - {err}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors.")
        sys.exit(1)
    else:
        print("Dataset Validation PASSED!")
        sys.exit(0)

if __name__ == "__main__":
    main()
