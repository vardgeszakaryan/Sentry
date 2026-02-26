import argparse
from pathlib import Path
from sentry_ai.dataset.ingest import ingest_local

def main():
    parser = argparse.ArgumentParser(description="Ingest a custom local dataset.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the local custom dataset")
    parser.add_argument("--target_dir", type=str, default="data/raw/custom_dataset/", help="Destination directory")
    
    args = parser.parse_args()
    
    ingest_local(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()
