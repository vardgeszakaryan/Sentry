import argparse
from pathlib import Path
from sentry_ai.dataset.ingest import ingest_github

def main():
    parser = argparse.ArgumentParser(description="Ingest a dataset from a GitHub repository.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository URL")
    parser.add_argument("--path", type=str, required=True, help="Subfolder path inside the repository to extract")
    parser.add_argument("--target_dir", type=str, default="data/raw/github_dataset/", help="Destination directory")
    parser.add_argument("--branch", type=str, default="main", help="Branch to clone")
    
    args = parser.parse_args()
    
    ingest_github(args.repo, args.path, args.target_dir, args.branch)

if __name__ == "__main__":
    main()
