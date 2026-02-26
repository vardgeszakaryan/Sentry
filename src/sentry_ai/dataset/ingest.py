import os
import shutil
import tempfile
from pathlib import Path
from git import Repo

def ingest_github(repo_url: str, subfolder_path: str, target_dir: str | Path, branch: str = "main"):
    """
    Ingest a dataset from a public GitHub repo by cloning and extracting a specific subfolder.
    
    Args:
        repo_url: URL to the GitHub repository.
        subfolder_path: Path within the repo containing the dataset (e.g., 'datasets/my_yolo').
        target_dir: Destination directory (e.g., 'data/raw/github_dataset/').
        branch: The branch to clone.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    
    print(f"Cloning {repo_url} (branch: {branch}) to a temporary directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
        
        source_subfolder = Path(temp_dir) / subfolder_path
        if not source_subfolder.exists():
            raise FileNotFoundError(f"Subfolder {subfolder_path} does not exist in the repository.")
            
        print(f"Extracting {subfolder_path} to {target}...")
        shutil.copytree(source_subfolder, target, dirs_exist_ok=True)
        
    print(f"Successfully ingested GitHub dataset to {target}")

def ingest_local(source_dir: str | Path, target_dir: str | Path):
    """
    Ingest a dataset from a local path by copying it to the target directory.
    
    Args:
        source_dir: Path to the local custom dataset.
        target_dir: Destination directory (e.g., 'data/raw/custom_dataset/').
    """
    source = Path(source_dir)
    target = Path(target_dir)
    
    if not source.exists():
        raise FileNotFoundError(f"Source directory {source} does not exist.")
        
    target.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying local dataset from {source} to {target}...")
    shutil.copytree(source, target, dirs_exist_ok=True)
    
    print(f"Successfully ingested local dataset to {target}")
