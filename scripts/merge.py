import argparse
from sentry_ai.config import load_config
from sentry_ai.dataset.merge import merge_datasets

def main():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    merge_datasets(config)

if __name__ == "__main__":
    main()
