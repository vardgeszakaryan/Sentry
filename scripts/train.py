import argparse
from sentry_ai.config import load_config
from sentry_ai.yolo.train import train_yolo

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    train_yolo(config)

if __name__ == "__main__":
    main()
