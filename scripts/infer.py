import argparse
from sentry_ai.config import load_config
from sentry_ai.yolo.infer import infer_yolo

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained YOLO model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, video, or directory")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    infer_yolo(config, args.source)

if __name__ == "__main__":
    main()
