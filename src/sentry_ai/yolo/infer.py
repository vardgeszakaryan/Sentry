from pathlib import Path
from ultralytics import YOLO

def infer_yolo(config: dict, source: str):
    """
    Run inference on images or video using trained YOLO model.
    """
    infer_cfg = config.get('inference', {})
    proj_cfg = config.get('project', {})
    
    weights = infer_cfg.get('weights', 'runs/detect/train/weights/best.pt')
    conf = infer_cfg.get('conf_threshold', 0.25)
    iou = infer_cfg.get('iou_threshold', 0.45)
    
    runs_dir = proj_cfg.get('runs_dir', 'runs/')
    proj_name = proj_cfg.get('name', 'sentry_yolo_model')
    
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    
    print(f"Running inference on source: {source}...")
    
    # Save results to a specific predict project/name for clean output
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=True,
        project=runs_dir,
        name=f"{proj_name}_infer"
    )
    print("Inference completed. Results saved in the runs directory.")
