from pathlib import Path
from ultralytics import YOLO

def generate_data_yaml(dataset_dir: Path, output_yaml: Path, num_classes: int = 80):
    """
    Generates a YOLO data.yaml for the merged dataset.
    """
    names = [f"'class_{i}'" for i in range(num_classes)]
    names_str = "[" + ", ".join(names) + "]"
    
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {names_str}
"""
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)

def train_yolo(config: dict):
    """
    Trains YOLO model using configuration.
    """
    train_cfg = config.get('training', {})
    ds_cfg = config.get('dataset', {})
    proj_cfg = config.get('project', {})
    
    merged_dir = Path(ds_cfg.get('merged_dir', 'data/processed/yolo_merged/'))
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged dataset directory not found at {merged_dir}. Please run the merge step first.")
        
    model_yaml = train_cfg.get('model_yaml', 'yolov8n.yaml')
    epochs = train_cfg.get('epochs', 100)
    batch = train_cfg.get('batch_size', 16)
    imgsz = train_cfg.get('imgsz', 640)
    device = train_cfg.get('device', '')
    workers = train_cfg.get('workers', 8)
    
    runs_dir = proj_cfg.get('runs_dir', 'runs/')
    proj_name = proj_cfg.get('name', 'sentry_yolo_model')
    
    # Check max class id to correctly generate names (or simply use 100, assuming max 80 for COCO)
    data_yaml_path = merged_dir / 'sentry_data.yaml'
    generate_data_yaml(merged_dir, data_yaml_path, num_classes=100)
    
    print(f"Initializing YOLO with {model_yaml}...")
    model = YOLO(model_yaml)
    
    print("Starting training...")
    kwargs = {
        'data': str(data_yaml_path.absolute()),
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'project': runs_dir,
        'name': proj_name,
        'workers': workers
    }
    if device:
        kwargs['device'] = device
        
    model.train(**kwargs)
    print("Training completed.")
