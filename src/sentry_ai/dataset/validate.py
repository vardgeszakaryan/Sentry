from pathlib import Path

def validate_yolo_dataset(dataset_dir: str | Path, max_class_id: int = 1000) -> list[str]:
    """
    Validates a YOLO format dataset recursively.
    Checks: missing pairs, invalid fields, out of range coords, negative sizes, class id.
    
    Args:
        dataset_dir: Root directory of the dataset.
        max_class_id: Upper bound for class IDs to check against.
        
    Returns:
        List of error messages.
    """
    root = Path(dataset_dir)
    errors = []
    
    if not root.exists():
        return [f"Dataset directory {root} does not exist."]
        
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    if not images_dir.exists():
        errors.append("Missing 'images' directory.")
    if not labels_dir.exists():
        errors.append("Missing 'labels' directory.")
        
    if not images_dir.exists() or not labels_dir.exists():
        return errors
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Traverse splits
    splits = [d.name for d in images_dir.iterdir() if d.is_dir()]
    if not splits:
        errors.append("No split directories found in 'images/'.")
        
    for split in splits:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split
            
        if not split_lbl_dir.exists():
            errors.append(f"Missing '{split}' label directory but images directory exists.")
            continue
            
        images = [f for f in split_img_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        for img_path in images:
            label_path = split_lbl_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                errors.append(f"Missing pair: Image {img_path.name} has no corresponding label file.")
                continue
                
            # Parse label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"{label_path.name}:{line_idx+1}: Expected 5 fields, found {len(parts)}.")
                        continue
                        
                    try:
                        cls_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        if cls_id < 0 or cls_id > max_class_id:
                            errors.append(f"{label_path.name}:{line_idx+1}: Class ID {cls_id} out of bounds.")
                            
                        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                            errors.append(f"{label_path.name}:{line_idx+1}: Coordinates (x, y) out of range [0, 1].")
                            
                        if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                            errors.append(f"{label_path.name}:{line_idx+1}: Size (w, h) must be in (0, 1].")
                            
                    except ValueError:
                        errors.append(f"{label_path.name}:{line_idx+1}: Invalid numeric fields.")
                        
            except Exception as e:
                errors.append(f"Error reading label {label_path.name}: {e}")
                
        # Check for labels without images
        labels = [f for f in split_lbl_dir.iterdir() if f.suffix == '.txt']
        for lbl_path in labels:
            # We don't know exact image extension, so we check if any matches
            found = False
            for ext in image_extensions:
                if (split_img_dir / (lbl_path.stem + ext)).exists():
                    found = True
                    break
            if not found:
                errors.append(f"Missing pair: Label {lbl_path.name} has no corresponding image file.")
                
    return errors
