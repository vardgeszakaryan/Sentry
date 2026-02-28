import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def audit_yolo_dataset(dataset_dir: str | Path) -> dict:
    """
    Audits a YOLO dataset and returns statistics.
    
    Args:
        dataset_dir: Root directory of the dataset.
        
    Returns:
        Dictionary containing audit statistics.
    """
    root = Path(dataset_dir)
    stats = {
        'total_images': 0,
        'labeled_images': 0,
        'empty_label_images': 0,
        'total_bboxes': 0,
        'class_distribution': defaultdict(int),
        'bbox_areas': []  # Will be converted to percentiles
    }
    
    if not root.exists():
        return stats
        
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        return stats
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    for split_img_dir in images_dir.iterdir():
        if not split_img_dir.is_dir():
            continue
        split = split_img_dir.name
        split_lbl_dir = labels_dir / split
            
        images = [f for f in split_img_dir.iterdir() if f.suffix.lower() in image_extensions]
        stats['total_images'] += len(images)
        
        for img_path in images:
            if not split_lbl_dir.exists():
                stats['empty_label_images'] += 1
                continue
                
            label_path = split_lbl_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                stats['empty_label_images'] += 1
                continue
                
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
            if not lines:
                stats['empty_label_images'] += 1
            else:
                stats['labeled_images'] += 1
                stats['total_bboxes'] += len(lines)
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            w = float(parts[3])
                            h = float(parts[4])
                            stats['class_distribution'][cls_id] += 1
                            stats['bbox_areas'].append(w * h)
                        except ValueError:
                            pass
                            
    # Compute percentiles
    if stats['bbox_areas']:
        areas = np.array(stats['bbox_areas'])
        stats['bbox_size_percentiles'] = {
            'min': float(np.min(areas)),
            '25th': float(np.percentile(areas, 25)),
            'median': float(np.median(areas)),
            '75th': float(np.percentile(areas, 75)),
            'max': float(np.max(areas))
        }
    else:
        stats['bbox_size_percentiles'] = {}
        
    del stats['bbox_areas'] # Clean up raw list
    stats['class_distribution'] = dict(stats['class_distribution'])
    
    return stats
