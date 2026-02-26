import os
import shutil
import random
from pathlib import Path

def remap_and_copy_label(src_label: Path, dst_label: Path, remap: dict):
    if not src_label.exists():
        return
        
    with open(src_label, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if remap and cls_id in remap:
                cls_id = remap[cls_id]
            parts[0] = str(cls_id)
            out_lines.append(" ".join(parts))
            
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, 'w') as f:
        f.write("\n".join(out_lines) + "\n")

def collect_items(dataset_dir: Path, prefix: str):
    items = [] # list of (img_path, lbl_path, split)
    if not dataset_dir.exists():
        return items
        
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    for split in ['train', 'val', 'test']:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split
        
        if not split_img_dir.exists():
            continue
            
        for img_path in split_img_dir.iterdir():
            if img_path.suffix.lower() in img_exts:
                lbl_path = split_lbl_dir / (img_path.stem + '.txt')
                items.append((img_path, lbl_path, split, prefix))
                
    return items

def merge_datasets(config: dict):
    """
    Merges github and custom datasets into a single YOLO dataset.
    """
    ds_config = config.get('dataset', {})
    github_dir = Path(ds_config.get('github_dir', 'data/raw/github_dataset/'))
    custom_dir = Path(ds_config.get('custom_dir', 'data/raw/custom_dataset/'))
    merged_dir = Path(ds_config.get('merged_dir', 'data/processed/yolo_merged/'))
    
    merge_mode = ds_config.get('merge_mode', 'preserve')
    remap = ds_config.get('class_remap', {})
    ratios = ds_config.get('rebuild_ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    
    print(f"Merging datasets to {merged_dir} using mode: {merge_mode}")
    
    if merged_dir.exists():
        print(f"Cleaning existing merged directory...")
        shutil.rmtree(merged_dir)
        
    items = []
    items.extend(collect_items(github_dir, "gh"))
    items.extend(collect_items(custom_dir, "cu"))
    
    if not items:
        print("No items found to merge.")
        return
        
    if merge_mode == 'rebuild':
        random.shuffle(items)
        n = len(items)
        train_end = int(n * ratios.get('train', 0.8))
        val_end = train_end + int(n * ratios.get('val', 0.1))
        
        for i in range(n):
            if i < train_end:
                items[i] = (items[i][0], items[i][1], 'train', items[i][3])
            elif i < val_end:
                items[i] = (items[i][0], items[i][1], 'val', items[i][3])
            else:
                items[i] = (items[i][0], items[i][1], 'test', items[i][3])
                
    # Copy files
    for img_path, lbl_path, split, prefix in items:
        new_stem = f"{prefix}_{img_path.stem}"
        new_img_name = new_stem + img_path.suffix
        new_lbl_name = new_stem + ".txt"
        
        v_img_dir = merged_dir / 'images' / split
        v_img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, v_img_dir / new_img_name)
        
        v_lbl_dir = merged_dir / 'labels' / split
        v_lbl_dir.mkdir(parents=True, exist_ok=True)
        remap_and_copy_label(lbl_path, v_lbl_dir / new_lbl_name, remap)
        
    print(f"Successfully merged {len(items)} items into {merged_dir}")
