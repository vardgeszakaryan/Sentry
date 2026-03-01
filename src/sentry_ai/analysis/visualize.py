import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import defaultdict

def analyze_yolo_dataset(dataset_dir: str | Path, output_dir: str | Path = None):
    """
    Performs a deep visual analysis of a YOLO dataset, generating graphs and heatmaps.
    
    Args:
        dataset_dir: Root directory of the dataset (should contain 'images' and 'labels' dirs).
        output_dir: Where to save the generated plots. If None, saves in dataset_dir/analysis.
    """
    root = Path(dataset_dir)
    if not root.exists():
        print(f"Dataset directory {root} does not exist.")
        return
        
    if output_dir is None:
        output_dir = root / 'analysis'
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Missing 'images' or 'labels' in {root}")
        return
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Traverse splits or flat structure
    split_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    if split_dirs:
        dirs_to_check = [(d, labels_dir / d.name) for d in split_dirs]
    else:
        dirs_to_check = [(images_dir, labels_dir)]
        
    # Collected statistics
    box_centers_x = []
    box_centers_y = []
    box_areas = []
    class_counts = defaultdict(int)
    image_widths = []
    image_heights = []
    
    total_images = 0
    
    print(f"Analyzing dataset at {root}...")
    
    for split_img_dir, split_lbl_dir in dirs_to_check:
        images = [f for f in split_img_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        total_images += len(images)
        
        for img_path in images:
            # Get Image dimensions
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    image_widths.append(w)
                    image_heights.append(h)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                
            # Process labels
            if split_lbl_dir.exists():
                label_path = split_lbl_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                cx = float(parts[1])
                                cy = float(parts[2])
                                box_w = float(parts[3])
                                box_h = float(parts[4])
                                
                                class_counts[cls_id] += 1
                                box_centers_x.append(cx)
                                box_centers_y.append(cy)
                                box_areas.append(box_w * box_h)
                            except ValueError:
                                pass
                                
    if not box_centers_x:
        print("No valid bounding boxes found to analyze.")
        return
        
    print(f"Processed {total_images} images and {len(box_centers_x)} bounding boxes. Generating plots...")
    
    # 1. Bounding Box Heatmap
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=box_centers_x, y=box_centers_y, cmap="inferno", fill=True, bw_adjust=0.5)
    plt.xlim(0, 1)
    plt.ylim(1, 0) # Invert Y axis to match image coordinates (0,0 is top-left)
    plt.title("Bounding Box Location Heatmap (Normalized)")
    plt.xlabel("X Center")
    plt.ylabel("Y Center")
    plt.savefig(output_dir / 'bbox_heatmap.png', dpi=300)
    plt.close()
    
    # 2. Box Area Distribution (Percentage of Image)
    # Box area in YOLO is already a ratio of the image area (since w and h are normalized [0, 1])
    # Convert to percentages
    area_percentages = np.array(box_areas) * 100 
    
    plt.figure(figsize=(10, 6))
    sns.histplot(area_percentages, bins=50, kde=True, color='skyblue')
    plt.title("Distribution of Bounding Box Area")
    plt.xlabel("Percentage of Total Image Area (%)")
    plt.ylabel("Count")
    plt.savefig(output_dir / 'bbox_area_distribution.png', dpi=300)
    plt.close()
    
    # 3. Image Dimensions Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=image_widths, y=image_heights, alpha=0.5, color='coral')
    plt.title("Image Dimensions (Width vs Height)")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.savefig(output_dir / 'image_dimensions.png', dpi=300)
    plt.close()
    
    # 4. Class Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.title("Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Number of Bounding Boxes")
    plt.xticks(classes)
    plt.savefig(output_dir / 'class_distribution.png', dpi=300)
    plt.close()
    
    print(f"Analysis complete! Graphs saved to: {output_dir}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset visually.")
    parser.add_argument("dataset_dir", type=str, help="Path to YOLO dataset root.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots.")
    args = parser.parse_args()
    analyze_yolo_dataset(args.dataset_dir, args.out)
