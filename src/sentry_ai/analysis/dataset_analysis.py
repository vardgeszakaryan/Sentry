from pathlib import Path
import json

class DatasetAnalyzer:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
    def _get_splits(self):
        split_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()]
        if split_dirs:
            return [(d.name, d, self.labels_dir / d.name) for d in split_dirs]
        return [("all", self.images_dir, self.labels_dir)]

    def analyze(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # We will conditionally import plotting and image libraries 
        # so that the CLI doesn't fail if they are missing unless this is explicitly run.
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PIL import Image

        image_sizes = [] # (width, height)
        box_centers = [] # (x, y)
        box_areas_relative = [] # w * h
        box_areas_absolute = [] # (w * img_w) * (h * img_h)
        class_counts = {}

        splits = self._get_splits()
        print(f"Analyzing dataset from {self.dataset_dir}...")
        
        for split_name, img_split_dir, lbl_split_dir in splits:
            if not lbl_split_dir.exists():
                continue
            
            images = [f for f in img_split_dir.iterdir() if f.is_file() and f.suffix.lower() in self.image_extensions]
            for img_path in images:
                label_path = lbl_split_dir / (img_path.stem + '.txt')
                
                # Get Image Size
                try:
                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                        image_sizes.append((img_w, img_h))
                except Exception as e:
                    print(f"Warning: Could not read image {img_path}: {e}")
                    continue

                if not label_path.exists():
                    continue

                # Parse label
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:])
                                
                                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                                box_centers.append((x, y))
                                box_areas_relative.append(w * h)
                                box_areas_absolute.append((w * img_w) * (h * img_h))
                except Exception as e:
                    print(f"Warning: Could not read label {label_path}: {e}")
                    
        # Plotting
        print(f"Generating plots and saving to {output_dir}")
        sns.set_theme(style="whitegrid")

        # 1. Image Sizes Scatter Plot
        if image_sizes:
            widths, heights = zip(*image_sizes)
            plt.figure(figsize=(10, 6))
            plt.scatter(widths, heights, alpha=0.5, c='blue')
            plt.title("Image Sizes (Width vs Height)")
            plt.xlabel("Width (px)")
            plt.ylabel("Height (px)")
            plt.savefig(output_dir / "image_sizes_scatter.png")
            plt.close()

        # 2. Box Centers Heatmap
        if box_centers:
            xs, ys = zip(*box_centers)
            plt.figure(figsize=(8, 8))
            # In YOLO y=0 is top, y=1 is bottom, so we invert Y-axis to match image coordinates visually
            plt.hist2d(xs, ys, bins=50, cmap='hot', density=True)
            plt.colorbar(label='Density')
            plt.title("Bounding Box Center Distribution")
            plt.xlabel("X Center (Normalized)")
            plt.ylabel("Y Center (Normalized)")
            plt.gca().invert_yaxis() 
            plt.savefig(output_dir / "box_centers_heatmap.png")
            plt.close()

        # 3. Box Sizes (Relative) Distribution
        if box_areas_relative:
            plt.figure(figsize=(10, 6))
            sns.histplot(box_areas_relative, bins=50, kde=True)
            plt.title("Bounding Box Area Distribution (Relative to Image Size)")
            plt.xlabel("Area (Normalized)")
            plt.ylabel("Count")
            plt.savefig(output_dir / "box_sizes_relative.png")
            plt.close()

        # 4. Box Sizes Categorization — dataset-relative using P33/P67 thresholds.
        # Thresholds are derived from the dataset's own box area distribution,
        # so "small/medium/large" always reflect THIS dataset's scale, not COCO's.
        small, medium, large = 0, 0, 0
        p33_threshold, p67_threshold = 0.0, 0.0
        if box_areas_absolute:
            areas = np.array(box_areas_absolute)
            p33_threshold = float(np.percentile(areas, 33))
            p67_threshold = float(np.percentile(areas, 67))

            small  = int(np.sum(areas < p33_threshold))
            medium = int(np.sum((areas >= p33_threshold) & (areas < p67_threshold)))
            large  = int(np.sum(areas >= p67_threshold))

            categories = [
                f'Small\n(<{p33_threshold:.0f} px²)',
                f'Medium\n({p33_threshold:.0f}–{p67_threshold:.0f} px²)',
                f'Large\n(≥{p67_threshold:.0f} px²)',
            ]
            cat_counts = [small, medium, large]

            plt.figure(figsize=(8, 6))
            sns.barplot(x=categories, y=cat_counts, palette="viridis")
            plt.title("Bounding Box Sizes (Dataset-Relative, P33/P67 thresholds)")
            plt.ylabel("Count")
            for i, v in enumerate(cat_counts):
                plt.text(i, v + max(cat_counts) * 0.01, str(v), ha='center')
            plt.savefig(output_dir / "box_size_categories.png")
            plt.close()

        # 5. Class Distribution
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            plt.figure(figsize=(12, 6))
            sns.barplot(x=classes, y=counts, palette="mako")
            plt.title("Class Distribution")
            plt.xlabel("Class ID")
            plt.ylabel("Number of Instances")
            plt.savefig(output_dir / "class_distribution.png")
            plt.close()

        # Save summary JSON
        summary = {
            "total_images": len(image_sizes),
            "total_boxes": len(box_centers),
            "class_counts": class_counts,
            "box_size_thresholds": {
                "method": "percentile_33_67",
                "small_upper_px2": round(p33_threshold, 2),
                "medium_upper_px2": round(p67_threshold, 2),
            },
            "box_size_stats": {
                "small": small,
                "medium": medium,
                "large": large,
            }
        }
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
            
        print("Analysis complete.")

def analyze_dataset(dataset_dir: str | Path, output_dir: str | Path):
    """
    Run the full dataset analysis and save plots to output_dir.
    """
    analyzer = DatasetAnalyzer(dataset_dir)
    analyzer.analyze(output_dir)
