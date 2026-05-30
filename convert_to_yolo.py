# convert_to_yolo_corrected.py
from datasets import load_dataset
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

def convert_described_flying_to_yolo():
    """
    Convert Described Flying Objects dataset to YOLO format.
    """
    print("Loading dataset from Hugging Face cache...")
    dataset = load_dataset("aliencaocao/described-flying-objects")
    
    # Create directory structure
    output_dir = Path("data/DescribedFlyingObjects")
    train_img_dir = output_dir / "train" / "images"
    train_label_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_label_dir = output_dir / "val" / "labels"
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check what's actually in the dataset
    print("\nExamining dataset structure...")
    sample = dataset['train'][0]
    print(f"Available keys: {sample.keys()}")
    
    # Look for annotation-related keys
    possible_keys = ['objects', 'labels', 'annotations', 'boxes', 'bbox', 'label', 'categories', 'objects_bbox']
    found_key = None
    
    for key in possible_keys:
        if key in sample:
            found_key = key
            print(f"Found annotations under key: '{key}'")
            break
    
    if found_key is None:
        print("\nNo standard annotation key found. Trying to access features...")
        # Try to get from features
        features = dataset["train"].features
        print(f"All features: {list(features.keys())}")
        
        # Check if the dataset has a 'objects' feature
        if 'objects' in features:
            print("Found 'objects' in features")
            found_key = 'objects'
        elif 'labels' in features:
            print("Found 'labels' in features")
            found_key = 'labels'
    
    # If still no annotations, create synthetic ones
    if found_key is None:
        print("\n⚠️ No annotations found. Creating synthetic annotations for testing...")
        create_synthetic = True
    else:
        create_synthetic = False
    
    # Extract unique classes
    print("\nExtracting unique classes...")
    unique_classes = set()
    
    if not create_synthetic:
        for item in dataset['train']:
            annotations = item[found_key]
            
            # Handle different annotation formats
            if isinstance(annotations, dict):
                # Check for label field
                if 'label' in annotations:
                    labels = annotations['label']
                elif 'labels' in annotations:
                    labels = annotations['labels']
                elif 'category' in annotations:
                    labels = annotations['category']
                else:
                    # Try to find any list field
                    for value in annotations.values():
                        if isinstance(value, list) and len(value) > 0:
                            labels = value
                            break
                    else:
                        continue
                
                # Add labels to unique set
                if isinstance(labels, list):
                    for label in labels:
                        unique_classes.add(str(label))
                else:
                    unique_classes.add(str(labels))
            
            elif isinstance(annotations, list):
                for ann in annotations:
                    if isinstance(ann, dict):
                        if 'label' in ann:
                            unique_classes.add(str(ann['label']))
                        elif 'labels' in ann:
                            unique_classes.add(str(ann['labels']))
                        elif 'category' in ann:
                            unique_classes.add(str(ann['category']))
                    else:
                        unique_classes.add(str(ann))
    
    # If no classes found, create synthetic classes
    if len(unique_classes) == 0:
        create_synthetic = True
        classes = ['drone', 'bird', 'airplane', 'helicopter', 'balloon']
        class_to_id = {cls: i for i, cls in enumerate(classes)}
        print(f"\nCreating synthetic dataset with {len(classes)} classes:")
        for cls, cid in class_to_id.items():
            print(f"  {cid}: {cls}")
    else:
        unique_classes = sorted(list(unique_classes))
        class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
        print(f"\nFound {len(unique_classes)} unique classes")
        for i, cls in enumerate(list(class_to_id.keys())[:10]):
            print(f"  {i}: {cls}")
    
    # Save class mapping
    with open(output_dir / "class_mapping.json", 'w') as f:
        json.dump({
            "class_to_id": class_to_id,
            "id_to_class": {i: cls for cls, i in class_to_id.items()},
            "num_classes": len(class_to_id)
        }, f, indent=2)
    
    # Split dataset
    full_dataset = dataset['train']
    split_idx = int(len(full_dataset) * 0.8)
    
    train_dataset = full_dataset.select(range(split_idx))
    val_dataset = full_dataset.select(range(split_idx, len(full_dataset)))
    
    print(f"\nTrain size: {len(train_dataset)} images")
    print(f"Val size: {len(val_dataset)} images")
    
    # Convert training set
    print("\nConverting training set...")
    for idx, item in enumerate(tqdm(train_dataset)):
        image = item['image']
        
        # Save image
        img_filename = f"frame_{idx:06d}.jpg"
        image.save(train_img_dir / img_filename)
        
        # Create labels
        boxes = []
        
        if create_synthetic:
            # Create synthetic bounding boxes
            import random
            width, height = image.size
            num_boxes = random.randint(1, 3)
            
            for _ in range(num_boxes):
                class_id = random.randint(0, len(class_to_id) - 1)
                box_width = random.uniform(0.1, 0.3)
                box_height = random.uniform(0.1, 0.3)
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.2, 0.8)
                boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        else:
            # Extract real annotations
            annotations = item[found_key]
            
            if isinstance(annotations, dict):
                # Handle dict format
                if 'bbox' in annotations and 'label' in annotations:
                    for bbox, label in zip(annotations['bbox'], annotations['label']):
                        class_id = class_to_id[str(label)]
                        x, y, w, h = bbox
                        img_w, img_h = image.size
                        center_x = (x + w/2) / img_w
                        center_y = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        boxes.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # Save labels
        if boxes:
            label_path = train_label_dir / f"frame_{idx:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(boxes))
    
    # Convert validation set
    print("\nConverting validation set...")
    for idx, item in enumerate(tqdm(val_dataset)):
        image = item['image']
        
        # Save image
        img_filename = f"val_{idx:06d}.jpg"
        image.save(val_img_dir / img_filename)
        
        # Create labels
        boxes = []
        
        if create_synthetic:
            # Create synthetic bounding boxes
            import random
            width, height = image.size
            num_boxes = random.randint(0, 2)
            
            for _ in range(num_boxes):
                class_id = random.randint(0, len(class_to_id) - 1)
                box_width = random.uniform(0.1, 0.3)
                box_height = random.uniform(0.1, 0.3)
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.2, 0.8)
                boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        else:
            # Extract real annotations
            annotations = item[found_key]
            
            if isinstance(annotations, dict):
                if 'bbox' in annotations and 'label' in annotations:
                    for bbox, label in zip(annotations['bbox'], annotations['label']):
                        class_id = class_to_id[str(label)]
                        x, y, w, h = bbox
                        img_w, img_h = image.size
                        center_x = (x + w/2) / img_w
                        center_y = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        boxes.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # Save labels
        if boxes:
            label_path = val_label_dir / f"val_{idx:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(boxes))
    
    # Create dataset.yaml
    yaml_content = f"""
# Described Flying Objects Dataset
path: {output_dir.absolute()}
train: train/images
val: val/images

nc: {len(class_to_id)}
names: {list(class_to_id.keys())}
"""
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Print statistics
    train_images = len(list(train_img_dir.glob("*.jpg")))
    train_labels = len(list(train_label_dir.glob("*.txt")))
    val_images = len(list(val_img_dir.glob("*.jpg")))
    val_labels = len(list(val_label_dir.glob("*.txt")))
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Classes: {len(class_to_id)}")
    print(f"   Train images: {train_images}")
    print(f"   Train label files: {train_labels}")
    print(f"   Val images: {val_images}")
    print(f"   Val label files: {val_labels}")
    
    if create_synthetic:
        print("\n⚠️ Note: Using synthetic annotations (no real bounding boxes available)")
    else:
        print("\n✅ Using real annotations from the dataset")

if __name__ == "__main__":
    convert_described_flying_to_yolo()