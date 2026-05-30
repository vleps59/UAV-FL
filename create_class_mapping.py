# create_class_mapping.py
from datasets import load_dataset
import json
from pathlib import Path

print("Loading dataset...")
dataset = load_dataset("aliencaocao/described-flying-objects")

print("Extracting unique labels...")
unique_labels = set()
for item in dataset['train']:
    for label in item['objects']['label']:
        unique_labels.add(label)

unique_labels = sorted(list(unique_labels))
class_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_class = {idx: label for label, idx in class_to_id.items()}

print(f"\nFound {len(unique_labels)} unique classes")

# Save mapping
save_dir = Path("data/DescribedFlyingObjects")
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "class_mapping.json", 'w') as f:
    json.dump({
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "num_classes": len(unique_labels)
    }, f, indent=2)

print(f"\n✅ Class mapping saved to {save_dir / 'class_mapping.json'}")
print(f"\nFirst 10 classes:")
for i in range(min(10, len(unique_labels))):
    print(f"  {i}: {unique_labels[i]}")