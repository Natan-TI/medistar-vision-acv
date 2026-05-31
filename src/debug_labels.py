from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

LABEL_DIR = Path("data/raw/labels")

all_colors_counter = Counter()

label_paths = sorted(LABEL_DIR.glob("*.png"))

print(f"Total de labels encontradas: {len(label_paths)}")

for label_path in label_paths:
    img = Image.open(label_path).convert("RGB")
    arr = np.array(img)

    pixels = arr.reshape(-1, 3)

    colors, counts = np.unique(pixels, axis=0, return_counts=True)

    print(f"\nArquivo: {label_path.name}")
    print(f"Total de cores: {len(colors)}")

    for color, count in zip(colors, counts):
        color_tuple = tuple(color.tolist())
        all_colors_counter[color_tuple] += int(count)
        print(f"{color_tuple} -> {count} pixels")

print("\n" + "=" * 60)
print("CORES GERAIS DO DATASET")
print("=" * 60)

for color, count in all_colors_counter.most_common():
    print(f"{color} -> {count} pixels")