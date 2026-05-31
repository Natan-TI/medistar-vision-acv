import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ==========================
# CONFIGURAÇÕES PRINCIPAIS
# ==========================

RAW_IMAGES_DIR = Path("data/raw/images")
RAW_LABELS_DIR = Path("data/raw/labels")

OUTPUT_DIR = Path("data/processed")

PATCH_SIZE = 128
STRIDE = 128

MIN_DOMINANT_RATIO = 0.60
MAX_IMAGES_PER_CLASS = 800

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ==========================
# CLASSES DO DATASET
# ==========================

CLASS_NAMES = {
    0: "urbano",
    1: "vegetacao_densa",
    2: "sombra",
    3: "vegetacao_esparsa",
    4: "agricultura",
    6: "terreno_exposto",
}


# Cores das máscaras conforme CLASSES.jpeg
CLASS_COLORS = {
    0: (255, 0, 0),        # Urbano
    1: (38, 115, 0),       # Vegetação Densa
    2: (0, 0, 0),          # Sombra
    3: (133, 199, 126),    # Vegetação Esparsa
    4: (255, 255, 0),      # Agricultura
    5: (128, 128, 128),    # Rocha
    6: (139, 69, 19),      # Solo Exposto
    7: (84, 117, 168),     # Água
}

ENABLED_CLASSES = [0, 1, 2, 3, 4, 6]

def reset_output_dirs():
    """
    Limpa e recria a pasta data/processed.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for class_id in ENABLED_CLASSES:
            class_name = CLASS_NAMES[class_id]
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

def rgb_mask_to_class_ids(mask_rgb):
    """
    Converte uma máscara RGB em uma matriz 2D com IDs das classes.
    Cada pixel recebe o ID da classe correspondente à sua cor.
    """
    height, width, _ = mask_rgb.shape
    class_mask = np.full((height, width), fill_value=-1, dtype=np.int16)

    for class_id, color in CLASS_COLORS.items():
        color_array = np.array(color)
        matches = np.all(mask_rgb == color_array, axis=-1)
        class_mask[matches] = class_id

    return class_mask


def get_dominant_class(mask_patch):
    """
    Descobre a classe dominante em um recorte da máscara.
    Retorna:
    - class_id dominante
    - proporção da classe dominante
    """
    valid_pixels = mask_patch[mask_patch >= 0]

    if len(valid_pixels) == 0:
        return None, 0

    class_ids, counts = np.unique(valid_pixels, return_counts=True)

    dominant_index = np.argmax(counts)
    dominant_class = int(class_ids[dominant_index])
    dominant_ratio = counts[dominant_index] / len(valid_pixels)

    return dominant_class, dominant_ratio


def find_matching_label(image_path):
    """
    Tenta encontrar a label correspondente para uma imagem.
    Exemplo:
    image_01.tif -> image_01.png
    """
    possible_label = RAW_LABELS_DIR / f"{image_path.stem}.png"

    if possible_label.exists():
        return possible_label

    # Caso o nome tenha algum padrão diferente, tenta buscar pelo começo do nome.
    candidates = list(RAW_LABELS_DIR.glob(f"{image_path.stem}*.png"))

    if candidates:
        return candidates[0]

    return None


def extract_patches():
    """
    Recorta as imagens originais em patches menores e agrupa por classe dominante.
    """
    random.seed(RANDOM_SEED)

    patches_by_class = {class_id: [] for class_id in ENABLED_CLASSES}

    image_paths = sorted(list(RAW_IMAGES_DIR.glob("*.tif")))

    if not image_paths:
        raise FileNotFoundError(
            f"Nenhuma imagem .tif encontrada em: {RAW_IMAGES_DIR}"
        )

    print(f"Total de imagens encontradas: {len(image_paths)}")

    for image_path in tqdm(image_paths, desc="Processando imagens"):
        label_path = find_matching_label(image_path)

        if label_path is None:
            print(f"[AVISO] Label não encontrada para: {image_path.name}")
            continue

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        image_np = np.array(image)
        label_np = np.array(label)

        class_mask = rgb_mask_to_class_ids(label_np)

        height, width, _ = image_np.shape

        for y in range(0, height - PATCH_SIZE + 1, STRIDE):
            for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                image_patch = image_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                mask_patch = class_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                dominant_class, dominant_ratio = get_dominant_class(mask_patch)
                # Agrupamento técnico:
                # A classe 5, originalmente "rocha", foi agrupada com solo exposto
                # por apresentar alta similaridade visual nos recortes gerados.
                if dominant_class == 5:
                    dominant_class = 6

                if dominant_class is None:
                    continue

                if dominant_class not in ENABLED_CLASSES:
                    continue

                if dominant_ratio < MIN_DOMINANT_RATIO:
                    continue

                patches_by_class[dominant_class].append(
                    {
                        "image_patch": image_patch,
                        "source_image": image_path.stem,
                        "x": x,
                        "y": y,
                        "ratio": dominant_ratio,
                    }
                )

    return patches_by_class


def split_and_save_patches(patches_by_class):
    """
    Balanceia, divide em train/val/test e salva os patches nas pastas finais.
    """
    summary = {}

    for class_id, patches in patches_by_class.items():
        if class_id not in ENABLED_CLASSES:
            continue

        class_name = CLASS_NAMES[class_id]

        random.shuffle(patches)

        if len(patches) > MAX_IMAGES_PER_CLASS:
            patches = patches[:MAX_IMAGES_PER_CLASS]

        total = len(patches)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        split_data = {
            "train": patches[:train_end],
            "val": patches[train_end:val_end],
            "test": patches[val_end:],
        }

        summary[class_name] = {
            "total": total,
            "train": len(split_data["train"]),
            "val": len(split_data["val"]),
            "test": len(split_data["test"]),
        }

        for split, split_patches in split_data.items():
            for index, patch_info in enumerate(split_patches):
                filename = (
                    f"{class_name}_"
                    f"{patch_info['source_image']}_"
                    f"x{patch_info['x']}_"
                    f"y{patch_info['y']}_"
                    f"{index}.png"
                )

                output_path = OUTPUT_DIR / split / class_name / filename

                patch_image = Image.fromarray(patch_info["image_patch"])
                patch_image.save(output_path)

    return summary


def print_summary(summary):
    """
    Exibe um resumo da quantidade de imagens geradas por classe.
    """
    print("\nResumo do dataset gerado:")
    print("-" * 70)
    print(f"{'Classe':25} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 70)

    for class_name, values in summary.items():
        print(
            f"{class_name:25} "
            f"{values['total']:8} "
            f"{values['train']:8} "
            f"{values['val']:8} "
            f"{values['test']:8}"
        )

    print("-" * 70)


def main():
    reset_output_dirs()

    patches_by_class = extract_patches()

    summary = split_and_save_patches(patches_by_class)

    print_summary(summary)

    print("\nDataset de classificação criado com sucesso!")
    print(f"Pasta final: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()