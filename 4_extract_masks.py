import time
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO, SAM
import torch
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

import config

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ROOT_POOL_PERSON = Path(config.ROOT_POOL_PERSON)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# YOLOv8x para detección (bounding box) + SAM2-L para segmentación precisa
DET_MODEL  = 'yolov8x.pt'
SAM_MODEL  = 'sam2_l.pt'


def extract_mask(crop_orig, det_model, sam_model):
    """
    Pipeline YOLO-detect → SAM2:
      1. YOLO localiza a la persona (bbox más confiable).
      2. SAM2 genera la máscara en resolución original a partir del bbox.
    Retorna la máscara binaria uint8 y metadatos de calidad.
    """
    h, w = crop_orig.shape[:2]

    # --- 1. Detección ---
    det_results = det_model.predict(
        source=crop_orig,
        classes=[0],
        verbose=False,
        device=DEVICE,
        conf=0.1,
        imgsz=640,
    )

    if not det_results or det_results[0].boxes is None or len(det_results[0].boxes) == 0:
        return None, {"is_valid": False, "reason": "No detection"}

    # Seleccionar la detección con mayor confianza
    boxes = det_results[0].boxes
    best_idx = int(boxes.conf.argmax())
    bbox = boxes.xyxy[best_idx].cpu().numpy().tolist()  # [x1, y1, x2, y2]

    # --- 2. Segmentación con SAM2 ---
    sam_results = sam_model(
        source=crop_orig,
        bboxes=[bbox],
        verbose=False,
        device=DEVICE,
    )

    if not sam_results or sam_results[0].masks is None or len(sam_results[0].masks.data) == 0:
        return None, {"is_valid": False, "reason": "SAM2 no mask"}

    mask_tensor = sam_results[0].masks.data[0].cpu().numpy()

    if mask_tensor.shape[:2] != (h, w):
        mask_binary = cv2.resize(
            mask_tensor.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_binary = mask_tensor

    mask_binary = (mask_binary * 255).astype(np.uint8)

    # --- 3. Heurísticas de calidad ---
    stats = {
        "area_ratio":         float(cv2.countNonZero(mask_binary) / (w * h)),
        "bottom_width_ratio": float(np.count_nonzero(mask_binary[-1, :]) / w),
        "top_width_ratio":    float(np.count_nonzero(mask_binary[0, :])  / w),
        "left_height_ratio":  float(np.count_nonzero(mask_binary[:, 0])  / h),
        "right_height_ratio": float(np.count_nonzero(mask_binary[:, -1]) / h),
        "is_valid": True,
    }

    if stats["area_ratio"] < 0.15:                                       stats["is_valid"] = False
    if stats["bottom_width_ratio"] > 0.45:                               stats["is_valid"] = False
    if stats["top_width_ratio"] > 0.35:                                  stats["is_valid"] = False
    if stats["left_height_ratio"] > 0.40 or stats["right_height_ratio"] > 0.40: stats["is_valid"] = False

    return mask_binary, stats


def process_pool():
    print("="*60)
    print("  Pre-segmentación YOLO-det + SAM2 para People Pool")
    print("="*60)

    print(f"[INFO] Cargando modelo de detección: {DET_MODEL} en {DEVICE}...")
    det_model = YOLO(DET_MODEL)

    print(f"[INFO] Cargando modelo de segmentación: {SAM_MODEL} en {DEVICE}...")
    sam_model = SAM(SAM_MODEL)

    pool_csv_p = ROOT_POOL_PERSON / 'pool.csv'
    if not pool_csv_p.exists():
        print(f"[WARN] No se encontró pool.csv en {ROOT_POOL_PERSON}")
        return

    masks_dir = ROOT_POOL_PERSON / 'masks'
    meta_dir  = ROOT_POOL_PERSON / 'metadata'
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df_pool = pd.read_csv(str(pool_csv_p))
    print(f"[INFO] Procesando pool ({len(df_pool)} parches)...")

    n_valid = 0
    n_invalid = 0
    n_skipped = 0

    for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="Segmentando"):
        img_path = Path(row['name'])
        if not img_path.exists():
            n_skipped += 1
            continue

        patch_name = img_path.stem
        mask_path  = masks_dir / f"{patch_name}.png"
        json_path  = meta_dir  / f"{patch_name}.json"

        if mask_path.exists() and json_path.exists():
            n_skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            n_skipped += 1
            continue

        mask, stats = extract_mask(img, det_model, sam_model)

        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)

        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
            n_valid += 1
        else:
            n_invalid += 1

    print(f"\n[SUCCESS] Pre-segmentación completada.")
    print(f"  Máscaras válidas:   {n_valid}")
    print(f"  Máscaras inválidas: {n_invalid}")
    print(f"  Omitidas (ya existían o no encontradas): {n_skipped}")


if __name__ == '__main__':
    t0 = time.perf_counter()
    process_pool()
    elapsed = time.perf_counter() - t0
    print(f"\nTiempo total: {int(elapsed//3600):02d}h {int(elapsed%3600//60):02d}m {elapsed%60:05.2f}s")
