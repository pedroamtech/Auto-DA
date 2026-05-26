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


def compute_completeness(mask_binary, pitch_deg):
    """
    Compute body-completeness metrics from a segmentation mask.

    Uses the source capture pitch (from 1_extract_information.py via pool.csv) to
    apply angle-aware thresholds:

      High oblique  (|pitch| < 35°): horizon visible, full standing body expected.
        → strict: head AND feet must be clearly present and not truncated,
          mask bounding box must be tall (aspect ≥ 1.5).

      Low oblique   (35° ≤ |pitch| < 55°): body mostly visible but foreshortened.
        → moderate thresholds.

      Nadir         (|pitch| ≥ 55°): person appears as a compact blob from above.
        → relaxed: anatomical top/bottom correspondence breaks down.

    Fields returned (merged into the stats dict):
      head_coverage  — fraction of top-7% rows occupied by mask pixels.
                       Low → head truncated at crop top; too high → head hits edge.
      feet_coverage  — fraction of bottom-7% rows occupied by mask pixels.
                       Low → feet truncated at crop bottom.
      mask_aspect    — height / width of mask bounding box.
                       > 1.5 → tall narrow silhouette (expected for oblique).
      complete_body  — True if all angle-appropriate criteria pass simultaneously.
    """
    h, w   = mask_binary.shape[:2]
    margin = max(3, int(h * 0.07))

    top_region    = mask_binary[:margin, :]
    bottom_region = mask_binary[-margin:, :]
    area = float(w * margin)

    head_cov = float(np.count_nonzero(top_region))    / (area + 1e-6)
    feet_cov = float(np.count_nonzero(bottom_region)) / (area + 1e-6)

    # Mask bounding-box aspect ratio (height / width of the person silhouette)
    rows_on = np.any(mask_binary > 127, axis=1)
    cols_on = np.any(mask_binary > 127, axis=0)
    if rows_on.any() and cols_on.any():
        r_lo, r_hi = np.where(rows_on)[0][[0, -1]]
        c_lo, c_hi = np.where(cols_on)[0][[0, -1]]
        mask_aspect = float(r_hi - r_lo + 1) / max(1, c_hi - c_lo + 1)
    else:
        mask_aspect = 1.0

    abs_p = abs(float(pitch_deg))
    if abs_p < 35:       # high oblique — horizon visible, full body required
        min_c, max_c = 0.05, 0.45
        min_aspect   = 1.5
    elif abs_p < 55:     # low oblique
        min_c, max_c = 0.03, 0.55
        min_aspect   = 1.0
    else:                # nadir — compact blob, anatomical mapping unreliable
        min_c, max_c = 0.02, 0.70
        min_aspect   = 0.5

    complete = bool(
        min_c < head_cov < max_c and
        min_c < feet_cov < max_c and
        mask_aspect >= min_aspect
    )

    return {
        'head_coverage': round(head_cov,    4),
        'feet_coverage': round(feet_cov,    4),
        'mask_aspect':   round(mask_aspect, 3),
        'complete_body': complete,
    }


def process_pool():
    print("="*60)
    print("  Pre-segmentación YOLO-det + SAM2 para People Pool")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  CUDA {torch.version.cuda}")
    else:
        print("[GPU] CUDA no disponible — se usará CPU (proceso lento)")

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

    # Build pitch lookup: crop_stem → source capture pitch.
    # 1_extract_information.py writes pitch to camera_data.csv;
    # 3_people_pool.py merges it into pool.csv — so each crop carries
    # the exact depression angle at which it was originally captured.
    pitch_lookup: dict = {}
    if 'pitch' in df_pool.columns:
        for _, r in df_pool.iterrows():
            try:
                stem = Path(r['name']).stem
                p    = float(r['pitch'])
                pitch_lookup[stem] = p if not np.isnan(p) else -45.0
            except Exception:
                pass
    if not pitch_lookup:
        print("[WARN] No se encontró columna 'pitch' en pool.csv — "
              "usando -45° por defecto para umbrales de completitud")

    n_valid   = 0
    n_invalid = 0
    n_skipped = 0
    n_updated = 0   # JSONs existentes actualizados con campos de completitud

    _NO_BODY = {'head_coverage': 0.0, 'feet_coverage': 0.0,
                'mask_aspect': 0.0,   'complete_body': False}

    for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="Segmentando", ncols=100):
        img_path   = Path(row['name'])
        patch_name = img_path.stem
        mask_path  = masks_dir / f"{patch_name}.png"
        json_path  = meta_dir  / f"{patch_name}.json"
        pitch_deg  = pitch_lookup.get(patch_name, -45.0)

        # ── Case A: JSON already exists ──────────────────────────────────────
        # Fast path: if completeness fields are missing, compute them from the
        # already-saved mask without re-running the expensive YOLO+SAM2 pipeline.
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            if 'complete_body' not in existing:
                if mask_path.exists():
                    saved_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    existing.update(
                        compute_completeness(saved_mask, pitch_deg)
                        if saved_mask is not None else _NO_BODY
                    )
                else:
                    existing.update(_NO_BODY)   # invalid crop — no mask
                with open(json_path, 'w') as f:
                    json.dump(existing, f, indent=4)
                n_updated += 1
            n_skipped += 1
            continue

        # ── Case B: full pipeline — YOLO detect → SAM2 segment ───────────────
        if not img_path.exists():
            n_skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            n_skipped += 1
            continue

        mask, stats = extract_mask(img, det_model, sam_model)

        # Angle-aware completeness metrics (uses pitch from 1_extract_information.py)
        stats.update(compute_completeness(mask, pitch_deg) if mask is not None else _NO_BODY)

        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)

        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
            n_valid += 1
        else:
            n_invalid += 1

    print(f"\n[SUCCESS] Pre-segmentación completada.")
    print(f"  Máscaras válidas:                       {n_valid}")
    print(f"  Máscaras inválidas:                     {n_invalid}")
    print(f"  JSONs actualizados (+ completeness):    {n_updated}")
    print(f"  Omitidas (ya al día o no encontradas):  {n_skipped}")


if __name__ == '__main__':
    t0 = time.perf_counter()
    process_pool()
    elapsed = time.perf_counter() - t0
    print(f"\nTiempo total: {int(elapsed//3600):02d}h {int(elapsed%3600//60):02d}m {elapsed%60:05.2f}s")
