import cv2
import numpy as np
import pandas as pd
import os
import sys
import random
import json
from pathlib import Path
from tqdm import tqdm
from glob import glob
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'people_pool'))
import config

# =============================================================================
# CONFIGURACIÓN V2 (Stratified Size Sampling)
# Cambios respecto a v1:
#   - Muestreo estratificado: cada imagen recibe TARGET_PER_BIN personas de cada
#     bin de tamaño, equilibrando activamente la distribución de salida.
#   - Pre-filtrado del pool por bin: solo se consideran crops cuya altura,
#     tras el escalado, pueda caer en el rango objetivo.
#   - Filtro por altura mínima final (HEIGHT_AUG_LOW) para descartar inserciones
#     imperceptibles.
#   - Reporte de distribución por partición al finalizar (consola + CSV).
#   - Eliminadas funciones no utilizadas (sharpen, harmonization, shadow).
# =============================================================================

NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 15)
HEIGHT_AUG_LOW       = getattr(config, 'HEIGHT_AUG_LOW', 5)
PITCH_TOLERANCE      = 25.0
MIN_SCALE            = 0.35
MAX_SCALE            = 1.30
MIN_CROP_SIZE        = 10
BORDER_MARGIN        = 20

ROOT_DATA         = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG   = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV     = Path(config.ROOT_POOL_PERSON)
DEPTH_MAPS_SUBDIR = 'depth_maps'
PARTITIONS        = config.PARTITIONS

# -----------------------------------------------------------------------------
# V2: Bins de tamaño objetivo (altura final en píxeles en la imagen de salida).
# Rango alcanzable: HEIGHT_MIN(28) * MIN_SCALE(0.35) ≈ 10px
#                   HEIGHT_MAX(79) * MAX_SCALE(1.30) ≈ 103px
# Cuatro bins iguales cubren ese rango. TARGET_PER_BIN personas por bin por imagen.
# -----------------------------------------------------------------------------
SIZE_BINS_PX   = [(10, 25), (25, 45), (45, 70), (70, 104)]
TARGET_PER_BIN = max(1, NUM_PEOPLE_PER_IMAGE // len(SIZE_BINS_PX))
MAX_ATTEMPTS   = 15  # intentos por candidato para encontrar posición en el bin

# =============================================================================
# FUNCIONES: ESCALADO MÉTRICO Y DETECCIÓN DE SUELO
# =============================================================================

def safe_float(val, default=1000.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except:
        return default


def calculate_metric_scale_v5(row_patch, bg_meta, d_target_norm):
    p_d_min  = safe_float(row_patch.get('depth_min', 0.1),   0.1)
    p_d_max  = safe_float(row_patch.get('depth_max', 100.0), 100.0)
    p_d_avg  = safe_float(row_patch.get('depth_avg', 0.5),   0.5)
    z_orig   = p_d_min + p_d_avg * (p_d_max - p_d_min)

    bg_d_min  = safe_float(bg_meta.get('depth_min', 0.1),   0.1)
    bg_d_max  = safe_float(bg_meta.get('depth_max', 100.0), 100.0)
    z_target  = bg_d_min + float(d_target_norm) * (bg_d_max - bg_d_min)

    f_orig = safe_float(row_patch.get('focal_y', 1000.0))
    f_bg   = safe_float(bg_meta.get('focal_y',   1000.0))

    return (z_orig / (z_target + 1e-6)) * (f_bg / f_orig)


def get_road_color_stats(bg_img, walkable_points):
    hsv_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    h, w    = bg_img.shape[:2]
    pixels  = []
    if walkable_points:
        for x, y in walkable_points:
            x, y = int(x), int(y)
            patch = hsv_img[max(0,y-5):min(h,y+5), max(0,x-5):min(w,x+5)].reshape(-1, 3)
            pixels.append(patch)
    if not pixels:
        pixels.append(hsv_img[int(h*0.8):int(h*0.95), int(w*0.3):int(w*0.7)].reshape(-1, 3))
    pixels   = np.vstack(pixels)
    mean_hsv = np.mean(pixels, axis=0)
    std_hsv  = np.maximum(np.std(pixels, axis=0), [10, 25, 25])
    return mean_hsv, std_hsv


def create_semantic_ground_mask(bg_img, d_map, walkable_points):
    hsv_img     = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    bg_h, bg_w  = bg_img.shape[:2]

    mean_hsv, std_hsv = get_road_color_stats(bg_img, walkable_points)
    lower_bound = np.clip(mean_hsv - std_hsv * 2.5, 0, 255).astype(np.uint8)
    upper_bound = np.clip(mean_hsv + std_hsv * 2.5, 0, 255).astype(np.uint8)
    if std_hsv[1] < 40 or mean_hsv[1] < 40:
        lower_bound[0] = 0; upper_bound[0] = 179

    valid_color_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    invalid_mask     = np.zeros((bg_h, bg_w), dtype=np.uint8)
    invalid_mask[d_map > 0.98] = 255

    lower_green  = np.array([30, 40, 40]); upper_green = np.array([90, 255, 255])
    green_mask   = cv2.dilate(cv2.inRange(hsv_img, lower_green, upper_green),
                              np.ones((9, 9), np.uint8), iterations=2)
    invalid_mask = cv2.bitwise_or(invalid_mask, green_mask)

    edges        = cv2.dilate(cv2.Canny((d_map * 255).astype(np.uint8), 15, 60),
                              np.ones((11, 11), np.uint8), iterations=2)
    invalid_mask = cv2.bitwise_or(invalid_mask, edges)

    final = cv2.bitwise_and(valid_color_mask, cv2.bitwise_not(invalid_mask))
    final[0:BORDER_MARGIN, :]  = 0; final[-BORDER_MARGIN:, :]  = 0
    final[:, 0:BORDER_MARGIN]  = 0; final[:, -BORDER_MARGIN:]  = 0
    return final

# =============================================================================
# UTILIDADES DE DISTRIBUCIÓN (V2)
# =============================================================================

def _print_distribution(partition, heights):
    total = len(heights)
    if total == 0:
        print(f"  [V2] {partition}: sin personas insertadas.")
        return
    print(f"\n[V2] Distribución de alturas insertadas — {partition} ({total} personas):")
    for bin_lo, bin_hi in SIZE_BINS_PX:
        count = sum(1 for h in heights if bin_lo <= h < bin_hi)
        pct   = 100 * count / total
        bar   = '█' * int(pct / 2)
        print(f"  [{bin_lo:3d}–{bin_hi:3d}px]: {count:5d} ({pct:5.1f}%) {bar}")


def _save_distribution_csv(out_dir, partition, heights):
    if not heights:
        return
    csv_path = out_dir / f'size_distribution_{partition}.csv'
    pd.DataFrame({'height_px': heights}).to_csv(str(csv_path), index=False)
    print(f"[V2] Distribución guardada en: {csv_path}")

# =============================================================================
# PIPELINE PRINCIPAL V2
# =============================================================================

def augment_partition(partition: str):
    images_dir = ROOT_DATA / partition / 'images'
    labels_dir = ROOT_DATA / partition / 'labels'
    pool_csv_p = ROOT_POOL_CSV / 'pool.csv'
    masks_dir  = ROOT_POOL_CSV / 'masks'
    meta_dir   = ROOT_POOL_CSV / 'metadata'
    meta_csv   = ROOT_DATA / partition / DEPTH_MAPS_SUBDIR / 'camera_data.csv'

    out_img_dir = ROOT_OUTPUT_AUG / partition / 'images'
    out_lbl_dir = ROOT_OUTPUT_AUG / partition / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not pool_csv_p.exists():
        return
    df_pool    = pd.read_csv(str(pool_csv_p))
    df_bg_meta = pd.read_csv(meta_csv).set_index('image_name')

    bg_images = [p for p in glob(str(images_dir / '*.jpg'))
                 if not os.path.basename(p).startswith('depth_')]

    partition_heights = []  # V2: acumula alturas de todas las personas insertadas

    for bg_path in tqdm(bg_images, desc=f'V2 Stratified {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_img  = cv2.imread(bg_path)
        if bg_img is None:
            continue
        bg_h, bg_w = bg_img.shape[:2]

        try:
            bg_meta = df_bg_meta.loc[bg_name]
            if isinstance(bg_meta, pd.DataFrame):
                bg_meta = bg_meta.iloc[0]
        except:
            bg_meta = pd.Series({'pitch': -45.0, 'depth_min': 0.1, 'depth_max': 100.0, 'focal_y': 1000.0})

        d_map = None
        for d_name in [f"depth_{bg_name}",
                       f"depth_{os.path.splitext(bg_name)[0]}.jpg",
                       f"depth_{os.path.splitext(bg_name)[0]}.png"]:
            path = ROOT_DATA / partition / DEPTH_MAPS_SUBDIR / d_name
            if path.exists():
                raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    d_map = cv2.resize(raw, (bg_w, bg_h),
                                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    break
        if d_map is None:
            d_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)

        original_labels = []
        walkable_points = []
        lbl_p = labels_dir / (os.path.splitext(bg_name)[0] + '.txt')
        if lbl_p.exists():
            with open(lbl_p, 'r') as f:
                original_labels = [l.strip() for l in f if l.strip()]
            for l in original_labels:
                p = l.split()
                if p[0] == '0':
                    walkable_points.append((float(p[1]) * bg_w,
                                            (float(p[2]) + float(p[4]) / 2) * bg_h))

        valid_mask        = create_semantic_ground_mask(bg_img, d_map, walkable_points)
        valid_y, valid_x  = np.where(valid_mask == 255)
        if len(valid_x) < 100:
            continue

        result_img    = bg_img.copy()
        final_labels  = original_labels.copy()
        aug_labels    = []  # solo las bboxes insertadas por aumentación

        bg_pitch     = safe_float(bg_meta.get('pitch', -45.0), -45.0)
        df_compatible = df_pool[abs(df_pool['pitch'] - bg_pitch) <= PITCH_TOLERANCE]

        aspect_ratio  = df_compatible['width_patch'] / df_compatible['height_patch']
        df_full_body  = df_compatible[(aspect_ratio >= 0.28) & (aspect_ratio <= 0.48)]
        if len(df_full_body) >= NUM_PEOPLE_PER_IMAGE:
            df_compatible = df_full_body

        if len(df_compatible) == 0:
            continue

        img_placed_heights = []

        # V2: bucle estratificado — TARGET_PER_BIN personas por bin de tamaño
        for bin_lo, bin_hi in SIZE_BINS_PX:
            placed_in_bin = 0

            # Pre-filtrar pool: solo crops que puedan producir alturas en [bin_lo, bin_hi]
            crop_h_min = bin_lo / MAX_SCALE
            crop_h_max = bin_hi / MIN_SCALE
            bin_pool   = df_compatible[
                (df_compatible['height_patch'] >= crop_h_min) &
                (df_compatible['height_patch'] <= crop_h_max)
            ]
            if len(bin_pool) == 0:
                continue

            candidates = bin_pool.sample(
                n=min(TARGET_PER_BIN * MAX_ATTEMPTS, len(bin_pool)),
                replace=True,
            )

            for _, row in candidates.iterrows():
                if placed_in_bin >= TARGET_PER_BIN:
                    break

                patch_path  = Path(row['name'])
                patch_stem  = patch_path.stem
                json_p      = meta_dir  / f"{patch_stem}.json"
                mask_p_file = masks_dir / f"{patch_stem}.png"

                if not json_p.exists() or not mask_p_file.exists():
                    continue
                with open(json_p, 'r') as f:
                    stats = json.load(f)
                if not stats.get('is_valid', False):
                    continue

                crop_orig = cv2.imread(str(patch_path))
                mask_orig = cv2.imread(str(mask_p_file), cv2.IMREAD_GRAYSCALE)
                if crop_orig is None or mask_orig is None:
                    continue

                for _ in range(MAX_ATTEMPTS):
                    idx = random.randint(0, len(valid_x) - 1)
                    cx, cy = int(valid_x[idx]), int(valid_y[idx])

                    scale = calculate_metric_scale_v5(row, bg_meta, d_map[cy, cx])
                    if scale < MIN_SCALE or scale > MAX_SCALE:
                        continue

                    nw = int(row['width_patch']  * scale)
                    nh = int(row['height_patch'] * scale)

                    # V2: verificar que la altura final cae en el bin objetivo
                    if not (bin_lo <= nh < bin_hi):
                        continue
                    # V2: descartar personas imperceptiblemente pequeñas
                    if nh < HEIGHT_AUG_LOW or nw < MIN_CROP_SIZE:
                        continue

                    x1, y1 = cx - nw // 2, cy - nh // 2
                    if x1 < 0 or y1 < 0 or x1 + nw >= bg_w or y1 + nh >= bg_h:
                        continue

                    crop   = cv2.resize(crop_orig, (nw, nh),
                                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
                    mask_p = cv2.resize(mask_orig, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    bg_roi = result_img[y1:y1+nh, x1:x1+nw].copy()

                    mask_blurred = cv2.GaussianBlur(mask_p, (3, 3), 0)
                    alpha        = mask_blurred.astype(float) / 255.0
                    alpha_3      = cv2.merge([alpha, alpha, alpha])

                    try:
                        blended = (crop.astype(float) * alpha_3) + (bg_roi.astype(float) * (1.0 - alpha_3))
                        result_img[y1:y1+nh, x1:x1+nw] = np.clip(blended, 0, 255).astype(np.uint8)
                        bbox_str = f"0 {cx/bg_w:.6f} {cy/bg_h:.6f} {nw/bg_w:.6f} {nh/bg_h:.6f}"
                        final_labels.append(bbox_str)
                        aug_labels.append(bbox_str)
                        placed_in_bin += 1
                        img_placed_heights.append(nh)
                        break
                    except:
                        continue

        out_name = f"{os.path.splitext(bg_name)[0]}_v2_{datetime.now().strftime('%H%M%S%f')}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f:
            f.write('\n'.join(final_labels))
        with open(str(out_lbl_dir / (out_name + '_aug.txt')), 'w') as f:
            f.write('\n'.join(aug_labels))

        partition_heights.extend(img_placed_heights)

    # V2: reporte final de distribución
    _print_distribution(partition, partition_heights)
    _save_distribution_csv(ROOT_OUTPUT_AUG / partition, partition, partition_heights)


if __name__ == '__main__':
    print("=" * 60)
    print("  Data Augmentation V2 — Stratified Size Sampling")
    print(f"  Bins: {SIZE_BINS_PX}")
    print(f"  Target por bin por imagen: {TARGET_PER_BIN}")
    print("=" * 60)

    for p in PARTITIONS:
        augment_partition(p)
