"""
Data Augmentation V2 — Geometrically-Consistent Patch Insertion for Drone Imagery

Core problems fixed vs V1:
  1. Cross-image VGGT depth comparison is INVALID (each VGGT run has its own arbitrary
     scale). Fix: scale is derived exclusively from the bounding boxes of already-labeled
     persons in the background image using the pinhole perspective law h ∝ 1/depth.

  2. Center-anchored placement causes persons to float. Fix: foot-anchored placement.

  3. No vanishing-point / horizon awareness in oblique views.
     Fix: horizon_row() excludes placements above the computed horizon line.

  4. Placements on non-flat surfaces (walls, car roofs).
     Fix: depth-consistency check around the foot region.

Quality constraints added in this version:
  QC-1. Resolution: reject crops that would need to be upscaled more than MAX_UPSCALE.
  QC-2. Scale ground-truth: predict target height from k-NN of labeled persons in the
         same image (depth-ratio interpolation). Falls back to global power-law model,
         then to depth-statistic estimate.
  QC-3. Distribution: track placed positions per image; reject placements that are
         too close to existing persons (original labels + already-placed augmentations).
  QC-6. Angle compensation: warp_crop_to_angle() applies a perspective transform to each
         crop to compensate for the difference between the source capture angle and the
         target background angle. Two components:
           · Vertical scale  ∝ cos(|dst_pitch|) / cos(|src_pitch|)
             — same standing person appears taller in oblique views, shorter near nadir.
           · Trapezoidal top-edge taper ∝ (1 − 0.45·sin(|pitch|))
             — at nadir the head/shoulders project as a narrow stripe (you see body depth,
               not width); at oblique the full front-facing silhouette fills the crop.
         The foot row (bottom) is always kept at full width. The PITCH_TOLERANCE filter
         limits the maximum correction needed to ≲ 25 % vertical / ≲ 15 % taper.
"""

import time, cv2, numpy as np, pandas as pd
import os, sys, random, json
from pathlib import Path
from tqdm import tqdm
from glob import glob
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'people_pool'))
import config

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch as _torch
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False

# ─── Configuration ────────────────────────────────────────────────────────────
NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 15)
HEIGHT_AUG_LOW       = getattr(config, 'HEIGHT_AUG_LOW', 5)
PITCH_TOLERANCE      = 20.0    # degrees — pool vs background pitch window (tighter for angle coherence)

# Scale / quality limits
MIN_SCALE            = 0.25    # minimum resize ratio (downscale floor)
MAX_SCALE            = 2.00    # physical maximum (perspective law)
MAX_UPSCALE          = 1.8     # QC-1: never upscale a source crop more than this
                               #   → avoids pixelated inserts in high-res backgrounds
MIN_SOURCE_HEIGHT    = 22      # QC-1: minimum source crop height in pixels
MIN_CROP_PX          = 10      # minimum dimension after resize

# Placement quality
BORDER_MARGIN        = 20      # ignore image border strip (px)
MAX_FOOT_DEPTH_STD   = 0.18    # QC-4: depth std threshold for flat-ground check
MAX_IOU_OVERLAP      = 0.15    # QC-3: max allowed bbox overlap fraction vs smaller person
MAX_CROP_REPEATS     = 2       # QC-5: max times same crop file is used in one image

# Scale calibration
MIN_REF_PERSONS      = 2       # min labeled persons for global model
KNN_REFS             = 7       # k-nearest labeled persons for depth-ratio interpolation
H_TYPICAL_FRAC       = 0.04    # last-resort: typical person ≈ 4% of image height

ROOT_DATA       = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)
DEPTH_SUBDIR    = 'depth_maps'
PARTITIONS      = config.PARTITIONS

import torch as _torch_check
DEVICE = 'cuda:0' if _torch_check.cuda.is_available() else 'cpu'

SIZE_BINS_PX = [(10, 25), (25, 45), (45, 70), (70, 104)]
MAX_ATTEMPTS = 40   # intentos máximos por unidad de target

# ── Segmentación semántica (SegFormer-B2 ADE20K) ──────────────────────────────
SEGFORMER_MODEL  = 'nvidia/segformer-b2-finetuned-ade-512-512'
SEG_DILATION_PX  = 20
# Clases ADE20K prohibidas para colocación de personas en imágenes de dron
FORBIDDEN_ADE20K = {
    1,   # building/edifice
    2,   # sky
    4,   # tree
    17,  # plant
    20,  # car
    21,  # water
    26,  # sea
    32,  # fence
    34,  # rock/stone
    38,  # railing
    42,  # column/pillar
    48,  # skyscraper
    53,  # stairs
    61,  # bridge
    68,  # hill
    72,  # palm
    80,  # bus
    83,  # truck
    84,  # tower
    93,  # pole
    102, # van
    109, # swimming pool
    140, # pier
}

# ── Distribución espacial en cuadrícula ───────────────────────────────────────
SPATIAL_GRID   = (4, 4)    # cuadrícula 4×4 para distribución más granular
N_CANDIDATES   = 500       # puntos candidatos a evaluar por selección de celda

# ── Zonas de exclusión por tamaño de objeto ───────────────────────────────────
# Los vehículos grandes necesitan mayor buffer para evitar que personas pequeñas
# aparezcan adyacentes a ellos (rompe perspectiva).
VEHICLE_CLASSES_ADE20K  = {20, 80, 83, 102}  # car, bus, truck, van
VEHICLE_EXTRA_DILATION  = 60                  # px extra sobre SEG_DILATION_PX

# ── Filtro anti-pixelación (SSIM) ─────────────────────────────────────────────
MIN_SSIM       = 0.55      # parches con SSIM simulado < umbral se rechazan

# ── Compatibilidad de tamaño con personas nativas ─────────────────────────────
NATIVE_MARGIN  = 1.8       # factor máximo sobre p90 nativo permitido


def safe_float(val, default=1000.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — SCALE PREDICTION FROM LABELED PERSONS (QC-2 primary method)
#
# Core idea: for a reference person with pixel height h_ref at depth d_ref and
# a target depth d_new, the pinhole perspective law gives:
#   h_new = h_ref * (d_ref / d_new)
#
# This requires NO metric depth — only that d_ref/d_new is consistent within
# the same VGGT depth map (which it is, even if absolute scale is arbitrary).
# ═══════════════════════════════════════════════════════════════════════════════

def parse_reference_persons(label_path, depth_map, depth_map_raw, img_w, img_h):
    """
    Extract reference scale anchors from existing labeled persons.
    depth_map      : normalized [0,1]  — usado solo para excluir píxeles extremos.
    depth_map_raw  : escala VGGT original — usado para los ratios de perspectiva.
    Returns list of dicts {x, y_feet, h, d} where d = depth_map_raw[y_feet, x].
    """
    refs = []
    if label_path is None or not label_path.exists():
        return refs
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != '0':
                continue
            xc   = float(parts[1]) * img_w
            yc   = float(parts[2]) * img_h
            h_px = float(parts[4]) * img_h
            if h_px < 6:
                continue
            x_ft  = int(np.clip(xc, 0, img_w - 1))
            y_ft  = int(np.clip(yc + h_px * 0.45, 0, img_h - 1))
            d_n   = float(depth_map[y_ft, x_ft])          # normalizado
            d_raw = float(depth_map_raw[y_ft, x_ft])      # escala VGGT
            if 0.02 < d_n < 0.97 and d_raw > 0:
                refs.append({'x': xc, 'y': float(y_ft), 'h': h_px, 'd': d_raw})
    return refs


def predict_height_from_refs(refs, x_q, y_q, depth_map, depth_map_raw, img_w, img_h):
    """
    Predict expected person height at (x_q, y_q).
    depth_map     : normalizado [0,1] — para la comprobación de umbral.
    depth_map_raw : escala VGGT — para los ratios de perspectiva.
    """
    if not refs:
        return None
    x_q = float(x_q); y_q = float(y_q)
    yi  = int(np.clip(y_q, 0, img_h - 1))
    xi  = int(np.clip(x_q, 0, img_w - 1))
    d_q_n   = float(depth_map[yi, xi])
    d_q_raw = float(depth_map_raw[yi, xi])
    if d_q_n < 0.02 or d_q_raw <= 0:
        return None

    dists = np.array([
        np.sqrt((x_q - r['x'])**2 + (y_q - r['y'])**2) + 1e-6
        for r in refs
    ])
    k   = min(KNN_REFS, len(refs))
    idx = np.argsort(dists)[:k]

    preds, weights = [], []
    for i in idx:
        r = refs[i]
        if r['d'] <= 0:
            continue
        # Ley de perspectiva con profundidades en escala VGGT (proporcional a Z real)
        h_pred = r['h'] * (r['d'] / d_q_raw)
        depth_penalty = abs(r['d'] - d_q_raw) / (r['d'] + 1e-6)
        w = 1.0 / (dists[i] * (1.0 + 2.0 * depth_penalty))
        preds.append(h_pred)
        weights.append(w)

    if not preds:
        return None
    w_arr = np.array(weights, dtype=float)
    w_arr /= w_arr.sum()
    return float(np.clip(np.dot(preds, w_arr), 4.0, img_h * 0.35))


# ─── Global power-law calibration (secondary fallback) ────────────────────────
def build_calibration(label_path, depth_map, depth_map_raw, img_w, img_h):
    """
    Fit h_px = k * depth_raw^n across all labeled persons (global per-image model).
    depth_map     : normalizado [0,1] — para el umbral de exclusión.
    depth_map_raw : escala VGGT — para el ajuste de potencia.
    """
    if label_path is None or not label_path.exists():
        return None
    samples = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != '0':
                continue
            xc, yc, h_px = float(parts[1])*img_w, float(parts[2])*img_h, float(parts[4])*img_h
            if h_px < 8:
                continue
            x_ft  = int(np.clip(xc, 0, img_w - 1))
            y_ft  = int(np.clip(yc + h_px * 0.45, 0, img_h - 1))
            d_n   = float(depth_map[y_ft, x_ft])
            d_raw = float(depth_map_raw[y_ft, x_ft])
            if 0.03 < d_n < 0.97 and d_raw > 0:
                samples.append((d_raw, h_px))
    if len(samples) < MIN_REF_PERSONS:
        return None
    depths  = np.array([s[0] for s in samples])
    heights = np.array([s[1] for s in samples])
    try:
        n, log_k = np.polyfit(np.log(depths), np.log(heights), 1)
        k = np.exp(log_k)
        if not (-3.0 <= n <= -0.05):
            raise ValueError
    except Exception:
        n = -1.0
        k = float(np.median(heights * depths))
    return {'k': float(k), 'n': float(n)}


def predict_height_from_model(cal, depth_raw):
    """cal fue ajustado sobre depth_raw → misma escala requerida."""
    if cal is None or depth_raw <= 0:
        return None
    return float(np.clip(cal['k'] * (depth_raw ** cal['n']), 4.0, 9999))


# ─── Last-resort depth-statistic fallback ─────────────────────────────────────
def build_depth_fallback(depth_map_raw, valid_y, valid_x, img_h):
    if len(valid_x) == 0:
        return None
    d_med = float(np.median(depth_map_raw[valid_y, valid_x]))
    if d_med <= 0:
        return None
    k = (img_h * H_TYPICAL_FRAC) * d_med
    return {'k': float(k), 'n': -1.0}


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — HORIZON CONSTRAINT (vanishing-point awareness)
# ═══════════════════════════════════════════════════════════════════════════════

def horizon_row(fy, cy, pitch_deg, img_h):
    """
    Y-pixel of the horizon for a camera at pitch_deg (< 0 = looking down).
    Formula: y_h = cy + fy * tan(pitch_rad).  Returns < 0 when off-screen.
    For most VisDrone images (pitch < -30°) this is off-screen → no constraint.
    """
    if pitch_deg >= 0:
        return -1.0
    yh = cy + fy * np.tan(np.radians(pitch_deg))
    return float(np.clip(yh, -img_h, img_h))


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — DEPTH-CONSISTENCY CHECK (reject non-flat surfaces)
# ═══════════════════════════════════════════════════════════════════════════════

def is_flat_ground(depth_map, x_ft, y_ft, nw, img_w, img_h):
    hw = max(2, nw // 4)
    region = depth_map[max(0, y_ft-3):min(img_h, y_ft+4),
                       max(0, x_ft-hw):min(img_w, x_ft+hw)]
    return region.size > 0 and float(np.std(region)) < MAX_FOOT_DEPTH_STD


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — SPATIAL DISTRIBUTION CHECK (QC-3)
# ═══════════════════════════════════════════════════════════════════════════════

def bbox_overlaps_existing(x_ft, y_ft, nw, nh, placed):
    """
    Rechaza si el candidato solapa o está demasiado cerca de una persona existente.

    Dos controles complementarios:
      1. Distancia mínima de pies: evita clustering incluso sin solapamiento de bbox.
         Umbral = max(nw, pnw) * 0.9 — las personas deben estar separadas al menos
         ~90% del ancho de la más ancha.
      2. IoU sobre área del bbox menor: rechaza solapamiento > MAX_IOU_OVERLAP.
    """
    x1n, y1n = x_ft - nw // 2, y_ft - nh
    x2n, y2n = x_ft + nw // 2, y_ft
    area_n   = nw * nh
    for px, py, pnw, pnh in placed:
        # ── Control 1: separación mínima entre centros ────────────────────
        min_sep_x = (nw + pnw) * 0.55          # ~55% de la suma de anchos
        min_sep_y = max(nh, pnh) * 0.40         # ~40% de la altura mayor
        if abs(x_ft - px) < min_sep_x and abs(y_ft - py) < min_sep_y:
            return True

        # ── Control 2: IoU sobre bbox ─────────────────────────────────────
        x1p, y1p = px - pnw // 2, py - pnh
        x2p, y2p = px + pnw // 2, py
        ix1, iy1 = max(x1n, x1p), max(y1n, y1p)
        ix2, iy2 = min(x2n, x2p), min(y2n, y2p)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter    = (ix2 - ix1) * (iy2 - iy1)
        min_area = min(area_n, pnw * pnh)
        if min_area > 0 and inter / min_area > MAX_IOU_OVERLAP:
            return True
    return False


def init_placed_from_labels(original_labels, img_w, img_h):
    """
    Seed the placed-persons tracker with the original ground-truth labels so
    new augmentations don't overlap existing annotated persons.
    Tuples: (x_feet, y_feet, w_px, h_px).
    """
    placed = []
    for l in original_labels:
        parts = l.split()
        if len(parts) < 5 or parts[0] != '0':
            continue
        xc   = float(parts[1]) * img_w
        yc   = float(parts[2]) * img_h
        h_px = int(float(parts[4]) * img_h)
        w_px = int(float(parts[3]) * img_w)
        placed.append((xc, yc + h_px * 0.5, w_px, h_px))
    return placed


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4.5 — SEMANTIC PLACEMENT MASK (SegFormer)
# ═══════════════════════════════════════════════════════════════════════════════

def load_segformer():
    """
    Carga SegFormer-B2 (ADE20K) una sola vez.
    Retorna (processor, model) o (None, None) si transformers no está instalado.
    """
    if not SEGFORMER_AVAILABLE:
        print('\n[WARN] SegFormer desactivado — transformers no instalado.')
        print('       Instalar: pip install transformers')
        print('       Fallback activo: detección de suelo por mapa de profundidad.\n')
        return None, None
    try:
        processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        model     = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        model     = model.to(DEVICE).eval()
        print(f'[INFO] SegFormer-B2 (ADE20K) listo en {DEVICE} — máscara semántica activada')
        return processor, model
    except Exception as e:
        print(f'[WARN] No se pudo cargar SegFormer: {e}')
        print('       Fallback activo: detección de suelo por mapa de profundidad.')
        return None, None


def get_semantic_forbidden_mask(bg_img, seg_processor, seg_model):
    """
    Ejecuta SegFormer sobre la imagen de fondo y devuelve una máscara uint8
    donde 255 = zona prohibida (árboles, edificios, vehículos, agua, etc.).
    Retorna None si SegFormer no está disponible.
    """
    if seg_processor is None or seg_model is None:
        return None
    try:
        bg_h, bg_w = bg_img.shape[:2]
        bg_rgb     = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        inputs = seg_processor(images=bg_rgb, return_tensors='pt')
        # Mover cada tensor al device explícitamente (evita fallo en BatchFeature.to())
        inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with _torch.no_grad():
            logits = seg_model(**inputs).logits      # (1, 150, H/4, W/4)

        seg_map = _torch.nn.functional.interpolate(
            logits, size=(bg_h, bg_w), mode='bilinear', align_corners=False
        ).argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)

        forbidden = np.zeros((bg_h, bg_w), dtype=np.uint8)
        vehicles  = np.zeros((bg_h, bg_w), dtype=np.uint8)
        for cls_id in FORBIDDEN_ADE20K:
            forbidden[seg_map == cls_id] = 255
        for cls_id in VEHICLE_CLASSES_ADE20K:
            vehicles[seg_map == cls_id] = 255

        # Dilación base para todas las clases prohibidas
        if SEG_DILATION_PX > 0:
            k         = np.ones((SEG_DILATION_PX, SEG_DILATION_PX), np.uint8)
            forbidden = cv2.dilate(forbidden, k, iterations=1)

        # Dilación extra para vehículos: evita que personas pequeñas queden
        # adyacentes a objetos grandes rompiendo la perspectiva
        if vehicles.any():
            total_v_dil = SEG_DILATION_PX + VEHICLE_EXTRA_DILATION
            k_v         = np.ones((total_v_dil, total_v_dil), np.uint8)
            forbidden   = cv2.bitwise_or(forbidden, cv2.dilate(vehicles, k_v))

        return forbidden
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — PLACEMENT MASK
# ═══════════════════════════════════════════════════════════════════════════════

def _road_color_stats(bg_img, walkable_pts):
    hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    h, w = bg_img.shape[:2]
    pxs  = []
    for x, y in (walkable_pts or []):
        x, y = int(x), int(y)
        pxs.append(hsv[max(0,y-5):min(h,y+5), max(0,x-5):min(w,x+5)].reshape(-1,3))
    if not pxs:
        pxs.append(hsv[int(h*.8):int(h*.95), int(w*.3):int(w*.7)].reshape(-1,3))
    pxs = np.vstack(pxs)
    mu  = np.mean(pxs, axis=0)
    std = np.maximum(np.std(pxs, axis=0), [10, 25, 25])
    return mu, std


def _sky_exclusion_mask(d_map, hsv, bg_h, bg_w, pitch_deg):
    """
    Adaptive sky detection combining depth and color cues.

    Root causes it addresses:
    - VGGT assigns sky depth in [0.85–0.97], not just > 0.98.
    - Hazy/gray sky shares HSV color with light-colored pavement.

    Strategy:
    1. Adaptive depth threshold: compare top-20% vs mid rows. If top is
       significantly deeper → sky is visible → lower the exclusion threshold.
    2. Sky color mask: sky/haze is high-brightness (V > 130) AND low-saturation
       (S < 45). Combined with a relaxed depth threshold it catches hazy sky.
    3. Near-nadir images (pitch < -55°): no visible sky → fall back to 0.97.
    """
    # ── 1. Adaptive depth threshold ──────────────────────────────────────
    if pitch_deg < -55:
        sky_thresh = 0.97           # nadir: barely any sky visible
    else:
        top_n      = max(1, bg_h // 5)
        mid_lo     = bg_h // 3
        mid_hi     = 2 * bg_h // 3
        top_d      = float(np.mean(d_map[:top_n, :]))
        mid_d      = float(np.median(d_map[mid_lo:mid_hi, :]))
        if top_d > mid_d + 0.08:   # clear sky/ground depth separation
            sky_thresh = max(0.82, top_d - 0.06)
        else:
            sky_thresh = 0.95       # no clear sky gradient → conservative threshold

    depth_sky = (d_map >= sky_thresh).astype(np.uint8) * 255

    # ── 2. Sky color mask: high brightness AND low saturation ─────────────
    s_ch      = hsv[:, :, 1].astype(float)
    v_ch      = hsv[:, :, 2].astype(float)
    color_sky = ((s_ch < 45) & (v_ch > 130)).astype(np.uint8) * 255

    # ── 3. Combined: union of strong depth evidence OR (depth + color) ───
    soft_depth = (d_map >= max(0.78, sky_thresh - 0.10)).astype(np.uint8) * 255
    combined   = cv2.bitwise_or(depth_sky,
                                cv2.bitwise_and(soft_depth, color_sky))
    combined   = cv2.dilate(combined, np.ones((15, 15), np.uint8), iterations=2)
    return combined


def _horizon_from_depth(d_map, bg_h):
    """
    Detect the horizon row from the depth map's row-wise mean profile.
    Uses a conservative threshold (30% of sky→ground range) so it errs on the
    side of excluding more sky rather than less.
    Returns: row index (0 = not detected / no clear sky-ground split).
    """
    row_means = np.mean(d_map, axis=1).astype(float)
    k      = max(3, bg_h // 40)          # slightly wider smoothing kernel
    kernel = np.ones(k) / k
    smooth = np.convolve(row_means, kernel, mode='same')

    top_n        = max(1, bg_h // 7)
    sky_level    = float(np.mean(smooth[:top_n]))
    mid_lo, mid_hi = bg_h // 3, 2 * bg_h // 3
    ground_level = float(np.median(smooth[mid_lo:mid_hi]))

    if sky_level - ground_level < 0.05:  # no clear sky/ground split
        return 0

    # 0.30 → conservative: detect the BEGINNING of the sky→ground transition
    # (excludes more rows than 0.45, reducing false placements in sky/haze)
    threshold  = sky_level - (sky_level - ground_level) * 0.30
    candidates = np.where(smooth < threshold)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def _min_exclusion_from_pitch(pitch_deg, bg_h):
    """
    Physical lower bound on the number of top rows that MUST be excluded.

    Derivation: for a camera at depression angle α = |pitch_deg| below horizontal,
    the sky occupies approximately (0.5 + sin(pitch_rad) * 0.65) of the image height.
    This formula is a smooth approximation calibrated on VisDrone pitch statistics.

    Why this is necessary: VGGT sometimes overestimates the pitch (returns −40° for an
    image that is visually −15°). When that happens, horizon_row() produces a value
    < 0 (off-screen) and _horizon_from_depth() may also underestimate the sky region.
    This function provides a guaranteed safety floor that does NOT depend on VGGT's
    pitch accuracy — it only needs to be in the correct ORDER OF MAGNITUDE.

    Examples:
      pitch = −10° → exclude top ~44%   (very oblique, large sky)
      pitch = −20° → exclude top ~28%
      pitch = −35° → exclude top ~13%
      pitch = −55° → exclude top  ~4%
      pitch = −90° → exclude top  ~3%  (near-nadir, minimal sky)
    """
    pitch_rad = np.radians(float(pitch_deg))
    sky_frac  = float(np.clip(0.5 + np.sin(pitch_rad) * 0.65, 0.03, 0.55))
    return int(bg_h * sky_frac)


def _depth_ground_mask(d_map_r, refs, bg_h, bg_w):
    """
    Detecta píxeles a nivel de suelo usando el mapa de profundidad VGGT.

    Principio físico para imágenes de dron:
      - Suelo (calle, plaza) = MAYOR profundidad (más lejos del dron)
      - Techos, copas de árboles, techos de vehículos = MENOR profundidad (más cerca)

    Estrategia: estimar la profundidad mínima del suelo usando las personas
    ya anotadas como anclajes. Rechazar todo lo que sea significativamente
    más cercano al dron que ese nivel.

    Sin referencias: estima la profundidad del suelo desde la zona inferior
    de la imagen (que en vistas de dron suele ser suelo).
    """
    # Estimar profundidad mínima del plano de suelo
    if refs:
        valid_depths = [r['d'] for r in refs if r['d'] > 0]
    else:
        valid_depths = []

    if valid_depths:
        # p10 de las profundidades de referencia × margen de seguridad
        d_min_ground = float(np.percentile(valid_depths, 10)) * 0.55
    else:
        # Sin referencias: usar cuarto inferior de la imagen como proxy del suelo
        r0 = int(bg_h * 0.60); r1 = int(bg_h * 0.92)
        c0 = int(bg_w * 0.10); c1 = int(bg_w * 0.90)
        region = d_map_r[r0:r1, c0:c1]
        if region.size == 0:
            return np.ones((bg_h, bg_w), dtype=np.uint8) * 255
        d_min_ground = float(np.percentile(region, 15)) * 0.50

    if d_min_ground <= 0:
        return np.ones((bg_h, bg_w), dtype=np.uint8) * 255

    # Aceptar solo píxeles ≥ d_min_ground (a nivel de suelo o más profundo)
    valid = (d_map_r >= d_min_ground).astype(np.uint8) * 255

    # Limpieza morfológica: cerrar huecos pequeños, eliminar ruido
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN,  np.ones((13, 13), np.uint8))
    return valid


def make_placement_mask(bg_img, d_map, d_map_r, refs, walkable_pts,
                         pitch_deg, fy, cy_principal, seg_forbidden=None):
    """
    Construye la máscara de zonas válidas para insertar personas.

    Capas de filtrado (orden de aplicación):
      1. Detección de nivel de suelo por profundidad  (siempre activa)
         → rechaza techos, copas de árboles, techos de vehículos
      2. SegFormer semántico                          (si disponible)
         → rechaza clases prohibidas: árbol, edificio, auto, agua, etc.
      3. Exclusión de cielo (profundidad + color)
      4. Vegetación densa (HSV verde)
      5. Márgenes de borde
      6. Exclusión de horizonte
    """
    bg_h, bg_w = bg_img.shape[:2]
    hsv        = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)

    # ── Capa 1: nivel de suelo por profundidad (siempre) ─────────────────────
    mask = _depth_ground_mask(d_map_r, refs, bg_h, bg_w)

    # ── Capa 2: semántica SegFormer (si disponible) ───────────────────────────
    if seg_forbidden is not None:
        mask[seg_forbidden > 0] = 0

    # ── Capa 3: exclusión de cielo ────────────────────────────────────────────
    sky = _sky_exclusion_mask(d_map, hsv, bg_h, bg_w, pitch_deg)
    mask[sky > 0] = 0

    # ── Capa 4: vegetación densa (HSV verde) ──────────────────────────────────
    green = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    mask[cv2.dilate(green, np.ones((15, 15), np.uint8), iterations=2) > 0] = 0

    # ── Capa 5: márgenes de borde ─────────────────────────────────────────────
    mask[:BORDER_MARGIN,  :]  = 0
    mask[-BORDER_MARGIN:, :]  = 0
    mask[:,  :BORDER_MARGIN]  = 0
    mask[:, -BORDER_MARGIN:]  = 0

    # ── Capa 6: horizonte (máximo de tres estimaciones independientes) ────────
    yh_geom    = horizon_row(fy, cy_principal, pitch_deg, bg_h)
    yh_depth   = _horizon_from_depth(d_map, bg_h)
    yh_physics = _min_exclusion_from_pitch(pitch_deg, bg_h)
    yh_final   = max(yh_geom if yh_geom >= 0 else 0, yh_depth, yh_physics)
    cutoff     = int(min(bg_h - 1, yh_final + bg_h * 0.04))
    mask[:cutoff, :] = 0

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6 — PERSPECTIVE CORRECTION FOR CROP ANGLE MISMATCH  (QC-6)
# ═══════════════════════════════════════════════════════════════════════════════

def warp_crop_to_angle(crop_img, mask_img, src_pitch_deg, dst_pitch_deg):
    """
    Aplica corrección de taper horizontal al parche para compensar la
    diferencia de ángulo de depresión entre la cámara fuente y la de destino.

    Solo se corrige el ANCHO DE LA CABEZA respecto a los pies (taper):
      - Nadir      → cabeza angosta (se ve la profundidad corporal, no los hombros)
      - Oblicuo    → cabeza ancha   (silueta frontal completa)

    La escala vertical NO se modifica: el resize posterior a (nw, nh) ya
    ajusta la altura aparente correcta derivada de la predicción por profundidad.
    Modificar la altura aquí causaría doble corrección y deformaría proporciones.

    Solo actúa para diferencias de pitch en [5°, 15°]; fuera de ese rango
    la corrección sería imperceptible (< 5°) o demasiado visible (> 15°).
    """
    if crop_img is None or mask_img is None:
        return crop_img, mask_img
    h, w = crop_img.shape[:2]
    if h < 2 or w < 2:
        return crop_img, mask_img

    delta = abs(float(dst_pitch_deg) - float(src_pitch_deg))
    if delta < 5.0 or delta > 15.0:
        return crop_img, mask_img

    sin_src = abs(np.sin(np.radians(src_pitch_deg)))
    sin_dst = abs(np.sin(np.radians(dst_pitch_deg)))

    # Fracción de ancho en la parte superior: 1 en oblicuo, ~0.65 en nadir
    top_frac_src = max(0.40, 1.0 - 0.35 * sin_src)
    top_frac_dst = max(0.40, 1.0 - 0.35 * sin_dst)
    taper        = float(np.clip(top_frac_dst / top_frac_src, 0.88, 1.12))

    if abs(taper - 1.0) < 0.03:      # corrección < 3% — no vale la pena
        return crop_img, mask_img

    top_w   = max(2, min(w, int(w * taper)))
    top_off = (w - top_w) // 2

    # Transformación trapezoidal: pies = ancho completo, cabeza = ancho reducido/ampliado
    src_pts = np.float32([[0,   0  ], [w-1, 0  ], [w-1, h-1], [0,   h-1]])
    dst_pts = np.float32([
        [top_off,           0  ],
        [top_off + top_w-1, 0  ],
        [w-1,               h-1],
        [0,                 h-1],
    ])

    M           = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img  = cv2.warpPerspective(crop_img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(mask_img, M, (w, h), flags=cv2.INTER_NEAREST)

    return warped_img, warped_mask


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7 — COMPOSITING
# ═══════════════════════════════════════════════════════════════════════════════

def paste_crop(canvas, crop, mask_gray, x1, y1, nw, nh):
    alpha = cv2.GaussianBlur(mask_gray, (3,3), 0).astype(float) / 255.0
    a3    = cv2.merge([alpha, alpha, alpha])
    roi   = canvas[y1:y1+nh, x1:x1+nw].astype(float)
    canvas[y1:y1+nh, x1:x1+nw] = np.clip(
        crop.astype(float) * a3 + roi * (1.0 - a3), 0, 255
    ).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7.5 — HELPERS: TARGET DISTRIBUTION · BALANCED PLACEMENT · SSIM QC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_target_per_bin(original_labels, img_h, bins, total_target):
    """
    Calcula cuántas personas insertar por bin para moverse hacia una distribución
    UNIFORME entre bins (atacar la cola larga).

    Estrategia: prioridad inversa al conteo existente.
      - Bin con 0 personas existentes  → prioridad máxima (1/1   = 1.00)
      - Bin con 9 personas existentes  → prioridad media  (1/10  = 0.10)
      - Bin con 99 personas existentes → prioridad baja   (1/100 = 0.01)

    El presupuesto total (total_target) se reparte proporcionalmente a las
    prioridades, de modo que los bins más subrepresentados reciben más intentos.
    """
    # Contar personas existentes por bin
    existing = {b: 0 for b in bins}
    for lbl in original_labels:
        parts = lbl.split()
        if len(parts) >= 5 and parts[0] == '0':
            h_px = float(parts[4]) * img_h
            for b in bins:
                if b[0] <= h_px < b[1]:
                    existing[b] += 1
                    break

    # Base mínima garantizada para cada bin (evita que bins alcanzables queden sin intentos)
    base = max(1, total_target // len(bins))

    # Bonus para bins subrepresentados: prioridad inversa al conteo existente
    priorities = {b: 1.0 / (existing[b] + 1.0) for b in bins}
    total_prio = sum(priorities.values())

    targets = {}
    for b in bins:
        bonus = int(total_target * priorities[b] / total_prio)
        # max(base, bonus): nunca por debajo del mínimo garantizado.
        # Bins subrepresentados reciben más; los sobrerepresentados conservan el base.
        targets[b] = max(base, bonus)

    return targets, existing

def compute_native_size_stats(original_labels, img_h):
    """
    Extrae la distribución de alturas de personas ya anotadas en la imagen.
    Retorna dict con p10, p90, mean, o None si no hay personas.
    Se usa para limitar los bins activos y evitar insertar personas de tamaño
    inconsistente con la perspectiva real de la imagen destino.
    """
    heights = [
        float(l.split()[4]) * img_h
        for l in original_labels
        if l.split() and l.split()[0] == '0' and float(l.split()[4]) * img_h >= 5
    ]
    if not heights:
        return None
    return {
        'p10':  float(np.percentile(heights, 10)),
        'p90':  float(np.percentile(heights, 90)),
        'mean': float(np.mean(heights)),
    }


def pick_placement_balanced(valid_y, valid_x, bg_h, bg_w, placed_feet):
    """
    Selecciona un punto de colocación favoreciendo celdas con menos personas.

    Estrategia en dos pases:
      Pase 1: sólo candidatos en celdas con conteo == mínimo absoluto.
      Pase 2 (fallback): candidatos en celdas con conteo <= media + 1.
    Esto evita el clustering que producía la condición min+1 original.
    """
    n = len(valid_x)
    if n == 0:
        return None, None

    rows, cols  = SPATIAL_GRID
    cell_counts = np.zeros((rows, cols), dtype=int)
    for px, py, _, _ in placed_feet:
        r = min(int(py / bg_h * rows), rows - 1)
        c = min(int(px / bg_w * cols), cols - 1)
        cell_counts[r, c] += 1

    min_count  = int(cell_counts.min())
    mean_count = float(cell_counts.mean())

    indices = random.sample(range(n), min(N_CANDIDATES, n))

    # Pase 1: preferir estrictamente las celdas menos pobladas (== mínimo)
    for i in indices:
        cx, cy = int(valid_x[i]), int(valid_y[i])
        r = min(int(cy / bg_h * rows), rows - 1)
        c = min(int(cx / bg_w * cols), cols - 1)
        if cell_counts[r, c] == min_count:
            return cx, cy

    # Pase 2: relajar a celdas <= media + 1 (cuando el mínimo no tiene candidatos)
    for i in indices:
        cx, cy = int(valid_x[i]), int(valid_y[i])
        r = min(int(cy / bg_h * rows), rows - 1)
        c = min(int(cx / bg_w * cols), cols - 1)
        if cell_counts[r, c] <= mean_count + 1:
            return cx, cy

    # Fallback final: punto aleatorio
    i = random.randint(0, n - 1)
    return int(valid_x[i]), int(valid_y[i])


def is_patch_quality_ok(crop_resized, scale_factor):
    """
    QC-7: Filtro anti-pixelación para parches upscaleados.
    Simula upscale→downscale y mide pérdida de información (pseudo-SSIM).
    Solo actúa cuando scale_factor > 1.2; downscale siempre pasa.
    """
    if scale_factor <= 1.2:
        return True
    h, w = crop_resized.shape[:2]
    if h < 4 or w < 4:
        return False

    small = cv2.resize(crop_resized, (max(2, w // 2), max(2, h // 2)),
                       interpolation=cv2.INTER_AREA)
    back  = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    g_orig = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY).astype(float)
    g_back = cv2.cvtColor(back,         cv2.COLOR_BGR2GRAY).astype(float)
    std_o  = g_orig.std() + 1e-6
    std_b  = g_back.std() + 1e-6
    ssim   = float(np.mean((g_orig - g_orig.mean()) * (g_back - g_back.mean()))) / (std_o * std_b)
    return ssim >= MIN_SSIM


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 8 — MAIN AUGMENTATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def augment_partition(partition: str):
    images_dir = ROOT_DATA / partition / 'images'
    labels_dir = ROOT_DATA / partition / 'labels'
    pool_csv_p = ROOT_POOL_CSV / partition / 'pool.csv'
    masks_dir  = ROOT_POOL_CSV / partition / 'masks'
    meta_dir   = ROOT_POOL_CSV / partition / 'metadata'
    meta_csv   = ROOT_DATA / partition / 'depth_maps' / 'camera_data.csv'

    out_img = ROOT_OUTPUT_AUG / partition / 'images'
    out_lbl = ROOT_OUTPUT_AUG / partition / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    if not pool_csv_p.exists():
        print(f'[ERROR] pool.csv not found: {pool_csv_p}'); return
    if not meta_csv.exists():
        print(f'[ERROR] camera_data.csv not found: {meta_csv}'); return

    df_pool    = pd.read_csv(str(pool_csv_p))
    df_bg_meta = pd.read_csv(str(meta_csv)).set_index('image_name')

    bg_images = sorted(
        p for p in glob(str(images_dir / '*.jpg'))
        if not os.path.basename(p).startswith('depth_')
    )

    # ── Cargar SegFormer una sola vez por partición ────────────────────────
    seg_processor, seg_model = load_segformer()

    # Per-partition statistics
    n_placed   = 0; n_skipped_img = 0
    stat_knn   = 0; stat_model    = 0; stat_fb    = 0
    stat_qc1   = 0; stat_qc3      = 0; stat_flat  = 0; stat_qc6b = 0
    stat_qc7   = 0  # pixelation rejections
    partition_heights      = []   # alturas insertadas
    partition_orig_heights = []   # alturas originales (para medir balance)

    print(f'\n[{partition}] {len(bg_images)} images | pool: {len(df_pool)} crops')

    for bg_path in tqdm(bg_images, desc=f'V2 {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_img  = cv2.imread(bg_path)
        if bg_img is None:
            continue
        bg_h, bg_w = bg_img.shape[:2]

        # ── Camera metadata ────────────────────────────────────────────────
        try:
            bg_meta = df_bg_meta.loc[bg_name]
            if isinstance(bg_meta, pd.DataFrame):
                bg_meta = bg_meta.iloc[0]
        except KeyError:
            bg_meta = pd.Series({
                'pitch': -45.0, 'focal_y': 1000.0,
                'depth_min': 0.1, 'depth_max': 100.0,
            })

        bg_pitch     = safe_float(bg_meta.get('pitch',    -45.0), -45.0)
        fy           = safe_float(bg_meta.get('focal_y', 1000.0), 1000.0)
        cy_principal = bg_h / 2.0   # VGGT always sets principal_y = H/2

        # Rango de profundidad VGGT para reconstrucción de escala real
        bg_d_min = safe_float(bg_meta.get('depth_min', 0.0), 0.0)
        bg_d_max = safe_float(bg_meta.get('depth_max', 1.0), 1.0)

        # ── Depth map ──────────────────────────────────────────────────────
        d_map = None
        for stem in [os.path.splitext(bg_name)[0] + '.png', bg_name]:
            p = ROOT_DATA / partition / DEPTH_SUBDIR / f'depth_{stem}'
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    max_val = 65535.0 if raw.dtype == np.uint16 else 255.0
                    d_map   = cv2.resize(raw, (bg_w, bg_h),
                                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / max_val
                    break
        if d_map is None:
            d_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)

        # Reconstruir escala VGGT original: norm_d = (raw-d_min)/(d_max-d_min)
        # → raw = norm_d*(d_max-d_min)+d_min   (necesario para ratios correctos)
        d_map_r = d_map * (bg_d_max - bg_d_min) + bg_d_min

        # ── Original labels ────────────────────────────────────────────────
        original_labels = []
        walkable_pts    = []
        lbl_path = labels_dir / (os.path.splitext(bg_name)[0] + '.txt')
        if lbl_path.exists():
            with open(lbl_path) as f:
                original_labels = [l.strip() for l in f if l.strip()]
            for l in original_labels:
                p = l.split()
                if p[0] == '0':
                    walkable_pts.append(
                        (float(p[1]) * bg_w, (float(p[2]) + float(p[4]) / 2) * bg_h)
                    )

        # ── Referencias de personas anotadas (necesario para nivel de suelo) ─
        # Se calcula ANTES de make_placement_mask para que _depth_ground_mask
        # pueda usar las profundidades de pies reales como ancla del plano de suelo.
        refs = parse_reference_persons(lbl_path, d_map, d_map_r, bg_w, bg_h)

        # ── Máscara semántica (SegFormer) — una vez por imagen ────────────
        seg_forbidden = get_semantic_forbidden_mask(bg_img, seg_processor, seg_model)

        # ── Placement mask ─────────────────────────────────────────────────
        valid_mask = make_placement_mask(bg_img, d_map, d_map_r, refs,
                                         walkable_pts, bg_pitch, fy, cy_principal,
                                         seg_forbidden=seg_forbidden)
        valid_y, valid_x = np.where(valid_mask == 255)
        if len(valid_x) < 100:
            n_skipped_img += 1
            continue

        # ── Scale calibration (two remaining tiers) ────────────────────────
        # Tier 2: global power-law model per image
        calibration = build_calibration(lbl_path, d_map, d_map_r, bg_w, bg_h)

        # Tier 3: depth-statistic fallback
        fb_cal = build_depth_fallback(d_map_r, valid_y, valid_x, bg_h)

        # ── Compatible pool ────────────────────────────────────────────────
        df_compat = df_pool[abs(df_pool['pitch'] - bg_pitch) <= PITCH_TOLERANCE].copy()
        ar = df_compat['width_patch'] / df_compat['height_patch']
        df_full = df_compat[(ar >= 0.28) & (ar <= 0.48)]
        if len(df_full) >= NUM_PEOPLE_PER_IMAGE:
            df_compat = df_full
        if len(df_compat) == 0:
            n_skipped_img += 1
            continue

        # QC-1: pre-filter pool — exclude crops too small to produce quality results
        # at reasonable upscale ratios. We keep crops where height_patch >= MIN_SOURCE_HEIGHT
        df_compat = df_compat[df_compat['height_patch'] >= MIN_SOURCE_HEIGHT]
        if len(df_compat) == 0:
            n_skipped_img += 1
            continue

        result_img   = bg_img.copy()
        final_labels = original_labels.copy()
        aug_labels   = []

        # QC-3: init placement tracker with original labeled persons
        placed_feet = init_placed_from_labels(original_labels, bg_w, bg_h)

        # QC-5: track how many times each crop file has been used this image
        crop_usage: dict = {}

        # ── Targets por bin: prioridad inversa al conteo existente ───────────
        # Los bins subrepresentados en esta imagen reciben más presupuesto,
        # moviendo activamente la distribución hacia la uniformidad.
        targets, existing_counts = compute_target_per_bin(
            original_labels, bg_h, SIZE_BINS_PX, NUM_PEOPLE_PER_IMAGE
        )

        # Acumular distribución original para estadísticas de partición
        for h_px in existing_counts.values():
            pass  # existing_counts ya tiene el conteo; se usa más abajo en stats
        for lbl in original_labels:
            parts = lbl.split()
            if len(parts) >= 5 and parts[0] == '0':
                partition_orig_heights.append(float(parts[4]) * bg_h)

        # ── Stratified placement ───────────────────────────────────────────
        for bin_lo, bin_hi in SIZE_BINS_PX:
            bin_target    = targets[(bin_lo, bin_hi)]
            placed_in_bin = 0
            attempts      = 0

            while placed_in_bin < bin_target and attempts < MAX_ATTEMPTS * bin_target:
                attempts += 1

                # Punto balanceado: favorece celdas con menos personas
                x_feet, y_feet = pick_placement_balanced(
                    valid_y, valid_x, bg_h, bg_w, placed_feet)
                if x_feet is None:
                    break

                d_feet   = float(d_map[y_feet, x_feet])    # normalizado — umbral
                d_feet_r = float(d_map_r[y_feet, x_feet])  # escala VGGT — ratios

                if d_feet < 0.02 or d_feet_r <= 0:
                    continue

                # ── Predict target height (three-tier hierarchy) ───────────
                exp_h = predict_height_from_refs(refs, x_feet, y_feet,
                                                 d_map, d_map_r, bg_w, bg_h)
                if exp_h is not None:
                    stat_knn += 1
                else:
                    exp_h = predict_height_from_model(calibration, d_feet_r)
                    if exp_h is not None:
                        stat_model += 1
                    else:
                        exp_h = predict_height_from_model(fb_cal, d_feet_r)
                        if exp_h is not None:
                            stat_fb += 1
                        else:
                            # Sin predicción válida: saltar este punto.
                            # Usar el midpoint del bin ignoraría la profundidad
                            # y produciría personas a escala incorrecta.
                            continue

                # This depth location belongs to a different bin — try elsewhere
                if not (bin_lo <= exp_h < bin_hi):
                    continue

                # QC-1: filter pool to crops that need scale ≤ MAX_UPSCALE
                min_src_h = max(MIN_SOURCE_HEIGHT, exp_h / MAX_UPSCALE)
                max_src_h = exp_h / (MIN_SCALE)        # can be shrunk freely
                bin_pool  = df_compat[
                    (df_compat['height_patch'] >= min_src_h) &
                    (df_compat['height_patch'] <= max_src_h)
                ]
                if len(bin_pool) == 0:
                    stat_qc1 += 1
                    continue

                # QC-5: prefer crops not yet used (or used < MAX_CROP_REPEATS)
                fresh = bin_pool[
                    bin_pool['name'].map(lambda n: crop_usage.get(n, 0) < MAX_CROP_REPEATS)
                ]
                row = (fresh if len(fresh) > 0 else bin_pool).sample(1).iloc[0]
                crop_usage[row['name']] = crop_usage.get(row['name'], 0) + 1
                patch_stem = Path(row['name']).stem
                json_p     = meta_dir  / f'{patch_stem}.json'
                mask_p     = masks_dir / f'{patch_stem}.png'

                if not json_p.exists() or not mask_p.exists():
                    continue
                with open(json_p) as f:
                    stats = json.load(f)
                if not stats.get('is_valid', False):
                    continue
                # Filtro de calidad de máscara (default=1.0 para JSONs sin estas métricas)
                if stats.get('solidity',   1.0) < 0.55:
                    continue
                if stats.get('smoothness', 1.0) < 0.10:
                    continue

                # QC-6b: oblique backgrounds require a complete body silhouette
                # (head + feet visible, tall aspect ratio). The complete_body flag
                # is written by 4_extract_masks.py using the source pitch from
                # 1_extract_information.py. `is False` leaves old JSONs unfiltered.
                if bg_pitch > -40.0 and stats.get('complete_body') is False:
                    stat_qc6b += 1
                    continue

                # El warp solo ajusta el taper (ancho de cabeza), no la altura.
                # h_eff es la altura original del parche; el resize lo lleva a exp_h.
                src_pitch = safe_float(row.get('pitch', bg_pitch), bg_pitch)
                h_eff     = int(row['height_patch'])

                scale = exp_h / max(h_eff, 1)
                if not (MIN_SCALE <= scale <= MAX_SCALE):
                    continue

                nw = int(row['width_patch'] * scale)
                nh = int(exp_h)
                if nh < HEIGHT_AUG_LOW or nw < MIN_CROP_PX:
                    continue
                if not (bin_lo <= nh < bin_hi):
                    continue

                # Foot-anchored placement: feet at (x_feet, y_feet)
                x1 = x_feet - nw // 2
                y1 = y_feet - nh
                if x1 < 0 or y1 < 0 or x1 + nw >= bg_w or y1 + nh >= bg_h:
                    continue

                # QC-4: depth consistency at feet (reject walls, car roofs, etc.)
                if not is_flat_ground(d_map, x_feet, y_feet, nw, bg_w, bg_h):
                    stat_flat += 1
                    continue

                # QC-3: spatial distribution — reject if bbox overlaps any existing person
                if bbox_overlaps_existing(x_feet, y_feet, nw, nh, placed_feet):
                    stat_qc3 += 1
                    continue

                # Load, apply perspective correction (QC-6), resize, paste
                crop_orig = cv2.imread(str(row['name']))
                mask_orig = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if crop_orig is None or mask_orig is None:
                    continue

                # Perspective warp: solo taper horizontal (ancho cabeza vs pies)
                # La altura no se modifica aquí — el resize a (nw, nh) es suficiente
                crop_orig, mask_orig = warp_crop_to_angle(
                    crop_orig, mask_orig, src_pitch, bg_pitch
                )
                if crop_orig is None or crop_orig.shape[0] < 2:
                    continue

                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                crop   = cv2.resize(crop_orig, (nw, nh), interpolation=interp)
                mask_r = cv2.resize(mask_orig, (nw, nh), interpolation=cv2.INTER_NEAREST)

                # QC-7: Anti-pixelación — rechaza parches borrosos tras upscale
                if not is_patch_quality_ok(crop, scale):
                    stat_qc7 += 1
                    continue

                try:
                    paste_crop(result_img, crop, mask_r, x1, y1, nw, nh)
                except Exception:
                    continue

                # YOLO label (bbox centre)
                cx_lbl = x_feet / bg_w
                cy_lbl = (y_feet - nh / 2) / bg_h
                bbox   = f'0 {cx_lbl:.6f} {cy_lbl:.6f} {nw/bg_w:.6f} {nh/bg_h:.6f}'
                final_labels.append(bbox)
                aug_labels.append(bbox)
                partition_heights.append(nh)
                placed_feet.append((x_feet, y_feet, nw, nh))  # update tracker
                n_placed += 1
                placed_in_bin += 1

        # ── Save result ────────────────────────────────────────────────────
        ts       = datetime.now().strftime('%H%M%S%f')
        out_stem = f"{os.path.splitext(bg_name)[0]}_v2_{ts}"
        cv2.imwrite(str(out_img / (out_stem + '.jpg')), result_img)
        with open(str(out_lbl / (out_stem + '.txt')), 'w') as f:
            f.write('\n'.join(final_labels))
        with open(str(out_lbl / (out_stem + '_aug.txt')), 'w') as f:
            f.write('\n'.join(aug_labels))

    # ── Partition summary ─────────────────────────────────────────────────
    sem_mode = 'SegFormer' if seg_model is not None else 'depth-only fallback'
    print(f'\n[{partition}] Placed: {n_placed} | Images skipped: {n_skipped_img}')
    print(f'  Placement mask: {sem_mode}')
    print(f'  Scale method  — kNN-depth: {stat_knn} | power-law: {stat_model} | fallback: {stat_fb}')
    print(f'  Rejected      — QC-1: {stat_qc1} | QC-3: {stat_qc3} | '
          f'QC-4: {stat_flat} | QC-6b: {stat_qc6b} | QC-7: {stat_qc7}')
    _print_distribution(partition, partition_orig_heights, partition_heights)
    _save_distribution(ROOT_OUTPUT_AUG / partition, partition,
                       partition_orig_heights, partition_heights)


def _print_distribution(partition, orig_heights, inserted_heights):
    """
    Muestra distribución original, insertada y combinada por bin.
    Permite verificar si el aumento de datos está corrigiendo la cola larga.
    """
    n_orig = len(orig_heights)
    n_ins  = len(inserted_heights)
    if n_ins == 0:
        print(f'\n[{partition}] No persons inserted.'); return

    print(f'\n{"─"*65}')
    print(f'  Distribución de tamaños — {partition}')
    print(f'  Original: {n_orig} | Insertadas: {n_ins} | Total: {n_orig + n_ins}')
    print(f'  {"Bin":>10}  {"Orig":>6} {"Orig%":>6}  {"Ins":>6} {"Ins%":>6}  {"Comb":>6} {"Comb%":>6}')
    print(f'  {"─"*60}')
    for lo, hi in SIZE_BINS_PX:
        c_o = sum(1 for h in orig_heights     if lo <= h < hi)
        c_i = sum(1 for h in inserted_heights if lo <= h < hi)
        c_c = c_o + c_i
        p_o = 100 * c_o / max(n_orig, 1)
        p_i = 100 * c_i / max(n_ins,  1)
        p_c = 100 * c_c / max(n_orig + n_ins, 1)
        bar = '█' * int(p_c / 3)
        print(f'  [{lo:3d}–{hi:3d}px]  {c_o:6d} {p_o:5.1f}%  {c_i:6d} {p_i:5.1f}%  {c_c:6d} {p_c:5.1f}%  {bar}')
    print(f'{"─"*65}')


def _save_distribution(out_dir, partition, orig_heights, inserted_heights):
    if not inserted_heights:
        return
    rows = (
        [{'height_px': h, 'source': 'original'} for h in orig_heights] +
        [{'height_px': h, 'source': 'inserted'} for h in inserted_heights]
    )
    pd.DataFrame(rows).to_csv(
        str(out_dir / f'size_distribution_{partition}.csv'), index=False
    )


if __name__ == '__main__':
    print('=' * 60)
    print('  Data Augmentation — Long-Tail Balanced Person Insertion')
    print(f'  Bins: {SIZE_BINS_PX}  |  Budget/imagen: {NUM_PEOPLE_PER_IMAGE} (repartido por prioridad inversa)')
    print(f'  Partitions: {PARTITIONS}')
    print(f'  Semantic mask: {"SegFormer-B2 (ADE20K)" if SEGFORMER_AVAILABLE else "depth-only fallback (pip install transformers)"}')
    print(f'  Spatial grid: {SPATIAL_GRID}  |  Max attempts/unit: {MAX_ATTEMPTS}')
    print(f'  QC-1 max_upscale={MAX_UPSCALE}x  |  QC-3 max_iou={MAX_IOU_OVERLAP}  |  QC-5 max_repeats={MAX_CROP_REPEATS}  |  QC-7 min_ssim={MIN_SSIM}')
    print('=' * 60)
    t0 = time.perf_counter()
    for p in PARTITIONS:
        augment_partition(p)
    elapsed = time.perf_counter() - t0
    print(f'\nTotal: {int(elapsed//3600):02d}h {int(elapsed%3600//60):02d}m {elapsed%60:05.2f}s')
