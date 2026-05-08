import cv2
import numpy as np
import pandas as pd
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
from glob import glob
from datetime import datetime

# Añadir el directorio people_pool al path para importar config
sys.path.insert(0, str(Path(__file__).parent / 'people_pool'))
import config

# =============================================================================
# PARÁMETROS (ajustables)
# =============================================================================

NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 10)
PITCH_TOLERANCE = 15.0
MIN_SCALE = 0.2
MAX_SCALE = 3.0
MIN_CROP_SIZE = 15
MAX_PLACEMENT_ATTEMPTS = 50
BORDER_MARGIN = 10
BLEND_FLAG = cv2.NORMAL_CLONE

# --- NUEVOS PARÁMETROS V3 ---
SKY_DEPTH_THRESHOLD = 0.97      # Profundidad normalizada por encima de la cual se considera cielo
GROUND_PLANE_TOLERANCE = 0.30   # Tolerancia relativa para el plano del suelo
ROUGHNESS_THRESHOLD = 0.02      # Desviación estándar máxima de profundidad (local) para ser suelo

# Rutas
ROOT_DATA       = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)
ROOT_META_CSV   = Path(config.ROOT_VGGT_METADATA)
DEPTH_MAPS_SUBDIR = 'depth_maps'

# Particiones
PARTITIONS = config.PARTITIONS

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_depth_map(bg_image_name: str, images_dir: Path) -> np.ndarray | None:
    depth_name = 'depth_' + bg_image_name
    depth_path = images_dir / DEPTH_MAPS_SUBDIR / depth_name
    if not depth_path.exists(): return None
    dm = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if dm is None: return None
    return dm.astype(np.float32) / 255.0

def calculate_scale(row_patch, bg_height, target_depth_val):
    h_orig = float(row_patch.get('height', 50.0))
    h_bg   = float(bg_height)
    d_orig = float(row_patch.get('depth_avg', 0.5))
    d_target = float(target_depth_val)
    
    scale_h = h_orig / (h_bg + 1e-6)
    scale_d = (1.1 - d_target) / (1.1 - d_orig + 1e-6)
    return np.clip(scale_h * scale_d, MIN_SCALE, MAX_SCALE)

def is_valid_surface(cx, cy, depth_map, bg_meta, bg_w, bg_h):
    """
    Valida si el punto es suelo usando:
    1. Máscara de Cielo
    2. Consistencia con Plano de Suelo
    3. Filtro de Rugosidad (Varianza local)
    """
    d_val = depth_map[cy, cx]
    
    # 1. ¿Es cielo?
    if d_val > SKY_DEPTH_THRESHOLD:
        return False
        
    # 2. ¿Es rugoso (árboles/follaje)?
    win = 5
    y1, y2 = max(0, cy-win), min(bg_h, cy+win)
    x1, x2 = max(0, cx-win), min(bg_w, cx+win)
    local_patch = depth_map[y1:y2, x1:x2]
    if local_patch.size > 0:
        if np.std(local_patch) > ROUGHNESS_THRESHOLD:
            return False

    # 3. Consistencia con Plano de Suelo (Opcional, requiere focal y height)
    # Si d_actual < d_teorico * 0.7, probablemente sea un objeto elevado (árbol)
    if 'depth_min' in bg_meta and 'depth_max' in bg_meta:
        try:
            f = bg_meta['focal_y']
            py = bg_meta['principal_y']
            h = bg_meta['height']
            pitch = np.radians(bg_meta['pitch'])
            
            # Ángulo del rayo con el eje óptico
            theta = np.arctan((cy - py) / f)
            phi = -(pitch + theta) # Ángulo total con la horizontal
            
            if phi > 0.02:
                z_theory = h / np.sin(phi)
                z_actual = bg_meta['depth_min'] + d_val * (bg_meta['depth_max'] - bg_meta['depth_min'])
                
                # Si está significativamente más cerca de lo que debería estar el suelo -> obstáculo
                if z_actual < z_theory * (1.0 - GROUND_PLANE_TOLERANCE):
                    return False
        except:
            pass

    return True

def build_mask(crop: np.ndarray) -> np.ndarray:
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[2:-2, 2:-2] = 255
    return mask

def yolo_bbox(cx, cy, w, h, img_w, img_h):
    return f"0 {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def augment_partition(partition: str):
    images_dir  = ROOT_DATA / partition / 'images'
    labels_dir  = ROOT_DATA / partition / 'labels'
    pool_csv_p  = ROOT_POOL_CSV / partition / 'pool.csv'
    out_img_dir = ROOT_OUTPUT_AUG / partition / 'images'
    out_lbl_dir = ROOT_OUTPUT_AUG / partition / 'labels'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not pool_csv_p.exists(): return

    df_pool = pd.read_csv(str(pool_csv_p))
    df_bg_meta = pd.read_csv(ROOT_META_CSV)
    df_bg_meta['image_name'] = df_bg_meta['image_name'].apply(os.path.basename)
    df_bg_meta = df_bg_meta.set_index('image_name')

    bg_images = sorted(glob(str(images_dir / '*.jpg')))
    bg_images = [p for p in bg_images if not os.path.basename(p).startswith('depth_')]

    for bg_path in tqdm(bg_images, desc=f'Augmenting {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_stem = os.path.splitext(bg_name)[0]
        
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]

        bg_meta = df_bg_meta.loc[bg_name] if bg_name in df_bg_meta.index else None
        bg_height_cam = bg_meta['height'] if bg_meta is not None else 50.0
        bg_pitch = bg_meta['pitch'] if bg_meta is not None else -45.0

        depth_map = load_depth_map(bg_name, images_dir)
        if depth_map is None:
            depth_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)
        else:
            if depth_map.shape[0] != bg_h or depth_map.shape[1] != bg_w:
                depth_map = cv2.resize(depth_map, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

        result_img = bg_img.copy()
        
        label_path = labels_dir / (bg_stem + '.txt')
        new_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                new_labels = [line.strip() for line in f if line.strip()]

        df_compatible = df_pool[
            (df_pool['pitch'] >= bg_pitch - PITCH_TOLERANCE) & 
            (df_pool['pitch'] <= bg_pitch + PITCH_TOLERANCE)
        ]
        if len(df_compatible) < NUM_PEOPLE_PER_IMAGE: df_compatible = df_pool.copy()

        placed = 0
        candidates = df_compatible.sample(n=min(NUM_PEOPLE_PER_IMAGE * 10, len(df_compatible)), replace=True)

        for _, person_row in candidates.iterrows():
            if placed >= NUM_PEOPLE_PER_IMAGE: break

            crop_path = person_row['name']
            if not os.path.exists(crop_path): continue
            crop_orig = cv2.imread(crop_path)
            if crop_orig is None: continue

            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                cx = random.randint(BORDER_MARGIN, bg_w - BORDER_MARGIN)
                cy = random.randint(BORDER_MARGIN, bg_h - BORDER_MARGIN)
                
                # --- NUEVA VALIDACIÓN V3 ---
                if not is_valid_surface(cx, cy, depth_map, bg_meta, bg_w, bg_h):
                    continue

                target_depth_val = depth_map[cy, cx]
                scale = calculate_scale(person_row, bg_height_cam, target_depth_val)
                new_w, new_h = int(person_row['width_patch'] * scale), int(person_row['height_patch'] * scale)
                
                if new_w < MIN_CROP_SIZE or new_h < MIN_CROP_SIZE: continue
                x1, y1 = cx - new_w // 2, cy - new_h // 2
                x2, y2 = x1 + new_w, y1 + new_h
                if x1 < 0 or y1 < 0 or x2 >= bg_w or y2 >= bg_h: continue
                
                crop_scaled = cv2.resize(crop_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask = build_mask(crop_scaled)
                
                try:
                    result_img = cv2.seamlessClone(crop_scaled, result_img, mask, (cx, cy), BLEND_FLAG)
                    new_labels.append(yolo_bbox(cx, cy, new_w, new_h, bg_w, bg_h))
                    placed += 1
                    break
                except: continue
            
        ts = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        out_name = f"{bg_stem}_v3_{ts}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f:
            f.write('\n'.join(new_labels))

if __name__ == '__main__':
    print("="*60)
    print("  Seamless Augmentation v3: Surface & Ground Aware")
    print("="*60)
    for p in PARTITIONS: augment_partition(p)
