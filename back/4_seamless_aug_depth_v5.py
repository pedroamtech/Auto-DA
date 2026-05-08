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
# CONFIGURACIÓN V5
# =============================================================================

NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 15)
PITCH_TOLERANCE = 25.0
MIN_SCALE = 0.02
MAX_SCALE = 1.5
MIN_CROP_SIZE = 5
BORDER_MARGIN = 20
BLEND_FLAG = cv2.NORMAL_CLONE

ROOT_DATA       = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)
ROOT_META_CSV   = Path(config.ROOT_VGGT_METADATA)
DEPTH_MAPS_SUBDIR = 'depth_maps'

PARTITIONS = config.PARTITIONS

# =============================================================================
# FUNCIONES: ESCALADO MÉTRICO Y DETECCIÓN DE SUELO
# =============================================================================

def safe_float(val, default=1000.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except: return default

def calculate_metric_scale_v5(row_patch, bg_meta, d_target_norm):
    """
    Escalado Métrico Absoluto:
    Convierte el mapa 0-1 a Metros Reales, y aplica la proyección:
    Tamaño = TamañoOriginal * (Z_original / Z_destino) * (FocalDestino / FocalOriginal)
    """
    # Profundidad de la persona original (en metros)
    p_d_min = safe_float(row_patch.get('depth_min', 0.1), 0.1)
    p_d_max = safe_float(row_patch.get('depth_max', 100.0), 100.0)
    p_d_avg = safe_float(row_patch.get('depth_avg', 0.5), 0.5)
    z_orig_m = p_d_min + p_d_avg * (p_d_max - p_d_min)
    
    # Profundidad del punto de destino en el fondo (en metros)
    bg_d_min = safe_float(bg_meta.get('depth_min', 0.1), 0.1)
    bg_d_max = safe_float(bg_meta.get('depth_max', 100.0), 100.0)
    z_target_m = bg_d_min + float(d_target_norm) * (bg_d_max - bg_d_min)
    
    # Focales
    f_orig = safe_float(row_patch.get('focal_y', 1000.0))
    f_bg   = safe_float(bg_meta.get('focal_y', 1000.0))
    
    scale = (z_orig_m / (z_target_m + 1e-6)) * (f_bg / f_orig)
    return np.clip(scale, MIN_SCALE, MAX_SCALE)

def get_road_color_stats(bg_img, walkable_points):
    """Extrae la media y desviación estándar del color HSV del suelo caminable."""
    hsv_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    h, w = bg_img.shape[:2]
    pixels = []
    
    if walkable_points:
        for x, y in walkable_points:
            x, y = int(x), int(y)
            y1, y2 = max(0, y-5), min(h, y+5)
            x1, x2 = max(0, x-5), min(w, x+5)
            patch = hsv_img[y1:y2, x1:x2].reshape(-1, 3)
            pixels.append(patch)
    
    if not pixels:
        # Fallback: Asumir que la parte inferior central es carretera
        y1, y2 = int(h*0.8), int(h*0.95)
        x1, x2 = int(w*0.3), int(w*0.7)
        patch = hsv_img[y1:y2, x1:x2].reshape(-1, 3)
        pixels.append(patch)
        
    pixels = np.vstack(pixels)
    mean_hsv = np.mean(pixels, axis=0)
    std_hsv = np.std(pixels, axis=0)
    
    # Asegurar una tolerancia mínima (el asfalto puede variar con la luz)
    std_hsv = np.maximum(std_hsv, [10, 25, 25])
    return mean_hsv, std_hsv

def create_semantic_ground_mask(bg_img, d_map, walkable_points):
    """
    Crea una máscara precisa identificando el color del suelo y filtrando
    árboles (color verde), cielo y edificios (bordes/discontinuidades).
    """
    hsv_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    bg_h, bg_w = bg_img.shape[:2]
    
    # 1. MÁSCARA DE COLOR DE SUELO (Carretera/Asfalto)
    mean_hsv, std_hsv = get_road_color_stats(bg_img, walkable_points)
    
    lower_bound = np.clip(mean_hsv - std_hsv * 2.5, 0, 255).astype(np.uint8)
    upper_bound = np.clip(mean_hsv + std_hsv * 2.5, 0, 255).astype(np.uint8)
    
    # Si la saturación es muy baja (gris/asfalto), relajamos el filtro de matiz (Hue)
    if std_hsv[1] < 40 or mean_hsv[1] < 40:
        lower_bound[0] = 0
        upper_bound[0] = 179
        
    valid_color_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    
    # 2. MÁSCARAS INVÁLIDAS (Cielo, Árboles, Edificios)
    invalid_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # A. Cielo: Profundidad muy alta
    invalid_mask[d_map > 0.98] = 255
    
    # B. Árboles y Vegetación: Filtro HSV para verde/amarillo verdoso
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    green_mask = cv2.dilate(green_mask, np.ones((9,9), np.uint8), iterations=2)
    invalid_mask = cv2.bitwise_or(invalid_mask, green_mask)
    
    # C. Edificios y Coches: Bordes afilados en el mapa de profundidad
    depth_8u = (d_map * 255).astype(np.uint8)
    edges = cv2.Canny(depth_8u, 15, 60)
    # Dilatamos los bordes para no pegar personas justo en paredes o coches
    edges = cv2.dilate(edges, np.ones((11,11), np.uint8), iterations=2)
    invalid_mask = cv2.bitwise_or(invalid_mask, edges)
    
    # 3. COMBINAR: Debe ser color de suelo Y NO estar en zona inválida
    final_valid_mask = cv2.bitwise_and(valid_color_mask, cv2.bitwise_not(invalid_mask))
    
    # 4. Eliminar Márgenes
    final_valid_mask[0:BORDER_MARGIN, :] = 0
    final_valid_mask[-BORDER_MARGIN:, :] = 0
    final_valid_mask[:, 0:BORDER_MARGIN] = 0
    final_valid_mask[:, -BORDER_MARGIN:] = 0
    
    return final_valid_mask

# =============================================================================
# PIPELINE PRINCIPAL V5
# =============================================================================

def augment_partition(partition: str):
    images_dir, labels_dir = ROOT_DATA / partition / 'images', ROOT_DATA / partition / 'labels'
    pool_csv_p = ROOT_POOL_CSV / partition / 'pool.csv'
    out_img_dir, out_lbl_dir = ROOT_OUTPUT_AUG / partition / 'images', ROOT_OUTPUT_AUG / partition / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True); out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not pool_csv_p.exists(): return
    df_pool = pd.read_csv(str(pool_csv_p))
    df_bg_meta = pd.read_csv(ROOT_META_CSV).set_index('image_name')

    bg_images = [p for p in glob(str(images_dir / '*.jpg')) if not os.path.basename(p).startswith('depth_')]

    for bg_path in tqdm(bg_images, desc=f'V5 Augment {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]

        try:
            bg_meta = df_bg_meta.loc[bg_name]
            if isinstance(bg_meta, pd.DataFrame): bg_meta = bg_meta.iloc[0]
        except: 
            bg_meta = pd.Series({'pitch': -45.0, 'depth_min': 0.1, 'depth_max': 100.0, 'focal_y': 1000.0})

        # Cargar Profundidad
        d_map = None
        for d_name in [f"depth_{bg_name}", f"depth_{os.path.splitext(bg_name)[0]}.jpg", f"depth_{os.path.splitext(bg_name)[0]}.png"]:
            path = images_dir / DEPTH_MAPS_SUBDIR / d_name
            if path.exists():
                d_map = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if d_map is not None:
                    d_map = cv2.resize(d_map, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    break
        if d_map is None: d_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)

        # Cargar etiquetas originales (para entrenamiento dinámico de color)
        original_labels = []
        lbl_p = labels_dir / (os.path.splitext(bg_name)[0] + '.txt')
        walkable_points = []
        if lbl_p.exists():
            with open(lbl_p, 'r') as f: 
                original_labels = [l.strip() for l in f if l.strip()]
                for l in original_labels:
                    p = l.split()
                    if p[0] == '0': walkable_points.append((float(p[1])*bg_w, (float(p[2])+float(p[4])/2)*bg_h))

        # PRE-COMPUTAR ZONAS VÁLIDAS (Segmentación Semántica)
        valid_mask = create_semantic_ground_mask(bg_img, d_map, walkable_points)
        valid_y, valid_x = np.where(valid_mask == 255)
        
        # Si la máscara es muy estricta y encuentra poco suelo, saltamos para evitar errores
        if len(valid_x) < 100: continue

        result_img, final_labels = bg_img.copy(), original_labels.copy()
        
        # Filtrado de Candidatos por Ángulo
        bg_pitch = safe_float(bg_meta.get('pitch', -45.0), -45.0)
        df_compatible = df_pool[abs(df_pool['pitch'] - bg_pitch) <= PITCH_TOLERANCE]
        if len(df_compatible) < NUM_PEOPLE_PER_IMAGE: df_compatible = df_pool

        placed = 0
        candidates = df_compatible.sample(n=min(NUM_PEOPLE_PER_IMAGE * 10, len(df_compatible)), replace=True)

        for _, row in candidates.iterrows():
            if placed >= NUM_PEOPLE_PER_IMAGE: break
            crop_orig = cv2.imread(row['name'])
            if crop_orig is None: continue

            # Pegado Instantáneo O(1)
            for _ in range(5):
                idx = random.randint(0, len(valid_x) - 1)
                cx, cy = int(valid_x[idx]), int(valid_y[idx])
                
                # ESCALADO MÉTRICO ABSOLUTO
                scale = calculate_metric_scale_v5(row, bg_meta, d_map[cy, cx])
                
                nw, nh = int(row['width_patch'] * scale), int(row['height_patch'] * scale)
                if nw < MIN_CROP_SIZE or nh < MIN_CROP_SIZE: continue

                x1, y1 = cx - nw//2, cy - nh//2
                if x1 < 0 or y1 < 0 or x1+nw >= bg_w or y1+nh >= bg_h: continue

                crop = cv2.resize(crop_orig, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
                mask = np.zeros(crop.shape[:2], dtype=np.uint8); mask[2:-2, 2:-2] = 255
                
                try:
                    result_img = cv2.seamlessClone(crop, result_img, mask, (cx, cy), BLEND_FLAG)
                    final_labels.append(f"0 {cx/bg_w:.6f} {cy/bg_h:.6f} {nw/bg_w:.6f} {nh/bg_h:.6f}")
                    placed += 1
                    break
                except: continue

        out_name = f"{os.path.splitext(bg_name)[0]}_v5_{datetime.now().strftime('%H%M%S%f')}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f: f.write('\n'.join(final_labels))

if __name__ == '__main__':
    print("="*60); print("  Seamless Augmentation V5: Metric Scale & Semantic Color Ground"); print("="*60)
    for p in PARTITIONS: augment_partition(p)
