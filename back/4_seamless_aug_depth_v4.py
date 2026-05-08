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
# CONFIGURACIÓN DE EFICIENCIA
# =============================================================================

NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 15)
PITCH_TOLERANCE = 15.0          # Tolerancia estricta para perspectiva coherente
MIN_SCALE = 0.1
MAX_SCALE = 3.0
MIN_CROP_SIZE = 10
BORDER_MARGIN = 20
BLEND_FLAG = cv2.NORMAL_CLONE

# Rutas
ROOT_DATA       = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)
ROOT_META_CSV   = Path(config.ROOT_VGGT_METADATA)
DEPTH_MAPS_SUBDIR = 'depth_maps'

PARTITIONS = config.PARTITIONS

# =============================================================================
# FUNCIONES DE PRE-COMPUTACIÓN Y ESCALADO (ESTILO V3 MEJORADO)
# =============================================================================

def load_depth_map(bg_image_name: str, images_dir: Path, bg_h: int, bg_w: int) -> np.ndarray:
    """Carga y redimensiona el mapa de profundidad."""
    stem = os.path.splitext(bg_image_name)[0]
    for d_name in [f"depth_{bg_image_name}", f"depth_{stem}.jpg", f"depth_{stem}.png"]:
        path = images_dir / DEPTH_MAPS_SUBDIR / d_name
        if path.exists():
            dm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if dm is not None:
                dm = cv2.resize(dm, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
                return dm.astype(np.float32) / 255.0
    # Fallback plano si no existe
    return np.full((bg_h, bg_w), 0.5, dtype=np.float32)

def create_valid_area_mask(bg_img, d_map):
    """
    PRE-COMPUTACIÓN EFICIENTE:
    Crea una máscara binaria (0 = válido, 255 = inválido) para evitar evaluar 
    miles de puntos en un bucle. Identifica cielo y árboles de una sola pasada.
    """
    bg_h, bg_w = bg_img.shape[:2]
    invalid_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # 1. Cielo: Profundidad muy alta
    invalid_mask[d_map > 0.98] = 255
    
    # 2. Árboles y Vegetación (Evita "copas de árboles"):
    # Detectamos color verde
    hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Detectamos rugosidad/bordes en el mapa de profundidad (los árboles son rugosos)
    depth_8u = (d_map * 255).astype(np.uint8)
    edges = cv2.Canny(depth_8u, 10, 50)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    
    # Árbol = Verde + Bordes/Rugosidad en profundidad
    tree_mask = cv2.bitwise_and(green_mask, edges)
    # Expandimos un poco la máscara del árbol para cubrir la copa completa
    tree_mask = cv2.dilate(tree_mask, np.ones((15,15), np.uint8), iterations=2)
    
    invalid_mask = cv2.bitwise_or(invalid_mask, tree_mask)
    
    # 3. Márgenes: Invalidar bordes para que los recortes no se salgan
    invalid_mask[0:BORDER_MARGIN, :] = 255
    invalid_mask[-BORDER_MARGIN:, :] = 255
    invalid_mask[:, 0:BORDER_MARGIN] = 255
    invalid_mask[:, -BORDER_MARGIN:] = 255
    
    return invalid_mask

def calculate_scale_v3(row_patch, bg_meta, target_depth_val):
    """
    Escalado robusto y funcional de la V3:
    Combina la relación de alturas de cámara con la relación de profundidades.
    """
    h_orig = float(row_patch.get('height', 50.0))
    h_bg   = float(bg_meta.get('height', 50.0))
    d_orig = float(row_patch.get('depth_avg', 0.5))
    d_target = float(target_depth_val)
    
    # Escala por diferencia de altura de dron
    scale_h = h_orig / (h_bg + 1e-6)
    
    # Escala por diferencia de profundidad en la escena
    # Usamos 1.1 para que el divisor no sea cero cuando d_orig se acerca a 1
    scale_d = (1.1 - d_target) / (1.1 - d_orig + 1e-6)
    
    return np.clip(scale_h * scale_d, MIN_SCALE, MAX_SCALE)

# =============================================================================
# PIPELINE PRINCIPAL ULTRA-EFICIENTE
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

    for bg_path in tqdm(bg_images, desc=f'Efficient Augment {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]

        try:
            bg_meta = df_bg_meta.loc[bg_name]
            if isinstance(bg_meta, pd.DataFrame): bg_meta = bg_meta.iloc[0]
        except: 
            bg_meta = pd.Series({'height': 50.0, 'pitch': -45.0})

        # 1. Cargar Profundidad
        d_map = load_depth_map(bg_name, images_dir, bg_h, bg_w)

        # 2. PRE-COMPUTAR ZONAS VÁLIDAS (O(1) placement)
        invalid_mask = create_valid_area_mask(bg_img, d_map)
        valid_y, valid_x = np.where(invalid_mask == 0)
        
        # Si no hay zonas válidas, saltar imagen
        if len(valid_x) == 0: continue

        # Cargar etiquetas originales
        original_labels = []
        lbl_p = labels_dir / (os.path.splitext(bg_name)[0] + '.txt')
        if lbl_p.exists():
            with open(lbl_p, 'r') as f: original_labels = [l.strip() for l in f if l.strip()]

        result_img, final_labels = bg_img.copy(), original_labels.copy()
        
        # Filtrado de Candidatos por Ángulo
        bg_pitch = float(bg_meta.get('pitch', -45.0))
        df_compatible = df_pool[abs(df_pool['pitch'] - bg_pitch) <= PITCH_TOLERANCE]
        if len(df_compatible) < NUM_PEOPLE_PER_IMAGE: df_compatible = df_pool

        placed = 0
        candidates = df_compatible.sample(n=min(NUM_PEOPLE_PER_IMAGE * 10, len(df_compatible)), replace=True)

        for _, row in candidates.iterrows():
            if placed >= NUM_PEOPLE_PER_IMAGE: break
            crop_orig = cv2.imread(row['name'])
            if crop_orig is None: continue

            # Al ya tener una lista de puntos válidos, elegimos uno al azar de forma instantánea
            # Hacemos unos pocos intentos por si el escalado o los bordes del parche fallan
            for _ in range(10):
                idx = random.randint(0, len(valid_x) - 1)
                cx, cy = int(valid_x[idx]), int(valid_y[idx])
                
                # Calcular escala V3
                d_target_val = d_map[cy, cx]
                scale = calculate_scale_v3(row, bg_meta, d_target_val)
                
                nw, nh = int(row['width_patch'] * scale), int(row['height_patch'] * scale)
                if nw < MIN_CROP_SIZE or nh < MIN_CROP_SIZE: continue

                x1, y1 = cx - nw//2, cy - nh//2
                x2, y2 = x1 + nw, y1 + nh
                
                # Validar que el recorte entre en la imagen
                if x1 < 0 or y1 < 0 or x2 >= bg_w or y2 >= bg_h: continue

                crop = cv2.resize(crop_orig, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
                mask = np.zeros(crop.shape[:2], dtype=np.uint8); mask[2:-2, 2:-2] = 255
                
                try:
                    result_img = cv2.seamlessClone(crop, result_img, mask, (cx, cy), BLEND_FLAG)
                    final_labels.append(f"0 {cx/bg_w:.6f} {cy/bg_h:.6f} {nw/bg_w:.6f} {nh/bg_h:.6f}")
                    placed += 1
                    break
                except: continue

        out_name = f"{os.path.splitext(bg_name)[0]}_v4_fast_{datetime.now().strftime('%H%M%S%f')}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f: f.write('\n'.join(final_labels))

if __name__ == '__main__':
    print("="*60); print("  Seamless Augmentation v4 Eficiente: Pre-computación y Filtro V3"); print("="*60)
    for p in PARTITIONS: augment_partition(p)
