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

# Número máximo de personas a pegar por imagen
NUM_PEOPLE_PER_IMAGE = getattr(config, 'NUM_PEOPLE_X_IMG', 10)

# Tolerancia de ángulo (pitch) en grados para considerar un parche compatible
PITCH_TOLERANCE = 15.0

# Límites de escalado para evitar deformaciones extremas
MIN_SCALE = 0.2
MAX_SCALE = 3.0

# Tamaño mínimo del recorte escalado (px)
MIN_CROP_SIZE = 15

# Número máximo de intentos para encontrar posición válida
MAX_PLACEMENT_ATTEMPTS = 30

# Margen de seguridad para bordes
BORDER_MARGIN = 10

# Tipo de blending OpenCV
BLEND_FLAG = cv2.NORMAL_CLONE

# Particiones
PARTITIONS = config.PARTITIONS

# Rutas
ROOT_DATA       = Path(config.ROOT_DATA1)
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)
ROOT_META_CSV   = Path(config.ROOT_VGGT_METADATA)

DEPTH_MAPS_SUBDIR = 'depth_maps'

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_depth_map(bg_image_name: str, images_dir: Path) -> np.ndarray | None:
    depth_name = 'depth_' + bg_image_name
    depth_path = images_dir / DEPTH_MAPS_SUBDIR / depth_name
    if not depth_path.exists():
        return None
    dm = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if dm is None:
        return None
    return dm.astype(np.float32) / 255.0  # [0, 1]

def calculate_scale(row_patch, bg_height, target_depth_val):
    """
    Calcula el factor de escala basado en:
    1. Altura de la cámara (metadato 'height')
    2. Profundidad (distancia al punto)
    """
    h_orig = float(row_patch.get('height', 50.0)) # altura cámara origen
    h_bg   = float(bg_height)                     # altura cámara destino
    
    d_orig = float(row_patch.get('depth_avg', 0.5))
    d_target = float(target_depth_val)
    
    # Factor de escala por altura de cámara (si el drone está más alto, la persona es más pequeña)
    # scale_h = h_orig / h_bg
    scale_h = h_orig / (h_bg + 1e-6)
    
    # Factor de escala por profundidad (Z)
    # Asumimos Z ~ 1 / (1.1 - D)
    # h_pixel ~ 1 / Z ~ (1.1 - D)
    # scale_d = (1.1 - D_target) / (1.1 - D_orig)
    scale_d = (1.1 - d_target) / (1.1 - d_orig + 1e-6)
    
    scale = scale_h * scale_d
    return np.clip(scale, MIN_SCALE, MAX_SCALE)

def build_mask(crop: np.ndarray) -> np.ndarray:
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    # Dejar un pequeño borde negro para seamlessClone
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

    if not pool_csv_p.exists():
        print(f"[ERROR] No pool.csv in {pool_csv_p}")
        return

    df_pool = pd.read_csv(str(pool_csv_p))
    
    # Cargar metadatos del dataset de fondo
    df_bg_meta = pd.read_csv(ROOT_META_CSV)
    df_bg_meta['image_name'] = df_bg_meta['image_name'].apply(os.path.basename)
    df_bg_meta = df_bg_meta.set_index('image_name')

    bg_images = sorted(glob(str(images_dir / '*.jpg')))
    bg_images = [p for p in bg_images if not os.path.basename(p).startswith('depth_')]

    print(f"\n[INFO] Procesando {len(bg_images)} imágenes de fondo...")

    for bg_path in tqdm(bg_images, desc=f'Augmenting {partition}', ncols=100):
        bg_name = os.path.basename(bg_path)
        bg_stem = os.path.splitext(bg_name)[0]
        
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]

        # Obtener metadatos de la imagen de fondo
        bg_meta = df_bg_meta.loc[bg_name] if bg_name in df_bg_meta.index else None
        bg_height_cam = bg_meta['height'] if bg_meta is not None else 50.0
        bg_pitch = bg_meta['pitch'] if bg_meta is not None else -45.0

        depth_map = load_depth_map(bg_name, images_dir)
        if depth_map is None:
            depth_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)
        else:
            # Asegurar que el mapa de profundidad coincida con el tamaño de la imagen de fondo
            if depth_map.shape[0] != bg_h or depth_map.shape[1] != bg_w:
                depth_map = cv2.resize(depth_map, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

        result_img = bg_img.copy()
        
        # Cargar etiquetas originales
        label_path = labels_dir / (bg_stem + '.txt')
        new_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                new_labels = [line.strip() for line in f if line.strip()]

        # Filtrar pool por ángulo (pitch)
        # Solo parches con ángulo similar
        df_compatible = df_pool[
            (df_pool['pitch'] >= bg_pitch - PITCH_TOLERANCE) & 
            (df_pool['pitch'] <= bg_pitch + PITCH_TOLERANCE)
        ]
        
        if len(df_compatible) < NUM_PEOPLE_PER_IMAGE:
            # Fallback: relajar tolerancia si hay pocos candidatos
            df_compatible = df_pool.copy()

        placed = 0
        # Muestrear candidatos
        candidates = df_compatible.sample(n=min(NUM_PEOPLE_PER_IMAGE * 5, len(df_compatible)), replace=True)

        for _, person_row in candidates.iterrows():
            if placed >= NUM_PEOPLE_PER_IMAGE: break

            crop_path = person_row['name']
            if not os.path.exists(crop_path): continue
            
            crop_orig = cv2.imread(crop_path)
            if crop_orig is None: continue

            # Intentar colocar en posición aleatoria con profundidad compatible
            success = False
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                # Elegir punto aleatorio en la imagen
                cx = random.randint(BORDER_MARGIN, bg_w - BORDER_MARGIN)
                cy = random.randint(BORDER_MARGIN, bg_h - BORDER_MARGIN)
                
                target_depth_val = depth_map[cy, cx]
                
                # Calcular escala basada en la profundidad de ese punto
                scale = calculate_scale(person_row, bg_height_cam, target_depth_val)
                
                new_w = int(person_row['width_patch'] * scale)
                new_h = int(person_row['height_patch'] * scale)
                
                if new_w < MIN_CROP_SIZE or new_h < MIN_CROP_SIZE: continue
                
                # Verificar si cabe
                x1, y1 = cx - new_w // 2, cy - new_h // 2
                x2, y2 = x1 + new_w, y1 + new_h
                
                if x1 < 0 or y1 < 0 or x2 >= bg_w or y2 >= bg_h: continue
                
                # Redimensionar parche
                crop_scaled = cv2.resize(crop_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask = build_mask(crop_scaled)
                
                try:
                    result_img = cv2.seamlessClone(crop_scaled, result_img, mask, (cx, cy), BLEND_FLAG)
                    new_labels.append(yolo_bbox(cx, cy, new_w, new_h, bg_w, bg_h))
                    placed += 1
                    success = True
                    break
                except cv2.error:
                    continue
            
        # Guardar resultados
        ts = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        out_name = f"{bg_stem}_v2_{ts}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f:
            f.write('\n'.join(new_labels))

    print(f"\n[DONE] Partition {partition} finished.")

if __name__ == '__main__':
    print("="*60)
    print("  Seamless Augmentation v2: Depth & Perspective Aware")
    print("="*60)
    for p in PARTITIONS:
        augment_partition(p)
