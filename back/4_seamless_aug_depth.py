# =============================================================================
# Augmentación de datos mediante Seamless Cloning con restricción de profundidad
# =============================================================================
#
# DESCRIPCIÓN:
#   Para cada imagen de fondo del dataset:
#     1. Lee su depth map (escala de grises: 0=cerca, 255=lejos).
#     2. Selecciona N recortes de personas del pool cuya altura relativa
#        (metadato `height`) sea compatible con la profundidad de la zona
#        de pegado → personas lejanas se pegan en zonas lejanas y viceversa.
#     3. Busca aleatoriamente posiciones de pegado válidas en la imagen de
#        fondo que tengan profundidad similar a la del recorte fuente.
#     4. Aplica cv2.seamlessClone (NORMAL_CLONE) para una fusión natural.
#     5. Genera anotaciones YOLO (.txt) para las personas pegadas +
#        las originales que se conservan.
#     6. Guarda la imagen aumentada y su .txt en ROOT_OUTPUT_AUG.
#
# DEPENDENCIAS:
#   pip install opencv-python numpy pandas tqdm
#
# USO:
#   python seamless_aug_depth.py
#
# CONFIGURACIÓN → editar people_pool/config.py
# =============================================================================

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

# Tamaño mínimo del recorte (px) para intentar pegarlo
MIN_CROP_SIZE = 20

# Tolerancia de profundidad normalizada [0-1]:
# Cuanto mayor, más zona de la imagen puede usarse para pegar.
DEPTH_TOLERANCE = 0.20

# Número máximo de intentos para encontrar posición válida por persona
MAX_PLACEMENT_ATTEMPTS = 50

# Añadir un margen (px) para que el recorte no quede cortado en los bordes
BORDER_MARGIN = 5

# Tipo de blending OpenCV
BLEND_FLAG = cv2.NORMAL_CLONE    # cv2.NORMAL_CLONE | cv2.MIXED_CLONE

# Particiones a procesar (de config)
PARTITIONS = config.PARTITIONS

# Rutas
ROOT_DATA       = Path(config.ROOT_DATA1)          # raíz del dataset
ROOT_OUTPUT_AUG = Path(config.ROOT_OUTPUT_AUG)     # salida augmentada
ROOT_POOL_CSV   = Path(config.ROOT_POOL_PERSON)    # pool de personas
ROOT_META_CSV   = Path(config.ROOT_VGGT_METADATA)  # CSV de cámara del dataset

# Directorio de depth maps (relativo al directorio de imágenes del dataset)
DEPTH_MAPS_SUBDIR = 'depth_maps'   # …/train/images/depth_maps/depth_<name>.jpg


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_depth_map(bg_image_name: str, images_dir: Path) -> np.ndarray | None:
    """
    Carga el depth map asociado a una imagen de fondo.
    El depth map se busca en images_dir/depth_maps/depth_<bg_image_name>
    Devuelve un array float32 normalizado [0,1] o None si no existe.
    """
    depth_name = 'depth_' + bg_image_name
    depth_path = images_dir / DEPTH_MAPS_SUBDIR / depth_name
    if not depth_path.exists():
        return None
    dm = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if dm is None:
        return None
    return dm.astype(np.float32) / 255.0   # normalizar a [0, 1]


def person_target_depth(row: pd.Series) -> float:
    """
    Calcula la profundidad objetivo de un recorte de persona usando el
    metadato `height` (altura relativa de la cámara) del CSV del pool.
    Valor normalizado [0,1]: 0=muy cerca, 1=muy lejos.
    """
    h = float(row.get('height', 0.5))
    # height en el CSV es la altura normalizada de la cámara.
    # A mayor altura relativa → persona más lejana → profundidad alta.
    return float(np.clip(h, 0.0, 1.0))


def find_valid_position(
    depth_map: np.ndarray,
    crop_h: int,
    crop_w: int,
    target_depth: float,
    tolerance: float = DEPTH_TOLERANCE,
    margin: int = BORDER_MARGIN,
    max_tries: int = MAX_PLACEMENT_ATTEMPTS,
) -> tuple[int, int] | None:
    """
    Busca aleatoriamente una posición (cx, cy) en la imagen de fondo donde
    la profundidad promedio de la región sea similar a target_depth ± tolerance.

    Devuelve (cx, cy) centro para seamlessClone, o None si no encuentra.
    """
    bg_h, bg_w = depth_map.shape

    # Límites de posición para que el recorte quede dentro de la imagen
    x_min = crop_w  // 2 + margin
    x_max = bg_w - crop_w // 2 - margin
    y_min = crop_h  // 2 + margin
    y_max = bg_h - crop_h // 2 - margin

    if x_min >= x_max or y_min >= y_max:
        return None   # recorte demasiado grande para la imagen

    for _ in range(max_tries):
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)

        # Región de la profundidad bajo el recorte
        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        region_depth = depth_map[y1:y2, x1:x2].mean()

        if abs(region_depth - target_depth) <= tolerance:
            return (cx, cy)

    return None   # no encontró posición compatible


def build_full_mask(crop: np.ndarray) -> np.ndarray:
    """
    Crea una máscara blanca completa del tamaño del recorte.
    seamlessClone necesita que la máscara sea 8UC1, no vacía y
    con al menos 1px de borde negro → dejamos 1px de margen.
    """
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[1:-1, 1:-1] = 255
    return mask


def yolo_bbox(cx: int, cy: int, w: int, h: int, img_w: int, img_h: int) -> str:
    """Convierte coordenadas de píxel a formato YOLO normalizado."""
    xc = cx / img_w
    yc = cy / img_h
    bw = w  / img_w
    bh = h  / img_h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def load_original_yolo_labels(label_path: Path) -> list[str]:
    """Lee las anotaciones YOLO originales del archivo .txt."""
    if not label_path.exists():
        return []
    with open(label_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def augment_partition(partition: str):
    """Procesa una partición completa del dataset."""

    images_dir  = ROOT_DATA / partition / 'images'
    labels_dir  = ROOT_DATA / partition / 'labels'
    pool_csv    = ROOT_POOL_CSV / partition / 'pool.csv'
    out_img_dir = ROOT_OUTPUT_AUG / partition / 'images'
    out_lbl_dir = ROOT_OUTPUT_AUG / partition / 'labels'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Cargar el pool de personas
    # ------------------------------------------------------------------
    if not pool_csv.exists():
        print(f"[ERROR] No se encontró pool.csv en: {pool_csv}")
        print("        Ejecuta primero 0_people_pool_v1.2.py")
        return

    df_pool = pd.read_csv(str(pool_csv))

    # Filtrar recortes con metadatos de cámara (los que tienen `height`)
    df_pool_meta = df_pool.dropna(subset=['height']).reset_index(drop=True)
    df_pool_no_meta = df_pool[df_pool['height'].isna()].reset_index(drop=True)

    print(f"\n[INFO] Partición '{partition}'")
    print(f"       Pool total           : {len(df_pool)} recortes")
    print(f"       Con metadatos cámara : {len(df_pool_meta)}")
    print(f"       Sin metadatos        : {len(df_pool_no_meta)}")

    # ------------------------------------------------------------------
    # 2. Obtener lista de imágenes de fondo
    # ------------------------------------------------------------------
    bg_images = sorted(glob(str(images_dir / '*.jpg')))
    bg_images = [p for p in bg_images if not os.path.basename(p).startswith('depth_')]

    print(f"       Imágenes de fondo    : {len(bg_images)}")

    if not bg_images:
        print("[ERROR] No se encontraron imágenes de fondo.")
        return

    # ------------------------------------------------------------------
    # 3. Procesar cada imagen de fondo
    # ------------------------------------------------------------------
    stats = {'procesadas': 0, 'personas_pegadas': 0, 'sin_depth': 0}

    for bg_path in tqdm(bg_images, desc=f'Augmentando [{partition}]', ncols=110):
        bg_name = os.path.basename(bg_path)
        bg_stem = os.path.splitext(bg_name)[0]

        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue
        bg_h, bg_w = bg_img.shape[:2]

        # Cargar depth map (puede ser None si no existe)
        depth_map = load_depth_map(bg_name, images_dir)
        has_depth = depth_map is not None

        if not has_depth:
            stats['sin_depth'] += 1
            # Si no hay depth map, usar mapa uniforme (profundidad media)
            depth_map = np.full((bg_h, bg_w), 0.5, dtype=np.float32)

        # Resultado acumulativo (se pegan varias personas sobre la misma imagen)
        result_img = bg_img.copy()

        # Anotaciones originales
        orig_labels = load_original_yolo_labels(labels_dir / (bg_stem + '.txt'))
        new_labels  = list(orig_labels)   # copiar las originales

        # Seleccionar candidatos del pool
        # Prioridad: recortes con metadatos → luego sin metadatos como fallback
        if len(df_pool_meta) >= NUM_PEOPLE_PER_IMAGE:
            candidates = df_pool_meta.sample(
                n=NUM_PEOPLE_PER_IMAGE * 3,   # muestrear más para compensar rechazos
                replace=len(df_pool_meta) < NUM_PEOPLE_PER_IMAGE * 3
            )
        else:
            candidates = pd.concat([df_pool_meta, df_pool_no_meta]).sample(
                n=min(NUM_PEOPLE_PER_IMAGE * 3, len(df_pool)),
                replace=True
            )

        placed = 0

        for _, person_row in candidates.iterrows():
            if placed >= NUM_PEOPLE_PER_IMAGE:
                break

            crop_path = person_row['name']
            if not os.path.exists(crop_path):
                continue

            crop = cv2.imread(crop_path)
            if crop is None:
                continue

            crop_h, crop_w = crop.shape[:2]

            # Ignorar recortes demasiado pequeños
            if crop_h < MIN_CROP_SIZE or crop_w < MIN_CROP_SIZE:
                continue

            # También ignorar recortes más grandes que la imagen de fondo
            if crop_h >= bg_h - 2 * BORDER_MARGIN or crop_w >= bg_w - 2 * BORDER_MARGIN:
                continue

            # Profundidad objetivo del recorte
            target_depth = person_target_depth(person_row)

            # Buscar posición en la imagen de fondo con profundidad compatible
            position = find_valid_position(
                depth_map, crop_h, crop_w, target_depth,
                tolerance=DEPTH_TOLERANCE,
                margin=BORDER_MARGIN,
            )

            if position is None:
                # Fallback: posición completamente aleatoria
                cx_min = crop_w  // 2 + BORDER_MARGIN
                cx_max = bg_w - crop_w  // 2 - BORDER_MARGIN
                cy_min = crop_h  // 2 + BORDER_MARGIN
                cy_max = bg_h - crop_h  // 2 - BORDER_MARGIN
                if cx_min >= cx_max or cy_min >= cy_max:
                    continue
                position = (random.randint(cx_min, cx_max),
                            random.randint(cy_min, cy_max))

            cx, cy = position

            # Crear máscara para seamlessClone
            mask = build_full_mask(crop)

            # Aplicar Seamless Cloning
            try:
                result_img = cv2.seamlessClone(
                    crop, result_img, mask,
                    (cx, cy),
                    BLEND_FLAG
                )
            except cv2.error as e:
                # seamlessClone puede fallar con recortes muy pequeños o en el borde
                continue

            # Registrar anotación YOLO de la persona pegada
            new_labels.append(yolo_bbox(cx, cy, crop_w, crop_h, bg_w, bg_h))
            placed += 1

        # Guardar imagen aumentada
        ts = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        out_name = f"{bg_stem}_aug_{ts}"
        cv2.imwrite(str(out_img_dir / (out_name + '.jpg')), result_img)

        # Guardar anotaciones YOLO
        with open(str(out_lbl_dir / (out_name + '.txt')), 'w') as f:
            f.write('\n'.join(new_labels))

        stats['procesadas']      += 1
        stats['personas_pegadas'] += placed

    # ------------------------------------------------------------------
    # 4. Resumen
    # ------------------------------------------------------------------
    print(f"\n[RESULTADO partición '{partition}']")
    print(f"  Imágenes augmentadas  : {stats['procesadas']}")
    print(f"  Personas pegadas total: {stats['personas_pegadas']}")
    print(f"  Imgs sin depth map    : {stats['sin_depth']} (se usó profundidad media)")
    print(f"  Salida imágenes       : {out_img_dir}")
    print(f"  Salida etiquetas      : {out_lbl_dir}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  Augmentación Seamless Cloning con Restricción de Profundidad")
    print("=" * 65)
    print(f"  Personas por imagen  : {NUM_PEOPLE_PER_IMAGE}")
    print(f"  Tolerancia profundidad: ±{DEPTH_TOLERANCE:.2f}")
    print(f"  Tipo de blending     : {'NORMAL_CLONE' if BLEND_FLAG == cv2.NORMAL_CLONE else 'MIXED_CLONE'}")
    print(f"  Particiones          : {PARTITIONS}")
    print("=" * 65)

    for partition in PARTITIONS:
        augment_partition(partition)

    print("\n[DONE] Augmentación completada.")
