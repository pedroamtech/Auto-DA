# Auto-DA

Pipeline de **Data Augmentation** para datasets de detección de personas desde drones. Combina estimación de cámara con IA (VGGT), segmentación de siluetas (YOLOv8x + SAM2-L) y muestreo estratificado por tamaño para corregir la distribución de cola larga en el tamaño de personas.

## Objetivo

Los datasets de vigilancia aérea (VisDrone, etc.) tienen una distribución de cola larga en el tamaño de personas: abundan las medianas, escasean las muy pequeñas y muy grandes. Este pipeline inserta recortes de personas reales con escala físicamente correcta (ley perspectiva + mapa de profundidad) y los distribuye uniformemente en bins de tamaño para equilibrar esa distribución antes de entrenar modelos YOLO.

## Pipeline

```
1_extract_information.py   →   (2_view_cluster.py)
                                       ↓
                              3_people_pool_v2.py
                                       ↓
                               4_extract_masks.py
                                       ↓
                           5_data_augmentation_v2.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros de cámara (pitch, focal, posición) y genera depth maps **16-bit PNG** con VGGT-1B. |
| `2_view_cluster.py` | *(Opcional)* Visualiza en 3D los grupos de cámaras por posición (KMeans). |
| `3_people_pool_v2.py` | Recorta personas del dataset, aplica filtros de tamaño (`HEIGHT_MIN`/`MAX`) y genera `pool.csv` con flags `complete_body` y `size_bin`. |
| `4_extract_masks.py` | Pipeline **YOLOv8x → SAM2-L**: detección de persona → máscara de alta precisión. Reanudable. Pesos descargados automáticamente. |
| `5_data_augmentation_v2.py` | Augmentación con muestreo estratificado por bins de tamaño, corrección de perspectiva (pitch), verificación de suelo plano y anti-solapamiento. |

### Scripts auxiliares

| Script | Función |
|---|---|
| `tools/video_to_frames.py` | Extrae frames de un video. |
| `tools/yolo_person_labeler.py` | Etiquetado manual/automático de personas en formato YOLO. |

## Requisitos

### Entorno

- **Python** 3.13.9 — entorno conda recomendado
- **CUDA** 13.2 — GPU NVIDIA recomendada
- **PyTorch** compatible con CUDA 13.2

### Instalación

```bash
git clone https://github.com/pedroamtech/Auto-DA.git
cd Auto-DA

conda create --name data_augmentation python=3.13.9
conda activate data_augmentation

pip install -r requirements.txt
```

### Autenticación Hugging Face (recomendado)

Evita límites de velocidad al descargar VGGT-1B:

1. Crea token **Read** en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Configúralo en Windows:
```powershell
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'TU_TOKEN_AQUI', 'User')
```

## Configuración

Edita `config.py` antes de ejecutar:

```python
ROOT_DATA1        = 'ruta/dataset'           # carpeta con train/images/ y train/labels/
ROOT_POOL_PERSON  = 'ruta/pool_person'       # salida del pool de recortes
ROOT_OUTPUT_AUG   = 'ruta/output_augmented'  # salida de las imágenes aumentadas
ROOT_VGGT_METADATA = 'ruta/dataset/train/images/camera_data.csv'  # generado en paso 1

HEIGHT_MIN        = 28     # altura mínima (px) de crops aceptados en el pool
HEIGHT_MAX        = 79     # altura máxima (px)
NUM_PEOPLE_X_IMG  = 30     # personas a insertar por imagen de fondo
PARTITIONS        = ['train']
```

## Uso

### Paso 1 — Extraer parámetros de cámara y depth maps

```bash
python 1_extract_information.py
```

Genera en `train/images/depth_maps/`: depth maps **16-bit PNG** (`depth_*.png`) y `camera_data.csv`.
Descarga VGGT-1B (~3.7 GB) la primera vez.

### Paso 2 — (Opcional) Visualizar clusters de cámara

```bash
python 2_view_cluster.py
```

### Paso 3 — Construir pool de personas

```bash
python 3_people_pool_v2.py
```

Genera `pool_person/train/pool.csv` con columnas `complete_body` y `size_bin`.
Muestra la distribución de recortes por bin al terminar.

### Paso 4 — Extraer máscaras de siluetas

```bash
python 4_extract_masks.py
```

Descarga `yolov8x.pt` (~130 MB) y `sam2_l.pt` (~428 MB) la primera vez.
Genera `pool_person/train/masks/` y `metadata/`. **Reanudable** si se interrumpe.

### Paso 5 — Augmentación estratificada

```bash
python 5_data_augmentation_v2.py
```

Genera imágenes y etiquetas en `ROOT_OUTPUT_AUG/train/`.
Muestra la distribución de alturas insertadas por bin al terminar.

## Estructura del Proyecto

```
Auto-DA/
├── 1_extract_information.py      # Paso 1: cámara + depth maps 16-bit
├── 2_view_cluster.py             # (Opcional) visualización 3D de clusters
├── 3_people_pool_v2.py           # Paso 3: pool con filtros de tamaño
├── 4_extract_masks.py            # Paso 4: YOLO + SAM2 segmentación
├── 5_data_augmentation_v2.py     # Paso 5: augmentación estratificada
├── config.py                     # Rutas y parámetros globales
├── vggt/                         # Código fuente modelo VGGT-1B
├── tools/                        # Herramientas auxiliares
├── back/                         # Versiones anteriores de los scripts
└── requirements_da.txt
```

## Salida esperada

```
ROOT_OUTPUT_AUG/
└── train/
    ├── images/
    │   └── <nombre>_v2_<timestamp>.jpg
    └── labels/
        ├── <nombre>_v2_<timestamp>.txt       # etiquetas completas (orig + aug)
        └── <nombre>_v2_<timestamp>_aug.txt   # solo personas insertadas
```

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y depth maps (Facebook Research)
- **Ultralytics YOLOv8x** — Detección de personas (bounding box)
- **Ultralytics SAM2-L** — Segmentación de siluetas de alta precisión (Meta AI)
- **OpenCV** — Procesamiento de imagen y alpha blending
- **Hugging Face Hub** — Descarga del modelo VGGT-1B

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---
⭐ Si este proyecto te ha sido útil, ¡dale una estrella!
