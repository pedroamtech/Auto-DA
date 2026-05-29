# Auto-DA

Pipeline avanzado de **Data Augmentation** para datasets de detección de personas desde drones. Combina estimación de cámara con VGGT, segmentación semántica con SegFormer, siluetas precisas con YOLOv8x + SAM2-L, y escalado métrico por profundidad para insertar personas de forma realista y equilibrar la distribución de cola larga en tamaños.

## Objetivo

Los datasets de vigilancia aérea (VisDrone, etc.) presentan una distribución de cola larga: abundan personas medianas, escasean las muy pequeñas y muy grandes. Este pipeline inserta recortes reales de personas a escala físicamente correcta derivada de la ley de perspectiva y mapas de profundidad 16-bit, garantizando que los parches se coloquen únicamente en zonas semánticamente válidas (suelo, plaza, acera) y con tamaño consistente con la perspectiva real de la escena.

## Pipeline

```
1_extract_information.py   →   (2_view_cluster.py)
                                       ↓
                               3_people_pool.py
                                       ↓
                               4_extract_masks.py
                                       ↓
                            5_data_augmentation.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros de cámara (pitch, focal, posición) y genera depth maps **16-bit PNG** con VGGT-1B. |
| `2_view_cluster.py` | *(Opcional)* Visualiza en 3D los grupos de cámaras. Auto-detecta el número óptimo de clusters (consenso Silhouette + Calinski-Harabasz + Davies-Bouldin). |
| `3_people_pool.py` | Recorta personas del dataset con filtros de tamaño (`HEIGHT_MIN`/`MAX`) y genera `pool.csv` con flags `complete_body`, `size_bin` y metadatos de cámara integrados. |
| `4_extract_masks.py` | Pipeline **YOLOv8x → SAM2-L**: detección → máscara de alta precisión con métricas de calidad (`solidity`, `smoothness`). Reanudable. |
| `5_data_augmentation.py` | Augmentación realista: máscara semántica **SegFormer-B2** para evitar zonas prohibidas, escalado k-NN por profundidad, distribución espacial balanceada, corrección de perspectiva (taper) y filtro anti-pixelación SSIM. |

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
ROOT_DATA1         = 'ruta/dataset'          # carpeta con train/images/ y train/labels/
ROOT_POOL_PERSON   = 'ruta/pool_person'      # salida del pool de recortes
ROOT_OUTPUT_AUG    = 'ruta/output_augmented' # salida de las imágenes aumentadas
ROOT_VGGT_METADATA = 'ruta/dataset/train/depth_maps/camera_data.csv'  # generado en paso 1

HEIGHT_MIN         = 28    # altura mínima (px) de crops aceptados en el pool
HEIGHT_MAX         = 79    # altura máxima (px)
NUM_PEOPLE_X_IMG   = 30    # personas a insertar por imagen de fondo
PARTITIONS         = ['train']
```

## Uso

### Paso 1 — Extraer parámetros de cámara y depth maps

```bash
python 1_extract_information.py
```

Genera en `train/depth_maps/`: depth maps **16-bit PNG** (`depth_*.png`) y `camera_data.csv`.
Descarga VGGT-1B (~3.7 GB) la primera vez.

### Paso 2 — (Opcional) Visualizar clusters de cámara

```bash
python 2_view_cluster.py
```

Selecciona `camera_data.csv` mediante diálogo. Muestra cones de FOV real, distribuciones de pitch/altura/focal por cluster y tabla de estadísticas. El número de clusters se calcula automáticamente.

### Paso 3 — Construir pool de personas

```bash
python 3_people_pool.py
```

Genera `pool_person/train/pool.csv`. Muestra distribución de recortes por bin de tamaño al terminar.

### Paso 4 — Extraer máscaras de siluetas

```bash
python 4_extract_masks.py
```

Descarga `yolov8x.pt` (~130 MB) y `sam2_l.pt` (~428 MB) la primera vez.
Genera `pool_person/train/masks/` y `metadata/` con métricas de calidad por máscara. **Reanudable** si se interrumpe.

### Paso 5 — Augmentación

```bash
python 5_data_augmentation.py
```

Genera imágenes y etiquetas en `ROOT_OUTPUT_AUG/train/`. Al terminar muestra:
- Distribución de alturas insertadas por bin
- Método de predicción usado (k-NN / power-law / fallback)
- Conteo de rechazos por cada QC

> **Nota:** Para activar la máscara semántica SegFormer instala `transformers>=4.40.0`.
> Si no está disponible el script funciona con detección de suelo por profundidad (fallback automático).

## Estructura del Proyecto

```
Auto-DA/
├── 1_extract_information.py   # Paso 1: cámara + depth maps 16-bit
├── 2_view_cluster.py          # (Opcional) visualización 3D de clusters
├── 3_people_pool.py           # Paso 3: pool con filtros de tamaño y métricas
├── 4_extract_masks.py         # Paso 4: YOLOv8x + SAM2-L + métricas de calidad
├── 5_data_augmentation.py     # Paso 5: augmentación realista con SegFormer
├── config.py                  # Rutas y parámetros globales
├── requirements.txt           # Dependencias del proyecto
├── vggt/                      # Código fuente modelo VGGT-1B
├── tools/                     # Herramientas auxiliares
└── back/                      # Versiones anteriores de los scripts
```

## Estructura de salida esperada

```
ROOT_OUTPUT_AUG/
└── train/
    ├── images/
    │   └── <nombre>_v2_<timestamp>.jpg
    └── labels/
        ├── <nombre>_v2_<timestamp>.txt      # etiquetas completas (orig + aug)
        └── <nombre>_v2_<timestamp>_aug.txt  # solo personas insertadas
```

## Controles de calidad implementados

| QC | Descripción |
|---|---|
| QC-1 | Resolución: rechaza crops que requieren upscale > 1.8× |
| QC-2 | Escala por k-NN de profundidad (personas anotadas como referencia) |
| QC-3 | Anti-solapamiento: IoU < 0.15 con personas existentes |
| QC-4 | Suelo plano: std de profundidad en zona de pies < 0.18 |
| QC-5 | Repetición: mismo crop máximo 2 veces por imagen |
| QC-6 | Corrección de perspectiva: taper ±12% para Δpitch 5°–15° |
| QC-7 | Anti-pixelación: SSIM simulado ≥ 0.55 para crops upscaleados |

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y depth maps 16-bit (Meta AI / Facebook Research)
- **Ultralytics YOLOv8x** — Detección de personas (bounding box)
- **Ultralytics SAM2-L** — Segmentación de siluetas de alta precisión (Meta AI)
- **SegFormer-B2 (ADE20K)** — Segmentación semántica para zonas de colocación válidas (NVIDIA / HuggingFace)
- **OpenCV** — Procesamiento de imagen, alpha blending y métricas de máscara
- **Hugging Face Hub** — Descarga del modelo VGGT-1B
- **scikit-learn** — Clustering de posiciones de cámara (KMeans + métricas de validez)

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---
⭐ Si este proyecto te ha sido útil, ¡dale una estrella!
