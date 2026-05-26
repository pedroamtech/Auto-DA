# Auto-DA

Pipeline de **Data Augmentation** para datasets de detección de personas desde UAV/dron, combinando estimación de cámara con IA (VGGT), segmentación de siluetas (YOLOv8x + SAM2-L) e inserción métrica basada en mapas de profundidad.

## Descripción

El proyecto extrae parámetros de cámara de imágenes reales usando el modelo **VGGT** (Facebook Research), construye un pool de recortes de personas y los inserta en nuevas imágenes de fondo con escalado métrico basado en profundidad y alpha blending. Las rutas se configuran en `config.py` sin necesidad de diálogos interactivos.

## Pipeline

```
1_extract_information.py   →   2_view_cluster.py
                                      ↓
                               3_people_pool.py
                                      ↓
                               4_extract_masks.py
                                      ↓
                            5_data_augmentation.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros de cámara y genera depth maps con VGGT. Lee rutas desde `config.py`. Guarda `depth_maps/` y `camera_data.csv` al mismo nivel que `images/`. Muestra info de GPU/CUDA al inicio. |
| `2_view_cluster.py` | Visualiza en 3D los grupos de cámaras (clustering KMeans). Pide el número de clusters por diálogo. |
| `3_people_pool.py` | Recorta personas del dataset (YOLO labels), filtra por rango de altura y genera `pool.csv` con metadatos de cámara integrados. |
| `4_extract_masks.py` | Pre-segmentación offline: pipeline **YOLOv8x → SAM2-L** para extraer siluetas de alta precisión. Muestra info de GPU/CUDA al inicio. |
| `5_data_augmentation.py` | Inserción de personas con escalado métrico por profundidad y alpha blending. Genera un archivo `_aug.txt` por imagen con solo las bboxes aumentadas. |

> Todos los scripts del pipeline imprimen el tiempo total de ejecución al finalizar (`00h 00m 00.00s`).

### Scripts auxiliares — `tools/`

| Script | Función |
|---|---|
| `tools/convert_okutama_to_yolo.py` | Convierte anotaciones Okutama-Action (tracking format) a archivos YOLO por frame. |
| `tools/convert_manipal_to_yolo.py` | Convierte anotaciones Manipal-UAV (MOT-style o YOLO normalizado) a formato YOLO. Auto-detecta el formato de entrada. |
| `tools/video_to_frames.py` | Extrae frames de un video y los guarda como imágenes. |
| `tools/yolo_person_labeler.py` | Herramienta de etiquetado de personas en formato YOLO. |

## Configuración — `config.py`

Todas las rutas del pipeline se definen aquí. No se requieren diálogos interactivos.

```python
ROOT_DATA1       = 'ruta/al/dataset'        # raíz del dataset (contiene train/, val/, ...)
ROOT_POOL_PERSON = 'ruta/al/pool_person'    # pool de recortes de personas
ROOT_OUTPUT_AUG  = 'ruta/a/salida'          # imágenes y labels aumentados
PARTITIONS       = ['train']                # particiones a procesar
HEIGHT_MIN       = 28   # altura mínima de crop aceptada (px)
HEIGHT_MAX       = 79   # altura máxima de crop aceptada (px)
HEIGHT_AUG_LOW   = 5    # altura mínima de inserción en imagen de salida (px)
NUM_PEOPLE_X_IMG = 30   # personas a insertar por imagen
```

### Estructura esperada del dataset

```
ROOT_DATA1/
└── train/
    ├── images/          ← imágenes originales
    ├── labels/          ← anotaciones YOLO (.txt)
    └── depth_maps/      ← generado por 1_extract_information.py
        └── camera_data.csv
```

## Requisitos

### Entorno

- **Python** 3.13 (recomendado entorno conda)
- **CUDA** 13.2 (recomendado, GPU NVIDIA)

### Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/pedroamtech/Auto-DA.git
cd Auto-DA
```

2. **Crear y activar el entorno conda:**
```bash
conda create --name auto_da python=3.13
conda activate auto_da
```

3. **Instalar dependencias:**
```bash
pip install -r requirements_da.txt
```

### Autenticación Hugging Face

Necesaria para descargar el modelo VGGT-1B en el paso 1.

1. Regístrate en [huggingface.co/join](https://huggingface.co/join).
2. Crea un token **Read** en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Configura la variable de entorno:

```powershell
# Windows (PowerShell)
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'TU_TOKEN_AQUI', 'User')
```

## Estructura del Proyecto

```
Auto-DA/
├── 1_extract_information.py     # Paso 1: Extracción de cámara y depth maps
├── 2_view_cluster.py            # Paso 2: Visualización de clusters
├── 3_people_pool.py             # Paso 3: Creación del pool de personas
├── 4_extract_masks.py           # Paso 4: Pre-segmentación (YOLOv8x + SAM2-L)
├── 5_data_augmentation.py       # Paso 5: Augmentación con escalado métrico
├── config.py                    # Configuración global de rutas y parámetros
├── vggt/                        # Código fuente del modelo VGGT
├── tools/                       # Conversión de datasets y utilidades
├── back/                        # Versiones anteriores de scripts
├── VGGT.pdf                     # Paper de referencia
└── README.md
```

## Salidas del pipeline

Por cada imagen aumentada se generan en `ROOT_OUTPUT_AUG/{partition}/`:

```
images/
└── {nombre}_aug_HHMMSSffffff.jpg            ← imagen con personas insertadas

labels/
├── {nombre}_aug_HHMMSSffffff.txt            ← todas las bboxes (original + aumentadas) en formato YOLO
└── {nombre}_aug_HHMMSSffffff_aug.txt        ← solo las bboxes insertadas por aumentación
```

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y profundidad (Facebook Research).
- **OpenCV** — Procesamiento de imagen y alpha blending.
- **Ultralytics YOLOv8x** — Detección de personas para pre-segmentación.
- **Ultralytics SAM2-L** — Segmentación de alta precisión (Meta AI). Pesos descargados automáticamente (`yolov8x.pt` ~130 MB, `sam2_l.pt` ~428 MB).
- **scikit-learn KMeans** — Agrupación de vistas de cámara.
- **Hugging Face Hub** — Descarga del modelo VGGT-1B.

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---
⭐ Si este proyecto te ha sido útil, ¡dale una estrella!
