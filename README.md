# DA-Seamless-Cloning

Pipeline de **Data Augmentation** para datasets de detección de personas, combinando estimación de cámara con IA (VGGT) y fusión de imágenes por Seamless Cloning (OpenCV).

## Descripción

El proyecto extrae parámetros de cámara de imágenes reales usando el modelo **VGGT** (Facebook Research), agrupa las vistas en clusters, construye un pool de recortes de personas y los inserta en nuevas imágenes de fondo con escalado métrico basado en profundidad. La segmentación de siluetas se realiza offline con un pipeline **YOLOv8x (detección) + SAM2-L (segmentación)** para obtener máscaras de alta precisión.

## Pipeline

```
1_extract_information.py   →   2_view_cluster.py
                                      ↓
                               3_people_pool.py
                                      ↓
                               4_extract_masks.py
                                      ↓
                            5_data_augmentation_ab.py
```

| Script | Función |
|---|---|
| `1_extract_information.py` | Extrae parámetros de cámara y genera depth maps (Visual y RAW) con VGGT. Soporta CLI y HF_TOKEN. |
| `2_view_cluster.py` | Visualiza en 3D los grupos de cámaras (clustering KMeans). |
| `3_people_pool.py` | Recorta personas del dataset (YOLO) y genera `pool.csv` con metadatos de cámara integrados. |
| `4_extract_masks.py` | **Pre-segmentación:** Pipeline **YOLOv8x → SAM2-L** para extraer siluetas de alta precisión de forma offline. Pesos descargados automáticamente en la primera ejecución. |
| `5_data_augmentation_ab.py` | **Principal:** Inserción de personas con escalado métrico por profundidad y alpha blending con feathering. Preserva el color original de cada parche. |

### Scripts auxiliares

| Script | Función |
|---|---|
| `tools/video_to_frames.py` | Extrae frames de un video y los guarda como imágenes. |
| `tools/yolo_person_labeler.py` | Herramienta de etiquetado manual/automático de personas en formato YOLO. |

## Requisitos

### Entorno

- **Python** 3.13.9 (Recomendado entorno virtual conda)
- **CUDA** 13.2 (recomendado, GPU NVIDIA)
- **PyTorch** compatible con CUDA 13.2.

### Instalación en Anaconda

1. **Clonar el repositorio:**
```bash
git clone https://github.com/pedroamtech/DA-Seamless-Cloning.git
cd DA-Seamless-Cloning
```

2. **Crear y activar el entorno conda:**
```bash
conda create --name data_augmentation python=3.13.9
conda activate data_augmentation
```

3. **Instalar requerimientos:**
El archivo `requirements_da.txt` está configurado para descargar la versión correcta de PyTorch con soporte para CUDA 13.2 y todas las dependencias del proyecto (`torch`, `numpy`, `opencv-python`, `ultralytics`, etc.):
```bash
pip install -r requirements_da.txt
```

### Autenticación Hugging Face (Recomendado)
Para evitar límites de velocidad y avisos, sigue estos pasos:
1. **Crear cuenta:** Regístrate en [huggingface.co/join](https://huggingface.co/join).
2. **Generar Token:** Ve a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y crea un token de tipo **Read**.
3. **Configurar en Windows:** Ejecuta en PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'TU_TOKEN_AQUI', 'User')
```

## Estructura del Proyecto

```
DA-Seamless-Cloning/
├── 1_extract_information.py     # Paso 1: Extracción de cámara y profundidad
├── 2_view_cluster.py            # Paso 2: Visualización de clusters
├── 3_people_pool.py             # Paso 3: Creación del pool de personas
├── 4_extract_masks.py           # Paso 4: Pre-segmentación de siluetas (YOLOv8x + SAM2-L)
├── 5_data_augmentation_ab.py    # Paso 5: Augmentación Principal (escalado métrico + alpha blending)
├── config.py                    # Configuración global de rutas
├── vggt/                        # Código fuente del modelo VGGT
├── people_pool/                 # Scripts de apoyo para el pool
├── tools/                       # Herramientas de video y etiquetado
├── back/                        # Respaldos de versiones anteriores
└── README.md
```

## Uso

### 1. Extraer información de cámara
Puedes usar el diálogo interactivo o pasar las rutas por CLI:
```bash
python 1_extract_information.py --img_dir "ruta/fotos" --out_dir "ruta/salida" --batch 50
```

### 2. Crear pool de personas e imágenes
Configura las rutas en `config.py` y ejecuta:
```bash
python 3_people_pool.py
```

### 3. Pre-segmentación de siluetas
Extrae las máscaras de las personas una sola vez (pipeline YOLOv8x → SAM2-L). Los pesos se descargan automáticamente en la primera ejecución (`yolov8x.pt` ~130 MB, `sam2_l.pt` ~428 MB):
```bash
python 4_extract_masks.py
```

### 4. Augmentación
Inserta los recortes en nuevas imágenes usando las máscaras pre-calculadas, con escalado métrico basado en mapa de profundidad y alpha blending con feathering:
```bash
python 5_data_augmentation_ab.py
```

## Tecnologías

- **[VGGT](https://github.com/facebookresearch/vggt)** — Estimación de cámara y profundidad (Facebook Research).
- **OpenCV** — Procesamiento de imagen y alpha blending con feathering.
- **Ultralytics YOLOv8x** — Detección de personas (bounding box).
- **Ultralytics SAM2-L** — Segmentación de alta precisión a partir del bounding box (Meta AI).
- **Hugging Face Hub** — Gestión del modelo pre-entrenado VGGT-1B.

## Autor

**Pedro AM** · [@pedroamtech](https://github.com/pedroamtech)

---
⭐ Si este proyecto te ha sido útil, ¡dale una estrella!