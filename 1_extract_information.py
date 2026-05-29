import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Añadir el directorio raíz de VGGT al path para encontrar el paquete vggt/
_VGGT_REPO = Path(__file__).parent / 'vggt'
if str(_VGGT_REPO) not in sys.path:
    sys.path.insert(0, str(_VGGT_REPO))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import login, get_token

import config

# Importaciones de VGGT
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError:
    print("Error: No se encuentran los módulos de VGGT. Ejecuta desde la raíz del proyecto.")
    sys.exit(1)

def extract_information(batch_size=32):
    # 1. Configuración del modelo (se carga una sola vez para todas las particiones)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        hf_token = get_token()
        if hf_token is None:
            import logging
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  CUDA {torch.version.cuda}")
    else:
        print("[GPU] CUDA no disponible — se usará CPU (proceso lento)")

    print(f"Cargando VGGT en {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", token=hf_token).to(device)
    model.eval()

    valid_exts = ('.png', '.jpg', '.jpeg')

    # 2. Procesar cada partición definida en config
    for partition in config.PARTITIONS:
        partition_dir = Path(config.ROOT_DATA1) / partition
        img_dir   = partition_dir / 'images'
        depth_dir = img_dir / 'depth_maps'
        csv_path  = img_dir / 'camera_data.csv'

        if not img_dir.exists():
            print(f"[{partition}] Carpeta de imágenes no encontrada: {img_dir}")
            continue

        depth_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([
            str(img_dir / f) for f in os.listdir(img_dir)
            if f.lower().endswith(valid_exts)
        ])
        if not image_files:
            print(f"[{partition}] No se encontraron imágenes en: {img_dir}")
            continue

        data_records = []
        print(f"\n[{partition}] {len(image_files)} imágenes | batch={batch_size}")
        print(f"  Depth maps -> {depth_dir}")
        print(f"  CSV        -> {csv_path}")

        # 3. Procesamiento por batches
        n_batches = len(range(0, len(image_files), batch_size))
        for batch_num, i in enumerate(range(0, len(image_files), batch_size), 1):
            batch_files = image_files[i : i + batch_size]
            images_tensor = load_and_preprocess_images(batch_files).to(device)

            with torch.inference_mode():
                predictions = model(images_tensor.unsqueeze(0))
                pose_enc   = predictions["pose_enc"]
                depth_data = predictions["depth"].squeeze(0).float().cpu().numpy()

            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
            extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
            intrinsics = intrinsics.squeeze(0).float().cpu().numpy()

            for j, img_path in tqdm(enumerate(batch_files), total=len(batch_files),
                                     desc=f"[{partition}] Batch {batch_num}/{n_batches}",
                                     unit="img", ncols=100):
                # A. Guardar mapa de profundidad (16-bit PNG — 256x más resolución que uint8)
                d_map = depth_data[j, :, :, 0]
                d_min, d_max = d_map.min(), d_map.max()
                norm_d = (d_map - d_min) / (d_max - d_min + 1e-8)
                depth_name = f"depth_{os.path.splitext(os.path.basename(img_path))[0]}.png"
                Image.fromarray((norm_d * 65535).astype(np.uint16)).save(depth_dir / depth_name)

                # B. Pose (Mundo <- Cámara)
                R, t = extrinsics[j][:3, :3], extrinsics[j][:3, 3]
                C_world = -np.dot(R.T, t)
                height_rel = abs(C_world[2])

                # C. Ángulo de inclinación (Pitch)
                pitch_rad = np.arcsin(np.clip(R[:, 2][2], -1.0, 1.0))
                pitch_deg = np.degrees(pitch_rad)

                data_records.append({
                    "image_name":     os.path.basename(img_path),
                    "depth_map_path": depth_name,
                    "depth_min":      d_min,
                    "depth_max":      d_max,
                    "focal_x":        intrinsics[j][0, 0],
                    "focal_y":        intrinsics[j][1, 1],
                    "principal_x":    intrinsics[j][0, 2],
                    "principal_y":    intrinsics[j][1, 2],
                    "pos_x":          C_world[0],
                    "pos_y":          C_world[1],
                    "pos_z":          C_world[2],
                    "height":         height_rel,
                    "pitch":          pitch_deg,
                    "R_world_flat":   R.T.flatten().tolist()
                })

            del images_tensor, predictions, pose_enc, depth_data
            torch.cuda.empty_cache()

        pd.DataFrame(data_records).to_csv(csv_path, index=False)
        print(f"[{partition}] CSV guardado en: {csv_path}")

    print("\nExtracción completada con éxito.")

if __name__ == "__main__":
    t0 = time.perf_counter()
    extract_information()
    elapsed = time.perf_counter() - t0
    print(f"\nTiempo total: {int(elapsed//3600):02d}h {int(elapsed%3600//60):02d}m {elapsed%60:05.2f}s")
