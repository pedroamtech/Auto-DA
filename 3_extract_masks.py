import cv2
import numpy as np
import pandas as pd
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch
import logging

# Configuración de logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Importar config del proyecto
import config

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ROOT_POOL_PERSON = Path(config.ROOT_POOL_PERSON)
PARTITIONS = config.PARTITIONS
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def extract_yolo_mask_original(crop_orig, yolo_model):
    """
    Usa YOLOv8x-Seg para extraer la máscara en resolución original.
    Retorna la máscara binaria y metadatos de calidad.
    """
    h, w = crop_orig.shape[:2]
    
    # Inferencia
    results = yolo_model.predict(
        source=crop_orig, 
        classes=[0], 
        verbose=False, 
        device=DEVICE, 
        conf=0.1, 
        retina_masks=True, 
        imgsz=640
    )
    
    if len(results) > 0 and results[0].masks is not None:
        mask_tensor = results[0].masks.data[0].cpu().numpy()
        # Redimensionar a la resolución original del parche si es necesario
        if mask_tensor.shape[:2] != (h, w):
            mask_binary = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_binary = mask_tensor
            
        mask_binary = (mask_binary * 255).astype(np.uint8)
        
        # Heurísticas de Calidad
        stats = {
            "area_ratio": float(cv2.countNonZero(mask_binary) / (w * h)),
            "bottom_width_ratio": float(np.count_nonzero(mask_binary[-1, :]) / w),
            "top_width_ratio": float(np.count_nonzero(mask_binary[0, :]) / w),
            "left_height_ratio": float(np.count_nonzero(mask_binary[:, 0]) / h),
            "right_height_ratio": float(np.count_nonzero(mask_binary[:, -1]) / h),
            "is_valid": True
        }
        
        # Aplicar los mismos criterios de validación que en el script original
        if stats["area_ratio"] < 0.15: stats["is_valid"] = False
        if stats["bottom_width_ratio"] > 0.45: stats["is_valid"] = False
        if stats["top_width_ratio"] > 0.35: stats["is_valid"] = False
        if stats["left_height_ratio"] > 0.40 or stats["right_height_ratio"] > 0.40: stats["is_valid"] = False
        
        return mask_binary, stats

    return None, {"is_valid": False, "reason": "No detection"}

def process_pool():
    print("="*60)
    print("  Pre-segmentación YOLOv8x-Seg para People Pool")
    print("="*60)
    
    print(f"[INFO] Cargando modelo YOLOv8x-Seg en {DEVICE}...")
    yolo_model = YOLO('yolov8x-seg.pt')

    for partition in PARTITIONS:
        pool_dir = ROOT_POOL_PERSON / partition
        pool_csv_p = pool_dir / 'pool.csv'
        
        if not pool_csv_p.exists():
            print(f"[WARN] No se encontró pool.csv en {pool_dir}")
            continue
            
        # Crear directorios de salida
        masks_dir = pool_dir / 'masks'
        meta_dir = pool_dir / 'metadata'
        masks_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        df_pool = pd.read_csv(str(pool_csv_p))
        
        print(f"[INFO] Procesando partición: {partition} ({len(df_pool)} parches)")
        
        for idx, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc=f"Segmentando {partition}"):
            img_path = Path(row['name'])
            if not img_path.exists(): continue
            
            patch_name = img_path.stem
            mask_path = masks_dir / f"{patch_name}.png"
            json_path = meta_dir / f"{patch_name}.json"
            
            # Saltar si ya existe (opcional, pero útil para reanudar)
            if mask_path.exists() and json_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            mask, stats = extract_yolo_mask_original(img, yolo_model)
            
            # Guardar metadatos
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=4)
                
            # Guardar máscara si es válida
            if mask is not None:
                cv2.imwrite(str(mask_path), mask)

    print("\n[SUCCESS] Pre-segmentación completada.")

if __name__ == '__main__':
    process_pool()
