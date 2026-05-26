from tqdm import tqdm
from glob import glob
from os.path import join, basename
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2
import config
import numpy as np

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

HEIGHT_MIN = config.HEIGHT_MIN  # px — altura mínima de crop aceptada
HEIGHT_MAX = config.HEIGHT_MAX  # px — altura máxima de crop aceptada


def _get_height_bin(h, h_min, h_max):
    span = h_max - h_min
    if h < h_min + span * 0.25: return 'xs'
    if h < h_min + span * 0.50: return 's'
    if h < h_min + span * 0.75: return 'm'
    return 'l'


def _poolCreation(args):
    root_data, anno_name, root_output, height_min, height_max = args

    anno_name = anno_name.replace('\\', '/')
    filename  = anno_name.split('/')[-1]

    img_name = filename.replace('.txt', '.jpg')
    img_path = str(root_data / 'images' / img_name)
    img = cv2.imread(img_path)
    if img is None:
        return []

    depth_dir  = root_data / 'depth_maps'
    depth_path = depth_dir / f"depth_{img_name}"
    depth_map  = None
    if depth_path.exists():
        depth_map_raw = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth_map_raw is not None:
            h_img, w_img = img.shape[:2]
            if depth_map_raw.shape[0] != h_img or depth_map_raw.shape[1] != w_img:
                depth_map = cv2.resize(depth_map_raw, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            else:
                depth_map = depth_map_raw

    height_img, width_img = img.shape[:2]
    cont       = 0
    crops_info = []

    with open(anno_name, 'r') as file:
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            if not row or len(row) < 5:
                continue
            if int(row[0]) == 0:
                x_center = float(row[1]) * width_img
                y_center = float(row[2]) * height_img
                w = int(float(row[3]) * width_img)
                h = int(float(row[4]) * height_img)
                x = int(x_center - w // 2)
                y = int(y_center - h // 2)

                if not (x >= 0 and y >= 0 and w > 0 and h > 0
                        and (x + w) <= width_img and (y + h) <= height_img):
                    continue

                # V2: filtro de tamaño — descarta crops fuera del rango útil
                if h < height_min or h > height_max:
                    continue

                crop_img  = img[y:y+h, x:x+w]
                crop_name = filename.replace('.txt', f'_{cont}.jpg')
                crop_path = str(root_output / crop_name)
                cv2.imwrite(crop_path, crop_img)

                depth_val = 0.5
                if depth_map is not None:
                    crop_depth = depth_map[y:y+h, x:x+w]
                    if crop_depth.size > 0:
                        depth_val = float(crop_depth.mean()) / 255.0

                crops_info.append({
                    'name':           crop_path,
                    'height_patch':   h,
                    'width_patch':    w,
                    'depth_avg':      depth_val,
                    'original_image': img_name,
                    'height_bin':     _get_height_bin(h, height_min, height_max),  # V2
                })
                cont += 1

    return crops_info


def poolCreation(root_data_list, root_output, num_process=10):
    root_output.mkdir(parents=True, exist_ok=True)

    meta_csv_path = root_data_list[0] / 'depth_maps' / 'camera_data.csv'
    if not meta_csv_path.exists():
        raise FileNotFoundError(f"No se encontró camera_data.csv en: {meta_csv_path}")

    print(f"\n[INFO] Cargando metadatos desde: {meta_csv_path}")
    df_meta = pd.read_csv(meta_csv_path)
    df_meta['image_name'] = df_meta['image_name'].apply(basename)
    df_meta_indexed = df_meta.set_index('image_name')

    all_crops = []

    for rd in root_data_list:
        annos     = glob(join(str(rd / 'labels'), '*.txt'))
        num_annos = len(annos)
        print(f"[INFO] Procesando {num_annos} imágenes en: {rd}")
        print(f"[INFO] Filtro de altura activo: {HEIGHT_MIN}px – {HEIGHT_MAX}px")

        args_list = list(zip(
            [rd]         * num_annos,
            annos,
            [root_output] * num_annos,
            [HEIGHT_MIN]  * num_annos,
            [HEIGHT_MAX]  * num_annos,
        ))

        with Pool(processes=num_process) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_poolCreation, args_list),
                    desc='Generando recortes (filtro de tamaño activo)',
                    total=num_annos,
                    ncols=100,
                )
            )

        for crop_list in results:
            all_crops.extend(crop_list)

    print(f"\n[INFO] Vinculando {len(all_crops)} recortes con metadatos de cámara...")

    records = []
    for crop in tqdm(all_crops, desc='Integrando metadatos', ncols=100):
        orig_img = crop['original_image']
        record   = dict(crop)
        if orig_img in df_meta_indexed.index:
            meta_row = df_meta_indexed.loc[orig_img]
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]
            record.update(meta_row.to_dict())
        records.append(record)

    df_final   = pd.DataFrame(records)
    output_csv = root_output / 'pool.csv'
    try:
        df_final.to_csv(str(output_csv), index=False)
    except PermissionError:
        ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = root_output / f'pool_{ts}.csv'
        df_final.to_csv(str(output_csv), index=False)

    # V2: distribución de tamaños del pool
    total = len(df_final)
    print(f"\nDistribución del pool por bin de altura ({HEIGHT_MIN}–{HEIGHT_MAX} px) — {total} crops:")
    bin_ranges = {
        'xs': (HEIGHT_MIN,                         HEIGHT_MIN + (HEIGHT_MAX - HEIGHT_MIN) // 4),
        's':  (HEIGHT_MIN + (HEIGHT_MAX - HEIGHT_MIN) // 4,  HEIGHT_MIN + (HEIGHT_MAX - HEIGHT_MIN) // 2),
        'm':  (HEIGHT_MIN + (HEIGHT_MAX - HEIGHT_MIN) // 2,  HEIGHT_MIN + 3*(HEIGHT_MAX - HEIGHT_MIN) // 4),
        'l':  (HEIGHT_MIN + 3*(HEIGHT_MAX - HEIGHT_MIN) // 4, HEIGHT_MAX),
    }
    for label, (lo, hi) in bin_ranges.items():
        count = (df_final['height_bin'] == label).sum()
        pct   = 100 * count / total if total > 0 else 0
        bar   = '█' * int(pct / 2)
        print(f"  {label} [{lo:2d}–{hi:2d}px]: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"\n[INFO] Pool guardado en: {output_csv}")
    print(f"       Columnas: {df_final.columns.tolist()}")


if __name__ == '__main__':
    for d in config.PARTITIONS:
        poolCreation(
            root_data_list=[Path(config.ROOT_DATA1) / d],
            root_output=Path(config.ROOT_POOL_PERSON),
        )
