"""
3_people_pool_v2.py  — Person crop pool builder (V2)

Changes from V1:
  · Height filter  : only crops with HEIGHT_MIN <= h <= HEIGHT_MAX are kept.
  · Boundary margin: crops whose bbox falls within BODY_MARGIN px of any image
    edge are discarded (person likely truncated by the frame border).
  · 'complete_body': True when the crop aspect ratio (w/h) is in the typical
    standing-person range; used by QC-6b in 5_data_augmentation_v2.py.
  · 'size_bin'     : pre-assigns each crop to the target bin of
    5_data_augmentation_v2.py for fast stratified pool lookups.
"""

from tqdm import tqdm
from glob import glob
import os
from os.path import join, basename
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import config

# ─── Filters (from config) ────────────────────────────────────────────────────
HEIGHT_MIN  = getattr(config, 'HEIGHT_MIN', 28)   # px — minimum crop height kept
HEIGHT_MAX  = getattr(config, 'HEIGHT_MAX', 79)   # px — maximum crop height kept
BODY_MARGIN = 5                                   # px — reject bbox within N px of edge

# Must match SIZE_BINS_PX in 5_data_augmentation_v2.py
SIZE_BINS_PX = [(10, 25), (25, 45), (45, 70), (70, 104)]

# Aspect ratio (w/h) range for a complete standing-body silhouette
AR_BODY_LO = 0.25
AR_BODY_HI = 0.52


def _size_bin_label(h: int) -> str:
    for lo, hi in SIZE_BINS_PX:
        if lo <= h < hi:
            return f'{lo}-{hi}'
    return 'out_of_range'


def _poolCreation_v2(args):
    root_data, anno_name, root_output = args

    anno_name = anno_name.replace('\\', '/')
    filename  = anno_name.split('/')[-1]
    img_name  = filename.replace('.txt', '.jpg')

    img = cv2.imread(str(root_data / 'images' / img_name))
    if img is None:
        return []

    # Depth map (optional) — soporta 8-bit y 16-bit PNG
    depth_dir    = root_data / 'depth_maps'
    img_stem     = os.path.splitext(img_name)[0]
    h_img, w_img = img.shape[:2]
    depth_map    = None
    for d_name in [f'depth_{img_stem}.png', f'depth_{img_name}']:
        d_path = depth_dir / d_name
        if d_path.exists():
            raw = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                max_val   = 65535.0 if raw.dtype == np.uint16 else 255.0
                resized   = cv2.resize(raw, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                depth_map = resized.astype(np.float32) / max_val
            break

    height_img, width_img = img.shape[:2]
    cont       = 0
    crops_info = []

    with open(anno_name, 'r') as file:
        for row in [x.split(' ') for x in file.read().strip().splitlines()]:
            if not row or len(row) < 5 or int(row[0]) != 0:
                continue

            x_c = float(row[1]) * width_img
            y_c = float(row[2]) * height_img
            w   = int(float(row[3]) * width_img)
            h   = int(float(row[4]) * height_img)
            x   = int(x_c - w // 2)
            y   = int(y_c - h // 2)

            # ── Bounding box validity ──────────────────────────────────────
            if not (x >= 0 and y >= 0 and w > 0 and h > 0
                    and (x + w) <= width_img and (y + h) <= height_img):
                continue

            # ── V2: height filter ──────────────────────────────────────────
            if not (HEIGHT_MIN <= h <= HEIGHT_MAX):
                continue

            # ── V2: boundary margin (reject near-edge / likely truncated) ──
            if (x < BODY_MARGIN or y < BODY_MARGIN
                    or (x + w) > (width_img - BODY_MARGIN)
                    or (y + h) > (height_img - BODY_MARGIN)):
                continue

            crop_name = filename.replace('.txt', f'_{cont}.jpg')
            crop_path = str(root_output / crop_name)
            cv2.imwrite(crop_path, img[y:y+h, x:x+w])

            # Depth average
            depth_val = 0.5
            if depth_map is not None:
                patch = depth_map[y:y+h, x:x+w]
                if patch.size > 0:
                    depth_val = float(patch.mean())

            # ── V2: complete_body flag (aspect ratio heuristic) ────────────
            ar            = w / max(h, 1)
            complete_body = bool(AR_BODY_LO <= ar <= AR_BODY_HI)

            crops_info.append({
                'name':           crop_path,
                'height_patch':   h,
                'width_patch':    w,
                'depth_avg':      depth_val,
                'original_image': img_name,
                'complete_body':  complete_body,
                'size_bin':       _size_bin_label(h),
            })
            cont += 1

    return crops_info


def poolCreation_v2(root_data_list, root_output, num_process=10):
    root_output.mkdir(parents=True, exist_ok=True)

    meta_csv_path = getattr(config, 'ROOT_VGGT_METADATA', None)
    if meta_csv_path is None:
        raise ValueError('ROOT_VGGT_METADATA not defined in config.py')

    print(f'\n[INFO] Loading camera metadata from: {meta_csv_path}')
    df_meta = pd.read_csv(meta_csv_path)
    df_meta['image_name'] = df_meta['image_name'].apply(basename)
    df_meta_indexed = df_meta.set_index('image_name')

    all_crops = []

    for rd in root_data_list:
        annos     = glob(join(str(rd / 'labels'), '*.txt'))
        num_annos = len(annos)
        print(f'[INFO] Processing {num_annos} images in: {rd}')
        print(f'       Height filter: [{HEIGHT_MIN}, {HEIGHT_MAX}] px | '
              f'Boundary margin: {BODY_MARGIN} px')

        with Pool(processes=num_process) as pool:
            results = list(tqdm(
                pool.imap_unordered(
                    _poolCreation_v2,
                    zip([rd] * num_annos, annos, [root_output] * num_annos)
                ),
                desc='Building pool V2',
                total=num_annos,
                ncols=100,
            ))

        for crop_list in results:
            all_crops.extend(crop_list)

    print(f'\n[INFO] Linking {len(all_crops)} crops with camera metadata...')
    records = []
    for crop in tqdm(all_crops, desc='Integrating metadata', ncols=100):
        record   = dict(crop)
        orig_img = crop['original_image']
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
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = root_output / f'pool_{ts}.csv'
        df_final.to_csv(str(output_csv), index=False)

    total    = len(df_final)
    body_pct = 100 * df_final['complete_body'].sum() / max(total, 1)
    print(f'\n[INFO] Pool V2 saved: {output_csv}')
    print(f'       Total crops  : {total}  (height {HEIGHT_MIN}–{HEIGHT_MAX} px)')
    print(f'       Complete body: {df_final["complete_body"].sum()} ({body_pct:.1f} %)')
    print('\n  Size-bin distribution:')
    for lo, hi in SIZE_BINS_PX:
        label = f'{lo}-{hi}'
        count = (df_final['size_bin'] == label).sum()
        pct   = 100 * count / max(total, 1)
        bar   = '█' * int(pct / 2)
        print(f'    [{lo:3d}–{hi:3d} px]: {count:5d}  ({pct:5.1f} %)  {bar}')
    oor = (df_final['size_bin'] == 'out_of_range').sum()
    if oor:
        print(f'    [out_of_range] : {oor}  (should be 0 — check HEIGHT_MIN/MAX)')
    print(f'\n  Columns: {df_final.columns.tolist()}')


if __name__ == '__main__':
    for d in config.PARTITIONS:
        poolCreation_v2(
            root_data_list=[Path(config.ROOT_DATA1) / d],
            root_output=Path(config.ROOT_POOL_PERSON) / d,
        )
