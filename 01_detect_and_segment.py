"""Delineate tree crowns using DeepForest/YOLO+SAM or Detectree2 masks.

Example:
 python 01_detect_and_segment.py forest.tif crowns_auto.gpkg \
   --rgb 3 2 1 --sam-checkpoint sam_vit_b_01ec64.pth --core 512 --margin 192

Install dependencies for the selected backend only. Detectree2 is a Mask R-CNN
instance segmenter: unlike DeepForest/YOLO, it does not require SAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape
from tqdm import tqdm

from crown_detectors import build_detector


def estimate_stretch(ds, bands, samples_per_axis=5, sample_size=256):
    """Estimate *one* 2-98 percentile stretch from sparse sample windows."""
    fragments = [[] for _ in bands]
    xs = np.linspace(0, max(0, ds.width - 1), samples_per_axis, dtype=int)
    ys = np.linspace(0, max(0, ds.height - 1), samples_per_axis, dtype=int)
    for cy in ys:
        for cx in xs:
            x0 = max(0, min(ds.width - sample_size, cx - sample_size // 2))
            y0 = max(0, min(ds.height - sample_size, cy - sample_size // 2))
            w = Window(x0, y0, min(sample_size, ds.width), min(sample_size, ds.height))
            arr = ds.read(list(bands), window=w, masked=True)
            for k in range(len(bands)):
                v = arr[k].compressed()
                v = v[np.isfinite(v)]
                if v.size:
                    fragments[k].append(v)
    limits = []
    for band, parts in zip(bands, fragments):
        if not parts:
            raise ValueError(f"No finite sample values for band {band}.")
        values = np.concatenate(parts).astype('float32')
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        limits.append([float(lo), float(max(hi, lo + 1e-10))])
    return limits


def make_rgb(arr, limits):
    # arr: 3 x H x W, order R G B
    out = []
    for b, (lo, hi) in zip(arr, limits):
        val = np.nan_to_num(b.astype('float32'), nan=lo, posinf=hi, neginf=lo)
        val = np.clip((val - lo) * 255.0 / (hi - lo), 0, 255).astype('uint8')
        out.append(val)
    return np.ascontiguousarray(np.moveaxis(np.stack(out), 0, -1))


def largest_component(mask):
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    if n <= 1:
        return mask, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    # Flag if another component is substantial relative to the first.
    second = np.sort(areas)[-2] if len(areas) > 1 else 0
    return components == idx, int(second > 0.25 * areas.max())


def mask_to_polygon(mask, window, ds, simplify_px=1.0):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub = mask[y0:y1, x0:x1]
    small_window = Window(window.col_off + x0, window.row_off + y0,
                          x1 - x0, y1 - y0)
    transform = ds.window_transform(small_window)
    polygons = [shape(geom) for geom, value in shapes(
        sub.astype('uint8'), mask=sub, transform=transform) if value == 1]
    if not polygons:
        return None
    geom = max(polygons, key=lambda p: p.area)
    # Tolerance in CRS coordinates corresponding roughly to a raster pixel.
    sx = math.hypot(ds.transform.a, ds.transform.d)
    sy = math.hypot(ds.transform.b, ds.transform.e)
    tolerance = simplify_px * min(sx, sy)
    geom = geom.simplify(tolerance, preserve_topology=True)
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('out_gpkg', type=Path)
    p.add_argument('--rgb', type=int, nargs=3, required=True, metavar=('RED', 'GREEN', 'BLUE'),
                   help='1-based band indices, as in Rasterio/QGIS')
    p.add_argument('--sam-checkpoint', type=Path, default=None,
                   help='required for deepforest/yolo only; Detectree2 predicts masks directly')
    p.add_argument('--detector', choices=['deepforest', 'yolo', 'detectree2', 'detectree'],
                   default='deepforest',
                   help='DeepForest/YOLO boxes + SAM, or Detectree2 direct instance masks')
    p.add_argument('--model-weights', type=Path, default=None,
                   help='weights for YOLO (.pt) or Detectree2 (.pth), required for these modes')
    p.add_argument('--yolo-imgsz', type=int, default=1024,
                   help='YOLO inference size; try 768 or 1024 for aerial images')
    p.add_argument('--yolo-iou', type=float, default=0.5,
                   help='YOLO nonmaximum suppression IoU threshold')
    p.add_argument('--yolo-class-id', type=int, default=None,
                   help='optional class index for tree crowns in a multiclass YOLO model')
    p.add_argument('--detectree-nms', type=float, default=0.3,
                   help='Detectree2 instance NMS threshold (default: 0.3)')
    p.add_argument('--detectree-base-model', type=str,
                   default='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                   help='Detectron2 config corresponding to Detectree2 checkpoint')
    p.add_argument('--core', type=int, default=512)
    p.add_argument('--margin', type=int, default=192)
    p.add_argument('--min-confidence', type=float, default=0.20)
    p.add_argument('--min-area', type=int, default=80, help='minimum pixel count in SAM mask')
    p.add_argument('--max-area', type=int, default=150_000)
    p.add_argument('--simplify-px', type=float, default=1.0)
    p.add_argument('--max-tiles', type=int, default=0,
                   help='limit processing for testing; 0 means entire selected range')
    p.add_argument('--start-tile', type=int, default=0,
                   help='0-based index in row-major scan; start processing here')
    p.add_argument('--end-tile', type=int, default=None,
                   help='0-based inclusive end index; exclusive with --max-tiles')
    p.add_argument('--tile-row', type=int, default=None, help='0-based core row to test')
    p.add_argument('--tile-col', type=int, default=None, help='0-based core column to test')
    p.add_argument('--debug-dir', type=Path, default=None,
                   help='save RGB previews, DeepForest boxes and per-tile diagnostics')
    p.add_argument('--df-score-thresh', type=float, default=None,
                   help='internal DeepForest prediction threshold (try 0.05 for diagnosis)')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = p.parse_args()

    if args.core < 64 or args.margin < 0:
        p.error('--core must be >=64 and --margin >= 0')
    if args.start_tile < 0 or args.max_tiles < 0:
        p.error('--start-tile and --max-tiles must be >=0')
    if args.end_tile is not None and args.end_tile < 0:
        p.error('--end-tile must be >=0')
    if args.end_tile is not None and args.max_tiles:
        p.error('use either --end-tile or --max-tiles')
    if args.end_tile is not None and args.tile_row is not None:
        p.error('--end-tile cannot be combined with --tile-row/--tile-col')
    if (args.tile_row is None) != (args.tile_col is None):
        p.error('provide BOTH --tile-row and --tile-col')
    if args.tile_row is not None and (args.tile_row < 0 or args.tile_col < 0 or args.start_tile):
        p.error('non-negative --tile-row/--tile-col cannot be combined with --start-tile')
    if args.df_score_thresh is not None and not (0 <= args.df_score_thresh <= 1):
        p.error('--df-score-thresh must be between 0 and 1')
    if args.detector == 'detectree':
        args.detector = 'detectree2'
    if args.detector == 'yolo':
        if args.model_weights is None or not args.model_weights.is_file():
            p.error('For --detector yolo pass an existing tree-trained checkpoint via --model-weights')
        if not (0 < args.yolo_iou <= 1):
            p.error('--yolo-iou must be in (0, 1]')
        if args.yolo_imgsz < 64:
            p.error('--yolo-imgsz must be >= 64')
        if args.yolo_class_id is not None and args.yolo_class_id < 0:
            p.error('--yolo-class-id must be nonnegative')
        if args.df_score_thresh is not None:
            p.error('--df-score-thresh only applies to --detector deepforest')
    if args.detector == 'detectree2':
        if args.model_weights is None or not args.model_weights.is_file():
            p.error('Detectree2 needs a local .pth checkpoint via --model-weights')
        if args.model_weights.suffix.lower() != '.pth':
            p.error('Detectree2 --model-weights must be a .pth Mask R-CNN checkpoint')
        if not (0 < args.detectree_nms <= 1):
            p.error('--detectree-nms must be in (0, 1]')
        if args.df_score_thresh is not None:
            p.error('--df-score-thresh only applies to --detector deepforest')
    else:
        if args.sam_checkpoint is None or not args.sam_checkpoint.is_file():
            p.error('DeepForest/YOLO require an existing --sam-checkpoint')
    if not (0 <= args.min_confidence <= 1):
        p.error('--min-confidence must be in [0, 1]')
    if args.min_area < 1 or args.max_area < args.min_area:
        p.error('--min-area/--max-area must define a positive non-empty range')
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but torch.cuda.is_available() is False')
    segmenter = None
    detector = None
    predictor = None
    if args.detector == 'detectree2':
        from crown_segmenters import Detectree2Segmenter
        segmenter = Detectree2Segmenter(args.model_weights, device,
                                       confidence=args.min_confidence,
                                       nms_iou=args.detectree_nms,
                                       base_model=args.detectree_base_model)
    else:
        from segment_anything import SamPredictor, sam_model_registry
        detector = build_detector(args, device)
        sam = sam_model_registry['vit_b'](checkpoint=str(args.sam_checkpoint))
        sam.to(device)
        sam.eval()
        predictor = SamPredictor(sam)
    records = []
    diagnostics = []
    with rasterio.open(args.image) as ds:
        if ds.crs is None:
            raise ValueError('The GeoTIFF has no CRS; this pipeline needs georeferencing.')
        if min(args.rgb) < 1 or max(args.rgb) > ds.count:
            raise ValueError('RGB bands must be within 1..' + str(ds.count))
        limits = estimate_stretch(ds, args.rgb)
        stretch_path = args.out_gpkg.with_name(args.out_gpkg.stem + '_stretch.json')
        stretch_path.parent.mkdir(parents=True, exist_ok=True)
        stretch_path.write_text(json.dumps({
            'rgb_bands_1_based': args.rgb, 'limits_2_98': limits,
            'source_image': str(args.image.resolve()), 'band_count': ds.count
        }, indent=2, ensure_ascii=False), encoding='utf-8')

        cores = [(x, y) for y in range(0, ds.height, args.core)
                        for x in range(0, ds.width, args.core)]
        cols = math.ceil(ds.width / args.core)
        rows = math.ceil(ds.height / args.core)
        if args.tile_row is not None:
            if args.tile_row >= rows or args.tile_col >= cols:
                p.error(f'tile outside raster; valid rows 0..{rows-1}, cols 0..{cols-1}')
            start = args.tile_row * cols + args.tile_col
        else:
            start = args.start_tile
        if start >= len(cores):
            p.error(f'--start-tile outside raster (0..{len(cores)-1})')
        if args.end_tile is not None:
            if not start <= args.end_tile < len(cores):
                p.error(f'--end-tile must be in {start}..{len(cores)-1}')
            cores = list(enumerate(cores))[start:args.end_tile + 1]
        else:
            cores = list(enumerate(cores))[start:]
            if args.max_tiles:
                cores = cores[:args.max_tiles]
            elif args.tile_row is not None:
                cores = cores[:1]
        print(f'Grid: {cols} columns x {rows} rows; selected {len(cores)} tile(s); '
               f'first tile index={start} (0-based).')
        for tile_index, (cx, cy) in tqdm(cores, desc='Tiles'):
            x0, y0 = max(0, cx - args.margin), max(0, cy - args.margin)
            x1 = min(ds.width, cx + args.core + args.margin)
            y1 = min(ds.height, cy + args.core + args.margin)
            window = Window(x0, y0, x1 - x0, y1 - y0)
            data = ds.read(list(args.rgb), window=window)
            rgb = make_rgb(data, limits)
            report = dict(tile_index=tile_index, tile_row=cy//args.core,
                          tile_col=cx//args.core, core_x=cx, core_y=cy,
                          rgb_std=round(float(rgb.std()), 3),
                          detector_boxes=0, rejected_confidence=0, rejected_box=0,
                          rejected_core=0, rejected_area=0,
                          rejected_geometry=0, kept=0, status='processed')
            tag = f'tile_{tile_index:05d}_r{cy//args.core}_c{cx//args.core}'
            if args.debug_dir:
                cv2.imwrite(str(args.debug_dir / f'{tag}_rgb.png'),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if report['rgb_std'] < 3:
                report['status'] = 'skipped_low_contrast'
                diagnostics.append(report)
                print(f'[{tag}] skipped: RGB std={report["rgb_std"]}')
                continue
            # A unified list of (bounding-box, mask, model score, SAM score).
            # For Detectree2 masks come directly from Mask R-CNN; for other
            # detectors, SAM refines each box individually.
            detections = None
            instances = None
            if segmenter is not None:
                instances = segmenter.predict_masks(rgb)
                report['detector_boxes'] = len(instances)
            else:
                detections = detector.predict(rgb)
                report['detector_boxes'] = len(detections) if detections is not None else 0

            if args.debug_dir:
                overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                if instances is not None:
                    boxes_for_debug = [instance.box for instance in instances]
                else:
                    boxes_for_debug = ([np.array([d[k] for k in
                        ('xmin', 'ymin', 'xmax', 'ymax')], dtype=float)
                        for _, d in detections.iterrows()] if detections is not None else [])
                for bx in boxes_for_debug:
                    a = [int(round(float(v))) for v in bx]
                    cv2.rectangle(overlay, tuple(a[:2]), tuple(a[2:]), (0, 0, 255), 2)
                if instances is not None:
                    for inst in instances:
                        contours, _ = cv2.findContours(inst.mask.astype('uint8'),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
                cv2.imwrite(str(args.debug_dir / f'{tag}_boxes.png'), overlay)

            if report['detector_boxes'] == 0:
                report['status'] = 'no_instance_masks' if segmenter else 'no_detector_boxes'
                diagnostics.append(report)
                print(f'[{tag}] {args.detector}: 0 instances')
                continue

            if segmenter is None:
                # SAM embedding is computed only for the box-based backends.
                predictor.set_image(rgb)
                candidates = ((np.array([d[k] for k in ('xmin','ymin','xmax','ymax')],
                                              dtype='float32'), float(d.get('score', 1.0)), None)
                              for _, d in detections.iterrows())
            else:
                candidates = ((inst.box, inst.score, inst.mask) for inst in instances)

            for bx, conf, initial_mask in candidates:
                if conf < args.min_confidence:
                    report['rejected_confidence'] += 1
                    continue
                bx = np.asarray(bx, dtype='float32')
                if not np.isfinite(bx).all() or bx.shape != (4,) or bx[2] <= bx[0] or bx[3] <= bx[1]:
                    report['rejected_box'] += 1
                    continue
                global_x = x0 + float((bx[0] + bx[2]) / 2)
                global_y = y0 + float((bx[1] + bx[3]) / 2)
                # One owner per tree: only accept box center in this tile's core.
                if not (cx <= global_x < min(cx + args.core, ds.width)
                        and cy <= global_y < min(cy + args.core, ds.height)):
                    report['rejected_core'] += 1
                    continue
                if initial_mask is None:
                    with torch.inference_mode():
                        masks, qualities, _ = predictor.predict(box=bx, multimask_output=True)
                    chosen = int(np.argmax(qualities))
                    input_mask = masks[chosen]
                    sam_score = float(qualities[chosen])
                else:
                    input_mask = initial_mask
                    sam_score = None

                if np.asarray(input_mask).shape != rgb.shape[:2]:
                    raise ValueError(f'Predicted mask shape does not match RGB tile for {tag}')
                mask, fragmented = largest_component(np.asarray(input_mask, dtype=bool))
                area = int(mask.sum())
                if area < args.min_area or area > args.max_area:
                    report['rejected_area'] += 1
                    continue
                polygon = mask_to_polygon(mask, window, ds, args.simplify_px)
                if polygon is None:
                    report['rejected_geometry'] += 1
                    continue

                flags = []
                if fragmented:
                    flags.append('fragmented')
                if sam_score is not None and sam_score < 0.8:
                    flags.append('low_sam_score')
                if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                    flags.append('touches_tile_edge')
                boxarea = max(1, float((bx[2] - bx[0]) * (bx[3] - bx[1])))
                if area > 1.8 * boxarea:
                    flags.append('large_mask_vs_box')
                records.append(dict(
                    tree_id=f'T{len(records) + 1:06d}', detector=args.detector,
                    detect_score=float(conf), sam_score=sam_score,
                    area_px=area, status='pending',
                    review_needed=int(bool(flags)), flags=','.join(flags),
                    geometry=polygon
                ))
                report['kept'] += 1
            diagnostics.append(report)
            print(f'[{tag}] {args.detector}={report["detector_boxes"]}, '
                  f'outside core={report["rejected_core"]}, '
                  f'area reject={report["rejected_area"]}, '
                  f'kept={report["kept"]}')
            if device == 'cuda':
                torch.cuda.empty_cache()

        if args.debug_dir:
            table_path = args.debug_dir / 'diagnostics.csv'
            with table_path.open('w', encoding='utf-8', newline='') as stream:
                if diagnostics:
                    writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
                    writer.writeheader()
                    writer.writerows(diagnostics)
            print(f'Debug files and diagnostics: {args.debug_dir}')
        if not records:
            total_boxes = sum(d['detector_boxes'] for d in diagnostics)
            print(f'Diagnostic summary: {args.detector} boxes={total_boxes}, '
                  f'kept={len(records)}, tiles={len(diagnostics)}')
            raise RuntimeError('No crowns were detected. Use --debug-dir to inspect RGB/boxes, '
                               'or pick a forest tile using --tile-row/--tile-col.')
        gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=ds.crs)
        gdf['sam_score'] = gdf['sam_score'].astype(float)  # NaN for Detectree2
        # Identify conspicuous overlaps. They may be multiple detections of one tree.
        idx = gdf.sindex
        for i, geom in enumerate(gdf.geometry):
            for j in idx.query(geom, predicate='intersects'):
                j = int(j)
                if j <= i:
                    continue
                other = gdf.geometry.iloc[j]
                if geom.intersection(other).area > 0.1 * min(geom.area, other.area):
                    for k in (i, j):
                        gdf.at[k, 'review_needed'] = 1
                        old = gdf.at[k, 'flags']
                        if 'overlap' not in old.split(','):
                            gdf.at[k, 'flags'] = (old + ',overlap').strip(',')
        args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        if args.out_gpkg.exists():
            args.out_gpkg.unlink()
        gdf.to_file(args.out_gpkg, layer='crowns', driver='GPKG')
        print(f'Saved {len(gdf)} candidate crowns to {args.out_gpkg}')
        print(f'Global RGB stretch: {stretch_path}')
        print(f'Flagged for review: {int(gdf.review_needed.sum())}')


if __name__ == '__main__':
    main()
