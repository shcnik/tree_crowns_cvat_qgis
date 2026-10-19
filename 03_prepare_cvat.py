"""Prepare non-destructive CVAT RGB tiles, initial COCO instance masks and tile manifest.

Images and COCO annotations are uploaded separately to a CVAT task.
Images are always included even if no tree was detected, to allow missed-tree review.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from shapely.geometry import mapping
from tqdm import tqdm

from crown_common import (bbox_center, geo_to_pixel, load_stretch, make_tiles,
                          mask_bbox, raster_signature, rgb_uint8, rle_encode)


def prepare(image: Path, gpkg: Path, stretch: Path, output: Path, *,
            core=512, margin=192, max_tiles=0, images_zip=True, only_with_annotations=False):
    if core < 64 or margin < 0:
        raise ValueError('core must be >=64 and margin nonnegative')
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / 'images'
    image_dir.mkdir(exist_ok=True)

    crowns = gpd.read_file(gpkg, layer='crowns')
    if crowns.crs is None:
        raise ValueError('GeoPackage must contain a CRS')
    if 'status' in crowns:
        crowns = crowns[~crowns['status'].fillna('').astype(str).str.lower().isin(
            ['reject', 'rejected', 'exclude'])].copy()
    if 'tree_id' not in crowns:
        raise ValueError('GeoPackage requires tree_id to preserve identities')
    if crowns.tree_id.isna().any() or crowns.tree_id.duplicated().any():
        raise ValueError('Missing or repeated tree_id in GeoPackage')
    if any(crowns.geometry.isna() | crowns.geometry.is_empty | ~crowns.geometry.is_valid):
        raise ValueError('GeoPackage contains empty or invalid crown geometries')

    with rasterio.open(image) as ds:
        if ds.crs is None:
            raise ValueError('GeoTIFF must have CRS')
        bands, limits = load_stretch(stretch, ds.count)
        if crowns.crs != ds.crs:
            crowns = crowns.to_crs(ds.crs)
        # Convert polygons to source-image pixel coords once, then use spatial index.
        pixel_geoms = [geo_to_pixel(g, ds.transform) for g in crowns.geometry]
        pixel_crowns = gpd.GeoDataFrame({'tree_id': crowns.tree_id.astype(str).tolist()},
                                         geometry=pixel_geoms)
        from shapely.geometry import Point, box
        center_points = gpd.GeoSeries([Point(*bbox_center(g)) for g in pixel_geoms])
        sindex = center_points.sindex
        images = []
        annotations = []
        tiles = []
        seed_records = []
        ann_id = 1
        iterable = make_tiles(ds.width, ds.height, core, margin)
        if max_tiles:
            from itertools import islice
            iterable = islice(iterable, max_tiles)
        for tile in tqdm(iterable, desc='Preparing CVAT tiles'):
            # Avoid reading outside raster; windows are already clipped.
            local_anns = []
            core_geom = box(tile.core_x0, tile.core_y0, tile.core_x1, tile.core_y1)
            for idx in sindex.query(core_geom, predicate='intersects'):
                geom = pixel_crowns.geometry.iloc[int(idx)]
                xc, yc = bbox_center(geom)
                if not tile.owns(xc, yc):
                    continue
                mask = rasterize([(mapping(geom), 1)], out_shape=(tile.height, tile.width),
                                 transform=rasterio.Affine.translation(tile.x, tile.y),
                                 fill=0, dtype='uint8')
                area = int(mask.sum())
                bbox = mask_bbox(mask)
                if not area or bbox is None:
                    continue
                local_anns.append((str(pixel_crowns.tree_id.iloc[int(idx)]), mask, area, bbox))
            if only_with_annotations and not local_anns:
                continue
            raw = ds.read(bands, window=tile.window)
            rgb = rgb_uint8(raw, list(range(1, 4)), limits)
            # Above rgb_uint8 expects indices into the passed array, which only has 3 bands.
            Image.fromarray(rgb).save(image_dir / tile.file_name)
            # Optional human-readable guide; NEVER upload these marked images to CVAT.
            guides = output / 'core_guides'
            guides.mkdir(exist_ok=True)
            guide = Image.fromarray(rgb).copy()
            draw = ImageDraw.Draw(guide)
            bx0, by0 = tile.core_x0 - tile.x, tile.core_y0 - tile.y
            bx1, by1 = tile.core_x1 - tile.x - 1, tile.core_y1 - tile.y - 1
            draw.rectangle((bx0, by0, bx1, by1), outline=(255, 30, 220), width=2)
            guide.save(guides / tile.file_name)
            image_id = len(images) + 1
            images.append({'id': image_id, 'file_name': tile.file_name,
                           'width': tile.width, 'height': tile.height})
            tiles.append(tile.to_dict())
            for tree_id, mask, area, bbox in local_anns:
                # iscrowd=1 is used solely to force COCO RLE representation of the
                # mask (not a statement that the tree represents a crowd).
                annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': 1,
                                    'segmentation': rle_encode(mask), 'area': area,
                                    'bbox': bbox, 'iscrowd': 1,
                                    'attributes': {'tree_id': tree_id}})
                seed_records.append({'tile': tile.file_name, 'annotation_id': ann_id,
                                     'tree_id': tree_id})
                ann_id += 1
        manifest = {
            'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'raster': raster_signature(ds), 'raster_path_at_export': str(image.resolve()),
            'seed_gpkg_at_export': str(gpkg.resolve()), 'stretch_path_at_export': str(stretch.resolve()),
            'rgb_bands_1_based': bands, 'limits_2_98': limits,
            'core_size': core,
            'margin_size': margin, 'tiles': tiles, 'seeds': seed_records,
        }

    info = {'description': 'Individual tree crowns for manual review in CVAT',
            'version': '1.0', 'date_created': datetime.now(timezone.utc).isoformat()}
    coco = {'info': info, 'licenses': [], 'categories': [{'id': 1, 'name': 'tree',
            'supercategory': 'vegetation'}], 'images': images, 'annotations': annotations}
    (output / 'instances_default.json').write_text(
        json.dumps(coco, ensure_ascii=False), encoding='utf-8')
    (output / 'tile_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    from collections import Counter
    count_by_image = Counter(row['tile'] for row in seed_records)
    with (output / 'tiles_for_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'initial_tree_count', 'core_local_x0', 'core_local_y0',
                         'core_local_x1', 'core_local_y1'])
        for t in tiles:
            writer.writerow([t['file_name'], count_by_image[t['file_name']],
                             t['core_x0'] - t['x'], t['core_y0'] - t['y'],
                             t['core_x1'] - t['x'], t['core_y1'] - t['y']])
    missed_ids = set(crowns.tree_id.astype(str)) - {r['tree_id'] for r in seed_records}
    if missed_ids and not max_tiles and not only_with_annotations:
        raise ValueError(f'{len(missed_ids)} source crown(s) not assigned to CVAT tiles '
                         f'(e.g. {sorted(missed_ids)[:3]}). Verify geometry is inside raster.')
    with zipfile.ZipFile(output / 'cvat_annotations.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output / 'instances_default.json', 'annotations/instances_default.json')
    if images_zip:
        with zipfile.ZipFile(output / 'cvat_images.zip', 'w', compression=zipfile.ZIP_STORED) as zf:
            for entry in images:
                zf.write(image_dir / entry['file_name'], entry['file_name'])
    print(f'Images: {len(images)}; pre-annotations: {len(annotations)}')
    print('Files: cvat_images.zip, cvat_annotations.zip, tile_manifest.json')
    return manifest, coco


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('crowns_gpkg', type=Path)
    p.add_argument('stretch', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--core', type=int, default=512)
    p.add_argument('--margin', type=int, default=192)
    p.add_argument('--max-tiles', type=int, default=0)
    p.add_argument('--no-images-zip', action='store_true')
    p.add_argument('--only-with-annotations', action='store_true',
                   help='NOT recommended for final annotation: hides tiles without detections')
    a = p.parse_args()
    prepare(a.image, a.crowns_gpkg, a.stretch, a.output_dir, core=a.core,
            margin=a.margin, max_tiles=a.max_tiles, images_zip=not a.no_images_zip,
            only_with_annotations=a.only_with_annotations)


if __name__ == '__main__':
    main()
