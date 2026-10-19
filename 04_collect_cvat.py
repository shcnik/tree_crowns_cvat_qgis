"""Import reviewed COCO 1.0 annotations from CVAT, reconstruct georeferenced crowns.

Important: this stage cannot guess which annotation is the same object after a large
edit. It performs one-to-one geometry-IoU matching with seed GPKG, writes a QA
report, and flags unmatched or overlapping results for manual checking.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union

from crown_common import (bbox_center, decode_segmentation,
                          geo_to_pixel, pixel_to_geo)


def read_coco(file: Path):
    """CVAT COCO .json or .zip containing instances_*.json (single subset/task)."""
    if file.suffix.lower() == '.json':
        return json.loads(file.read_text(encoding='utf-8'))
    if file.suffix.lower() != '.zip':
        raise ValueError('Expected COCO .json or .zip export')
    with zipfile.ZipFile(file) as zf:
        names = [n for n in zf.namelist()
                 if Path(n).name.startswith('instances_') and n.endswith('.json')]
        if len(names) != 1:
            raise ValueError(f'Expected one instances_*.json in ZIP, found {names}. '
                             'Export one CVAT task, not a multi-subset project.')
        return json.loads(zf.read(names[0]).decode('utf-8'))


def mask_to_pixel_geometry(mask, x0, y0, *, min_part_area=1):
    geoms = [shape(g) for g, val in shapes(mask.astype('uint8'), mask=mask != 0,
             transform=Affine.translation(int(x0), int(y0)))
             if val == 1]
    geoms = [g for g in geoms if g.area >= min_part_area]
    if not geoms:
        return None
    return unary_union(geoms)


def load_seed_gpkg(path, crs):
    seed = gpd.read_file(path, layer='crowns')
    if seed.crs is None:
        raise ValueError('Seed GeoPackage has no CRS')
    if seed.crs != crs:
        seed = seed.to_crs(crs)
    if 'tree_id' not in seed:
        raise ValueError('Seed polygons require tree_id')
    if seed.tree_id.isna().any() or seed.tree_id.duplicated().any():
        raise ValueError('Seed tree IDs are not unique')
    return seed


def collect(cvat_annotations: Path, manifest_path: Path, seed_path: Path, output_gpkg: Path,
            *, match_iou=0.30, allow_missing_images=False, min_area=8):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported tile manifest schema')
    coco = read_coco(cvat_annotations)
    categories = {int(c['id']): c['name'] for c in coco['categories']}
    # Check all image names exactly: avoid silent loss of undetected trees / tile coverage.
    expected = {entry['file_name']: entry for entry in manifest['tiles']}
    images = {}
    for entry in coco['images']:
        name = Path(entry['file_name'].replace('\\', '/')).name
        if name in images:
            raise ValueError(f'Ambiguous repeated image basename: {name}')
        if name not in expected:
            raise ValueError(f'Unexpected image in CVAT COCO: {name}')
        tile = expected[name]
        if int(entry['width']) != tile['width'] or int(entry['height']) != tile['height']:
            raise ValueError(f'Wrong tile size for {name}: pixel georeferencing unsafe')
        images[name] = int(entry['id'])
    missing = set(expected) - set(images)
    if missing and not allow_missing_images:
        raise ValueError(f'COCO lacks {len(missing)} tile(s) from manifest, e.g. '
                         f'{sorted(missing)[:4]}; use --allow-missing-images only for partial exports.')
    by_id = {image_id: name for name, image_id in images.items()}
    transform = Affine(*manifest['raster']['transform'])
    crs = manifest['raster']['crs']
    if crs is None:
        raise ValueError('Manifest lacks CRS')
    seed = load_seed_gpkg(seed_path, crs)
    manifest_seed_ids = {str(s['tree_id']) for s in manifest.get('seeds', [])}
    if not manifest_seed_ids.issubset(set(seed.tree_id.astype(str))):
        raise ValueError('Seed GeoPackage does not contain all source tree IDs from manifest')
    pixel_seed = [geo_to_pixel(g, transform) for g in seed.geometry]
    tree_ids = [str(v) for v in seed.tree_id.tolist()]

    # Seeds assigned only by bbox center to the tile's core, independent of
    # edited/deleted CVAT annotations and regardless of geometry intersections.
    owner_by_id = {}
    core = int(manifest['core_size'])
    num_cols = (int(manifest['raster']['width']) + core - 1) // core
    name_by_rc = {(t['row'], t['col']): t['file_name'] for t in manifest['tiles']}
    for i, geometry in enumerate(pixel_seed):
        cx, cy = bbox_center(geometry)
        row, col = int(cy // core), int(cx // core)
        owner_by_id[tree_ids[i]] = name_by_rc.get((row, col))

    seed_by_tile = defaultdict(list)
    for i, tree_id in enumerate(tree_ids):
        seed_by_tile[owner_by_id[tree_id]].append(i)

    report = {'images_in_export': len(images), 'images_missing': sorted(missing),
              'ignored_non_tree': [], 'ignored_no_segmentation': [],
              'ignored_outside_core': [], 'ignored_tiny_or_empty': [],
              'duplicate_mask_warnings': [], 'unmatched_seed_ids': [],
              'matched': [], 'new_ids': [], 'preserved_unreviewed': []}
    candidates = []
    annotation_ids = [int(ann['id']) for ann in coco['annotations']]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError('COCO annotation IDs must be unique')
    for ann in coco['annotations']:
        image_id = int(ann['image_id'])
        if image_id not in by_id:
            raise ValueError(f'Annotation refers to missing image id={image_id}')
        name = by_id[image_id]
        if categories.get(int(ann['category_id'])) != 'tree':
            report['ignored_non_tree'].append(int(ann['id']))
            continue
        segmentation = ann.get('segmentation')
        if not segmentation:
            # A bbox is NOT a tree crown; do not silently turn it into a mask.
            report['ignored_no_segmentation'].append(int(ann['id']))
            continue
        tile = expected[name]
        mask = decode_segmentation(segmentation, tile['height'], tile['width'])
        area = int(mask.sum())
        if area < min_area:
            report['ignored_tiny_or_empty'].append(int(ann['id']))
            continue
        geom = mask_to_pixel_geometry(mask, tile['x'], tile['y'])
        if geom is None or geom.is_empty:
            report['ignored_tiny_or_empty'].append(int(ann['id']))
            continue
        cx, cy = bbox_center(geom)
        if not (tile['core_x0'] <= cx < tile['core_x1'] and
                tile['core_y0'] <= cy < tile['core_y1']):
            report['ignored_outside_core'].append(
                {'annotation_id': int(ann['id']), 'image': name})
            continue
        flags = []
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            flags.append('touches_image_edge')
        candidates.append({'tile': name, 'ann_id': int(ann['id']),
                           'pixel_geom': geom, 'flags': flags, 'area_px': area})

    # One-to-one matching; seeds and all candidate geometries are in source pixel coordinates.
    proposals = []
    for i, cand in enumerate(candidates):
        for si in seed_by_tile[cand['tile']]:
            old_geom = pixel_seed[si]
            if not cand['pixel_geom'].intersects(old_geom):
                continue
            intersection = cand['pixel_geom'].intersection(old_geom).area
            union = cand['pixel_geom'].area + old_geom.area - intersection
            iou = intersection / union if union > 0 else 0
            if iou >= match_iou:
                proposals.append((iou, i, si))
    proposals.sort(reverse=True)
    assignments = {}
    used_seed = set()
    for iou, ci, si in proposals:
        if ci in assignments or si in used_seed:
            continue
        assignments[ci] = (si, iou)
        used_seed.add(si)

    existing_ids = set(tree_ids)
    new_id_counter = max((int(m.group(1)) for t in existing_ids
                          if (m := re.fullmatch(r'T(\d+)', t))), default=0) + 1
    rows = []
    for i, cand in enumerate(candidates):
        if i in assignments:
            si, iou = assignments[i]
            tree_id = tree_ids[si]
            report['matched'].append({'tree_id': tree_id, 'annotation_id': cand['ann_id'],
                                      'iou': round(iou, 4)})
            original = seed.iloc[si]
            score = original.get('detect_score', None)
            sam_score = original.get('sam_score', None)
            flags = cand['flags']
        else:
            while f'T{new_id_counter:06d}' in existing_ids:
                new_id_counter += 1
            tree_id = f'T{new_id_counter:06d}'
            existing_ids.add(tree_id)
            new_id_counter += 1
            score, sam_score = None, None
            flags = cand['flags'] + ['new_in_cvat']
            report['new_ids'].append(tree_id)
        if 'touches_image_edge' in flags:
            status = 'review'
        else:
            status = 'edited'
        rows.append({'tree_id': tree_id, 'status': status,
                     'detect_score': score, 'sam_score': sam_score,
                     'area_px': cand['area_px'], 'review_needed': int(bool(flags)),
                     'flags': ','.join(flags), 'tile_name': cand['tile'],
                     'cvat_ann_id': cand['ann_id'],
                     'geometry': pixel_to_geo(cand['pixel_geom'], transform)})

    # Seeds from tiles absent in reviewed package remain untouched, to prevent
    # a test/partial roundtrip from deleting all other original detections.
    completed = set(images)
    for si, tree_id in enumerate(tree_ids):
        owner = owner_by_id[tree_id]
        if owner in completed:
            if si not in used_seed:
                report['unmatched_seed_ids'].append(tree_id)
            continue
        original = seed.iloc[si]
        rows.append({'tree_id': tree_id, 'status': str(original.get('status', 'pending')),
                     'detect_score': original.get('detect_score', None),
                     'sam_score': original.get('sam_score', None),
                     'area_px': original.get('area_px', None),
                     'review_needed': 1, 'flags': 'not_reviewed', 'tile_name': owner or '',
                     'cvat_ann_id': None, 'geometry': original.geometry})
        report['preserved_unreviewed'].append(tree_id)

    if not rows:
        raise ValueError('No crowns to export: check labels, segmentations, and task completeness')
    out = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)
    if out.tree_id.duplicated().any():
        raise AssertionError('Duplicate tree IDs after import')
    out_sindex = out.sindex
    for i, geometry in enumerate(out.geometry):
        for j in out_sindex.query(geometry, predicate='intersects'):
            j = int(j)
            if j <= i:
                continue
            other = out.geometry.iloc[j]
            intersection = geometry.intersection(other).area
            if intersection <= 0:
                continue
            ratio = intersection / min(geometry.area, other.area)
            if ratio > 0.20:
                report['duplicate_mask_warnings'].append({
                    'first': out.tree_id.iloc[i], 'second': out.tree_id.iloc[j],
                    'overlap_fraction_smaller': round(ratio, 4)})
                for k in (i, j):
                    out.at[k, 'review_needed'] = 1
                    flags = out.at[k, 'flags']
                    if 'overlap' not in flags.split(','):
                        out.at[k, 'flags'] = (flags + ',overlap').strip(',')
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()
    out.to_file(output_gpkg, layer='crowns', driver='GPKG')
    report['crowns_output'] = len(out)
    report['deleted_in_review_count'] = len(report['unmatched_seed_ids'])
    report['new_in_review_count'] = len(report['new_ids'])
    report_path = output_gpkg.with_name(output_gpkg.stem + '_qa.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(out)} crowns: {output_gpkg}')
    print(f'Matched: {len(report["matched"])}, new: {len(report["new_ids"])}, '
          f'not present in reviewed imagery: {len(report["preserved_unreviewed"])}, '
          f'unmatched seeds in reviewed imagery: {len(report["unmatched_seed_ids"])}')
    print(f'QA: {report_path}')
    return out, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cvat_coco', type=Path, help='COCO exported from CVAT: ZIP or JSON')
    p.add_argument('tile_manifest', type=Path)
    p.add_argument('seed_gpkg', type=Path)
    p.add_argument('out_gpkg', type=Path)
    p.add_argument('--match-iou', type=float, default=.30)
    p.add_argument('--allow-missing-images', action='store_true')
    p.add_argument('--min-area', type=int, default=8)
    a = p.parse_args()
    if not 0 <= a.match_iou <= 1:
        p.error('match-iou must be in [0, 1]')
    collect(a.cvat_coco, a.tile_manifest, a.seed_gpkg, a.out_gpkg,
            match_iou=a.match_iou, allow_missing_images=a.allow_missing_images,
            min_area=a.min_area)


if __name__ == '__main__':
    main()
