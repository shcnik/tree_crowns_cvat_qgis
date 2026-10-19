"""Export corrected crown polygons to PNG masks, contours, RGB and lossless multispectral GeoTIFF.

Reads ONLY the window around each tree, never the full orthomosaic.
The bands/ PNG files contain exact values for uint8/uint16 inputs; for any
other dtype, visual previews are emitted, while exact values stay in TIFF.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.errors import WindowError
from rasterio.features import geometry_window, rasterize
from shapely.affinity import affine_transform
from shapely.geometry import mapping
from tqdm import tqdm


def safe_id(value, fallback):
    value = str(value).strip() if value is not None else ''
    if value.lower() in ('', 'none', 'nan'):
        value = fallback
    return re.sub(r'[^A-Za-z0-9_-]', '_', value)


def rgb_uint8(raw, rgb_indices, stretch):
    parts = []
    for band, (low, high) in zip(rgb_indices, stretch):
        values = raw[band - 1].astype('float32')
        scaled = 255 * (np.nan_to_num(values, nan=low, posinf=high, neginf=low) - low) / (high - low)
        parts.append(np.clip(scaled, 0, 255).astype('uint8'))
    return np.stack(parts, axis=-1)


def write_band_png(band, target):
    """Return whether saved values are exactly unchanged."""
    if band.dtype == np.dtype('uint8') or band.dtype == np.dtype('uint16'):
        Image.fromarray(band).save(target)
        return True
    # For other types, only a preview; raw values remain lossless in .tif.
    data = band.astype('float64')
    finite = data[np.isfinite(data)]
    if not finite.size:
        preview = np.zeros(band.shape, dtype='uint16')
    else:
        mn, mx = float(finite.min()), float(finite.max())
        preview = np.clip(
            65535 * (np.nan_to_num(data, nan=mn) - mn) / max(mx - mn, 1e-12),
            0, 65535
        ).astype('uint16')
    Image.fromarray(preview).save(target.with_name(target.stem + '_preview.png'))
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('crowns_gpkg', type=Path, help='QGIS-corrected GPKG')
    p.add_argument('output_dir', type=Path)
    p.add_argument('--stretch', required=True, type=Path,
                   help='*_stretch.json emitted by 01_detect_and_segment.py')
    p.add_argument('--pad', type=int, default=16, help='context pixels around crown')
    p.add_argument('--manifest', type=Path, help='optional CVAT manifest to validate raster identity')
    p.add_argument('--approved-only', action='store_true',
                   help='only export status=approved or status=edited')
    p.add_argument('--limit', type=int, default=0, help='export only first N trees for testing')
    args = p.parse_args()

    stretch_config = json.loads(args.stretch.read_text(encoding='utf-8'))
    rgb = [int(x) for x in stretch_config['rgb_bands_1_based']]
    limits = stretch_config['limits_2_98']
    gdf = gpd.read_file(args.crowns_gpkg, layer='crowns')
    if gdf.crs is None:
        raise ValueError('GPKG has no CRS.')
    if 'status' in gdf:
        gdf = gdf[~gdf['status'].fillna('').str.lower().isin(['reject', 'rejected', 'exclude'])]
        if args.approved_only:
            gdf = gdf[gdf['status'].fillna('').str.lower().isin(['approved', 'edited'])]
    elif args.approved_only:
        raise ValueError('No status field for --approved-only.')
    if args.limit:
        gdf = gdf.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.image) as ds:
        if args.manifest:
            from crown_common import validate_signature
            validate_signature(ds, json.loads(args.manifest.read_text(encoding='utf-8'))['raster'])
        if ds.crs is None:
            raise ValueError('GeoTIFF has no CRS.')
        if max(rgb) > ds.count:
            raise ValueError('RGB indices refer to nonexistent band.')
        if gdf.crs != ds.crs:
            gdf = gdf.to_crs(ds.crs)

        seen = set()
        summary = []
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc='Exporting crowns'):
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                print(f'WARNING: skipping invalid crown {idx}')
                continue
            label = safe_id(row.get('tree_id', None), f'T{int(idx) + 1:06d}')
            if label in seen:
                raise ValueError(f'Duplicate tree_id: {label}')
            seen.add(label)
            try:
                window = geometry_window(ds, [mapping(geom)], pad_x=args.pad, pad_y=args.pad)
            except (WindowError, ValueError):
                print(f'WARNING: crown {label} does not intersect raster.')
                continue
            window = window.round_offsets().round_lengths()
            raw = ds.read(window=window)  # ALL original spectral channels
            _, h, w = raw.shape
            transform = ds.window_transform(window)
            mask = rasterize([(mapping(geom), 1)], out_shape=(h, w),
                             transform=transform, fill=0, dtype='uint8')
            if not np.any(mask):
                print(f'WARNING: empty rasterized mask for {label}')
                continue
            folder = args.output_dir / label
            bands_dir = folder / 'bands'
            bands_dir.mkdir(parents=True, exist_ok=True)
            rgb_image = rgb_uint8(raw, rgb, limits)
            Image.fromarray(rgb_image).save(folder / 'rgb.png')
            Image.fromarray(mask * 255).save(folder / 'mask.png')

            overlay = rgb_image.copy()
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  # RGB red
            Image.fromarray(overlay).save(folder / 'contour.png')
            masked_rgb = rgb_image.copy()
            masked_rgb[mask == 0] = 0
            Image.fromarray(masked_rgb).save(folder / 'masked_rgb.png')

            tif_path = folder / 'multispectral.tif'
            # Original values and dtype preserved, all bands, with CRS and affine transform.
            profile = dict(driver='GTiff', width=w, height=h, count=ds.count,
                           dtype=raw.dtype, crs=ds.crs, transform=transform,
                           compress='deflate', nodata=ds.nodata)
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(raw)
                for band_i, desc in enumerate(ds.descriptions, start=1):
                    if desc:
                        dst.set_band_description(band_i, desc)

            exact_bands = []
            for b in range(ds.count):
                band_path = bands_dir / f'band_{b + 1:02d}.png'
                exact_bands.append(write_band_png(raw[b], band_path))

            # Affine: Shapely expects (a,b,d,e,xoff,yoff) for x'=a*x+b*y+xoff.
            inverse = ~transform
            local_geom = affine_transform(geom, [inverse.a, inverse.b,
                                                 inverse.d, inverse.e,
                                                 inverse.c, inverse.f])
            meta = {
                'tree_id': label,
                'status': str(row.get('status', '')),
                'source_tif': str(args.image),
                'crs': ds.crs.to_string(),
                'source_pixel_window_col_row_width_height': [int(window.col_off),
                    int(window.row_off), w, h],
                'crop_affine_coefficients': list(transform)[:6],
                'number_of_bands': ds.count,
                'bands_1_based': list(range(1, ds.count + 1)),
                'band_descriptions': list(ds.descriptions),
                'rgb_bands_1_based': rgb,
                'per_band_png_contains_exact_values': exact_bands,
                'geographic_polygon': mapping(geom),
                'polygon_in_crop_pixel_coordinates': mapping(local_geom),
                'mask_convention': '255=inside crown, 0=outside',
            }
            (folder / 'metadata.json').write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.append([label, w, h, int(mask.sum()), int(all(exact_bands))])

    import csv
    with (args.output_dir / 'index.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tree_id', 'width', 'height', 'mask_pixels', 'all_band_png_exact'])
        writer.writerows(summary)
    print(f'Exported {len(summary)} crowns into {args.output_dir}')


if __name__ == '__main__':
    main()
