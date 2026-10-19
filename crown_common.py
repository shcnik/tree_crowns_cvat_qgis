"""Shared raster/CVAT/COCO helpers; only common geospatial and image dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    file_name: str
    x: int
    y: int
    width: int
    height: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int

    @property
    def window(self):
        return Window(self.x, self.y, self.width, self.height)

    def owns(self, x, y):
        return self.core_x0 <= x < self.core_x1 and self.core_y0 <= y < self.core_y1

    def to_dict(self):
        return vars(self).copy()


def make_tiles(width: int, height: int, core: int, margin: int):
    if core <= 0 or margin < 0:
        raise ValueError('core must be positive and margin nonnegative')
    for cy in range(0, height, core):
        for cx in range(0, width, core):
            x = max(0, cx - margin)
            y = max(0, cy - margin)
            x1 = min(width, cx + core + margin)
            y1 = min(height, cy + core + margin)
            yield Tile(cy // core, cx // core, f'tile_r{cy // core:05d}_c{cx // core:05d}.png',
                       x, y, x1 - x, y1 - y, cx, cy,
                       min(cx + core, width), min(cy + core, height))


def bbox_center(geom):
    x0, y0, x1, y1 = geom.bounds
    return (x0 + x1) / 2, (y0 + y1) / 2


def geo_to_pixel(geom, transform):
    from shapely.affinity import affine_transform
    inv = ~transform
    return affine_transform(geom, [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f])


def pixel_to_geo(geom, transform):
    from shapely.affinity import affine_transform
    return affine_transform(geom, [transform.a, transform.b,
                                   transform.d, transform.e, transform.c, transform.f])


def rgb_uint8(raw, rgb_indices, limits):
    parts = []
    for i, (lo, hi) in zip(rgb_indices, limits):
        vals = raw[i - 1].astype('float32')
        vals = np.nan_to_num(vals, nan=lo, posinf=hi, neginf=lo)
        parts.append(np.clip((vals - lo) * 255. / (hi - lo), 0, 255).astype('uint8'))
    return np.stack(parts, axis=-1)


def load_stretch(path, count):
    conf = json.loads(Path(path).read_text(encoding='utf-8'))
    bands = [int(v) for v in conf['rgb_bands_1_based']]
    limits = conf['limits_2_98']
    if len(bands) != 3 or len(limits) != 3 or min(bands) < 1 or max(bands) > count:
        raise ValueError('invalid RGB band mapping/stretch for raster')
    if any(hi <= lo for lo, hi in limits):
        raise ValueError('stretch upper bound must exceed lower bound')
    return bands, limits


def raster_signature(ds):
    return {'width': ds.width, 'height': ds.height, 'band_count': ds.count,
            'crs': ds.crs.to_string() if ds.crs else None,
            'transform': list(ds.transform)[:6]}


def validate_signature(ds, signature):
    got = raster_signature(ds)
    if (got['width'] != signature['width'] or got['height'] != signature['height']
            or got['crs'] != signature['crs'] or got['band_count'] != signature['band_count']
            or not np.allclose(got['transform'], signature['transform'], rtol=0, atol=1e-9)):
        raise ValueError('Current raster does not match CVAT tile manifest; refusing georeferencing')


def rle_encode(binary):
    """Uncompressed COCO RLE, Fortran/column-major, starts with background run."""
    h, w = binary.shape
    flat = np.asarray(binary, dtype='uint8').ravel(order='F')
    if not np.isin(flat, [0, 1]).all():
        raise ValueError('COCO mask must be binary')
    changes = np.flatnonzero(np.diff(flat) != 0) + 1
    starts = np.r_[0, changes, flat.size]
    lengths = np.diff(starts).astype(int).tolist()
    if flat.size and flat[0] == 1:
        lengths.insert(0, 0)
    return {'size': [int(h), int(w)], 'counts': lengths}


def rle_counts_from_compressed(s):
    """Read COCO compressed RLE (maskApi.c rleFrString) without pycocotools."""
    if isinstance(s, bytes):
        s = s.decode('ascii')
    counts = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        while True:
            c = ord(s[p]) - 48
            p += 1
            if c < 0 or c > 63:
                raise ValueError('invalid compressed COCO RLE character')
            x |= (c & 0x1f) << (5 * k)
            k += 1
            if not (c & 0x20):
                if c & 0x10:
                    x |= -1 << (5 * k)
                break
        if len(counts) > 2:
            x += counts[-2]
        if x < 0:
            raise ValueError('negative RLE run')
        counts.append(x)
    return counts


def rle_decode(segmentation, height, width):
    """Decode COCO compressed or uncompressed RLE to uint8 HxW mask."""
    h, w = [int(v) for v in segmentation['size']]
    if (h, w) != (height, width):
        raise ValueError(f'RLE size {(h, w)} does not match image {(height, width)}')
    counts = segmentation['counts']
    if isinstance(counts, (str, bytes)):
        counts = rle_counts_from_compressed(counts)
    if any(not isinstance(v, int) or v < 0 for v in counts):
        raise ValueError('invalid RLE counts')
    if sum(counts) != h * w:
        raise ValueError(f'RLE sums to {sum(counts)}, expected {h * w}')
    flat = np.zeros(h * w, dtype='uint8')
    pos = 0
    for idx, run in enumerate(counts):
        if idx % 2:
            flat[pos:pos + run] = 1
        pos += run
    return flat.reshape((h, w), order='F')


def decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, dict):
        return rle_decode(segmentation, height, width)
    # COCO polygons are lists of flattened x,y lists; use pixel-center rasterization.
    import cv2
    out = np.zeros((height, width), dtype='uint8')
    for poly in segmentation:
        pts = np.asarray(poly, dtype='float64')
        if pts.size < 6 or pts.size % 2:
            continue
        xy = np.rint(pts.reshape(-1, 2)).astype('int32')
        cv2.fillPoly(out, [xy], color=1)
    return out


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
