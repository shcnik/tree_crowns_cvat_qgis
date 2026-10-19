"""Network- and ML-free end-to-end tests using a synthetic 7-band GeoTIFF."""
import importlib.util
import sys
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), ROOT / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_codec():
    common = load('crown_common.py')
    for mask in [np.zeros((12, 15), dtype='uint8'), np.ones((12, 15), dtype='uint8'),
                 np.array([[1, 0, 1], [0, 1, 1]], dtype='uint8')]:
        assert np.array_equal(mask, common.rle_decode(common.rle_encode(mask), *mask.shape))
    # C reference compresses first run 2 to '2', one foreground pixel '1' etc.
    assert common.rle_counts_from_compressed('213') == [2, 1, 3]


def make_synthetic(tmp_path):
    tif = tmp_path / 'synthetic_7band.tif'
    tx = Affine(0.075, 0.004, 400000, 0.002, -0.075, 5800000)
    rng = np.random.default_rng(4)
    raw = rng.integers(5, 6000, size=(7, 128, 133), dtype='uint16')
    with rasterio.open(tif, 'w', driver='GTiff', width=133, height=128,
                       count=7, dtype='uint16', transform=tx, crs='EPSG:32632') as ds:
        ds.write(raw)
    pix_poly = [box(20, 17, 45, 43), box(70, 72, 107, 105),
                Polygon([(47, 7), (72, 7), (74, 25), (50, 25)])]
    geoms = [affine_transform(g, [tx.a, tx.b, tx.d, tx.e, tx.c, tx.f]) for g in pix_poly]
    gdf = gpd.GeoDataFrame([
        {'tree_id': f'T{i+1:06d}', 'status': 'pending', 'detect_score': .75,
         'sam_score': .8, 'geometry': g}
        for i, g in enumerate(geoms)], crs='EPSG:32632')
    seed = tmp_path / 'seed.gpkg'
    gdf.to_file(seed, layer='crowns', driver='GPKG')
    stretch = tmp_path / 'stretch.json'
    stretch.write_text(json.dumps({'rgb_bands_1_based': [3, 2, 1],
                                   'limits_2_98': [[0, 6000]] * 3}), encoding='utf-8')
    return tif, seed, stretch


def test_cvat_roundtrip(tmp_path):
    prep = load('03_prepare_cvat.py')
    imp = load('04_collect_cvat.py')
    tif, gpkg, stretch = make_synthetic(tmp_path)
    output = tmp_path / 'cvat'
    manifest, coco = prep.prepare(tif, gpkg, stretch, output, core=64, margin=32)
    assert len(coco['images']) == 6  # 3 cols * 2 rows
    assert len(coco['annotations']) == 3
    assert len(list((output / 'images').glob('*.png'))) == 6
    with zipfile.ZipFile(output / 'cvat_images.zip') as zf:
        assert len(zf.namelist()) == 6
    with zipfile.ZipFile(output / 'cvat_annotations.zip') as zf:
        assert 'annotations/instances_default.json' in zf.namelist()

    # Exact mask RLE reconstruction, including partial imagery + reversed axis projection.
    with rasterio.open(tif) as ds:
        first = coco['annotations'][0]
        tile = {i['id']: i for i in coco['images']}[first['image_id']]
        assert first['area'] > 0
        mask = load('crown_common.py').rle_decode(
            first['segmentation'], tile['height'], tile['width'])
        assert int(mask.sum()) == first['area']
        assert tile['width'] <= 128
    # Simulate CVAT: remove tree T000001, add one tree, keep the rest.
    removed = coco['annotations'][0]
    coco['annotations'] = coco['annotations'][1:]
    img = next(i for i in coco['images'] if i['id'] == removed['image_id'])
    from crown_common import rle_encode
    new_mask = np.zeros((img['height'], img['width']), dtype='uint8')
    new_mask[9:19, 12:25] = 1
    coco['annotations'].append({'id': 10000, 'image_id': img['id'], 'category_id': 1,
                                'segmentation': rle_encode(new_mask),
                                'area': int(new_mask.sum()), 'bbox': [12, 9, 13, 10],
                                'iscrowd': 1})
    # The added tree must be centered in the tile's owning core.
    t = next(t for t in manifest['tiles'] if t['file_name'] == img['file_name'])
    xmin = t['core_x0'] - t['x']
    ymin = t['core_y0'] - t['y']
    new_mask[:] = 0
    new_mask[ymin+4:ymin+14, xmin+4:xmin+14] = 1
    coco['annotations'][-1]['segmentation'] = rle_encode(new_mask)
    coco['annotations'][-1]['area'] = int(new_mask.sum())
    edited_path = tmp_path / 'reviewed.json'
    edited_path.write_text(json.dumps(coco), encoding='utf-8')
    corrected = tmp_path / 'corrected.gpkg'
    out, report = imp.collect(edited_path, output / 'tile_manifest.json', gpkg, corrected,
                              match_iou=.25)
    assert corrected.exists()
    assert removed['attributes']['tree_id'] in report['unmatched_seed_ids']
    assert len(report['matched']) == 2
    assert len(report['new_ids']) == 1
    assert len(out) == 3
    assert out.crs.to_epsg() == 32632
    assert out.tree_id.is_unique
    assert (tmp_path / 'corrected_qa.json').exists()

    # Downstream original multispectral exporter accepts round-tripped GPKG.
    export = load('02_export_dataset.py')
    import subprocess, sys
    cp = subprocess.run([sys.executable, str(ROOT / '02_export_dataset.py'),
                         str(tif), str(corrected), str(tmp_path / 'dataset'),
                         '--stretch', str(stretch), '--pad', '8'],
                        capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    folders = list((tmp_path / 'dataset').glob('T*'))
    assert len(folders) == 3
    for folder in folders:
        with rasterio.open(folder / 'multispectral.tif') as f:
            assert f.count == 7 and f.dtypes[0] == 'uint16'
            assert (f.read(1) == np.asarray(Image.open(folder / 'bands/band_01.png'))).all()
        rgb = np.asarray(Image.open(folder / 'rgb.png'))
        mask = np.asarray(Image.open(folder / 'mask.png'))
        outline = np.asarray(Image.open(folder / 'contour.png'))
        assert rgb.shape[:2] == mask.shape == outline.shape[:2]


def test_partial_keeps_unreviewed(tmp_path):
    prep = load('03_prepare_cvat.py')
    imp = load('04_collect_cvat.py')
    tif, gpkg, stretch = make_synthetic(tmp_path)
    output = tmp_path / 'partial'
    manifest, coco = prep.prepare(tif, gpkg, stretch, output,
                                  core=64, margin=32, max_tiles=1)
    path = output / 'instances_default.json'
    out, report = imp.collect(path, output / 'tile_manifest.json', gpkg,
                              tmp_path / 'partial.gpkg')
    assert len(out) == 3
    assert len(report['preserved_unreviewed']) >= 1
