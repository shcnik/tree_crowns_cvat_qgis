"""Print band metadata and geospatial properties of a multispectral raster."""
import argparse
import rasterio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image')
a = parser.parse_args()
with rasterio.open(a.image) as ds:
    print('Size:', ds.width, 'x', ds.height)
    print('Number of bands:', ds.count)
    print('Band dtypes:', ds.dtypes)
    print('CRS:', ds.crs)
    print('Pixel resolution:', ds.res)
    print('Nodata:', ds.nodata)
    for i in ds.indexes:
        print(f'Band {i:02d}: description={ds.descriptions[i - 1]!r}, '
              f'interpretation={ds.colorinterp[i - 1]}, tags={ds.tags(i)}')
