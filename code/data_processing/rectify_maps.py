
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os

def recify_raster(input_raster, output_raster):
    
    with rasterio.open(input_raster) as src:
        # Get the source CRS, transform, and metadata
        src_crs = src.crs
        src_transform = src.transform
        src_meta = src.meta

        # Calculate the transform and dimensions for the destination
        dst_transform, width, height = calculate_default_transform(
            src_crs, src_crs, src.width, src.height, *src.bounds)

        # Update the metadata with the new transform and dimensions
        dst_meta = src_meta.copy()
        dst_meta.update({
            'crs': src_crs,
            'transform': dst_transform,
            'width': width,
            'height': height
        })

        # Open the destination file for writing
        with rasterio.open(output_raster, 'w', **dst_meta) as dst:
            # Reproject the raster to remove rotation
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=src_crs,
                    resampling=Resampling.nearest
                )

    print(f"Corrected raster saved to {output_raster}")
    return src_transform, dst_transform, src.width, src.height

def recify_raster_reverse(input_raster, src_transform, dst_transform, width, height, output_raster):
    
    with rasterio.open(input_raster) as src:
        # Get the source CRS, transform, and metadata
        src_crs = src.crs
        src_meta = src.meta

        # Update the metadata with the new transform and dimensions
        dst_meta = src_meta.copy()
        dst_meta.update({
            'crs': src_crs,
            'transform': dst_transform,
            'width': width,
            'height': height
        })
        
        # Open the destination file for writing
        with rasterio.open(output_raster, 'w', **dst_meta) as dst:
            # Reproject the raster to remove rotation
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=src_crs,
                    resampling=Resampling.nearest
                )
            
    print(f"Corrected raster saved to {output_raster}")


# we use 1878 as reference
reference_dir = '../../output/georeferenced/1878/map'
reference_files = os.listdir(reference_dir)
years = ['1888']
dirs= ['../../output/georeferenced/{}/map'.format(year) for year in years]

for i in range(len(reference_files)): 
    # we use 1878 as reference
    raster_1878 = rasterio.open(reference_dir + '/' + reference_files[i])
    name = reference_files[i].split('.')[0].split('_')[-1]

    
    image1 = np.transpose(raster_1878.read(), (1, 2, 0))
    if not os.path.exists('../../output/georeferenced/rectified/'):
        os.makedirs('../../output/georeferenced/rectified/')
    if not os.path.exists('../../output/georeferenced/rectified/{}'.format(name)):
        os.makedirs('../../output/georeferenced/rectified/{}'.format(name))
    plt.imsave('../../output/georeferenced/rectified/{}/{}_1878.png'.format(name, name), image1)
    
    for year, dir in zip(years, dirs):
        files = os.listdir(dir)
        src_transform1 = raster_1878.transform
        width1 = raster_1878.width
        height1 = raster_1878.height
        raster2 = rasterio.open(dir + '/' + files[i])
        src_transform2 = raster2.transform

        recify_raster_reverse(dir + '/' + files[i], src_transform2, src_transform1, width1, height1, '../../output/georeferenced/temp.tif') #rectified/{}_{}.tif'.format(name, year)
        src_transform1 = raster2.transform

        image2 = gdal.Open('../../output/georeferenced/temp.tif'.format(name, year))
        image2 = np.transpose(image2.ReadAsArray(), (1, 2, 0))
        plt.imsave('../../output/georeferenced/rectified/{}/{}_{}.png'.format(name, name, year), image2)
        # overlay = image1*0.5 + image2*0.5
        # plt.imshow(overlay.astype(int))
        # plt.show()
