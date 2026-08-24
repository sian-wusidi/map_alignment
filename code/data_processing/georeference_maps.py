import json
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from rasterio.mask import mask as m
import os
from rasterio.features import shapes
import geopandas
import cv2



year = '1888' # you do the same for other years
dirs = ".../../data/gcps/"
crs = CRS.from_string("EPSG:4326")

for filename in os.listdir(dirs):
    pixel_coords = []
    map_coords = []
    gcps = []
    year = filename.split('_')[3][:4]
    outdir = "../../output/georeferenced/" + str(year)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir + '/map'):
        os.makedirs(outdir + '/map')
    if not os.path.exists(outdir + '/mask'):
        os.makedirs(outdir + '/mask')
    if not os.path.exists(outdir + '/cropped'):
        os.makedirs(outdir + '/cropped')
    if not os.path.exists(outdir + '/mask_vect'):
        os.makedirs(outdir + '/mask_vect')

    if filename.startswith("BhdV") or filename.startswith("BHdV"):
        json_path = os.path.join(dirs, filename)
        with open(json_path) as json_file:
            data = json.load(json_file)
        image_path = os.path.join("../../data/maps/", filename.replace("json", "jpg"))
        image = Image.open(image_path)
        image_array = np.array(image)
        
        id = data['items'][0]['id']
        for feature in data['items'][0]['body']['features']:
            pix_coordx = feature['properties']['pixelCoords'][1]
            pix_coordy = feature['properties']['pixelCoords'][0]
            map_coordx = feature['geometry']['coordinates'][0]  # lon
            map_coordy = feature['geometry']['coordinates'][1]  # lat
            # map_coordx, map_coordy = convert_wgs84_to_lambert93(map_coordx, map_coordy)
            pixel_coords.append([pix_coordx, pix_coordy])
            map_coords.append([map_coordx, map_coordy])
            gcps.append(GCP(pix_coordx, pix_coordy, map_coordx, map_coordy))
        geotransform = from_gcps(gcps)
        output_tiff_path = outdir + "/map/" + filename.replace("json", "tiff")
        print(f'GeoTIFF saved to {output_tiff_path}')
        # Write the image data to a GeoTIFF file
        with rasterio.open(
            output_tiff_path,
            'w',
            driver='GTiff',
            height=image_array.shape[0],
            width=image_array.shape[1],
            count=3,  # Assuming the JPG is an RGB image
            dtype=image_array.dtype,
            crs=crs,
            transform=geotransform,
        ) as dst:
            for i in range(1, 4):  # Write each band (R, G, B)
                dst.write(image_array[:, :, i-1], i)    
        
        # read mask
        mask = geopandas.read_file('../../data/masks/bhdv_atlas_municipal_{}_masks.gpkg'.format(year), layer=id) # index is -1
        
        mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
        coords = np.asarray(mask.geometry.get_coordinates())
        coords = np.abs(coords)
        cv2.fillPoly(mask_array,[coords.astype(int)], color=(255, 255, 255))
        output_tiff_path = outdir + "/mask/" + filename.replace("json", "tiff")

        with rasterio.open(
            output_tiff_path,
            'w',
            driver='GTiff',
            height=mask_array.shape[0],
            width=mask_array.shape[1],
            count=1,  # Assuming the JPG is an RGB image
            dtype=mask_array.dtype,
            crs=crs,
            transform=geotransform,
        ) as dst:
            dst.write(mask_array[:, :], 1)

        # convert mask to polygon
        with rasterio.open(outdir + "/mask/" + filename.replace("json", "tiff")) as src:
            image = src.read(1)
            image = (image > 0).astype(np.uint8)
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(image, mask=image, transform=src.transform)))
        geoms = list(results)
        gdf = geopandas.GeoDataFrame.from_features(geoms, crs=src.crs)
        output_path = outdir + "/mask_vect/" + filename.replace("json", "shp")
        gdf.to_file(output_path)


        # crop the image
        output_tiff_path = outdir + "/cropped/" + filename.replace("json", "tiff")
        # save image cropped by mask into tif
        with rasterio.open(outdir + "/map/" + filename.replace("json", "tiff")) as src:
            out_image, out_transform = m(src, [gdf['geometry'][0]], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
                dest.write(out_image)
