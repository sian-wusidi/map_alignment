import sys
sys.path.insert(1, '../')
import coco_pano_ext_demo_new as coco
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.affinity import affine_transform
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Configuration
name = '0002'
src_year = '1888'
tgt_year = '1900'
basedir = "../../data/aligned_instances/"
simplification_size = 4

print('Processing', name, src_year, tgt_year)

# Load instance maps
source_gt_warped = np.load(basedir + "instance_seg_{}_{}_{}_warp.npy".format(name, src_year, tgt_year))
target_gt = np.load(basedir + "instance_seg_{}_{}.npy".format(name, tgt_year))

# Create output directory
savedir = '../output/building_blocks_output/'
os.makedirs(savedir + 'colormaps/', exist_ok=True)
os.makedirs(savedir + 'vectors/', exist_ok=True)

# Load building blocks
building_blocks = cv2.imread(basedir + 'building_blocks_{}_{}.png'.format(name, tgt_year), cv2.IMREAD_GRAYSCALE)



# Calculate IoU (precision and recall)
T = (target_gt + 1).astype(int)
P = (source_gt_warped + 1).astype(int)

print('T shape:', T.shape, 'T type:', T.dtype)
print('P shape:', P.shape, 'P type:', P.dtype)

precision, recall = coco.precision_recall_only(T, P, mode='labelmap')

# Initialize arrays for aggregation
recall_per_building_blocks = np.zeros_like(building_blocks, dtype=np.float32)
results = np.zeros_like(building_blocks, dtype=np.uint8)

# Get connected components
num_labels, labels_im = cv2.connectedComponents(building_blocks.astype(np.uint8), connectivity=8)
geometries = []
ids = []
ious = []

print(f"Processing {num_labels - 1} building blocks...")

# Process each building block
for i in tqdm(range(1, num_labels)):
    # Get the mask of the current connected component
    component_mask = (labels_im == i).astype(np.uint8)
    
    # Calculate the recall for the current connected component, excluding roads (-1)
    valid_mask = (component_mask == 1) & (recall != -1)
    recall_component = np.mean(recall[valid_mask])
    if np.isnan(recall_component):
        recall_component = -1
    recall_per_building_blocks[component_mask == 1] = recall_component
    
    # Extract contours
    contours = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    if len(contours) == 0:
        continue
    
    for con in contours:
        if len(con) < 3:
            continue
        
        cntr = con
        cv2.drawContours(results, [cntr], 0, 255, 2)
        
        polygon = Polygon(cntr[:, 0, :])
        
        if not polygon.is_valid:
            print(f"Fixing geometry for component {i}")
            polygon = polygon.buffer(0)
        
        if polygon.is_valid and not polygon.is_empty and polygon.area >= 200:
            geometries.append(polygon.simplify(simplification_size))
            ids.append(i)
            ious.append(recall_component)

# Save colormap visualization
plt.figure(figsize=(10, 10))
plt.imshow(recall_per_building_blocks, cmap='hot', vmin=0, vmax=1)
plt.colorbar()
plt.title(f'IoU per Building Block - {name} {tgt_year}')
plt.savefig(savedir + 'colormaps/iou_per_building_blocks_{}_{}.png'.format(name, tgt_year), dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved colormap with {len(geometries)} valid building blocks")

# Load reference TIFF for georeferencing (update path as needed)
reference_tif_path = basedir + "reference_{}_{}.tiff".format(name, tgt_year)

# If reference TIFF exists, apply georeferencing
if os.path.exists(reference_tif_path):
    with rasterio.open(reference_tif_path) as ref_tif:
        transform = ref_tif.transform
        reference_crs = ref_tif.crs
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': geometries, 
        'id': ids, 
        'iou_{}'.format(src_year): ious
    }, crs=reference_crs)
    
    # Apply affine transformation
    affine_matrix = [transform.a, transform.b, transform.d, transform.e, transform.xoff, transform.yoff]
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: affine_transform(geom, affine_matrix))
    
    # Transform to Lambert (EPSG:2154)
    gdf_lambert = gdf.to_crs(epsg=2154)
    
    # Save shapefile
    gdf_lambert.to_file(savedir + "vectors/building_blocks_{}_{}.shp".format(name, tgt_year))
    print(f"Saved georeferenced shapefile")
else:
    # Create GeoDataFrame without georeferencing (pixel coordinates)
    gdf = gpd.GeoDataFrame({
        'geometry': geometries, 
        'id': ids, 
        'iou_{}'.format(src_year): ious
    })
    
    # Save as GeoJSON (works without CRS)
    gdf.to_file(savedir + "vectors/building_blocks_{}_{}.geojson".format(name, tgt_year), driver='GeoJSON')
    print(f"Saved shapefile (pixel coordinates, no georeferencing)")

print('Processing complete!')
print(f'Total building blocks: {len(geometries)}')
print(f'Mean IoU: {np.mean([iou for iou in ious if iou != -1]):.3f}')