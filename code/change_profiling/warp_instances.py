import cv2
import numpy as np
import os

def warp_instance_map(instance_map, optical_flow):
    # Get the shape of the instance map
    height, width = instance_map.shape[:2]

    # Create a grid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    # Apply the optical flow to the grid
    warped_grid_x = grid_x + optical_flow[..., 0]
    warped_grid_y = grid_y + optical_flow[..., 1]

    # Warp the instance map using the warped grid
    warped_instance_map = cv2.remap(instance_map, warped_grid_x, warped_grid_y, cv2.INTER_NEAREST)

    return warped_instance_map

basedir = "../output/georeferenced/"
flowdir = '../output/flow_predictions/'  
savedir = '../output/instance_predictions/'

names = os.listdir(basedir + 'rectified/')
years = ['1868a', 1878, 1888, 1900, '1912b', 1925, 1937]
for name in names:
    for i in range(len(years)-1):
        src_year = years[i]
        tgt_year = years[i+1]
        if os.path.exists(savedir + 'instance_seg_{}_{}_{}_warp.npy'.format(name, src_year, tgt_year)):
            print('Skipping', name, src_year, tgt_year)
            continue
        # Load the instance map and optical flow from numpy files
        instance_map = np.load(basedir + 'instance_predictions/instance_seg_{}_{}.png.npy'.format(name, src_year))
        optical_flow = np.load(flowdir + '{}/{}_{}/test_evaluation/global_optical_flow_ts.npy'.format(name, src_year, tgt_year))
        # Warp the instance map using the optical flow
        warped_instance_map = warp_instance_map(instance_map, optical_flow)
        # Save the warped instance map to a file
        np.save(savedir + 'instance_seg_{}_{}_{}_warp.npy'.format(name, src_year, tgt_year), warped_instance_map)