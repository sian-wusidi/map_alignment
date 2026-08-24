import sys
sys.path.insert(1, '../')
import coco_pano_ext_demo_new as coco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import os

def show_precision_recall(t, GT, Pred):
    precision, recall = coco.precision_recall_maps(GT, Pred, lower_bound=t)
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,4))
    ax1.imshow(precision)
    ax1.set_title("Precision")
    ax2.imshow(recall)
    ax2.set_title("Recall")
    f.colorbar(matplotlib.cm.ScalarMappable(cmap=coco._cmap), ax=[ax1, ax2])
    plt.show()

def get_small_connected_components(image, min_size):
    image = image.astype(np.int32)
    image_areas = np.zeros_like(image)
    # Compute areas for ground truth
    areas_counts = np.bincount(image.flatten())
    nonzeros = image > 0  # Mask to skip zero values in gt
    image_areas[nonzeros] = areas_counts[image[nonzeros]]
    image_small = np.where(image_areas <= min_size, image, 0)
    labels, image_mapping = np.unique(image_small, return_inverse=True)
    image_small = image_mapping.reshape(image_small.shape)
    image_small = (image_small > 0).astype(int)
    return image_small

def return_precision_recall(t, GT, Pred, savedir, img_name):
    precision, recall = coco.precision_recall_only(GT, Pred, mode='labelmap')
    modify_A = (precision <= t) & (precision > -1)  # remove roads at -1
    modify_B = (recall <= t) & (recall > -1)
    small_instance_GT = get_small_connected_components(GT, 55*55)
    small_instance_Pred = get_small_connected_components(Pred, 55*55)
    modify_A = modify_A & ~small_instance_Pred
    modify_B = modify_B & ~small_instance_GT
    change_map = modify_A | modify_B
    plt.imsave(savedir + 'change_map_{}.png'.format(img_name), change_map, cmap='gray')
    plt.imsave(savedir + 'change_map_A_{}.png'.format(img_name), modify_A, cmap='gray')
    plt.imsave(savedir + 'change_map_B_{}.png'.format(img_name), modify_B, cmap='gray')
    print(f"Saved change maps for {img_name}")

# Configuration
basedir = "../data/aligned_instances/"
name = '0002'
src_year = '1888'
tgt_year = '1900'

print('Processing', name, src_year, tgt_year)

# Load the aligned instance files
source_gt_warped = np.load(basedir + "instance_seg_{}_{}_{}_warp.npy".format(name, src_year, tgt_year))
target_gt = np.load(basedir + "instance_seg_{}_{}.npy".format(name, tgt_year))

# Create output directory
savedir = '../output/derived_changes/{}/'.format(name)
if not os.path.exists(savedir):
    os.makedirs(savedir)

img_name = '{}_{}_{}'.format(name, src_year, tgt_year)

# Check if images are same size or need padding
if source_gt_warped.shape != target_gt.shape:
    print(f"Source shape: {source_gt_warped.shape}, Target shape: {target_gt.shape}")
    # Apply padding if needed
    target_gt_pad = np.zeros(source_gt_warped.shape)
    offset = 24
    target_gt_pad[offset:-offset, offset:-offset] = target_gt
    target_gt = target_gt_pad

# Convert to appropriate format
T = (target_gt + 1).astype(int)
P = (source_gt_warped + 1).astype(int)

print('T shape:', T.shape, 'T type:', T.dtype)
print('P shape:', P.shape, 'P type:', P.dtype)

# Calculate change maps
return_precision_recall(t=0.5, GT=T, Pred=P, savedir=savedir, img_name=img_name)

print('Processing complete!')