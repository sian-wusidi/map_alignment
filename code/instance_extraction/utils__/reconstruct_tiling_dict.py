import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import pdb

def reconstruct_tiling(original_image_path, test_pred_dict, tile_save_path, w_size, image_debug=None, save_image=True):
    in_patches = list(test_pred_dict.keys())
    in_patches.sort()
    patches_images =  list(test_pred_dict.values())

    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)

    if save_image:
        cv2.imwrite(tile_save_path, new_img)

    if image_debug is not None:
        cv2.imwrite(image_debug, new_img * 255)
    return new_img


def reconstruct_tiling_array(original_image_path, patches_images, w_size):
    win_size = w_size
    pad_px = win_size // 2

    in_img = cv2.imread(original_image_path)

    if in_img is None:
        print('original image{}'.format(original_image_path))

    new_img = reconstruct_from_patches(patches_images, win_size, pad_px, in_img.shape, np.float32)
    return new_img


def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype):
    '''Adjust to take patch images directly.
    patch_size is the size of the tiles
    step_size should be patch_size//2
    image_size_2d is the size of the original image
    image_dtype is the data type of the target image

    Most of this could be guessed using an array of patches
    (except step_size but, again, it should be should be patch_size//2)
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    
    if len(patches_images.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    patch_offset = step_size//2
    patch_inner = p_h-step_size
    for row in range(numrows):
        for col in range(numcols):
            tt = patches_images[row*numcols+col]
            tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset]
            img[row*step_size:row*step_size+patch_inner,
                col*step_size:col*step_size+patch_inner] = tt_roi # +1?? 
    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]



def reconstruct_from_patches_blend(patches_images, patch_size, step_size, image_size_2d, image_dtype, reverse=False):
    '''Adjust to take patch images directly.
    patch_size is the size of the tiles
    step_size should be patch_size//2
    image_size_2d is the size of the original image
    image_dtype is the data type of the target image

    Most of this could be guessed using an array of patches
    (except step_size but, again, it should be should be patch_size//2)
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    
    if len(patches_images.shape) == 4:
        img = np.ones((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)*(-1)
    else:
        img = np.ones((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)*(-1)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    overlap = patch_size//2 #p_h-step_size
    global_id_counter = 1
    patches_images_new = patches_images.copy()

    for row in range(numrows):
        for col in range(numcols):
            tt = patches_images[row*numcols+col]
            # tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset].copy()
            tt_roi = tt.copy()  # tile mask
            instance_ids = np.unique(tt_roi[tt_roi != -1])

            for local_id in instance_ids:
                instance_mask = (tt_roi == local_id)
                # matched = False
                best_match_id = None
                if row > 0:
                    overlap_region = img[row*step_size:row*step_size+overlap,
                                        col*step_size:col*step_size+patch_size]
                    overlap_instance_ids = np.unique(overlap_region[overlap_region != -1])  #
                    best_match_id = find_best_match_unique(overlap_instance_ids, overlap_region, instance_mask[:overlap,:], 0.1)

                if col > 0 and best_match_id is None:
                    overlap_region = img[row*step_size:row*step_size+patch_size,
                                        col*step_size:col*step_size+overlap]
                    overlap_instance_ids = np.unique(overlap_region[overlap_region != -1])#
                    best_match_id = find_best_match_unique(overlap_instance_ids, overlap_region, instance_mask[:,:overlap], 0.1)

                if best_match_id is not None:
                    global_id = best_match_id
                else:
                    global_id = global_id_counter
                    global_id_counter += 1
                img[row*step_size:row*step_size+patch_size,
                                        col*step_size:col*step_size+patch_size][instance_mask]= global_id
    return img[step_size:-patch_size,step_size:-patch_size,...]

def _best_match(region, mask, iou_threshold):
    '''Lines 13 / 16: the instance id in `region` with the highest IoU with `mask`,
    if that IoU > iou_threshold; otherwise None. Background (0) and unwritten (-1)
    pixels are never candidates.'''
    best_id, best_iou = None, iou_threshold
    for cand in np.unique(region[region > 0]):
        cand_mask = region == cand
        union = np.logical_or(cand_mask, mask).sum()
        score = np.logical_and(cand_mask, mask).sum() / union if union else 0.0
        if score > best_iou:
            best_id, best_iou = int(cand), score
    return best_id


def reconstruct_from_patches_blend(patches_images, patch_size, step_size, image_size_2d,
                                   image_dtype=np.int32, iou_threshold=0.1):
    '''Algorithm 1: stitching instance predictions across overlapping tiles.

    Patches: (N, s, s) label tiles, -1 = unknown, 0 = background, 1+ = local ids.
    Patches are taken from the image padded by step_size on the top/left, row-major.
    Returns I (h, w): 0 = background, positive = global instance ids. Ids are not
    necessarily consecutive, since merged ids (line 19) leave gaps.
    '''
    if patches_images.ndim != 3:
        raise ValueError("instance stitching expects (N, H, W) label patches")
    h, w = int(image_size_2d[0]), int(image_size_2d[1])
    s, t = patch_size, step_size
    o = s - t                                                   # overlap

    # Line 3. The canvas is padded, so r and c count tiles over the padded size.
    # int64 internally: needs -1, and avoids overflow; cast to image_dtype at the end.
    i_h, i_w = h + s, w + s
    I = np.full((i_h + s // 2, i_w + s // 2), -1, dtype=np.int64)
    r = i_h // t - 1
    c = i_w // t - 1
    g = 1
    if len(patches_images) != r * c:
        raise ValueError(f"Expected {r * c} patches, got {len(patches_images)}")

    for i in range(r):                                          # line 4
        for j in range(c):                                      # line 5
            p = patches_images[i * c + j]                       # line 6
            L = np.unique(p[p > 0])                             # line 7

            for l in L:                                         # line 8
                M = p == l                                      # line 9
                id_u = id_l = None                              # line 10

                if i > 0:                                       # lines 11-13
                    R_u = I[i * t:i * t + o, j * t:j * t + s]
                    id_u = _best_match(R_u, M[:o, :], iou_threshold)

                if j > 0:                                       # lines 14-16
                    R_l = I[i * t:i * t + s, j * t:j * t + o]
                    id_l = _best_match(R_l, M[:, :o], iou_threshold)

                if id_u is not None and id_l is not None and id_u != id_l:   # line 17
                    id_ = min(id_u, id_l)                       # line 18
                    I[I == max(id_u, id_l)] = id_               # line 19
                elif id_u is not None:                          # lines 20-21
                    id_ = id_u
                elif id_l is not None:                          # lines 22-23
                    id_ = id_l
                else:                                           # lines 24-26
                    id_ = g
                    g += 1

                # Line 27: write id_ without overwriting another instance.
                # Background (0) committed by an earlier tile's line 28 is not an
                # instance, so it may be overwritten.
                tile = I[i * t:i * t + s, j * t:j * t + s]      # view into I
                tile[M & (tile <= 0)] = id_

            tile = I[i * t:i * t + s, j * t:j * t + s]          # line 28
            tile[tile == -1] = 0

    I = I[t:t + h, t:t + w]                                     # remove padding
    return I.astype(image_dtype)      

def stitch_predictions(patches_images, patch_size, step_size, image_size_2d, image_dtype, patch_offset):
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size

    # Initialize output image (adjusted to image size and dtype)
    img = np.ones((i_h + p_h // 2, i_w + p_w // 2, 3), dtype=image_dtype) * (-1) if len(patches_images.shape) == 4 else np.ones((i_h + p_h // 2, i_w + p_w // 2), dtype=image_dtype) * (-1)

    # Calculate the number of rows and columns of patches
    numrows = (i_h) // step_size - 1
    numcols = (i_w) // step_size - 1
    expected_patches = numrows * numcols

    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    # Define overlap size
    overlap = step_size

    global_id_counter = 1  # Global ID counter for unique instances
    for row in range(numrows):
        for col in range(numcols):
            # Extract current patch and apply ROI if offset exists
            patch_index = row * numcols + col
            tt_roi = patches_images[patch_index][patch_offset:-patch_offset, patch_offset:-patch_offset] if patch_offset > 0 else patches_images[patch_index].copy()
            
            # Identify unique instance IDs in the patch, ignoring background (-1)
            instance_ids = np.unique(tt_roi[tt_roi != -1])

            for local_id in instance_ids:
                instance_mask = (tt_roi == local_id)
                best_match_id = None

                # Row-wise overlap (top)
                if row > 0:
                    overlap_region = img[row * step_size + patch_offset:row * step_size + overlap - patch_offset, col * step_size + patch_offset:col * step_size + patch_size - patch_offset]
                    overlap_instance_ids = np.unique(overlap_region)
                    best_match_id = find_best_match(overlap_instance_ids, overlap_region, instance_mask[:overlap - 2 * patch_offset, :], 0.1)

                # Column-wise overlap (left) only if no match was found
                if col > 0 and best_match_id is None:
                    overlap_region = img[row * step_size + patch_offset:row * step_size + patch_size - patch_offset, col * step_size + patch_offset:col * step_size + overlap - patch_offset]
                    overlap_instance_ids = np.unique(overlap_region)
                    best_match_id = find_best_match(overlap_instance_ids, overlap_region, instance_mask[:, :overlap - 2 * patch_offset], 0.1)

                # Assign the appropriate global ID
                if best_match_id is not None:
                    if len(best_match_id) == 1:
                        global_id = best_match_id[0]
                    else:
                        global_id = best_match_id[0] if -1 in best_match_id else global_id_counter
                        global_id_counter += 1
                        img[(img == best_match_id[0]) | (img == best_match_id[1])] = global_id
                else:
                    global_id = global_id_counter
                    global_id_counter += 1

                # Assign the global ID to the corresponding region in the stitched image
                img[row * step_size + patch_offset:row * step_size + patch_size - patch_offset,
                    col * step_size + patch_offset:col * step_size + patch_size - patch_offset][instance_mask] = global_id

    # Return the final stitched image
    return img[step_size + patch_offset:-(patch_size + patch_offset), step_size + patch_offset:-(patch_size + patch_offset), ...]



def reconstruct_from_patches_bidirectional(patches_images, patch_size, step_size, image_size_2d, image_dtype):
    image, patches_images_new = reconstruct_from_patches_blend(patches_images, patch_size, step_size, image_size_2d, image_dtype, reverse=False)
    image = reconstruct_from_patches_blend(patches_images_new[::-1,::-1, ::-1], patch_size, step_size, image_size_2d, image_dtype, reverse=True)

    return image 
def find_best_match(overlap_instance_ids, overlap_region, instance_mask, iou_threshold):
    """Find the best matching instance in the overlap region based on IoU."""
    best_match_id = None
    iou_list = []
    for overlap_id in overlap_instance_ids:
        overlap_mask = (overlap_region == overlap_id)
        current_iou = iou(overlap_mask, instance_mask)
        if current_iou > iou_threshold:
            iou_list.append((overlap_id, current_iou))
        
        if len(iou_list):
            iou_list.sort(key=lambda x: x[1], reverse=True)
            
            if instance_mask.sum()>1000:
                # get the ids of top 2 iou
                best_match_id = np.array(iou_list)[0:2,0]
            else:
                best_match_id = np.array(iou_list)[0:1,0]
           
    
    return best_match_id

def find_best_match_unique(overlap_instance_ids, overlap_region, instance_mask, iou_threshold, overlap_threshold=0.6):
    """Find the best matching instance in the overlap region based on IoU."""
    best_iou = 0
    best_match_id = None
    iou_list = []
    for overlap_id in overlap_instance_ids:
        overlap_mask = (overlap_region == overlap_id)
        current_iou = iou(overlap_mask, instance_mask)
        if current_iou > best_iou and current_iou > iou_threshold:
            best_iou = current_iou
            best_match_id = overlap_id
           
    
    return best_match_id



def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return intersection / mask1.sum() if mask1.sum() != 0 else 0

