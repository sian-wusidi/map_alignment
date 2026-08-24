import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
from PIL import Image
# Define the path to the directory containing the images
path = '../../output/georeferenced/rectified/'
save_path = '../../output/displacement_training/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path + 'Img_src/')
    os.makedirs(save_path + 'Img_tar/')

dirs = os.listdir(path)
masks = os.listdir('../output/georeferenced/1878/mask')  # 1878 as reference  
w_size = 520
def get_bounding_box(image):
    # Get the coordinates of all non-zero pixels
    coords = np.argwhere(image > 0)
    
    # Find the minimum and maximum coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def generate_tiling(image_path, w_size, mode, resize=None):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    if isinstance(image_path, str):
        in_img = np.array(Image.open(image_path))
    else:
        in_img = image_path

    if resize is not None:
        in_img = in_img[:resize[0], :resize[1], ...]

    if len(in_img.shape) == 2:
        if mode == 'img':
            img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'edge')
        else:
            img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'constant')
        tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    else:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'constant')

        tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            if len(in_img.shape) == 2:
                tt = tiles[row, col, ...].copy()
            else:
                tt = tiles[row, col, 0, ...].copy()
            tiles_lst.append(tt)
    return tiles_lst

for i in range(0, len(dirs)):
    # if i == 2 or i == 3 or i==1: # district 3.4.5.6.7
    #     continue
    imgs = os.listdir(path + dirs[i]) 
    mask = plt.imread('../output/georeferenced/1878/mask/' + masks[i])
    x_min, y_min, x_max, y_max = get_bounding_box(mask)
    print(masks[i], dirs[i])
    for j in range(0,len(imgs)-1):
        img1 = plt.imread(path + dirs[i] + '/' + imgs[j])
        img2 = plt.imread(path + dirs[i] + '/' + imgs[j+1])
        img1 = img1[x_min:x_max, y_min:y_max]
        img2 = img2[x_min:x_max, y_min:y_max]
        src_img = np.array(generate_tiling(img1, w_size, mode='img'))
        tar_img  = np.array(generate_tiling(img2, w_size, mode='img'))
        for k in range(len(src_img)):
            plt.imsave(save_path + 'Img_src/' + dirs[i] + '_' + str(j) + '_' + str(k) + '.png', src_img[k])
            plt.imsave(save_path + 'Img_tar/' + dirs[i] + '_' + str(j) + '_' + str(k) + '.png', tar_img[k])
        break  # just save one image pair for demo code
