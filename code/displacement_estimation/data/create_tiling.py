import numpy as np
from skimage.util import view_as_windows
from PIL import Image
import argparse
import pdb

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
        if mode == 'flow':
            tiles = view_as_windows(img_pad, (win_size,win_size,2), step=pad_px)
        else:
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

def main():
    parser = argparse.ArgumentParser(description='Create Tillings.')
    parser.add_argument('input_path', help='Path of original image.')
    parser.add_argument('gt_path', help='Path of ground truth.')
    parser.add_argument('save_image_path', help='Directory of images for saving the tilings.')
    parser.add_argument('save_gt_path', help='Directory of ground truth for saving the tilings.')
    parser.add_argument('width', help='Width of the tillings')

    args = parser.parse_args()
    generate_tiling(args.input_path, args.gt_path, args.save_image_path, args.save_gt_path, args.width)


if __name__ == '__main__':
    main()
