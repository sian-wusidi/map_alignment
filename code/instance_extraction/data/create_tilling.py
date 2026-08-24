import numpy as np
from skimage.util import view_as_windows
from PIL import Image
import argparse
import cv2

import pdb

def pad_image(image, target_size=(2048, 2048), fill_color=(0, 0, 0)):
    """Pads an image to the specified target size using OpenCV.

    Args:
        image: The input image as a NumPy array.
        target_size: A tuple specifying the desired width and height.
        fill_color: The color used to fill the padded regions.

    Returns:
        The padded image as a NumPy array.
    """

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    target_height, target_width = target_size

    # Calculate the padding required for each dimension
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left

    # Create a new image with the target size and fill color
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    
    # Create a mask for the padded regions
    mask = np.zeros_like(padded_image)
    mask[top+10:top + height-10, left+10:left + width-10] = 1

    return padded_image, mask


def generate_tiling(image_path, w_size, mode='img', margin=[0,0]):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    # in_img = np.array(Image.open(image_path))[:7400, :10000, ...]

    # padding image to 2048x2048
    w = np.array(Image.open(image_path)).shape[0]
    if w < 1000:
        in_img = np.array(Image.open(image_path))
        in_img, mask  = pad_image(in_img)
    else:
        in_img = np.array(Image.open(image_path))#[margin[0]:7400-margin[0], margin[1]:10000-margin[1], ...]

    if mode == 'gt' and len(in_img.shape) == 3:
        in_img = in_img[:,:,0]

    if len(in_img.shape) == 2:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'constant', constant_values=0) #edge
        tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    else:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'constant', constant_values=0)  #edge
        tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            if len(in_img.shape) == 2:
                tt = tiles[row, col, ...].copy()
            else:
                tt = tiles[row, col, 0, ...].copy()
            # If you want black boarder, set the value to 25 (Suggest not using balck boarder)
            # bordersize=1005
            # tt[:bordersize,:, 2] = 255
            # tt[-bordersize:,:, 2] = 255
            # tt[:,:bordersize, 2] = 255
            # tt[:,-bordersize:, 2] = 255
            # skio.imsave(os.path.join(save_image_path, f"t_r{row:02d}_c{col:02d}.jpg"), tt)
            tiles_lst.append(tt)

    if w < 1000:
        return tiles_lst, mask
    else:
        return tiles_lst, None

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
