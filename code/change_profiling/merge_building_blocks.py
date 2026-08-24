import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
import os

basedir = "../../output/instance_inference/"

years = [1880] 


for y in years:
    name = 'instance_seg_map_{}.npy'.format(y)
    print('Processing', name, y)
    if os.path.isdir('../../output/building_blocks_inference/') is False:
        os.makedirs('../../output/building_blocks_inference/')
    # Load the instance map
    image = np.load(basedir + name)
    roads = (image==-1)
    
    building_blocks = 1 - roads
    selem = disk(3)  
    eroded_image = closing(building_blocks, selem)

    plt.imsave('../../output/building_blocks_inference/building_blocks_{}.png'.format(y), eroded_image, cmap='gray')