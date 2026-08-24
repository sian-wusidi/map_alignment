import numpy as np
import os 
import geopandas
import cv2
from random import randint
import matplotlib.pyplot as plt
import pdb
from PIL import Image


basedir = "../../data/vector_annotations/"
savedir = "../../output/displacement_training/" 

if os.path.exists(savedir + "/Foreground_img/") is False:
    os.makedirs(savedir + "/Foreground_img/")
if os.path.exists(savedir + "/Foreground_mask/") is False:
    os.makedirs(savedir + "/Foreground_mask/")
sample_size = 520

sheets = os.listdir(basedir + "map/")
vectors = [s.replace("png", "shp") for s in sheets]

num_foreground_max = 5
sample_number = 3000
count = 0

for i in range(len(sheets)):
    input_sheet_location = basedir+ "map/" + sheets[i]
    # open png
    sheet = np.array(Image.open(input_sheet_location))
    shape = sheet.shape

    GT_vec = geopandas.read_file(basedir + "vector/" + vectors[i])
    areas = GT_vec.area
    foreground = np.where(areas>=5000) and np.where(areas<=20000)
    print(i, 'number of objetcs:', len(foreground[0]))
    for i in range(0, len(foreground[0])):
        masks = np.zeros((sample_size, sample_size))
        overlay = np.zeros((sample_size, sample_size, 3))
        num_foreground = randint(1, num_foreground_max)
        for f in range(0, num_foreground): 
            index = randint(0, len(foreground[0])-1)
            sample_foreground = foreground[0][index]
            centroid_x_f = GT_vec.geometry[sample_foreground].centroid.x
            centroid_y_f = GT_vec.geometry[sample_foreground].centroid.y
            if GT_vec.geometry[sample_foreground].type is not 'Polygon':
                continue
            centroid_f = np.array(GT_vec.geometry[sample_foreground].centroid.xy).T
            exterior = np.array(GT_vec.geometry[sample_foreground].exterior.xy).T

            x_min_f = max(centroid_x_f - sample_size//2, 0)
            y_max_f = min(- (centroid_y_f - sample_size//2), shape[0])
            x_max_f = min(centroid_x_f + sample_size//2, shape[1])
            y_min_f = max(- (centroid_y_f + sample_size//2), 0)
            
            foreground_img = sheet[int(y_min_f):int(y_max_f), int(x_min_f):int(x_max_f), :]
            mask = np.zeros((int(y_max_f) - int(y_min_f), int(x_max_f) - int(x_min_f)))

            poly = exterior - centroid_f 
            # translate to image coordinates (local patch)
            poly[:,0] = poly[:,0] + mask.shape[1]//2
            poly[:,1] = mask.shape[0]//2 - poly[:,1] 
            cv2.fillPoly(mask,[poly.astype(int)], color=(255, 255, 255))
            mask = cv2.dilate(mask, np.ones((3,3)), iterations=1)
           
            # random translation
            offset_x = randint(-100,100)
            offset_y = randint(-100,100)   
            T = np.float32([[1, 0, offset_x], [0, 1, offset_y]]) 

            foreground_img = cv2.warpAffine(foreground_img, T, (sample_size, sample_size))
            mask = cv2.warpAffine(mask, T, (sample_size, sample_size)) 
            mask = np.expand_dims(mask, axis=-1)
            foreground_img = np.uint8(foreground_img * mask//255)[:,:,0:3]
            
            overlay = foreground_img * mask//255 + overlay* (1-mask//255)

            masks = masks + mask[:,:,0]

        masks[masks>=255] == 255    
        plt.imsave(savedir + "/Foreground_img/"+ str(count) + ".png", np.uint8(overlay))
        plt.imsave(savedir + "/Foreground_mask/"+ str(count) + ".png", np.uint8(masks), cmap='gray')
        count += 1
        pdb.set_trace()
        
    