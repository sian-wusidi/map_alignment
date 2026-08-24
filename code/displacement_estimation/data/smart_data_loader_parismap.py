# import sys
# sys.path.insert(1, '../../')

import numpy as np
from torch.utils import data
import torch
from torchvision import transforms
import os
from PIL import Image
import pdb
from data.create_tiling import generate_tiling

class Data_dual(data.Dataset):
    def __init__(self, src_image_path, tar_image_path, w_size, mode='test', rgb=True):
        self.src_image_path = src_image_path
        self.tar_image_path = tar_image_path

        self.rgb = rgb
        self.w_size = w_size

        if type(src_image_path) is list:
            src_image_lst = []
            for path in src_image_path:
                src_image_lst += generate_tiling(path, w_size=self.w_size, mode='img')
            self.src_img  = np.array(src_image_lst)
        else:
            self.src_img  = np.array(generate_tiling(self.src_image_path, w_size=self.w_size, mode='img'))
        if type(tar_image_path) is list:
            tar_image_lst = []
            for path in tar_image_path:
                tar_image_lst += generate_tiling(path, w_size=self.w_size, mode='img')
            self.tar_img  = np.array(tar_image_lst)
        else:
            self.tar_img  = np.array(generate_tiling(self.tar_image_path, w_size=self.w_size, mode='img'))

        if mode == 'train':
            self.src_img = self.src_img[:int(0.8*len(self.src_img))] #0.8
            self.tar_img = self.tar_img[:int(0.8*len(self.tar_img))]
        elif mode == 'val':
            self.src_img = self.src_img[int(0.8*len(self.src_img)):]
            self.tar_img = self.tar_img[int(0.8*len(self.tar_img)):]
   
    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index]
        img_src = np.array(img_src, dtype=np.float32)

        img_tar = self.tar_img[index]
        img_tar = np.array(img_tar, dtype=np.float32)


        changes = np.zeros((self.w_size, self.w_size, 3))
        I1Tensor = transforms.ToTensor()(img_src)
        I2Tensor = transforms.ToTensor()(img_tar)
        changeTensor = transforms.ToTensor()(changes)[:2]
 
        output = {
            'source_image': I1Tensor,
            'target_image': I2Tensor,
            'changes': changeTensor,
            'epoch': 0,
            'iter': 0,
            'settings': 0
        }
        return output

class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)  #min

class Data_dual_load(data.Dataset):
    def __init__(self, src_image_path, tar_image_path, w_size, mode='test', rgb=True):
        self.src_image_path = src_image_path
        self.tar_image_path = tar_image_path

        self.rgb = rgb
        self.w_size = w_size

        self.src_img = sorted(os.listdir(self.src_image_path))
        self.tar_img = sorted(os.listdir(self.tar_image_path))

        np.random.seed(10000)
        indices = np.arange(len(self.src_img))
        np.random.shuffle(indices)

        self.src_img = [self.src_img[i] for i in indices]
        self.tar_img = [self.tar_img[i] for i in indices]

        if mode == 'train':
            self.src_img = self.src_img[:int(0.8*len(self.src_img))] #0.8
            self.tar_img = self.tar_img[:int(0.8*len(self.tar_img))]
        elif mode == 'val':
            self.src_img = self.src_img[int(0.8*len(self.src_img)):]
            self.tar_img = self.tar_img[int(0.8*len(self.tar_img)):]
   
    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index % len(self.src_img)]
        img_src = Image.open(self.src_image_path + img_src)
        img_src = np.array(img_src, dtype=np.float32)[:,:,0:3]

        img_tar = self.tar_img[index % len(self.tar_img)]
        img_tar = Image.open(self.tar_image_path + img_tar)
        img_tar = np.array(img_tar, dtype=np.float32)[:,:,0:3]

        changes = np.zeros((self.w_size, self.w_size, 3))
        I1Tensor = transforms.ToTensor()(img_src)
        I2Tensor = transforms.ToTensor()(img_tar)
        changeTensor = transforms.ToTensor()(changes)[:2]
 
        output = {
            'source_image': I1Tensor,
            'target_image': I2Tensor,
            'changes': changeTensor,
            'epoch': 0,
            'iter': 0,
            'settings': 0
        }
        return output
   
class Data_dual_load_change(data.Dataset):
    def __init__(self, image_path, w_size, mode='test', rgb=True):
        self.image_path = image_path

        self.rgb = rgb
        self.w_size = w_size

        self.src_img = sorted(os.listdir(self.image_path + 'Img_src'))
        self.tar_img = sorted(os.listdir(self.image_path + 'Img_tar'))
        
        self.text_path = sorted(os.listdir(self.image_path + '/Text/'))
        self.text_mask_path = sorted(os.listdir(self.image_path + '/Text_mask/'))
        self.fore_mask_path = sorted(os.listdir(self.image_path + '/Foreground_mask/'))
        self.fore_img_path = sorted(os.listdir(self.image_path + '/Foreground_img/'))
        self.fore_img_mask = list(zip(self.fore_mask_path, self.fore_img_path))
        np.random.seed(10000)
        np.random.shuffle(self.fore_img_mask)
        self.fore_mask_path, self.fore_img_path = zip(*self.fore_img_mask)

        if mode == 'train':
            self.src_img = self.src_img[:int(0.8*len(self.src_img))]
            self.tar_img = self.tar_img[:int(0.8*len(self.tar_img))]
        elif mode == 'val':
            self.src_img = self.src_img[int(0.8*len(self.src_img)):]
            self.tar_img = self.tar_img[int(0.8*len(self.tar_img)):]
        self.mode = mode
    def __len__(self):
        return min(len(self.src_img), len(self.tar_img))    

    def __getitem__(self, index):
        img_src = self.src_img[index] # % len(self.src_img)
        img_src = Image.open(self.image_path + '/Img_src/' + img_src)
        img_src = np.array(img_src, dtype=np.float32)[:,:,:3]  

        img_tar = self.tar_img[index]
        img_tar = Image.open(self.image_path + '/Img_tar/' + img_tar)
        img_tar = np.array(img_tar, dtype=np.float32)[:,:,:3]

        img_tar_prime = img_tar.copy() 

        if self.mode == 'train':
            img_fore_mask = Image.open(self.image_path + '/Foreground_mask/' + self.fore_mask_path[index % len(self.fore_mask_path)])
            img_fore_mask = np.array(img_fore_mask, dtype=np.float32)[:, :, 0:3]
            mask = img_fore_mask.any(axis=-1, keepdims=True)

            img_fore = Image.open(self.image_path + '/Foreground_img/' + self.fore_img_path[index % len(self.fore_img_path)])
            img_fore = np.array(img_fore, dtype=np.float32)[:, :, :3]

            img_text = Image.open(self.image_path + '/Text/' + self.text_path[index % len(self.text_path)])
            img_text = np.array(img_text, dtype=np.float32)[:, :, :3]

            im_text_mask = Image.open(self.image_path + '/Text_mask/' + self.text_mask_path[index % len(self.text_mask_path)])
            im_text_mask = np.array(im_text_mask, dtype=np.float32)[:, :, :3]/255

            if mask.sum() > 0 and mask.sum() < self.w_size*self.w_size*0.5 and index % 2 == 0: # every two images
                img_tar_prime = img_tar*(1-mask) + img_fore*mask
                img_tar_prime = img_tar_prime.astype(np.float32)
            if index % 3 == 1: # every three images
                img_tar_prime = img_tar_prime * (1 - im_text_mask) + img_text * im_text_mask
                img_tar_prime = img_tar_prime.astype(np.float32)

            if mask.sum() > 0 and mask.sum() < self.w_size*self.w_size*0.5 and index % 2 == 1: # every five images
                img_tar = img_tar*(1-mask) + img_fore*mask
                img_tar = img_tar.astype(np.float32)
            if index % 3 == 2: # every three images
                img_tar = img_tar * (1 - im_text_mask) + img_text * im_text_mask
                img_tar = img_tar.astype(np.float32)
        else:
            img_fore_mask = np.zeros((self.w_size, self.w_size, 3), dtype=np.float32)
        # changes = np.zeros((self.w_size, self.w_size, 3))
        I1Tensor = transforms.ToTensor()(img_src)
        I2Tensor = transforms.ToTensor()(img_tar)
        I3Tensor = transforms.ToTensor()(img_tar_prime)
        changeTensor = transforms.ToTensor()(img_fore_mask)[:2]
 
        output = {
            'source_image': I1Tensor,
            'target_image': I2Tensor,
            'target_image_prime': I3Tensor, 
            'changes': changeTensor,
            'epoch': 0,
            'iter': 0,
            'settings': 0
        }

        return output
