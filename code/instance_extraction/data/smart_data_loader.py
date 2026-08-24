import pdb
import numpy as np
import torch
from torch.utils import data
from skimage.measure import label
from PIL import Image
import os
from data.create_tilling import generate_tiling



class Data_instance(data.Dataset):
    def __init__(self, src_image_path, gt_path, mask_path, processor, w_size, resize=None, rgb=True, mode='test', margin=0):
        self.src_image_path = src_image_path

        self.rgb = rgb
        self.w_size = w_size
        self.gt_path = gt_path
        self.mask_path = mask_path  

        self.src_img,_  = generate_tiling(self.src_image_path, w_size=self.w_size)  #, margin=[600,1400]
        self.src_img = np.array(self.src_img)
        self.gt,_      = generate_tiling(self.gt_path, w_size=self.w_size)
        self.gt = np.array(self.gt)
        self.road_mask,_ = generate_tiling(self.mask_path, w_size=self.w_size)
        self.road_mask = np.array(self.road_mask)
        self.processor = processor

        if mode == 'train':
            self.src_img = self.src_img[:int(len(self.src_img)*0.8)]
            self.gt = self.gt[:int(len(self.gt)*0.8)]
            self.road_mask = self.road_mask[:int(len(self.road_mask)*0.8)]

        elif mode == 'val':
            self.src_img = self.src_img[int(len(self.src_img)*0.8):]
            self.gt = self.gt[int(len(self.gt)*0.8):]
            self.road_mask = self.road_mask[int(len(self.road_mask)*0.8):]

    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index]
        img_src = np.array(img_src)/255
        img_src = np.moveaxis(img_src, -1, 0)  

        edge = self.gt[index]
        edge = edge.astype(np.uint8)[:,:,0]/255
    
        class_id_map = (1-edge).astype(np.uint8)

        mask = self.road_mask[index]
        mask = (mask.astype(np.uint8)[:,:,0]/255).astype(int)
        class_id_map[mask==1] = 0
        instance_seg = label(class_id_map, connectivity=2)

        if np.unique(instance_seg).shape[0] == 1 and np.unique(class_id_map).shape[0] == 1:
            inputs = self.processor([img_src], return_tensors="pt")
            inputs = {k:v.squeeze().float() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            # convert labels of instance_seg to one-hot encoding
            # mask_labels = []  # for debugging
            inst2class = {}
            for i in range(np.max(instance_seg)+1):
                # mask_labels.append((instance_seg == i)[np.newaxis, ...])
                classl = np.unique(class_id_map[instance_seg == i])[0]
                inst2class.update({i: classl})
            # print(inst2class)
            # mask_labels = np.concatenate(mask_labels, axis=0)
    
            inputs = self.processor([img_src], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze().float() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        return inputs

class Data_instance_seq(data.Dataset):
    def __init__(self, src_image_paths, gt_path, mask_path, processor, w_size, resize=None, rgb=True, mode='test', margin=0):
        self.src_image_paths = src_image_paths

        self.rgb = rgb
        self.w_size = w_size
        self.gt_path = gt_path
        self.mask_path = mask_path  

        src_img_list = []
        years = []
        for src_image_path in os.listdir(self.src_image_paths):
            year = src_image_path.split('_')[-1].split('.')[0]
            src_img,_  = generate_tiling(os.path.join(self.src_image_paths,src_image_path), w_size=self.w_size)  #, margin=[600,1400]
            src_img_list.append(np.array(src_img))
            years.append(year)  
        
        self.src_img = np.array(src_img_list)
        self.src_img = np.moveaxis(self.src_img, 0, -1) # H, W, C, t

        self.gt,_      = generate_tiling(self.gt_path, w_size=self.w_size)
        self.gt = np.array(self.gt)
        self.road_mask,_ = generate_tiling(self.mask_path, w_size=self.w_size)
        self.road_mask = np.array(self.road_mask)
        self.processor = processor

        if mode == 'train':
            self.src_img = self.src_img[:int(len(self.src_img)*0.8)]
            self.gt = self.gt[:int(len(self.gt)*0.8)]
            self.road_mask = self.road_mask[:int(len(self.road_mask)*0.8)]

        elif mode == 'val':
            self.src_img = self.src_img[int(len(self.src_img)*0.8):]
            self.gt = self.gt[int(len(self.gt)*0.8):]
            self.road_mask = self.road_mask[int(len(self.road_mask)*0.8):]

    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index]  # H, W, C, t
        img_src = np.array(img_src)/255
        img_src = np.transpose(img_src, (3, 2, 0, 1))  # t, C, H, W
        img_src_center = img_src[0]
        edge = self.gt[index]
        edge = edge.astype(np.uint8)[:,:,0]/255
    
        class_id_map = (1-edge).astype(np.uint8)

        mask = self.road_mask[index]
        mask = (mask.astype(np.uint8)[:,:,0]/255).astype(int)
        class_id_map[mask==1] = 0
        instance_seg = label(class_id_map, connectivity=2)
        images = torch.tensor(img_src, dtype = torch.float32) # t, C, H, W
        if np.unique(instance_seg).shape[0] == 1 and np.unique(class_id_map).shape[0] == 1:
            inputs = self.processor([img_src_center], return_tensors="pt")
            inputs = {k:v.squeeze().float() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
            inputs["pixel_values"] = images
        else:
            # convert labels of instance_seg to one-hot encoding
            # mask_labels = []  # for debugging
            inst2class = {}
            for i in range(np.max(instance_seg)+1):
                # mask_labels.append((instance_seg == i)[np.newaxis, ...])
                classl = np.unique(class_id_map[instance_seg == i])[0]
                inst2class.update({i: classl})
            # print(inst2class)
            # mask_labels = np.concatenate(mask_labels, axis=0)
             # t, C, H, W
            inputs = self.processor([img_src_center], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze().float() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
            inputs["pixel_values"] = images
        
        return inputs

    
class Data_instance_seq_val(data.Dataset):
    def __init__(self, src_image_paths, gt_path, processor, w_size, resize=None, rgb=True, margin=0):
        self.src_image_paths = src_image_paths

        self.rgb = rgb
        self.w_size = w_size
        self.gt_path = gt_path

        src_img_list = []
        for src_image_path in self.src_image_paths:
            src_img,_  = generate_tiling(src_image_path, w_size=self.w_size)  #, margin=[600,1400]
            src_img_list.append(np.array(src_img))
        self.src_img = np.array(src_img_list)
        self.src_img = np.moveaxis(self.src_img, 0, -1) # H, W, C, t
        self.processor = processor

    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index]
        img_src = np.array(img_src)/255
        img_src = np.transpose(img_src, (3, 2, 0, 1))  # t, C, H, W

        # edge = self.gt[index]
        # if len(edge.shape) == 3:
        #     edge = edge.astype(np.uint8)[:,:,0]/255
        # else:
        #     edge = edge.astype(np.uint8)/255
    
        # class_id_map = (1-edge).astype(np.uint8)
        images = torch.tensor(img_src, dtype = torch.float32)
        inputs = self.processor([img_src[0]], return_tensors="pt")
        inputs = {k:v.squeeze().float() for k,v in inputs.items()}
        inputs["class_labels"] = torch.tensor([0])
        inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        inputs["pixel_values"] = images
        return inputs
    
class Data_instance_val(data.Dataset):
    def __init__(self, src_image_path, gt_path, processor, w_size, resize=None, rgb=True, margin=0):
        self.src_image_path = src_image_path

        self.rgb = rgb
        self.w_size = w_size
        self.gt_path = gt_path

        self.src_img,_  = generate_tiling(self.src_image_path, w_size=self.w_size)  # [600,1400] 3500,4500 [1000,2000] 5400,6000
        self.src_img = np.array(self.src_img)
        # self.gt,_      = generate_tiling(self.gt_path, w_size=self.w_size)
        # self.gt = np.array(self.gt)
        self.processor = processor

    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):
        img_src = self.src_img[index]
        img_src = np.array(img_src)/255
        img_src = np.moveaxis(img_src, -1, 0)

        # edge = self.gt[index]
        # if len(edge.shape) == 3:
        #     edge = edge.astype(np.uint8)[:,:,0]/255
        # else:
        #     edge = edge.astype(np.uint8)/255
    
        # class_id_map = (1-edge).astype(np.uint8)

        inputs = self.processor([img_src], return_tensors="pt")
        inputs = {k:v.squeeze().float() for k,v in inputs.items()}
        inputs["class_labels"] = torch.tensor([0])
        inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))

        return inputs