import os
from utils__.reconstruct_tiling_dict import reconstruct_from_patches_blend
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from data.smart_data_loader import Data_instance_seq_val
from torch.utils.data import DataLoader
import torch
from model import TemporalMask2Former
from transformers import  MaskFormerImageProcessor, Mask2FormerConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch


def collate_fn(batch):
    pixel_values = [example["pixel_values"] for example in batch] 
    pixel_mask = [example["pixel_mask"] for example in batch] 
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}



if __name__ == '__main__':
    id2label = {0: 'background', 1: 'building'}
    w_size = 256 # original: 512

    state = torch.load("../../pretrained_weights/instance_extraction/best_model.pth", map_location="cpu")
    # for k in state.keys():
    #     if "fusion" in k.lower() or "attn" in k.lower():
    #         print(k)

    cfg = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-large-coco-panoptic", id2label=id2label, ignore_mismatched_sizes=True)
    cfg.num_queries = 200
    model = TemporalMask2Former(cfg)
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = MaskFormerImageProcessor(do_reduce_labels=False, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)

    basedir = '../../data/instance_extraction/'

    test_img_list = os.listdir(os.path.join(basedir, 'map_1880'))
    test_img_list = [os.path.join(basedir, 'map_1880', f) for f in test_img_list if f.endswith('.jpg')]
    data = Data_instance_seq_val(test_img_list, None, processor, w_size)
   
    image_original = plt.imread(test_img_list[0])

    infer_dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.to(device)
    model.eval()
    patches_images = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(infer_dataloader)):# test on a subset for speed
            outputs = model(
                    pixel_values= [x.to(device) for x in batch["pixel_values"]],  #.to(device)
                    mask_labels=[x.to(device) for x in batch["mask_labels"]],
                    class_labels=[x.to(device) for x in batch["class_labels"]],
            )

            # Backward propagation
            target_sizes = (w_size,w_size)
            tensor_list = [target_sizes for _ in range(len(batch["pixel_values"]))]
            pred_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=tensor_list)[0]['segmentation']
            pred_instance_map[batch["pixel_mask"][0]==0] = 0
            instance_map = pred_instance_map.detach().cpu().numpy()+1
            image = batch["pixel_values"][0][0].permute(1,2,0).detach().cpu().numpy()
            instance_map = np.stack([instance_map] * 3, axis=-1)/np.max(instance_map)
            concatenated_image = np.concatenate((instance_map, image), axis=1)
            patches_images.append(pred_instance_map[np.newaxis, ...])

    patches_images = np.concatenate(patches_images, axis=0)
    pad_px = w_size//2
    in_img_shape = image_original.shape[0:2]
    new_img = reconstruct_from_patches_blend(patches_images, w_size, pad_px, in_img_shape, np.float32)  

    uniques = np.unique(new_img)
    for i in range(1, len(uniques)):
        new_img[new_img == uniques[i]] = -1 if uniques[i] == -1 else i

    np.save('../../output/instance_inference/instance_seg_map_1880.npy', new_img)
    plt.imsave('../../output/instance_inference/instance_seg_map_1880.png', new_img/np.max(new_img))