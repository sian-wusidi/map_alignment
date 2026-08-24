import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from data.smart_data_loader import Data_instance_seq
from torch.utils.data import DataLoader
import torch
from model import TemporalMask2Former
from transformers import  MaskFormerImageProcessor, Mask2FormerConfig
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import torch


def collate_fn(batch):
    pixel_values = [example["pixel_values"] for example in batch] #torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = [example["pixel_mask"] for example in batch] #torch.stack([example["pixel_mask"] for example in batch])
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

    test_img_list = [os.path.join(basedir, 'map_1880')]

    test_gt_list = [os.path.join(basedir, 'map_1880_edgemap.png')]

    test_mask_list = [os.path.join(basedir, 'map_1880_road_mask.png')]
    all_test_data = []
    for i, (test_img, test_gt, mask) in enumerate(zip(test_img_list, test_gt_list, test_mask_list)):
        data = Data_instance_seq(test_img, test_gt, mask, processor, w_size, mode='val')  # we use all data for illustration purpose
        all_test_data.append(data)
    
    all_test_data = ConcatDataset(all_test_data) 
    test_dataloader = DataLoader(all_test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):# test on a subset for speed
            outputs = model(
                    pixel_values= [x.to(device) for x in batch["pixel_values"]],  #.to(device)
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            target_sizes = (w_size,w_size)
            tensor_list = [target_sizes for _ in range(len(batch["pixel_values"]))]
            pred_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=tensor_list)[0]['segmentation']
            pred_instance_map[batch["pixel_mask"][0]==0] = 0
            instance_map =pred_instance_map.detach().cpu().numpy()+1
            image = batch["pixel_values"][0][0].permute(1,2,0).detach().cpu().numpy()
            instance_map = np.stack([instance_map] * 3, axis=-1)/np.max(instance_map)
            concatenated_image = np.concatenate((instance_map, image), axis=1)
            plt.imsave( 'concatenated_test.png', concatenated_image)