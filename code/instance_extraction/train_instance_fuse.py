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

def collate_fn(batch):
    pixel_values = [example["pixel_values"] for example in batch] #torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = [example["pixel_mask"] for example in batch] #torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

    
if __name__ == '__main__':
    id2label = {0: 'background', 1: 'building'}
    w_size = 256 # original: 512

    cfg = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-large-coco-panoptic", id2label=id2label, ignore_mismatched_sizes=True)
    cfg.num_queries = 200
    model = TemporalMask2Former(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = MaskFormerImageProcessor(do_reduce_labels=False, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor =MaskFormerImageProcessor(do_reduce_labels=False, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)


    basedir = '../../data/instance_extraction/'


    train_img_list = [os.path.join(basedir, 'map_1880')]

    train_gt_list = [os.path.join(basedir, 'map_1880_edgemap.png')]

    train_mask_list = [os.path.join(basedir, 'map_1880_road_mask.png')]
    all_training_data = []
    all_val_data = []
    # all_val_data = []
    for i, (train_img, train_gt, mask) in enumerate(zip(train_img_list, train_gt_list, train_mask_list)):
        data = Data_instance_seq(train_img, train_gt, mask, processor, w_size, mode='train')
        val_data = Data_instance_seq(train_img, train_gt, mask, processor, w_size, mode='val')
        all_training_data.append(data)
        all_val_data.append(val_data)


    all_training_data = ConcatDataset(all_training_data)
    train_dataloader = DataLoader(all_training_data, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)

    all_val_data = ConcatDataset(all_val_data)
    val_dataloader = DataLoader(all_val_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    checkpoint_dir = './training_log/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    best_loss = np.inf
    model.to(device)
    for epoch in range(100):
        print("Epoch:", epoch)
        model.train()
        num_samples = 0
        running_loss = 0.0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(
                    pixel_values=[x.to(device) for x in batch["pixel_values"]],
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()
            batch_size = len(batch["pixel_values"])
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)
                target_sizes = (w_size,w_size)
                tensor_list = [target_sizes for _ in range(batch_size)]
                pred_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=tensor_list)[0]['segmentation']
                instance_map =pred_instance_map.detach().cpu().numpy()+1
                image = batch["pixel_values"][0][0].permute(1,2,0).detach().cpu().numpy()
                instance_map = np.stack([instance_map] * 3, axis=-1)/np.max(instance_map)
                concatenated_image = np.concatenate((instance_map, image), axis=1)
                plt.imsave(checkpoint_dir + 'concatenated_train.png', concatenated_image)
            # Optimization
            optimizer.step()

        val_loss = 0
        num_samples = 0
        
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                outputs = model(
                    pixel_values=[x.to(device) for x in batch["pixel_values"]],
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
                loss = outputs.loss
                val_loss += loss.item()
                num_samples += len(batch["pixel_values"])
                if idx % 100 == 0:
                    target_sizes = (w_size,w_size)
                    tensor_list = [target_sizes for _ in range(batch_size)]
                    pred_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=tensor_list)[0]['segmentation']
                    instance_map =pred_instance_map.detach().cpu().numpy()+1
                    image = batch["pixel_values"][0][0].permute(1,2,0).detach().cpu().numpy()
                    instance_map = np.stack([instance_map] * 3, axis=-1) / np.max(instance_map)
                    
                    concatenated_image = np.concatenate((instance_map, image), axis=1)
                    plt.imsave(checkpoint_dir + 'concatenated_val.png', concatenated_image)
            print("Validation Loss:", val_loss/num_samples)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), checkpoint_dir + 'best_model_epoch_{}.pth'.format(epoch))
            with open(checkpoint_dir + 'loss_log.txt', 'a') as f:
                f.write(f"Epoch: {epoch}, Training Loss: {running_loss/num_samples}, Validation Loss: {val_loss/num_samples}\n")