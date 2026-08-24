
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import log
from torch.nn import functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from model.warpC import GLUNetModel

from data.smart_data_loader_parismap import Data_dual_load_change
from data.synthetic_warps_sampling import SynthecticAffHomoTPSTransfo, GetRandomSyntheticAffHomoTPSFlow
from data.create_triplet import BatchedImageTripletCreation

from losses import L1, MultiScaleFlow, WBipathLoss

from plot import plot_flows_warpc

import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter #, RandomGaussianBlur
from data.color_augmentation_torch import RandomGaussianBlur

import datetime



def pre_process_image_glunet(source_img, apperance_transform=None):
    source_img_copy = source_img.float().cuda().div(255.0)
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().cuda(), size=(256, 256), mode='area')
    source_img_256 = source_img_256.float().div(255.0)
    if apperance_transform is not None:
        source_img_copy = apperance_transform(source_img_copy)
        source_img_256 = apperance_transform(source_img_256)
    return source_img_copy.cuda(), source_img_256.cuda()


if __name__ == '__main__':
    #### MODEL ####
    model = GLUNetModel()  
    model.cuda()
   
    #### DATALOADER ####
    image_path = '../../output/displacement_training/'

    w_size = 520
    train_dataset = Data_dual_load_change(image_path,  w_size, mode='train')
    val_dataset    = Data_dual_load_change(image_path,  w_size, mode='val')

    # 1. Apperance transform
    apperance_transform = transforms.Compose([ColorJitter(brightness=0.2,saturation=0.2, contrast=0.2), 
                                                                   RandomGaussianBlur(sigma=(0.2, 0.5),
                                                                                      probability=0.2)])  #sigma=(0.2, 1.0)
    # 2. Define dataloaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    n_train = len(train_loader)
    n_val = len(val_loader)
    print('Number of training batches: {}'.format(n_train))
    # 3. Create synthetic flow
    sample_transfo = SynthecticAffHomoTPSTransfo(size_output_flow=w_size)
    synthetic_flow_generator = GetRandomSyntheticAffHomoTPSFlow(size_output_flow=w_size, transfo_sampling_module=sample_transfo)

    # 4. Create data augmentation
    triplet_creator = BatchedImageTripletCreation(synthetic_flow_generator=synthetic_flow_generator, output_size=w_size, crop_size=w_size)

    # 5. optimizer
    steps= [] #[20, 40, 60]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, weight_decay=0.0004)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.5)$
    lr_scheduler = None
    lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step'}

    # #### LOSS FUNCTION ####
    # actual objective is L1, with different weights for the different levels.
    objective = L1(reduction='mean')
    weights_level_loss = [0.32, 0.08, 0.02, 0.01]
    loss_objective = MultiScaleFlow(level_weights=weights_level_loss[2:], loss_function=objective, downsample_gt_flow=True)
    loss_objective_256 = MultiScaleFlow(level_weights=weights_level_loss[:2], loss_function=objective, downsample_gt_flow=True)

    # Pre-defined parameters
    loss_weight_default = {'w_bipath': 1.0, 'warp_supervision': 1.0,
            'w_bipath_constant': 1.0, 'warp_supervision_constant': 1.0,
            'cc_mask_alpha_1': 0.03, 'cc_mask_alpha_2': 0.5}

    loss_weight = {'warp_supervision': 1.0, 'w_bipath': 1.0,
            'warp_supervision_constant': 1.0, 'w_bipath_constant': 1.0}
    
    detach_flow_for_warping = True
    unsupervised_objective = WBipathLoss(loss_objective, detach_flow_for_warping=detach_flow_for_warping,
                                            compute_cyclic_consistency=False,
                                            alpha_1=loss_weight_default['cc_mask_alpha_1'], 
                                            alpha_2=loss_weight_default['cc_mask_alpha_2'])
    unsupervised_objective_256 = WBipathLoss(loss_objective_256, detach_flow_for_warping=detach_flow_for_warping, 
                                             compute_cyclic_consistency=False, 
                                             alpha_1=loss_weight_default['cc_mask_alpha_1'],
                                             alpha_2=loss_weight_default['cc_mask_alpha_2'])
    best_loss = np.inf
    training_or_validation = 'train'
    base_save_dir = './results/' + str(datetime.datetime.now().strftime("%b-%d_%H_%M_%S")) 
    if not os.path.isdir('./results/'):
        os.makedirs('./results/')
    if not os.path.isdir(base_save_dir):
        os.makedirs(base_save_dir)
    if not os.path.isdir(base_save_dir + '/train'):
        os.makedirs(base_save_dir + '/train')
    if not os.path.isdir(base_save_dir + '/val'):
        os.makedirs(base_save_dir + '/val')

    logger = log.get_logger(os.path.join(base_save_dir, '{}.txt'.format('log')))
    epochs = 50
    ### write config #####
    config = {
        'batch_size': batch_size,
        'optimizer': optimizer,
        'epochs': epochs,
        'steps':steps,
        'apperance_transforms': apperance_transform,
        'path': image_path,
        'run': model.name,
        'notes': 'Unsupervised and supervised training',
    }
    file_path = os.path.join(base_save_dir, '{}.txt'.format('config'))

    with open(file_path, 'w') as file:
        file.write("Neural Network Configuration Settings\n")
        file.write("="*40 + "\n")
    
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

epochs = 100
for epoch in range(epochs):
    model.train()
    mean_loss = []
    mean_acc = []
    with tqdm(total=int(n_train*batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for i, ori_batch in enumerate(train_loader):
            mini_batch = triplet_creator(ori_batch)

            # pre process data (normalize the images and creates input for L-Net of GLU-Net)
            source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'])
            target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'])
            target_image_prime, target_image_prime_256 = pre_process_image_glunet(mini_batch['target_image_prime'])

            # Prepare the flow ground truth
            flow_gt = mini_batch['flow_map'].cuda()
            bs, _, h_original, w_original = flow_gt.shape

            # ground-truth flow field for 256x256 module
            flow_gt_256 = F.interpolate(flow_gt, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)

            # apply correspondence mask
            mask = mini_batch['correspondence_mask'].cuda()
            mask = F.interpolate(mask.float().unsqueeze(1), (h_original, w_original), mode='bilinear', align_corners=False).squeeze(1).floor()  # bxhxw
            mask = mask.bool()
            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear', align_corners=False).squeeze(1).floor()  # bx256x256, flooring
            mask_256 = mask_256.bool()

            mini_batch['source_image'] = source_image
            mini_batch['target_image'] = target_image
            mini_batch['target_image_prime'] = target_image_prime
            mini_batch['source_image_256'] = source_image_256
            mini_batch['target_image_256'] = target_image_256
            mini_batch['target_image_prime_256'] = target_image_prime_256

            mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].cuda()
            mini_batch['mask'] = mask
            mini_batch['flow_map'] = flow_gt  # between target_prime and target
            mini_batch['mask_256'] = mask_256
            mini_batch['flow_map_256'] = flow_gt_256  # between target_prime and target

            # forward pass
            # extract features to avoid recomputing them
            im_source_pyr = model.pyramid(mini_batch['source_image'], eigth_resolution=True)
            im_target_pyr = model.pyramid(mini_batch['target_image'], eigth_resolution=True)
            im_target_prime_pyr = model.pyramid(mini_batch['target_image_prime'], eigth_resolution=True)
            im_source_pyr_256 = model.pyramid(mini_batch['source_image_256'])
            im_target_pyr_256 = model.pyramid(mini_batch['target_image_256'])
            im_target_prime_pyr_256 = model.pyramid(mini_batch['target_image_prime_256'])

            output_target_to_source_256, output_flow_target_to_source = model(
            mini_batch['target_image'], mini_batch['source_image'],
            mini_batch['target_image_256'], mini_batch['source_image_256'],
            im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
            estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
            estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

            # Compute losses
            un_loss_256, un_loss_o, ss_loss_o, ss_loss_256 = 0.0, 0.0, 0.0, 0.0
            output_target_prime_to_target_directly_256, output_target_prime_to_target_directly = model(mini_batch['target_image_prime'], mini_batch['target_image'],
                                                                                                    mini_batch['target_image_prime_256'], mini_batch['target_image_256'],
                                                                                                    im_target_prime_pyr, im_target_pyr, im_target_prime_pyr_256, im_target_pyr_256)
            estimated_flow_target_prime_to_target_directly_256 = output_target_prime_to_target_directly_256['flow_estimates']
            estimated_flow_target_prime_to_target_directly = output_target_prime_to_target_directly['flow_estimates']
            ss_loss_o, ss_stats_o = loss_objective(estimated_flow_target_prime_to_target_directly, mini_batch['flow_map'], mask=mini_batch['mask'])
            ss_loss_256, ss_stats_256 = loss_objective_256(estimated_flow_target_prime_to_target_directly_256, mini_batch['flow_map_256'], mask=mini_batch['mask_256'])
            un_stats_256, un_stats_o, output_un_256, output_un = {}, {}, {}, {}
            
            # compute the unsupervised loss
            output_target_prime_to_source_256, output_target_prime_to_source = model(mini_batch['target_image_prime'], mini_batch['source_image'],
                                                                                    mini_batch['target_image_prime_256'], mini_batch['source_image_256'], im_target_prime_pyr,
                                                                                    im_source_pyr, im_target_prime_pyr_256, im_source_pyr_256)
            estimated_flow_target_prime_to_source_256 = output_target_prime_to_source_256['flow_estimates']
            estimated_flow_target_prime_to_source = output_target_prime_to_source['flow_estimates']

            output_source_to_target_256, output_source_to_target = model(mini_batch['source_image'], mini_batch['target_image'],
                                                                        mini_batch['source_image_256'], mini_batch['target_image_256'], im_source_pyr, im_target_pyr,
                                                                        im_source_pyr_256, im_target_pyr_256)
            estimated_flow_source_to_target_256 = output_source_to_target_256['flow_estimates']
            estimated_flow_source_to_target = output_source_to_target['flow_estimates']

            un_loss_o_, un_stats_o_, output_un_ = unsupervised_objective(mini_batch['flow_map'], mini_batch['mask'], 
                                                                        estimated_flow_target_prime_to_source,
                                                                        estimated_flow_source_to_target)

            un_loss_256_, un_stats_256_, output_un_256_ = unsupervised_objective_256(mini_batch['flow_map_256'], mini_batch['mask_256'], 
                                                                        estimated_flow_target_prime_to_source_256,
                                                                        estimated_flow_source_to_target_256)

            un_loss_o += un_loss_o_
            un_loss_256 += un_loss_256_ 
            loss = ss_loss_o + ss_loss_256 + un_loss_o + un_loss_256 # Supervised loss plus unsupervised loss

            if i % 10 == 0:  
                h, w = w_size, w_size
                output_target_to_source_256, output_flow_target_to_source = model(
                mini_batch['target_image'], mini_batch['source_image'],
                mini_batch['target_image_256'], mini_batch['source_image_256'],
                im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
                estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
                estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

                plot_flows_warpc(base_save_dir + '/train', 'un_batch{}_iter{}_reso_520'.format(batch_size, i), h, w,
                                image_source=mini_batch['source_image'],
                                image_target=mini_batch['target_image'],
                                image_target_prime=mini_batch['target_image_prime'],
                                estimated_flow_target_to_source=estimated_flow_target_to_source[-1],
                                estimated_flow_target_prime_to_source=estimated_flow_target_prime_to_source[-1],
                                estimated_flow_target_prime_to_target=estimated_flow_target_prime_to_target_directly[-1],
                                estimated_flow_target_prime_to_target_directly=
                                estimated_flow_target_prime_to_target_directly[-1],
                                gt_flow_target_prime_to_target=mini_batch['flow_map'],
                                mask=mini_batch['mask'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss.append(loss.item())

            pbar.update(batch_size)
            pbar.set_postfix(**{'total_loss': loss.item()})
    train_mean_loss = np.mean(mean_loss)

    model.eval()
    val_mean_loss = []
    val_mean_acc = []
    with tqdm(total=int(n_val*batch_size)-1, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for i, ori_batch in enumerate(val_loader):
            with torch.no_grad():
                mini_batch = triplet_creator(ori_batch)

                # pre process data (normalize the images and creates input for L-Net of GLU-Net)
                source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'])
                target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'])
                target_image_prime, target_image_prime_256 = pre_process_image_glunet(mini_batch['target_image_prime'])

                # Prepare the flow ground truth
                flow_gt = mini_batch['flow_map'].cuda()
                bs, _, h_original, w_original = flow_gt.shape

                # ground-truth flow field for 256x256 module
                flow_gt_256 = F.interpolate(flow_gt, (256, 256), mode='bilinear', align_corners=False)
                flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
                flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)

                # apply correspondence mask
                mask = mini_batch['correspondence_mask'].cuda()
                mask = F.interpolate(mask.float().unsqueeze(1), (h_original, w_original), mode='bilinear', align_corners=False).squeeze(1).floor()  # bxhxw
                mask = mask.bool()
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear', align_corners=False).squeeze(1).floor()  # bx256x256, flooring
                mask_256 = mask_256.bool()

                mini_batch['source_image'] = source_image
                mini_batch['target_image'] = target_image
                mini_batch['target_image_prime'] = target_image_prime
                mini_batch['source_image_256'] = source_image_256
                mini_batch['target_image_256'] = target_image_256
                mini_batch['target_image_prime_256'] = target_image_prime_256

                mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].cuda()
                mini_batch['mask'] = mask
                mini_batch['flow_map'] = flow_gt  # between target_prime and target
                mini_batch['mask_256'] = mask_256
                mini_batch['flow_map_256'] = flow_gt_256  # between target_prime and target

                # forward pass
                # extract features to avoid recomputing them
                im_source_pyr = model.pyramid(mini_batch['source_image'], eigth_resolution=True)
                im_target_pyr = model.pyramid(mini_batch['target_image'], eigth_resolution=True)
                im_target_prime_pyr = model.pyramid(mini_batch['target_image_prime'], eigth_resolution=True)
                im_source_pyr_256 = model.pyramid(mini_batch['source_image_256'])
                im_target_pyr_256 = model.pyramid(mini_batch['target_image_256'])
                im_target_prime_pyr_256 = model.pyramid(mini_batch['target_image_prime_256'])

                output_target_to_source_256, output_flow_target_to_source = model(
                mini_batch['target_image'], mini_batch['source_image'],
                mini_batch['target_image_256'], mini_batch['source_image_256'],
                im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
                estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
                estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

                # Compute losses
                un_loss_256, un_loss_o, ss_loss_o, ss_loss_256 = 0.0, 0.0, 0.0, 0.0
                output_target_prime_to_target_directly_256, output_target_prime_to_target_directly = model(mini_batch['target_image_prime'], mini_batch['target_image'],
                                                                                                        mini_batch['target_image_prime_256'], mini_batch['target_image_256'],
                                                                                                        im_target_prime_pyr, im_target_pyr, im_target_prime_pyr_256, im_target_pyr_256)
                estimated_flow_target_prime_to_target_directly_256 = output_target_prime_to_target_directly_256['flow_estimates']
                estimated_flow_target_prime_to_target_directly = output_target_prime_to_target_directly['flow_estimates']
                ss_loss_o, ss_stats_o = loss_objective(estimated_flow_target_prime_to_target_directly, mini_batch['flow_map'], mask=mini_batch['mask'])
                ss_loss_256, ss_stats_256 = loss_objective_256(estimated_flow_target_prime_to_target_directly_256, mini_batch['flow_map_256'], mask=mini_batch['mask_256'])
                un_stats_256, un_stats_o, output_un_256, output_un = {}, {}, {}, {}
                
                # compute the unsupervised loss
                output_target_prime_to_source_256, output_target_prime_to_source = model(mini_batch['target_image_prime'], mini_batch['source_image'],
                                                                                        mini_batch['target_image_prime_256'], mini_batch['source_image_256'], im_target_prime_pyr,
                                                                                        im_source_pyr, im_target_prime_pyr_256, im_source_pyr_256)
                estimated_flow_target_prime_to_source_256 = output_target_prime_to_source_256['flow_estimates']
                estimated_flow_target_prime_to_source = output_target_prime_to_source['flow_estimates']

                output_source_to_target_256, output_source_to_target = model(mini_batch['source_image'], mini_batch['target_image'],
                                                                            mini_batch['source_image_256'], mini_batch['target_image_256'], im_source_pyr, im_target_pyr,
                                                                            im_source_pyr_256, im_target_pyr_256)
                estimated_flow_source_to_target_256 = output_source_to_target_256['flow_estimates']
                estimated_flow_source_to_target = output_source_to_target['flow_estimates']

                un_loss_o_, un_stats_o_, output_un_ = unsupervised_objective(mini_batch['flow_map'], mini_batch['mask'], 
                                                                            estimated_flow_target_prime_to_source,
                                                                            estimated_flow_source_to_target)

                un_loss_256_, un_stats_256_, output_un_256_ = unsupervised_objective_256(mini_batch['flow_map_256'], mini_batch['mask_256'], 
                                                                            estimated_flow_target_prime_to_source_256,
                                                                            estimated_flow_source_to_target_256)

                un_loss_o += un_loss_o_
                un_loss_256 += un_loss_256_ 
                val_loss = ss_loss_o + ss_loss_256 + un_loss_o + un_loss_256 # Supervised loss plus unsupervised loss

                if i % 10 == 0:
                    h, w = w_size, w_size
                    output_target_to_source_256, output_flow_target_to_source = model(
                    mini_batch['target_image'], mini_batch['source_image'],
                    mini_batch['target_image_256'], mini_batch['source_image_256'],
                    im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
                    estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
                    estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

                    plot_flows_warpc(base_save_dir + '/val', 'un_batch{}_iter{}_reso_520'.format(batch_size, i), h, w,
                                image_source=mini_batch['source_image'],
                                image_target=mini_batch['target_image'],
                                image_target_prime=mini_batch['target_image_prime'],
                                estimated_flow_target_to_source=estimated_flow_target_to_source[-1],
                                estimated_flow_target_prime_to_source=estimated_flow_target_prime_to_source[-1],
                                estimated_flow_target_prime_to_target=estimated_flow_target_prime_to_target_directly[-1],
                                estimated_flow_target_prime_to_target_directly=
                                estimated_flow_target_prime_to_target_directly[-1],
                                gt_flow_target_prime_to_target=mini_batch['flow_map'],
                                mask=mini_batch['mask'])

                val_mean_loss.append(val_loss.item())

            pbar.update(batch_size)
            pbar.set_postfix(**{'val_total_loss': val_loss.item()})

    val_mean_loss = np.mean(val_mean_loss)
    logger.info('lr: %e, train_total_loss: %f, val_total_loss: %f' % 
            (optimizer.param_groups[0]['lr'],
                torch.from_numpy(np.array(train_mean_loss)).cuda(),
                torch.from_numpy(np.array(val_mean_loss)).cuda()
                )
            )
    # lr_scheduler.step()
    if val_mean_loss <= best_loss:
        best_loss = val_mean_loss
        print('Best loss:{}'.format(best_loss), "epoch:{}".format(epoch))
        torch.save(model.state_dict(), os.path.join(base_save_dir, 'best_weight.pth'.format(epoch)))

