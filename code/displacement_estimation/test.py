import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
from matplotlib import pyplot as plt
from model.warpC import GLUNetModel
from data.reconstruct_tiling_dict import reconstruct_from_patches
import numpy as np
import torch
from tqdm.auto import tqdm
import argparse
import torch.nn.functional as F  
from data.smart_data_loader_parismap import Data_dual
import flow_vis
from model.pixel_wise_mapping import remap_using_flow_fields


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def length_sq(x):
    return np.sum(x**2, axis=-1)


def pre_process_image_glunet(source_img, apperance_transform=None):
    source_img_copy = source_img.float().cuda().div(255.0)
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().cuda(), size=(256, 256), mode='area')
    source_img_256 = source_img_256.float().div(255.0)
    if apperance_transform is not None:
        source_img_copy = apperance_transform(source_img_copy)
        source_img_256 = apperance_transform(source_img_256)
    return source_img_copy.cuda(), source_img_256.cuda()

def test(network, args):

    batch_size = 1
    w_size = args.w_size
    test_img  = Data_dual(args.src_image, args.tar_image, w_size)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    n_test = len(testloader)
    patches_image_flow = []  #target to source
    patches_image_flow_st = [] # source to target
    patches_image_flow_change = []
    patches_image_change_warp = []
    patches_image = []
    patches_image_ori = []
    patches_image_est = []
    patches_gt_est = []
    patches_src_est = []
    network.eval()
    for i, mini_batch in enumerate(tqdm(testloader)):
        if args.cuda:
            img1, img2 = mini_batch['source_image'].cuda(), mini_batch['target_image'].cuda()
            src_gt = mini_batch['changes'].cuda()

        source_image, source_image_256 = pre_process_image_glunet(img1)
        target_image, target_image_256 = pre_process_image_glunet(img2)

        im_source_pyr = network.pyramid(source_image, eigth_resolution=True)
        im_target_pyr = network.pyramid(target_image, eigth_resolution=True)

        im_source_pyr_256 = network.pyramid(source_image_256)
        im_target_pyr_256 = network.pyramid(target_image_256)

        with torch.no_grad():
            output_target_to_source_256, output_target_to_source = network(target_image, source_image,
                                                                            target_image_256, source_image_256,
                                                                            im_target_pyr, im_source_pyr,
                                                                            im_target_pyr_256, im_source_pyr_256)  # change from source to target w.r.t. target
            output_source_to_target_256, output_source_to_target  = network(source_image, target_image,
                                                                            source_image_256, target_image_256,
                                                                            im_source_pyr, im_target_pyr,
                                                                            im_source_pyr_256, im_target_pyr_256) # change from target to source w.r.t. source
            
        estimated_flow_target_to_source = output_target_to_source['flow_estimates'][-1]
        estimated_flow_target_to_source = F.interpolate(estimated_flow_target_to_source, (w_size, w_size), mode='bilinear', align_corners=False)
        estimated_flow_target_to_source = estimated_flow_target_to_source.detach().permute(0, 2, 3, 1).cpu().numpy()

        estimated_flow_source_to_target = output_source_to_target['flow_estimates'][-1]
        estimated_flow_source_to_target = F.interpolate(estimated_flow_source_to_target, (w_size, w_size), mode='bilinear', align_corners=False)
        estimated_flow_source_to_target = estimated_flow_source_to_target.detach().permute(0, 2, 3, 1).cpu().numpy()

        source_image = source_image[0].permute(1, 2, 0).detach().cpu().numpy()

        target_est = remap_using_flow_fields(source_image, estimated_flow_target_to_source[0][:,:,0],
                                                           estimated_flow_target_to_source[0][:,:,1])
        source_est = remap_using_flow_fields(target_image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                              estimated_flow_source_to_target[0][:,:,0], estimated_flow_source_to_target[0][:,:,1])


        source_image = (source_image * 255.).astype(np.uint8)
        target_est = (target_est * 255.).astype(np.uint8)
        source_est = (source_est * 255.).astype(np.uint8)

        target = ((torch.permute(img2[0], (1, 2, 0))).byte()).detach().cpu().numpy()

        overlay_ori = (source_image  * 0.5 + target * 0.5).astype(np.uint8)
        overlay     = (target_est * 0.5 + target * 0.5).astype(np.uint8)
    
        overlay_all = np.concatenate([source_image, target, overlay_ori, overlay], axis=1)
        if i % 20 == 0:
            plt.imsave(args.path + '/inf_predictions/overlay_{}.jpg'.format(i), overlay_all.astype(np.uint8))

        patches_image.append(np.expand_dims(overlay, axis=0))
        patches_image_ori.append(np.expand_dims(overlay_ori, axis=0))
        patches_image_est.append(np.expand_dims(target_est, axis=0))
        patches_src_est.append(np.expand_dims(source_est, axis=0))
        patches_image_flow.append(estimated_flow_target_to_source)
        patches_image_flow_st.append(estimated_flow_source_to_target)
            
    patches_images = np.concatenate((patches_image), axis=0)
    patches_image_ori = np.concatenate((patches_image_ori), axis=0)
    patches_image_est = np.concatenate((patches_image_est), axis=0)
    patches_image_src_est = np.concatenate((patches_src_est), axis=0)
    patches_images_flow = np.concatenate((patches_image_flow), axis=0)
    patches_images_flow_st = np.concatenate((patches_image_flow_st), axis=0)

    return patches_images_flow, patches_images_flow_st, patches_images, patches_image_ori, patches_image_est, patches_image_src_est


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    network = GLUNetModel()
    path = args.path
    modelpath = path + '/' + args.model
    # loading Network
    network.load_state_dict(torch.load(modelpath))
    msg = 'Loading pretrained model from {}'.format(args.model)
    network.cuda()
    print(msg)
    print('Source year: {}'.format(args.year_sc), 'Target year: {}'.format(args.year_tar))

    args.src_image = plt.imread('../../data/displacement_estimation/test_{}.png'.format(args.year_sc))   #1895
    size = args.src_image.shape
    args.tar_image = plt.imread('../../data/displacement_estimation/test_{}.png'.format(args.year_tar))[:size[0], :size[1], :]  #1905  need to read as array below
    if args.src_image.max() < 2:
        args.src_image = (args.src_image*255).astype(np.uint8)
    if args.tar_image.max() < 2:
        args.tar_image = (args.tar_image*255).astype(np.uint8)
    

    path = os.path.join(path, 'test_{}_{}'.format(args.year_sc, args.year_tar))
    args.path = path
    if not os.path.exists(path):
        os.makedirs(path)
    res_dir = os.path.join(path, 'inf_displacement')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(path +'/inf_img/'):
        os.makedirs(path +'/inf_img/')
    if not os.path.exists(path +'/inf_img/displacement/'):
        os.makedirs(path +'/inf_img/displacement/')
    if not os.path.exists(path +'/inf_img/overlay/'):
        os.makedirs(path +'/inf_img/overlay/')
    if not os.path.exists(path +'/inf_predictions/'):
        os.makedirs(path +'/inf_predictions/')

    patches_images_ws, patches_images_wt, patches_images, patches_image_ori, patches_image_est, pactches_image_src = test(network, args)

    # Reconstruct image
    tile_save_image_path_ts = os.path.join(res_dir, 'flow_ts_{}_{}.npy'.format(args.year_tar, args.year_sc))
    tile_save_image_path_st = os.path.join(res_dir, 'flow_st_{}_{}.npy'.format(args.year_sc, args.year_tar))
    in_img =  args.src_image  
    w_size = args.w_size
    pad_px = w_size // 2
    flowUp_vis = reconstruct_from_patches(patches_images_ws, w_size, pad_px, in_img.shape, np.float32, mode='flow')
    flowUp_vis_st = reconstruct_from_patches(patches_images_wt, w_size, pad_px, in_img.shape, np.float32, mode='flow')
    overlay_vis = reconstruct_from_patches(patches_images, w_size, pad_px, in_img.shape, np.float32, mode='image')
    overlay_vis_ori = reconstruct_from_patches(patches_image_ori, w_size, pad_px, in_img.shape, np.float32, mode='image')
    est_target_vis= reconstruct_from_patches(patches_image_est, w_size, pad_px, in_img.shape, np.float32, mode='image')
    est_src_vis= reconstruct_from_patches(pactches_image_src, w_size, pad_px, in_img.shape, np.float32, mode='image')

    plt.imsave(path +'/inf_img/overlay/overlay_{}_{}.jpg'.format(args.year_sc, args.year_tar), overlay_vis.astype(np.uint8))
    plt.imsave(path +'/inf_img/overlay/overlay_ori_{}_{}.jpg'.format(args.year_sc, args.year_tar), overlay_vis_ori.astype(np.uint8))
    plt.imsave(path +'/inf_img/overlay/est_target_vis_{}_{}.jpg'.format(args.year_sc, args.year_tar), est_target_vis.astype(np.uint8))
    plt.imsave(path +'/inf_img/overlay/target_ori.jpg', (args.tar_image).astype(np.uint8))
    plt.imsave(path +'/inf_img/overlay/src_ori.jpg', (args.src_image).astype(np.uint8))
    plt.imsave(path +'/inf_img/overlay/est_src_vis_{}_{}.jpg'.format(args.year_tar, args.year_sc), est_src_vis.astype(np.uint8))

     # write raster
    with open(tile_save_image_path_ts, 'wb') as f:
        np.save(tile_save_image_path_ts, flowUp_vis)
    with open(tile_save_image_path_st, 'wb') as f:
        np.save(tile_save_image_path_st, flowUp_vis_st)

    
    flow_rgb_ts = flow_vis.flow_to_color(flowUp_vis.astype(np.float32), convert_to_bgr=True)
    cv2.imwrite(path +'/inf_img/displacement/flow_ts_{}_{}_vis.jpg'.format(args.year_tar, args.year_sc), flow_rgb_ts)
    flow_rgb_st = flow_vis.flow_to_color(flowUp_vis_st, convert_to_bgr=True)
    cv2.imwrite(path +'/inf_img/displacement/flow_st_{}_{}_vis.jpg'.format(args.year_sc, args.year_tar), flow_rgb_st)


def parse_args():
    parser = argparse.ArgumentParser('Test UNET')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed control.')
    parser.add_argument('--year_sc', type=str, default=50,
                        help='Year.')
    parser.add_argument('--year_tar', type=str, default=50,
                        help='Year.')
    parser.add_argument('--model_type', type=str, default='copy_net',
                        help='The type of the model')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str,
                        default=None, help='the model to test')
    parser.add_argument('--path', type=str, help='folder of checkpoints')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
                        help='number of classes in the output')
    parser.add_argument('--mode', type=str, default='Residual',
                        help='The mode of nCONV layer')
    parser.add_argument('--w_size', type=int, default=520)
    parser.add_argument('--warp_change', action='store_true',
                        help='warp change')
    return parser.parse_args()


if __name__ == '__main__':
    main()
