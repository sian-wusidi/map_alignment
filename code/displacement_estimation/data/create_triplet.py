import torch
from packaging import version
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pdb

class BatchedImageTripletCreation:
    def __init__(self, synthetic_flow_generator, crop_size, output_size,
                 compute_mask_zero_borders=False, min_percent_valid_corr=0.1, padding_mode='zeros'):
        self.compute_mask_zero_borders = compute_mask_zero_borders

        self.synthetic_flow_generator = synthetic_flow_generator
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_percent_valid_corr = min_percent_valid_corr
        self.padding_mode = padding_mode

    def __call__(self, mini_batch, net=None, training=True, *args, **kwargs):
        output_mini_batch = mini_batch.copy()
        # take original images
        if 'source_image' in list(mini_batch.keys()):
            source_image = output_mini_batch['source_image'].cuda()
        target_image = output_mini_batch['target_image'].cuda()
        if 'target_image_prime' in list(mini_batch.keys()):
            target_image_prime = output_mini_batch['target_image_prime'].cuda()
        b, _, h, w = target_image.shape

        if h < self.output_size[0] or w < self.output_size[1]:
            # should be larger, otherwise the warping will create huge black borders
            print('Image or flow is the same size than desired output size in warping dataset ! ')

        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        flow_gt_trgprime_to_trg = self.synthetic_flow_generator(mini_batch=mini_batch)
        flow_gt_trgprime_to_trg.require_grad = False
        bs, _, h_f, w_f = flow_gt_trgprime_to_trg.shape

        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the size of the original images
            flow_gt_trgprime_to_trg = F.interpolate(flow_gt_trgprime_to_trg, (h, w), mode='bilinear',
                                                    align_corners=False)
            flow_gt_trgprime_to_trg[:, 0] *= float(w) / float(w_f)
            flow_gt_trgprime_to_trg[:, 1] *= float(h) / float(h_f)

        if 'changes' in list(mini_batch.keys()):
            changes = output_mini_batch['changes'].cuda()

        if 'target_image_prime' in list(mini_batch.keys()):
            target_image_prime, mask_zero_borders = warp(target_image_prime, flow_gt_trgprime_to_trg,
                                                     padding_mode=self.padding_mode,
                                                     return_mask=True)
            if 'changes' in list(mini_batch.keys()):
                changes, mask_zero_borders = warp(changes, flow_gt_trgprime_to_trg,
                                                        padding_mode=self.padding_mode,
                                                        return_mask=True)
        else:
            target_image_prime, mask_zero_borders = warp(target_image, flow_gt_trgprime_to_trg,
                                                     padding_mode=self.padding_mode,
                                                     return_mask=True)
        # target_image_prime = target_image_prime.byte()
        # because here, i still have my images in [0-255] so for augmentations to work well, the images
        # should be uint8 will be interpolated with 'area'

        # if there exists a ground-truth flow between the source and target image, also modify it so it corresponds
        # to the new source and target images.
        if 'flow_map' in list(mini_batch.keys()):
            if isinstance(mini_batch['flow_map'], list):
                flow_gt_target_to_source = mini_batch['flow_map'][-1].to(self.device)
                mask_gt_target_to_source = mini_batch['correspondence_mask'][-1].to(self.device)
            else:
                flow_gt_target_to_source = mini_batch['flow_map'].to(self.device)
                mask_gt_target_to_source = mini_batch['correspondence_mask'].to(self.device)
        else:
            flow_gt_target_to_source = None
            mask_gt_target_to_source = None

        

        # crop a center patch from the images and the ground-truth flow field, so that black borders are removed
        x_start = w // 2 - self.crop_size[1] // 2
        y_start = h // 2 - self.crop_size[0] // 2
        if 'source_image' in list(mini_batch.keys()):
            source_image_resized = source_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        if 'changes' in list(mini_batch.keys()):
            changes_resized = changes[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_resized = target_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_prime_resized = target_image_prime[:, :, y_start: y_start + self.crop_size[0],
                                                        x_start: x_start + self.crop_size[1]]
        flow_gt_trgprime_to_trg_resized = flow_gt_trgprime_to_trg[:, :, y_start: y_start + self.crop_size[0],
                                                                  x_start: x_start + self.crop_size[1]]
        mask_zero_borders = mask_zero_borders[:, y_start: y_start + self.crop_size[0],
                                              x_start: x_start + self.crop_size[1]]

        if 'flow_map' in list(mini_batch.keys()):
            flow_gt_target_to_source = flow_gt_target_to_source[:, :, y_start: y_start + self.crop_size[0],
                                                                x_start: x_start + self.crop_size[1]]
            if mask_gt_target_to_source is not None:
                mask_gt_target_to_source = mask_gt_target_to_source[:, y_start: y_start + self.crop_size[0],
                                                                    x_start: x_start + self.crop_size[1]]

        if 'target_kps' in mini_batch.keys():
            target_kp = mini_batch['target_kps'].to(self.device).clone()  # b, N, 2
            source_kp = mini_batch['source_kps'].to(self.device).clone()  # b, N, 2
            source_kp[:, :, 0] = source_kp[:, :, 0] - x_start   # will just make the not valid part even smaller
            source_kp[:, :, 1] = source_kp[:, :, 1] - y_start
            target_kp[:, :, 0] = target_kp[:, :, 0] - x_start
            target_kp[:, :, 1] = target_kp[:, :, 1] - y_start

        # resize to final output size, this is to prevent the crop from removing all common areas
        if self.output_size != self.crop_size:
            if 'source_image' in list(mini_batch.keys()):
                source_image_resized = F.interpolate(source_image_resized, self.output_size,
                                                    mode='bilinear')
            target_image_resized = F.interpolate(target_image_resized, self.output_size,
                                                 mode='bilinear')
            target_image_prime_resized = F.interpolate(target_image_prime_resized, self.output_size,
                                                       mode='bilinear')
            changes_resized = F.interpolate(changes_resized, self.output_size,
                                                    mode='bilinear')
            flow_gt_trgprime_to_trg_resized = F.interpolate(flow_gt_trgprime_to_trg_resized, self.output_size,
                                                            mode='bilinear', align_corners=False)
            flow_gt_trgprime_to_trg_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_trgprime_to_trg_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders = F.interpolate(mask_zero_borders.float().unsqueeze(1), self.output_size,
                                              mode='bilinear', align_corners=False).floor().squeeze(1)

            if 'flow_map' in list(mini_batch.keys()):
                flow_gt_target_to_source = F.interpolate(flow_gt_target_to_source, self.output_size,
                                                         mode='bilinear', align_corners=False)
                flow_gt_target_to_source[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                flow_gt_target_to_source[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                if mask_gt_target_to_source is not None:
                    mask_gt_target_to_source = F.interpolate(mask_gt_target_to_source.unsqueeze(1).float(),
                                                             self.output_size,
                                                             mode='bilinear', align_corners=False).floor()
                    mask_gt_target_to_source = mask_gt_target_to_source.bool() if \
                        version.parse(torch.__version__) >= version.parse("1.1") else mask_gt_target_to_source.byte()

            # if target kps, also resize them
            if 'target_kps' in mini_batch.keys():
                source_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                source_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                target_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                target_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

        # create ground truth correspondence mask for flow between target prime and target
        mask_gt = create_border_mask(flow_gt_trgprime_to_trg_resized)
        mask_gt = mask_gt.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_gt.byte()

        if self.compute_mask_zero_borders:
            # if mask_gt has too little commun areas, overwrite to use that mask in anycase
            if mask_gt.sum() < mask_gt.shape[-1] * mask_gt.shape[-2] * self.min_percent_valid_corr:
                mask_zero_borders = mask_gt
            '''
            else:
                # if padding mode is zeros, can compute the mask_zero_borders like this, from the intensity of the image
                # mask black borders that might have appeared from the warping, when creating target_image_prime
                mask_zero_borders = define_mask_zero_borders(target_image_prime_resized)
            '''
            output_mini_batch['mask_zero_borders'] = mask_zero_borders.bool() if \
                version.parse(torch.__version__) >= version.parse("1.1") else mask_zero_borders.byte()

        if 'target_kps' in mini_batch.keys():
            output_mini_batch['target_kps'] = target_kp  # b, N, 2
            output_mini_batch['source_kps'] = source_kp  # b, N, 2

        if 'flow_map' in list(mini_batch.keys()):
            # gt between target and source (what we are trying to estimate during training unsupervised)
            # it corresponds to 'flow_map' of the ORIGINAL mini_batch.
            output_mini_batch['flow_map_target_to_source'] = flow_gt_target_to_source
            output_mini_batch['correspondence_mask_target_to_source'] = mask_gt_target_to_source

        # save the new batch information
        if 'source_image' in list(mini_batch.keys()):
            output_mini_batch['source_image'] = source_image_resized.byte()
        if 'changes' in list(mini_batch.keys()):
            output_mini_batch['changes'] = changes_resized.byte()
        output_mini_batch['target_image'] = target_image_resized.byte()
        output_mini_batch['target_image_prime'] = target_image_prime_resized.byte()  # if apply transfo after
        output_mini_batch['correspondence_mask'] = mask_gt
        output_mini_batch['flow_map'] = flow_gt_trgprime_to_trg_resized


        # between target_prime and target, replace the old one
        return output_mini_batch

def create_border_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image), and convert to float. """
    return get_gt_correspondence_mask(flow).float()

def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w-1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h-1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w-1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h-1)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w-1) & mapping[1].ge(0) & mapping[1].le(h-1)
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    return mask

def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    args:
        x: [B, C, H, W]
        flo: [B, 2, H, W] flow
    outputs:
        output: warped x [B, C, H, W]
    """
    B, C, H, W = flo.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if version.parse(torch.__version__) >= version.parse("1.3"):
        output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode=padding_mode)
    else:
        output = nn.functional.grid_sample(x, vgrid, padding_mode=padding_mode)

    if return_mask:
        vgrid = vgrid.permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output

def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1,2,0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)