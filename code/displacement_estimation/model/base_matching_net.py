import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from packaging import version

from model.modules.consensus_network_modules import MutualMatching
from model.model_utils import conv, predict_flow
from model.modules.feature_correlation_layer import FeatureL2Norm, GlobalFeatureCorrelationLayer
from model.flow_and_mapping_operations import convert_mapping_to_flow, convert_flow_to_mapping


class BaseMultiScaleMatchingNet(nn.Module):
    """
    Common to all multiscale dense matching architectures
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = None

    @staticmethod
    def resize_and_rescale_flow(flow, output_size):
        # output size is h, w
        b, _, h, w = flow.shape
        if h == output_size[0] and w == output_size[1]:
            return flow

        flow = F.interpolate(input=flow, size=output_size, mode='bilinear', align_corners=False)
        flow[:, 0] *= float(output_size[1]) / float(w)
        flow[:, 1] *= float(output_size[0]) / float(h)
        return flow

    @staticmethod
    def scale_flow_to_resolution(flow, ratio_x, ratio_y=None, div=1.0):
        if ratio_y is None:
            ratio_y = ratio_x
        flow_warping = flow * div
        flow_warping[:, 0, :, :] *= ratio_x
        flow_warping[:, 1, :, :] *= ratio_y
        return flow_warping

    @staticmethod
    def get_nbr_features_pyramid(pyramid_type):
        # reso 1/4, 1/8, 1/16
        if pyramid_type == 'VGG':
            nbr_features = [128, 256, 512]
        elif pyramid_type == 'ResNet':
            nbr_features = []
        elif pyramid_type == 'PWCNet':
            nbr_features = []
        else:
            raise NotImplementedError('The feature extractor that you selected in not implemented: {}'
                                      .format(pyramid_type))
        return nbr_features


    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if version.parse(torch.__version__) >= version.parse("1.3"):
            # to be consistent to old version, I put align_corners=True.
            # to investigate if align_corners False is better.
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)
        return output

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, *input):
        raise NotImplementedError


class BaseGLUMultiScaleMatchingNet(BaseMultiScaleMatchingNet):
    """Base class for GLU-Net based networks."""

    def __init__(self, pyramid=None, pyramid_256=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.params = params
        self.visdom = None
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=False)

        if pyramid is not None:
            self.pyramid = pyramid
        if pyramid_256 is not None:
            self.pyramid_256 = pyramid_256

    def initialize_global_corr(self):
        # Feature correlation layer
        self.corr = GlobalFeatureCorrelationLayer(shape='3D', normalization=False,
                                                    put_W_first_in_channel_dimension=False)


    def get_global_correlation(self, c14, c24):
        """
        Based on feature maps c14 and c24, computes the global correlation from c14 to c24.
        c14 basically corresponds to target/reference image and c24 to source/query image.

        Here, in the simplest version (without GOCor), we follow DGC-Net.
        the features are usually L2 normalized, and the cost volume is post-processed with relu, and l2 norm
        Args:
            c14:  B, c, h_t, w_t
            c24:  B, c, h_s, w_s

        Returns:
            corr4: global correlation B, h_s*w_s, h_t, w_t
        """
        b = c14.shape[0]
        # directly obtain the 3D correlation
        corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target

        # to add on top of the correlation ! (already included in NC-Net)
        corr4d = MutualMatching(corr4.view(b, c24.shape[2], c24.shape[3], c14.shape[2], c14.shape[3]).unsqueeze(1))
        corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])

        corr4 = self.l2norm(F.relu(corr4))
        return corr4

    def initialize_last_level_refinement_module(self, input_to_refinement, batch_norm):
        # refinement module from PWC-Net
        self.l_dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                               batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

    def initialize_adaptive_reso_refinement_module(self,  input_to_refinement, batch_norm):
        # refinement module from PWC-Net
        self.dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                             batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

    def PWCNetRefinementAdaptiveReso(self, x):
        # refinement module from PWC-Net
        x = self.dc_conv6(self.dc_conv5(self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))))
        res = self.dc_conv7(x)
        return x, res

    def PWCNetRefinementFinal(self, x):
        # refinement module from PWC-Net
        x = self.l_dc_conv6(self.l_dc_conv5(self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))))
        res = self.l_dc_conv7(x)
        return x, res

    def extract_pyramid(self, im, im_256):
        # pyramid for the high resolution image
        im_pyr = self.pyramid(im, eigth_resolution=True)

        # pyramid for the 256x256 image
        im_pyr_256 = self.pyramid(im_256)
        return im_pyr, im_pyr_256

    def extract_features(self, im_target, im_source, im_target_256, im_source_256,
                         im_target_pyr=None, im_source_pyr=None, im_target_pyr_256=None, im_source_pyr_256=None):
        # pyramid, original reso
        if im_target_pyr is None:
            im_target_pyr = self.pyramid(im_target, eigth_resolution=True)
        if im_source_pyr is None:
            im_source_pyr = self.pyramid(im_source, eigth_resolution=True)
        c11 = im_target_pyr[-2]  # size original_res/4xoriginal_res/4
        c21 = im_source_pyr[-2]
        c12 = im_target_pyr[-1]  # size original_res/8xoriginal_res/8
        c22 = im_source_pyr[-1]

        # pyramid, 256 reso
        if im_target_pyr_256 is None:
            im_target_pyr_256 = self.pyramid(im_target_256)
        if im_source_pyr_256 is None:
            im_source_pyr_256 = self.pyramid(im_source_256)
        c13 = im_target_pyr_256[-2]  # size 256/8 x 256/8
        c23 = im_source_pyr_256[-2]  # size 256/8 x 256/8
        c14 = im_target_pyr_256[-1]  # size 256/16 x 256/16
        c24 = im_source_pyr_256[-1]  # size 256/16 x 256/16

        return c14, c24, c13, c23, c12, c22, c11, c21


    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first'):
        """
        Estimates the flow field relating the target to the source image. Returned flow has output_shape if provided,
        otherwise the same dimension than the target image. If scaling is provided, the output shape is the
        target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
            = self.pre_process_data(source_img, target_img)
        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)

    # FOR FLIPPING CONDITION
    def flipping_condition(self, im_source_base, im_target_base, device):
        condition = 'min_avg_flow'

        list_average_flow = []
        false_true = [False, True]
        for apply_flipping in false_true:
            im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping)
            b, _, h_original, w_original = im_target.size()
            b, _, h_256, w_256 = im_target_256.size()

            # pyramid, 256 reso
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
            c14 = im1_pyr_256[-1]
            c24 = im2_pyr_256[-1]

            corr4, flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
            if condition == 'min_avg_flow':
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
            else:
                value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                average_flow = value.sum()
            list_average_flow.append(average_flow.item())
        if condition == 'min_avg_flow':
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
        else:
            target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]
        if target_image_is_flipped:
            list_average_flow = []

            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                    self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping)
                b, _, h_original, w_original = im_target.size()
                b, _, h_256, w_256 = im_target_256.size()

                # pyramid, 256 reso
                im1_pyr_256 = self.pyramid(im_target_256)
                im2_pyr_256 = self.pyramid(im_source_256)
                c14 = im1_pyr_256[-1]
                c24 = im2_pyr_256[-1]

                corr4, flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                if condition == 'min_avg_flow':
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                else:
                    value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                    average_flow = value.sum()
                list_average_flow.append(average_flow.item())
            if condition == 'min_avg_flow':
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            else:
                target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]

        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x_original, ratio_y_original = \
            self.pre_process_data(im_source_base, im_target_base, apply_flip=target_image_is_flipped)
        return im_source.to(device).contiguous(), im_target.to(device).contiguous(), \
               im_source_256.to(device).contiguous(), im_target_256.to(device).contiguous(), \
               ratio_x_original, ratio_y_original

    def estimate_flow_with_flipping_condition(self, source_img, target_img, output_shape=None,
                                              scaling=1.0, mode='channel_first'):
        """
        Estimates the flow field relating the target to the source image with flipping condition.
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y = \
            self.flipping_condition(source_img, target_img, self.device)

        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        if self.target_image_is_flipped:
            flipped_mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)\
                .permute(0, 2, 3, 1).cpu().numpy()
            b = flipped_mapping.shape[0]
            mapping_per_batch = []
            for i in range(b):
                map = np.copy(np.fliplr(flipped_mapping[i]))
                mapping_per_batch.append(map)

            mapping = torch.from_numpy(np.float32(mapping_per_batch)).permute(0, 3, 1, 2).to(self.device)
            flow_est = convert_mapping_to_flow(mapping, self.device)

        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)
