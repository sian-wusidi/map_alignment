import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.VGG_features import VGGPyramid
from model.base_matching_net import BaseGLUMultiScaleMatchingNet
from model.model_utils import conv, conv_head, predict_flow, predict_mask, unnormalise_and_convert_mapping_to_flow, deconv
from model.modules.local_correlation import correlation
from model.modules.feature_correlation_layer import FeatureL2Norm


class GLUNetModel(BaseGLUMultiScaleMatchingNet):
    """
    GLU-Net model.
    The flows (flow2, flow1) are predicted such that they are scaled to the input image resolution. To obtain the flow
    from target to source at original resolution, one just needs to bilinearly upsample (without further scaling).
    """
    def __init__(self):
        super().__init__()

        self.name = 'GLUNetModel'
        # level 4, 16x16
        nd = 16*16  # global correlation
        od = nd + 2
        # here, I follow DGC-Net global correlation module + decoder
        self.decoder4 = CMDTopResidualConnections(in_channels=od, batch_norm=True)

        # level 3, 32x32
        nd = (2*4+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 2 + 0
        self.decoder3 = OpticalFlowEstimatorResidualConnection(in_channels=od, batch_norm=True)
        input_to_refinement_3 = 32

        # weights for refinement module
        self.initialize_adaptive_reso_refinement_module(input_to_refinement_3, True)

        # level 2, 1/8 of original resolution
        nd = (2*4+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 2
        self.decoder2 = OpticalFlowEstimatorResidualConnection(in_channels=od, batch_norm=True)
        input_to_refinement_2 = 32

        self.upfeat2 = deconv(input_to_refinement_2, 2, kernel_size=4,
                                stride=2, padding=1)

        # level 1, 1/4 of original resolution
        nd = (2*4+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 2 + 2  # 2 for flow, 2 for upfeat
        self.decoder1 = OpticalFlowEstimatorResidualConnection(in_channels=od, batch_norm=True)
        input_to_refinement_1 = 32

        self.initialize_last_level_refinement_module(input_to_refinement_1, True)

        # initialize modules
        for m in self.modules():
            # deconv initialized in its own module
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

        # initialize the global and local correlation modules
        # comprises GOCor or feature correlation layer
        self.initialize_global_corr()

        # features back-bone extractor
        feature_extractor = VGGPyramid(train=False)

        self.pyramid = feature_extractor

    def forward(self, im_target, im_source, im_target_256, im_source_256, im_target_pyr=None, im_source_pyr=None,
                im_target_pyr_256=None, im_source_pyr_256=None):
        """
        Args:
            im_target: torch Tensor Bx3xHxW, normalized with imagenet weights
            im_source: torch Tensor Bx3xHxW, normalized with imagenet weights
            im_target_256: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_source_256: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_target_pyr: in case the pyramid features are already computed.
            im_source_pyr: in case the pyramid features are already computed.
            im_target_pyr_256: in case the pyramid features are already computed.
            im_source_pyr_256: in case the pyramid features are already computed.

        Returns:
            output_256: dict with keys 'flow_estimates'. It contains the flow field of the two deepest levels
                        corresponding to the L-Net (flow4 and flow3), they are scaled for input resolution of 256x256.
            output: dict with keys 'flow_estimates'. It contains the flow field of the two shallowest levels
                    corresponding to the H-Net (flow2 and flow1), they are scaled for original (high resolution)
                    input resolution.
        """
        # im1 is target image, im2 is source image
        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()  # fixed size of 256x256
        div = 1.0

        c14, c24, c13, c23, c12, c22, c11, c21 = self.extract_features(im_target, im_source, im_target_256,
                                                                       im_source_256, im_target_pyr,
                                                                       im_source_pyr, im_target_pyr_256,
                                                                       im_source_pyr_256)
        # RESOLUTION 256x256
        # level 4: 16x16
        # here same procedure then DGC-Net global mapping decoder.
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        # init_map is fed to the decoder to be consistent with decoder of DGC-Net (but not particularly needed)
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)
        # we want flow4 to be scaled for h_256xw_256, so we multiply by these ratios.
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y

        up_flow4 = F.interpolate(input=flow4, size=(32, 32), mode='bilinear', align_corners=False)

        # level 3: 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        # the flow needs to be scaled for the current resolution in order to warp.
        # otherwise flow4 and up_flow4 are scaled for h_256xw_256.
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)

        # constrained correlation now
        # feature correlation layer
        corr3 = correlation.FunctionCorrelation(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(corr3)

        # PWC-Net refinement context module
        x_, res_flow3_ = self.PWCNetRefinementAdaptiveReso(x3)
        res_flow3 = res_flow3 + res_flow3_
        flow3 = res_flow3 + up_flow4

        # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
        # do iterative upsampling so that gap is always smaller than 2.
        R_w = float(w_original)/8.0/32.0
        R_h = float(w_original)/8.0/32.0
        if R_w > R_h:
            R = R_w
        else:
            R = R_h

        minimum_ratio = 3.0
        nbr_extra_layers = max(0, int(round(np.log(R/minimum_ratio)/np.log(2))))

        if nbr_extra_layers == 0:
            flow3[:, 0, :, :] *= float(w_original) / float(256)
            flow3[:, 1, :, :] *= float(h_original) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]
        else:
            # adding extra layers
            flow3[:, 0, :, :] *= float(w_original) / float(256)
            flow3[:, 1, :, :] *= float(h_original) / float(256)
            for n in range(nbr_extra_layers):
                ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n ))
                up_flow3 = F.interpolate(input=flow3, size=(int(h_original * ratio), int(w_original * ratio)),
                                            mode='bilinear',
                                            align_corners=False)
                c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_original * ratio),
                                                                        int(w_original * ratio)), mode='area')
                c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_original * ratio),
                                                                        int(w_original * ratio)), mode='area')
                warp3 = self.warp(c23_bis, up_flow3 * div * ratio)

                # feature correlation layer
                corr3 = correlation.FunctionCorrelation(c13_bis, warp3)
                corr3 = self.leakyRELU(corr3)

                corr3 = torch.cat((corr3, up_flow3), 1)
                x, res_flow3 = self.decoder2(corr3)
                flow3 = res_flow3 + up_flow3

        # ORIGINAL RESOLUTION
        up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                    align_corners=False)

        # level 2 : 1/8 of original resolution feature_corr_layer
        ratio = 1.0 / 8.0
        warp2 = self.warp(c22, up_flow3*div*ratio)

        # feature correlation layer
        corr2 = correlation.FunctionCorrelation(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3), 1)

        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = F.interpolate(input=flow2, size=(int(h_original / 4.0), int(w_original / 4.0)), mode='bilinear',
                                     align_corners=False)

        up_feat2 = self.upfeat2(x2)

        # level 1: 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = self.warp(c21, up_flow2*div*ratio)

        # feature correlation layer
        corr1 = correlation.FunctionCorrelation(c11, warp1)
        corr1 = self.leakyRELU(corr1)
        corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        x, res_flow1 = self.decoder1(corr1)

        x_, res_flow1_ = self.PWCNetRefinementFinal(x)
        res_flow1 = res_flow1 + res_flow1_
        flow1 = res_flow1 + up_flow2

        # prepare output dict
        output = {'flow_estimates': [flow2, flow1]}
        output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4}
        return output_256, output


######################################################################################################################################

class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)
        return x

class CMDTopResidualConnections(CorrespondenceMapBase):
    """
    DGC-Net mapping decoder, with residual connections.
    """
    def __init__(self, in_channels, batch_norm=False, output_x=False):
        super().__init__(in_channels, batch_norm)

        self.output_x = output_x
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv0_skip = conv(128, 96, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv2_skip = conv(96, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.final = conv_head(32)

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x0 = self.conv_0(x)
        x0_relu = self.leakyRELU(x0)
        x2 = self.conv_2(self.conv_1(x0_relu))
        x2_skip = x2 + self.conv0_skip(x0)
        x2_skip_relu = self.leakyRELU(x2_skip)
        x4 = self.conv_4(self.conv_3(x2_skip_relu))
        x4_skip = x4 + self.conv2_skip(x2_skip)
        x = self.leakyRELU(x4_skip)

        mapping = self.final(x)
        if self.output_x:
            return x, mapping
        else:
            return mapping

######################################################################################################################################

class OpticalFlowEstimatorResidualConnection(nn.Module):
    """
    PWCNet optical flow decoder modified with residual connections.
    """
    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorResidualConnection, self).__init__()

        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv0_skip = conv(128, 96, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv2_skip = conv(96, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x0 = self.conv_0(x)
        x0_relu = self.leakyRELU(x0)
        x2 = self.conv_2(self.conv_1(x0_relu))
        x2_skip = x2 + self.conv0_skip(x0)
        x2_skip_relu = self.leakyRELU(x2_skip)
        x4 = self.conv_4(self.conv_3(x2_skip_relu))
        x4_skip = x4 + self.conv2_skip(x2_skip)
        x = self.leakyRELU(x4_skip)
        flow = self.predict_flow(x)

        return x, flow



