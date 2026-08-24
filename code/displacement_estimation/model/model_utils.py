import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batch_norm=False, relu=True):
    if batch_norm:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1, inplace=True))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes))
    else:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias),
                                nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias))

def conv_head(in_channels, nbr_out_channels=2):
    return nn.Conv2d(in_channels, nbr_out_channels, kernel_size=3, padding=1)

def predict_flow(in_planes, nbr_out_channels=2):
    return nn.Conv2d(in_planes, nbr_out_channels, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, out_planes=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    deconv_ = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    nn.init.kaiming_normal_(deconv_.weight.data, mode='fan_in')
    if deconv_.bias is not None:
        deconv_.bias.data.zero_()
    return deconv_

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
    mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow