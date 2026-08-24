import torch.nn.functional as F
import torch
from packaging import version

from model.pixel_wise_mapping import warp
from model.flow_and_mapping_operations import get_gt_correspondence_mask


def weights_supervised_and_unsupervised(loss_su, loss_un, loss_weight, apply_constant_weights=False):
    if not apply_constant_weights:
        L_supervised = loss_su.detach()
        L_unsupervised = loss_un.detach()
        ratio = loss_weight['warp_supervision'] / loss_weight['w_bipath']
        if L_unsupervised > L_supervised:
            u_l_w = 1
            s_l_w = L_unsupervised / (L_supervised + 1e-8) * ratio
        else:
            u_l_w = L_supervised / (L_unsupervised + 1e-8) / ratio
            s_l_w = 1
        loss = loss_un * u_l_w + loss_su * s_l_w
    
    else:
        loss = loss_weight['w_bipath_constant'] * loss_un + \
               loss_weight['warp_supervision_constant'] * loss_su
    return loss


class MultiScaleFlow:
    """ Module for multi-scale matching loss computation.
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """

    def __init__(self, level_weights, loss_function, downsample_gt_flow: bool):
        """
        Args:
            level_weights: weights to apply to computed loss at each level (from coarsest to finest pyramid level)
            loss_function: actual loss computation module, used for all levels
            downsample_gt_flow: bool, downsample gt flow to estimated flow resolution? otherwise, the estimated flow
                                of each level is instead up-sampled to ground-truth resolution for loss computation
        """
        self.level_weights = level_weights
        self.loss_function = loss_function
        self.downsample_gt_flow = downsample_gt_flow

    def one_scale(self, est_flow, gt_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """
        if self.downsample_gt_flow:
            b, _, h, w = est_flow.size()
            gt_flow = F.interpolate(gt_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).floor()
                    # floor not to include the borders
                    mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow size
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).floor()
                    # floor not to include the borders
                    mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

        return self.loss_function(est_flow, gt_flow, mask=mask)

    def __call__(self, network_output, gt_flow, mask=None):
        """
        Args:
            network_output: network predictions, can either be a dictionary, where network_output['flow_estimates']
                            is a list containing the estimated flow fields at all levels. or directly the list 
                            of flow fields.
            gt_flow: ground-truth flow field
            mask: bool tensor, valid mask, 1 indicates valid pixels where the loss is computed.

        Returns:
            loss: computed loss
            stats: dict with stats from the loss computation

        """

        if isinstance(network_output, dict):
            # it is a dictionary, extract the flow estimates
            flow_output = network_output['flow_estimates']
        else:
            flow_output = network_output  # the flow was directly given
        if type(flow_output) not in [tuple, list]:
            flow_output = [flow_output]

        loss = 0
        stats = {}
        for i, (flow, weight) in enumerate(zip(flow_output, self.level_weights)):
            # from smallest size to biggest size (last one is a quarter of input image size
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[i]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, mask=mask_used)
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss.item()
            loss += level_loss
        return loss, stats


class L1:
    """ Computes L1 loss. """
    def __init__(self, reduction='weighted_sum_normalized'):
        """
        Args:
            reduction: str, type of reduction to apply to loss
        """
        super().__init__()
        self.reduction = reduction

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        L1 = torch.sum(torch.abs(est_flow-gt_flow), 1, keepdim=True)

        if mask is not None:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach()) & mask
        else:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach())

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(L1, mask).mean()
            else:
                loss = L1.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                L1 = L1 * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += L1[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return L1.sum()/b
            return L1
        else:
            raise ValueError


def length_sq(x):
    return torch.sum(x**2, dim=1)


class WBipathLoss:
    """
    Main module computing the W-bipath loss. The W-bipath constraints computes the flow composition from the
    target prime to the target image.
    """
    def __init__(self, objective, detach_flow_for_warping=True, compute_cyclic_consistency=False,
                 alpha_1=0.01, alpha_2=0.5):
        """

        Args:
            objective: final objective, like multi-scale EPE or L1 loss
            loss_weight: weights used
            detach_flow_for_warping: bool, prevent back-propagation through the flow used for warping.
            compute_cyclic_consistency:
            alpha_1: hyper-parameter for the visibility mask
            alpha_2: hyper-parameter for the visibility mask
        """
        self.objective = objective
        # self.loss_weight = loss_weight
        self.detach_flow_for_warping = detach_flow_for_warping

        self.compute_cyclic_consistency = compute_cyclic_consistency
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def get_cyclic_consistency_mask(self, estimated_flow_target_prime_to_source_per_level,
                                    warping_flow_source_to_target, synthetic_flow):

        b, _, h, w = synthetic_flow.shape
        b, _, h_, w_ = estimated_flow_target_prime_to_source_per_level.shape

        synthetic_flow = F.interpolate(synthetic_flow, (h_, w_), mode='bilinear', align_corners=False)

        # defines occluded pixels (or just flow is not good enough)
        mag_sq_fw = length_sq(estimated_flow_target_prime_to_source_per_level) + \
                length_sq(warping_flow_source_to_target) + length_sq(synthetic_flow)
        occ_thresh_fw = self.alpha_1 * mag_sq_fw + self.alpha_2
        fb_occ_fw = length_sq(estimated_flow_target_prime_to_source_per_level + warping_flow_source_to_target -
                              synthetic_flow) > occ_thresh_fw

        # defines the mask of not occluded pixels
        mask_fw = ~fb_occ_fw  # shape bxhxw
        return mask_fw

    def __call__(self, flow_map, mask_used, estimated_flow_target_prime_to_source,
                 estimated_flow_source_to_target, *args, **kwargs):
        """

        Args:
            flow_map: corresponds to known flow relating the target prime image to the target
            mask_used: mask indicating in which regions the flow_map is valid
            estimated_flow_target_prime_to_source: list of estimated flows
            estimated_flow_source_to_target: list of estimated flows
        Returns:
            loss_un: final loss
            stats_un: stats from loss computation
            output: dictionary containing some intermediate results, for example the composition flow.
        """
        b, _, h, w = flow_map.shape  # size of the gt flow, meaning size of the input images

        output = {}
        estimated_flow_target_prime_to_target_through_composition = []
        masks = []
        if self.compute_cyclic_consistency:
            mask_cyclic_list = []

        if not isinstance(estimated_flow_target_prime_to_source, list):
            estimated_flow_target_prime_to_source = [estimated_flow_target_prime_to_source]
        if not isinstance(estimated_flow_source_to_target, list):
            estimated_flow_source_to_target = [estimated_flow_source_to_target]

        for nbr, (estimated_flow_target_prime_to_source_per_level, estimated_flow_source_to_target_per_level) \
                in enumerate(zip(estimated_flow_target_prime_to_source, estimated_flow_source_to_target)):
            b, _, h_, w_ = estimated_flow_target_prime_to_source_per_level.shape

            if self.detach_flow_for_warping:
                estimated_flow_target_prime_to_source_per_level_warping = \
                    estimated_flow_target_prime_to_source_per_level.detach().clone()
            else:
                estimated_flow_target_prime_to_source_per_level_warping = \
                    estimated_flow_target_prime_to_source_per_level.clone()
            estimated_flow_target_prime_to_source_per_level_warping[:, 0, :, :] *= float(w_) / float(w)
            estimated_flow_target_prime_to_source_per_level_warping[:, 1, :, :] *= float(h_) / float(h)

            warping_flow_source_to_target = warp(estimated_flow_source_to_target_per_level,
                                                 estimated_flow_target_prime_to_source_per_level_warping)
            estimated_flow = estimated_flow_target_prime_to_source_per_level + warping_flow_source_to_target
            estimated_flow_target_prime_to_target_through_composition.append(estimated_flow)

            # need to also compute the mask according to warping
            mask = get_gt_correspondence_mask(estimated_flow_target_prime_to_source_per_level_warping.detach())
            if mask_used is not None:
                # interpolate the mask to low resolution (floor is important here, because the mask can correspond
                # to invalid flow regions.
                mask_used_resized = F.interpolate(mask_used.unsqueeze(1).float(), (h_, w_), mode='bilinear',
                                                  align_corners=False).floor().squeeze(1)
                mask_used_resized = mask_used_resized.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                    else mask_used_resized.byte()
                mask = mask & mask_used_resized

            if self.compute_cyclic_consistency:
                mask_cyclic = self.get_cyclic_consistency_mask(estimated_flow_target_prime_to_source_per_level.detach(),
                                                               warping_flow_source_to_target.detach(), flow_map)
                mask = mask & mask_cyclic
                mask_cyclic_list.append(mask_cyclic)
            masks.append(mask)

        output['mask_training'] = masks
        if self.compute_cyclic_consistency:
            output['mask_cyclic'] = mask_cyclic_list
        output['estimated_flow_target_prime_to_target_through_composition'] = \
            estimated_flow_target_prime_to_target_through_composition

        loss_un, stats_un = self.objective(estimated_flow_target_prime_to_target_through_composition,
                                           flow_map, mask=masks)
        return loss_un, stats_un, output


def calculate_unsupervised_change_loss(estimated_change_source_to_target, estimated_flow_target_prime_to_source, estimated_change_target_prime_to_source, h, w, weight=3, detach_flow_for_warping=True):
    # only when target and target prime does not have changes
    b, _, h_, w_ = estimated_flow_target_prime_to_source.shape
    estimated_flow_target_prime_to_source_warping = estimated_flow_target_prime_to_source.clone()
    estimated_change_source_to_target_warping = estimated_change_source_to_target.clone()
    if detach_flow_for_warping:
        estimated_flow_target_prime_to_source_warping = \
                    estimated_flow_target_prime_to_source_warping.detach()
        estimated_change_source_to_target_warping = estimated_change_source_to_target_warping.detach()
    
    estimated_flow_target_prime_to_source_warping[:, 0, :, :] *= float(w_) / float(w)
    estimated_flow_target_prime_to_source_warping[:, 1, :, :] *= float(h_) / float(h)
    
    warping_change_target_prime_to_source = warp(estimated_change_source_to_target_warping, estimated_flow_target_prime_to_source_warping)

    loss = warpc_change_loss(warping_change_target_prime_to_source + 1e-10, estimated_change_target_prime_to_source + 1e-10) * weight
    return loss, warping_change_target_prime_to_source

def calculate_diceloss(inputs, targets, smooth=1e-3, downsample_gt=True):
    if len(targets.size()) == 3:
        targets = targets.unsqueeze(1)
    if len(inputs.size()) == 3:
        inputs = inputs.unsqueeze(1)
    if downsample_gt:
        b, _, h, w = inputs.size()  #estimated change
        targets = F.interpolate(targets, (h, w), mode='bilinear', align_corners=True)
  
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    
    return 1 - dice

def warpc_change_loss(change1, change2, smooth=1e-3):
    change2_flip = torch.flip(change2, dims=[1])
    dice = calculate_diceloss(change1, change2_flip, smooth, downsample_gt=False)
    return dice