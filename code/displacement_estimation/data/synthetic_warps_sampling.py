""" Modified from NC-Net. """
import torch
import numpy as np
import random

from data.aff_homo_tps_generation import HomographyGridGen, TpsGridGen, AffineGridGen, ComposedGeometricTnf


class SynthecticAffHomoTPSTransfo:
    """Generates a flow field of given size, corresponding to a randomly sampled Affine, Homography, TPS or Affine-TPS
    transformation. """
    def __init__(self, size_output_flow, random_t=0.25, random_s=0.45, random_alpha=np.pi / 12,
                 random_t_tps_for_afftps=60/750, random_t_hom=250/750, random_t_tps=250/750, tps_grid_size=3, tps_reg_factor=0,
                 transformation_types=['hom', 'tps', 'afftps'], parametrize_with_gaussian=False, proba_horizontal_flip=0.0, use_cuda=True):
        """
        For all transformation parameters, image is taken as in interval [-1, 1]. Therefore all parameters must be
        within [0, 1]. The range of sampling is then [-parameter, parameter] or [1-parameter, 1+parameter] for the
        scale.
        Args:
            size_output_flow: desired output size for generated flow field
            random_t: max translation for affine transform.
            random_s: max scale for affine transform
            random_alpha: max rotation and shearing angle for the affine transform
            random_t_tps_for_afftps: max translation parameter for the tps transform generation, used for the
                         affine-tps transforms
            random_t_hom: max translation parameter for the homography transform generation
            random_t_tps: max translation parameter for the tps transform generation
            tps_grid_size: tps grid size
            tps_reg_factor:
            transformation_types: list of transformations to samples.
                                  Must be selected from ['affine', 'hom', 'tps', 'afftps']
            parametrize_with_gaussian: sampling distribution for the transformation parameters. Gaussian ? otherwise,
                                       uses a uniform distribution
            use_cuda: use_cuda?
        """

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.out_h, self.out_w = size_output_flow

        self.proba_horizontal_flip = proba_horizontal_flip

        self.parametrize_with_gaussian = parametrize_with_gaussian
        # for homo
        self.random_t_hom = random_t_hom
        # for tps
        self.random_t_tps = random_t_tps
        # for affine
        self.random_t = random_t
        self.random_alpha = random_alpha
        self.random_s = random_s
        if random_t_tps_for_afftps is None:
            random_t_tps_for_afftps = random_t_tps
        self.random_t_tps_for_afftps = random_t_tps_for_afftps
        self.use_cuda = use_cuda
        if transformation_types is None:
            transformation_types = ['affine', 'hom', 'tps', 'afftps']
        self.transformation_types = transformation_types
        if 'hom' in self.transformation_types:
            self.homo_grid_sample = HomographyGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        if 'affine' in self.transformation_types:
            self.aff_grid_sample = AffineGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        # self.aff_grid_sample_2 = AffineGridGenV2(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda) for offset
        if 'tps' in self.transformation_types:
            self.tps_grid_sample = TpsGridGen(out_h=self.out_h, out_w=self.out_w, grid_size=tps_grid_size,
                                              reg_factor=tps_reg_factor, use_cuda=use_cuda)
        if 'afftps' in self.transformation_types:
            self.tps_aff_grid_sample = ComposedGeometricTnf(tps_grid_size=tps_grid_size, tps_reg_factor=tps_reg_factor,
                                                            out_h=self.out_h, out_w=self.out_w, offset_factor=1.0,
                                                            padding_crop_factor=None, use_cuda=use_cuda)

    def __call__(self, *args, **kwargs):
        """Generates a flow_field (flow_gt) from sampling a geometric transformation. """

        geometric_model = self.transformation_types[random.randrange(0, len(self.transformation_types))]
        # sample the theta
        theta_tps, theta_hom, theta_aff = 0.0, 0.0, 0.0
        if self.parametrize_with_gaussian:
            if geometric_model == 'affine' or geometric_model == 'afftps':
                rot_angle = np.random.normal(0, self.random_alpha, 1)
                sh_angle = np.random.normal(0, self.random_alpha, 1)
                lambda_1 = np.random.normal(1, self.random_s, 1)
                lambda_2 = np.random.normal(1, self.random_s, 1)
                tx = np.random.normal(0, self.random_t, 1)
                ty = np.random.normal(0, self.random_t, 1)

                R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

                D = np.diag([lambda_1[0], lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
                theta_aff = torch.Tensor(theta_aff.astype(np.float32)).unsqueeze(0)
                theta_aff = theta_aff.cuda() if self.use_cuda else theta_aff
            if geometric_model == 'hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom + np.random.normal(0, self.random_t_hom, 8)
                theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
                theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom
            if geometric_model == 'tps':
                theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta_tps = theta_tps + np.random.normal(0, self.random_t_tps, 18)
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
            if geometric_model == 'afftps':
                theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta_tps = theta_tps + np.random.normal(0, self.random_t_tps_for_afftps, 18)
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
        else:
            if geometric_model == 'affine' or geometric_model == 'afftps':
                rot_angle = (np.random.rand(1) - 0.5) * 2 * self.random_alpha
                # between -np.pi/12 and np.pi/12 for random_alpha = np.pi/12
                sh_angle = (np.random.rand(1) - 0.5) * 2 * self.random_alpha
                lambda_1 = 1 + (2 * np.random.rand(1) - 1) * self.random_s  # between 0.75 and 1.25 for random_s = 0.25
                lambda_2 = 1 + (2 * np.random.rand(1) - 1) * self.random_s  # between 0.75 and 1.25
                tx = (2 * np.random.rand(1) - 1) * self.random_t  # between -0.25 and 0.25 for random_t=0.25
                ty = (2 * np.random.rand(1) - 1) * self.random_t

                R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

                D = np.diag([lambda_1[0], lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
                theta_aff = torch.Tensor(theta_aff.astype(np.float32)).unsqueeze(0)
                theta_aff = theta_aff.cuda() if self.use_cuda else theta_aff
            if geometric_model == 'hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom + (np.random.rand(8) - 0.5) * 2 * self.random_t_hom
                theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
                theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom

            if geometric_model == 'tps':
                theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
            if geometric_model == 'afftps':
                theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps_for_afftps
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps

        if geometric_model == 'hom':
            mapping = self.homo_grid_sample.forward(theta_hom)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 1, 2,h,w
        elif geometric_model == 'affine':
            mapping = self.aff_grid_sample.forward(theta_aff)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2,h,w
        elif geometric_model == 'tps':
            mapping = self.tps_grid_sample.forward(theta_tps)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2,h,w
        elif geometric_model == 'afftps':
            mapping = self.tps_aff_grid_sample(theta_aff, theta_tps)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2,h,w
        else:
            raise NotImplementedError

        # 1, 2, h, w
        if random.random() < self.proba_horizontal_flip:
            mapping_gt = convert_flow_to_mapping(flow_gt)
            mapping_gt = torch.from_numpy(np.copy(
                np.fliplr(mapping_gt.squeeze().permute(1, 2, 0).cpu().numpy()))).permute(2, 0, 1).unsqueeze(0)
            flow_gt = convert_mapping_to_flow(mapping_gt).cuda()
        '''
        flow_inverse = None
        if self.compute_inverse_flow:
            if geometric_model == 'hom':
                mapping_inverse = self.homo_grid_sample.forward(torch.inverse(theta_hom))
                flow_gt_inverse = unormalise_and_convert_mapping_to_flow(mapping_inverse, output_channel_first=True)
            elif geometric_model == 'affine':
        '''
        return flow_gt


class GetRandomSyntheticAffHomoTPSFlow:
    """ For all image in a batch, computes flow fields corresponding to random Affine, homography or TPS transforms.
    They have a size corresponding to argument size_output_flow.
    A different geometric transformation is sampled for each image pair in the batch.
    The range of sampling parameters are provided through the transfo sampling module. """
    def __init__(self, size_output_flow, transfo_sampling_module=None):
        """
        Args:
            size_output_flow:  size of the outputted flow field
            transfo_sampling_module: module to sample the transform parameters. If None, we use the default module.
        """
        self.size_output_flow = size_output_flow

        if transfo_sampling_module is None:
            transfo_sampling_module = SynthecticAffHomoTPSTransfo(size_output_flow=self.size_output_flow, random_t=0.25,
                                                                  random_s=0.5, random_alpha=np.pi / 12,
                                                                  random_t_hom=0.4, random_t_tps=0.4,
                                                                  transformation_types=['affine', 'hom',
                                                                                        'tps', 'afftps'])
            # this is quite strong transformations
        self.sample_transfo = transfo_sampling_module

    def __call__(self,  mini_batch):
        with torch.no_grad():
            target_image = mini_batch['target_image']
            b = target_image.shape[0]
            synthetic_flow = []

            for b_ in range(b):
                flow_gt_ = self.sample_transfo()
                synthetic_flow.append(flow_gt_)

            flow_gt = torch.cat(synthetic_flow, dim=0)
        return flow_gt.cuda().requires_grad_(False)


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

def convert_mapping_to_flow(mapping, output_channel_first=True):
    if not isinstance(mapping, np.ndarray):
        # torch tensor
        if len(mapping.shape) == 4:
            if mapping.shape[1] != 2:
                # size is BxHxWx2
                mapping = mapping.permute(0, 3, 1, 2)

            B, C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if mapping.shape[0] != 2:
                # size is HxWx2
                mapping = mapping.permute(2, 0, 1)

            C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()

            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(mapping.shape) == 4:
            if mapping.shape[3] != 2:
                # size is Bx2xHxW
                mapping = mapping.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = mapping.shape[:3]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if mapping.shape[0] == 2:
                # size is 2xHxW
                mapping = mapping.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = mapping.shape[:2]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)

def unormalise_and_convert_mapping_to_flow(map, output_channel_first=True):

    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
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
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[1, :, :] = (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        flow = np.copy(map)
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.transpose(0, 2, 3, 1)

            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            mapping[:,:,:,0] = (map[:,:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,:,1] = (map[:,:,:,1] + 1) * (h_scale - 1) / 2
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.transpose(1, 2, 0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:,:,0] = (map[:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,1] = (map[:,:,1] + 1) * (h_scale - 1) / 2
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = mapping[:,:,0]-X
            flow[:,:,1] = mapping[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)