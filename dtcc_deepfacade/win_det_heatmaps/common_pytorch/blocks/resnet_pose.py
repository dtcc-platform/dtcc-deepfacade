import os
from easydict import EasyDict as edict

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet as r


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


from common_pytorch.base_modules.deconv_head import DeconvHead
from common_pytorch.base_modules.resnet import resnet_spec, ResnetBackbone
from common_pytorch.base_modules.architecture import PoseNet_1branch, PoseNet_2branch

def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 18
    # default head setting
    config.num_deconv_layers = 3
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    # input
    config.input_channel = 3
    # output
    config.depth_dim = 1

    #AE
    config.head_branch = 2

    return config


def init_pose_net(pose_net, cfg):
    if cfg.from_model_zoo:
        _, _, _, name = resnet_spec[cfg.num_layers]
        org_resnet = model_zoo.load_url(model_urls[name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        pose_net.backbone.load_state_dict(org_resnet)
        # print("Init Network from model zoo")
    else:
        if os.path.exists(cfg.pretrained):
            model = torch.load(cfg.pretrained)
            pose_net.load_state_dict(model['network'])
            # print("Init Network from pretrained", cfg.pretrained)


def get_pose_net(network_cfg, ae_feat_dim, num_point_types):
    block_type, layers, channels, name = resnet_spec[network_cfg.num_layers]
    backbone_net = ResnetBackbone(block_type, layers, network_cfg.input_channel)

    # one branch, double output channel
    out_channel = num_point_types * 2
    if network_cfg.head_branch == 2:
        out_channel = num_point_types

    heatmap_head = DeconvHead(channels[-1], network_cfg.num_deconv_layers, network_cfg.num_deconv_filters,
                              network_cfg.num_deconv_kernel, network_cfg.final_conv_kernel, out_channel, network_cfg.depth_dim)
    # NOTE: to specify 4 to avoid center tag
    tagmap_head = DeconvHead(channels[-1], network_cfg.num_deconv_layers, network_cfg.num_deconv_filters,
                             network_cfg.num_deconv_kernel, network_cfg.final_conv_kernel, 4, ae_feat_dim)

    if network_cfg.head_branch == 1:
        return PoseNet_1branch(backbone_net, heatmap_head)
    elif network_cfg.head_branch == 2:
        return PoseNet_2branch(backbone_net, heatmap_head, tagmap_head)
    else:
        assert 0