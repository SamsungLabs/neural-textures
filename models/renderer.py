import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn.functional import grid_sample

from utils.common import to_tanh, to_sigm


class Renderer(nn.Module):
    def __init__(self, in_channels, segm_channels, texsegm):
        super().__init__()

        n_out = 16
        self.model = smp.Unet('resnet18', encoder_weights=None, in_channels=in_channels, classes=n_out)
        norm_layer = nn.BatchNorm2d

        padding = nn.ZeroPad2d

        self.rgb_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, n_out, 3, 1, 0, bias=False),
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, n_out, 3, 1, 0, bias=False),
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, 3, 3, 1, 0, bias=True),
            nn.Tanh())

        self.segm_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, segm_channels, 3, 1, 0, bias=True),
            nn.Sigmoid())

        texsegm = texsegm.clone()
        self.texsegm = torch.nn.Parameter(texsegm.unsqueeze(2), requires_grad=False)
        self.i = 0

    def forward(self, data_dict):
        uv = data_dict['uv']

        if 'nrender' not in data_dict:
            ntex = data_dict['ntex']
            N = uv.shape[0]
            ntex_summed = ntex.sum(1)

            texsegm = self.texsegm.clone()
            if 'hands_vis' in data_dict:
                hands_vis = data_dict['hands_vis']
                lhand_segm = texsegm[:, 4:5, 0]
                rhand_segm = texsegm[:, 3:4, 0]

                ntex_summed = ntex_summed * (1. - lhand_segm) + ntex_summed * lhand_segm * hands_vis[:, :1]
                ntex_summed = ntex_summed * (1. - rhand_segm) + ntex_summed * rhand_segm * hands_vis[:, 1:2]

            neural_render = grid_sample(ntex_summed, uv.permute(0, 2, 3, 1))
        else:
            neural_render = data_dict['nrender']

        inp = torch.cat([neural_render, uv], dim=1)
        out = self.model(inp)

        segm = self.segm_head(out)
        rgb = self.rgb_head(out)
        rgb_ = rgb

        segm_fg = segm[:, :1]
        rgb_bbg = to_tanh(to_sigm(rgb) * segm_fg)

        if 'background' in data_dict:
            background = data_dict['background']
            rgb_segm = to_sigm(rgb) * segm_fg + background * (1. - segm_fg)
            rgb_segm = to_tanh(rgb_segm)
        else:
            rgb_segm = rgb_bbg

        out_dict = dict(fake_rgb=rgb_segm, fake_segm=segm, fake_rgb_bb=rgb_bbg)

        return out_dict
