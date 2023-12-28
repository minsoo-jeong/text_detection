import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import cv2

from .Transformer import Transformer


class midlinePredictor(nn.Module):
    def __init__(self, seg_channel,node_num,mid_node_num):
        super(midlinePredictor, self).__init__()
        self.seg_channel = seg_channel
        self.clip_dis = 100
        self.num_polygon_points = node_num
        self.num_midline_points = mid_node_num
        self.midline_preds = nn.ModuleList()
        self.contour_preds = nn.ModuleList()
        self.iter = 3
        for i in range(self.iter):
            self.midline_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8,
                    dim_feedforward=1024, drop_rate=0.0,
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )
            self.contour_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8,
                    dim_feedforward=1024, drop_rate=0.0,
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )
        if not self.training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, init_polys, inds):
        if not self.training:
            self.iter = 1
        def get_node_feature(features, polys, inds):
            batch_size = features.shape[0]
            # normalize [-1, 1]
            polys = polys.clone().float()
            polys[..., 0] = (2. * polys[..., 0] / features.shape[3]) - 1.
            polys[..., 1] = (2. * polys[..., 1] / features.shape[2]) - 1.
            gcn = []
            for i in range(batch_size):
                grid = polys[inds == i].unsqueeze(0).to(features.device)
                grid_feat = torch.nn.functional.grid_sample(features[i:i + 1], grid)[0].permute(1, 0, 2)
                gcn.append(grid_feat)
            gcn_feature = torch.cat(gcn, dim=0).to(features.device)

            return gcn_feature

        polys = init_polys
        features = x
        if not len(polys):
            return init_polys, init_polys

        for i in range(self.iter):
            node_feature = get_node_feature(features, polys, inds)

            pred = self.midline_preds[i](node_feature).permute(0, 2, 1)
            midlines = polys[:, :self.num_midline_points] + torch.clamp(pred, -self.clip_dis, self.clip_dis)[:,
                                                            :self.num_midline_points]

            mid_node_feature = get_node_feature(features, midlines, inds)

            merged_feat = torch.cat((node_feature, mid_node_feature), dim=2)
            pred = self.contour_preds[i](merged_feat).permute(0, 2, 1)

            new_polys = polys + torch.clamp(pred, -self.clip_dis, self.clip_dis)[:, :self.num_polygon_points]

            polys = new_polys

        return polys, midlines
