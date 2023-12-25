import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import math
import cv2

from .Transformer import Transformer
from .gcn_utils import get_node_feature


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def split_edge_seqence(points, n_parts):
    pts_num = points.shape[0]
    long_edge = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def get_sample_point(text_mask, num_points, approx_factor, scales=None):
    # get sample point in contours
    contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = approx_factor * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
    # approx = contours[0].reshape((-1, 2))
    if scales is None:
        ctrl_points = split_edge_seqence(approx, num_points)
    else:
        ctrl_points = split_edge_seqence(approx * scales, num_points)
    ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

    return ctrl_points


class Evolution(nn.Module):
    def __init__(self, node_num, seg_channel):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.seg_channel = seg_channel
        self.clip_dis = 100

        self.iter = 3
        for i in range(self.iter):
            evolve_gcn = Transformer(seg_channel, 128, num_heads=8, dim_feedforward=1024, drop_rate=0.0, if_resi=True,
                                     block_nums=3)
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        if not self.training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, init_polys, inds):

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
            return init_polys

        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))

            node_feature = get_node_feature(features, polys, inds)

            pred = evolve_gcn(node_feature).permute(0, 2, 1)
            polys = polys + torch.clamp(pred, -self.clip_dis, self.clip_dis)[:, :self.node_num]

        return polys
