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
        if not self.is_training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask * tr_masks[bid]) / np.sum(text_mask)) - 1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50 / (cfg.scale * cfg.scale) or confidence < cfg.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])

                poly = get_sample_point(text_mask, cfg.num_points,
                                        cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences

    # def get_boundary_proposal_eval_cuda(self, input=None, seg_preds=None):
    #     # print ("using cuda ccl")
    #     cls_preds = seg_preds[:, 0, :, :].detach()
    #     dis_preds = seg_preds[:, 1, :, :].detach()

    #     inds = []
    #     init_polys = []
    #     confidences = []
    #     for bid, dis_pred in enumerate(dis_preds):
    #         dis_mask = dis_pred > cfg.dis_threshold
    #         dis_mask = dis_mask.type(torch.cuda.ByteTensor)
    #         labels = cc_torch.connected_components_labeling(dis_mask)
    #         key = torch.unique(labels, sorted = True)
    #         for l in key:
    #             text_mask = labels == l
    #             confidence = round(torch.mean(cls_preds[bid][text_mask]).item(), 3)
    #             if confidence < cfg.cls_threshold or torch.sum(text_mask) < 50/(cfg.scale*cfg.scale):
    #                 continue
    #             confidences.append(confidence)
    #             inds.append([bid, 0])

    #             text_mask = text_mask.cpu().numpy()
    #             poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
    #             init_polys.append(poly)

    #     if len(inds) > 0:
    #         inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #     else:
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #         inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

    #     return init_polys, inds, confidences

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        num_point = i_it_poly.shape[1]
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2) * cfg.scale, cnn_feature.size(3) * cfg.scale
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,
                             :num_point]
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w - 1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt", embed=None):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            # init_polys, inds, confidences = self.get_boundary_proposal_eval_cuda(input=input, seg_preds=seg_preds - embed)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter + 1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences


    def forward2(self,feature):
        h, w = feature.size(2) * cfg.scale, feature.size(3) * cfg.scale


        for i in range(iter):


