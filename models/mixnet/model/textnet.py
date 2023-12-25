import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .backbones.FSNet import FSNet
from .layers.Transformer import Transformer
from .layers.midline import midlinePredictor
from .layers.evolution import Evolution
from .layers.model_block import FPN


class TextNet(nn.Module):
    def __init__(self, backbone='FSM', embed=True, mid=True, pos=True, num_points=20):
        super().__init__()

        self.fpn = FPN(backbone)
        self.embed = embed
        self.mid = mid
        self.pos = pos
        self.num_points = num_points

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        if embed:
            self.embed_head = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            )
        if mid:
            self.BPN = midlinePredictor(seg_channel=32 + 4)
        elif pos:
            self.BPN = Evolution(num_points, seg_channel=32 + 4 + 2)
        else:
            self.BPN = Evolution(num_points, seg_channel=32 + 4)

    def forward(self, x, ):
        b, c, h, w = x.shape

        up1 = self.fpn(x)

        preds = self.seg_head(up1)

        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        cls_preds = fy_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = fy_preds[:, 1, :, ].detach().cpu().numpy()
        init_polys, confs, index = generate_boundary(dis_preds, cls_preds, 0.3, 0.85)

        if self.mid:
            pred_polys, mid_polys = self.BPN(cnn_feats, init_polys.cuda(), index.cuda())
            midlines = [mid_polys[index == batch] for batch in range(b)]
        else:
            pred_polys = self.BPN(cnn_feats, init_polys.cuda(), index.cuda())

        init_polys = [init_polys[index == batch] for batch in range(b)]
        pred_polys = [pred_polys[index == batch] for batch in range(b)]

        return init_polys, pred_polys, midlines if self.mid else None


import numpy as np
import cv2

from utils.polygons import sampling_points_from_mask


def generate_boundary(dist_preds, cls_preds, dist_threshold, cls_threshold):
    confidences = []
    polys = []
    inds = []
    for b, dist_pred in enumerate(dist_preds):
        mask = dist_pred > dist_threshold
        ret, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
        for idx in range(1, ret):
            text_mask = labels == idx
            confidence = np.mean(cls_preds[b][text_mask])
            if confidence < cls_threshold or np.sum(text_mask) < 10:
                continue

            points = [[p.x, p.y] for p in sampling_points_from_mask(text_mask, 20)[0]]
            points = np.array(points).astype(np.int32)

            confidences.append(confidence)
            polys.append(points)
            inds.append(b)

    return torch.from_numpy(np.array(polys)), torch.from_numpy(np.array(confidences)), torch.from_numpy(np.array(inds))


if __name__ == '__main__':
    a = TextNet()
