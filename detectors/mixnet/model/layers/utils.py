import numpy as np
import torch


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
