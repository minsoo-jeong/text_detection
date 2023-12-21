import torch.nn as nn
import torch


class CraftLoss(nn.Module):
    def __init__(self, pos_thres=.1, neg_thres=.0):
        super(CraftLoss, self).__init__()
        self.positive_threshold = pos_thres
        self.negative_threshold = neg_thres

    def forward(self, output, character_map, affinity_map):
        batch_size, height, width, channels = output.shape

        output = output.contiguous().view([batch_size * height * width, channels])

        character = output[:, 0]
        affinity = output[:, 1]

        affinity_map = affinity_map.view([batch_size * height * width])
        character_map = character_map.view([batch_size * height * width])

        loss_character = hard_negative_mining(character, character_map, self.positive_threshold,self.negative_threshold)
        loss_affinity = hard_negative_mining(affinity, affinity_map, self.positive_threshold,self.negative_threshold)

        # TODO: weight character twice then affinity
        all_loss = loss_character * 2 + loss_affinity

        return all_loss * 100


def hard_negative_mining(pred, target, pos_thres=.1, neg_thres=.0):
    """
    Online hard mining on the entire batch
    :param pred: predicted character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :param target: target character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :return: Online Hard Negative Mining loss
    """
    all_loss = (pred - target) ** 2

    positive = target >= pos_thres
    negative = target <= neg_thres

    positive_loss = all_loss * positive
    negative_loss = all_loss * negative

    indices = min(max(1000, 3 * positive.sum()), negative.sum())
    negative_loss_t = torch.sort(negative_loss, descending=True).values
    negative_loss_t = negative_loss_t[:indices]
    # negative_loss = torch.sort(negative_loss, descending=True).values[0:min(max(1000, 3 * torch.sum(positive)), torch.sum(negative))]
    return (positive_loss.sum() + negative_loss.sum()) / (
            torch.sum(positive) + negative_loss.shape[0])
