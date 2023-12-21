
import numpy as np


from utils.bbox import scale_bboxes, xyxy2points,points2xyxy

from .craft_utils import getDetBoxes


def postprocess(pred, input_size, bbox_format='xyxy', char_threshold=0.6, link_threshold=0.3, word_threshold=0.7, poly=False):

    score_text = pred[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = pred[0, :, :, 1].cpu().data.numpy().astype(np.float32)

    bboxes, polys = getDetBoxes(score_text, score_link, word_threshold, link_threshold, char_threshold, False)

    # bboxes: list of array(4,2) or empty

    # convert free rectangle to 4-points
    bboxes += [xyxy2points(points2xyxy(points)) for points in polys if points is not None]
    bboxes = np.array(bboxes)


    out_size = pred.shape[1:3]
    if len(bboxes):
        bboxes = scale_bboxes(bboxes, out_size, input_size, bbox_format='points')

    if bbox_format == 'xyxy':
        bboxes = points2xyxy(bboxes)

    return bboxes

