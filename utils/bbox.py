import numpy as np

def xyxy2points(boxes):
    """
    :param boxes: numpy array with shape [num_boxes, 4] or [4]
    :return: numpy array with shape [num_boxes, 4, 2]
    """
    if len(boxes)==0:
        return np.array([])

    if isinstance(boxes,list):
        boxes = np.array(boxes)

    origin_dim = boxes.ndim
    if boxes.ndim==1:
        boxes = np.expand_dims(boxes, 0)

    boxes = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for x1, y1, x2, y2 in boxes])
    if origin_dim==1:
        boxes = boxes[0]
    return boxes

def points2xyxy(boxes):
    """
    :param boxes: numpy array with shape [num_boxes, 4, 2] or [4,2]
    :return: numpy array with shape [num_boxes, 4] or [4]
    """
    if len(boxes)==0:
        return np.array([])

    if isinstance(boxes,list):
        boxes = np.array(boxes)

    origin_dim = boxes.ndim
    if boxes.ndim==2:
        boxes = np.expand_dims(boxes, 0)

    boxes= np.array([(points[..., 0].min(), points[..., 1].min(), points[..., 0].max(), points[..., 1].max()) for points in boxes])
    if origin_dim==2:
        boxes = boxes[0]
    return boxes


def scale_bboxes(bboxes, image_size, target_size, bbox_format='xyxy'):
    """
    change bboxes coordinate for target image
    :param bboxes: bounding box on image [num boxes, 4] or [num boxes, 4, 2]
    :param image_size: origin image size [height, width]
    :param target_size: target image size [height, width]

    :return: [num boxes, 4] or [num boxes, 4, 2]
    """
    assert ((bbox_format == 'xyxy' and bboxes.ndim == 2) or (bbox_format == 'points' and bboxes.ndim == 3))

    ratio = min(image_size[0] / target_size[0], image_size[1] / target_size[1])
    pad = ((image_size[0] - target_size[0] * ratio) // 2, (image_size[1] - target_size[1] * ratio) // 2)

    if bbox_format == 'xyxy':
        bboxes[:, 0] -= pad[1]
        bboxes[:, 1] -= pad[0]
        bboxes[:, 2] -= pad[1]
        bboxes[:, 3] -= pad[0]
    elif bbox_format == 'points':
        bboxes[..., 0] -= pad[1]
        bboxes[..., 1] -= pad[0]
    bboxes /= ratio

    return bboxes



def check_validity(bbox,size=None, bbox_format='xyxy'):
    """
    Check the validity of a bounding box.

    Args:
    - bbox (list or numpy.ndarray): Bounding box coordinates in the format specified by bbox_format.
                                    For bbox_format='xyxy', format is [x1, y1, x2, y2].
                                    For bbox_format='xy', format is [num_points, 2].
    - size (tuple, optional): Image size in the format (width, height). Defaults to None.
    - bbox_format (str, optional): Format of the bounding box coordinates. Possible values: 'xyxy', 'xy'.
                                   'xyxy' represents [x1, y1, x2, y2].
                                   'xy' represents [num_points, 2]. Defaults to 'xyxy'.
    """
    if isinstance(bbox,list):
        bbox = np.array(bbox)


    if bbox_format=='xyxy':
        valid = bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
        if size:
            width,height=size
            valid = valid and (0 <= bbox[0] <= width) and (0 <= bbox[1] <= height) and (
                0 <= bbox[2] <= width) and (0 <= bbox[3] <= height)
    elif bbox_format=='xy':
        if size is None:
            valid = np.all(0 <= bbox[:, 0]) and np.all(0 <= bbox[:, 1])
        else:
            valid = np.all((0 <= bbox[:, 0]) < size[0]) and np.all((0 <= bbox[:, 1]) < size[1])

    return valid


def check_bbox_validity(bbox,size=None, bbox_format='xyxy'):
    """
    Check the validity of a bounding box.

    Args:
    - bbox (list or numpy.ndarray): Bounding box coordinates in the format specified by bbox_format.
                                    For bbox_format='xyxy', format is [x1, y1, x2, y2].
                                    For bbox_format='xy', format is [num_points, 2].
    - size (tuple, optional): Image size in the format (width, height). Defaults to None.
    - bbox_format (str, optional): Format of the bounding box coordinates. Possible values: 'xyxy', 'xy'.
                                   'xyxy' represents [x1, y1, x2, y2].
                                   'xy' represents [num_points, 2]. Defaults to 'xyxy'.
    """
    if isinstance(bbox,list):
        bbox = np.array(bbox)


    if bbox_format=='xyxy':
        valid = bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
        if size:
            width,height=size
            valid = valid and (0 <= bbox[0] <= width) and (0 <= bbox[1] <= height) and (
                0 <= bbox[2] <= width) and (0 <= bbox[3] <= height)
    elif bbox_format=='xy':
        if size is None:
            valid = np.all(0 <= bbox[:, 0]) and np.all(0 <= bbox[:, 1])
        else:
            valid = np.all((0 <= bbox[:, 0]) < size[0]) and np.all((0 <= bbox[:, 1]) < size[1])

    return valid

