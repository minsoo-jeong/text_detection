import cv2
import torch
import numpy as np


def hconcat_images(images, height=None):
    """
    :param images: list of images
    :return: horizontally concatenated images
    """
    height = height if height is not None else max([image.shape[0] for image in images])
    resized = []
    for im in images:
        ratio = im.shape[0] / height
        width = int(im.shape[1] / ratio)
        if isinstance(im, torch.Tensor):
            im = im.numpy()

        if len(im.shape) == 2:
            im = convert_to_heatmap(im)

        resized.append(cv2.resize(im, (width, height)))
    target = cv2.hconcat(resized)
    return target


def vconcat_images(images, width=None):
    width= width if width is not None else max([image.shape[1] for image in images])

    resized = []
    for im in images:
        ratio = im.shape[1] / width
        height = int(im.shape[0] / ratio)
        if isinstance(im, torch.Tensor):
            im = im.numpy()

        if len(im.shape) == 2:
            im = convert_to_heatmap(im)

        resized.append(cv2.resize(im, (width, height)))
    target = cv2.vconcat(resized)

    return target


def convert_to_heatmap(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def draw_text(image, text, leftbottom=None, color=(0, 0, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    # scale = .7 * (image.shape[0] * image.shape[1]) / (400 * 535)
    FONT_SCALE=2e-3
    THICKNESS_SCALE=2e-3
    scale = min(image.shape[0], image.shape[1]) * FONT_SCALE
    lw=int(min(image.shape[0], image.shape[1]) * THICKNESS_SCALE+.9)
    (fw, fh), _ = cv2.getTextSize('1', font, scale, lw)

    if leftbottom is None:
        leftbottom = (10, image.shape[0] - 10)
    image = cv2.putText(image, text, leftbottom, font, scale, color, lw)


    return image
