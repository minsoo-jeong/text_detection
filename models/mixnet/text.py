from model import TextNet
from torchsummary import summary
import torch

from torch.utils.data import DataLoader, Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from utils.misc import set_seed, filter_word
from utils.bbox import points2xyxy, check_bbox_validity, check_validity
from utils.images import unnormalize_image_tensor

import matplotlib.pyplot as plt
from collections import defaultdict

from shapely.geometry import Polygon, LineString
from shapely.ops import clip_by_rect

from scipy import ndimage as ndimg
import torch

from datasets.ic15 import IC15

set_seed(42)

root = '/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01'
# root='/mldisk2/ms/datasets/OCR/AOS_OCR/AOS_Test'
size = 1536

transform = A.Compose(
    [
        A.LongestMaxSize(size),
        # A.RandomScale(scale_limit=(1., 2.), interpolation=cv2.INTER_CUBIC),
        # A.SafeRotate(45, border_mode=cv2.BORDER_CONSTANT, p=1),
        # A.RandomResizedCrop(size, size, scale=(0.03, 0.1), interpolation=cv2.INTER_CUBIC),
        # A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2, always_apply=False, p=0.5),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(),
        ToTensorV2()
    ]  # , keypoint_params=A.KeypointParams(format='xy', label_fields=['words','pidx']))
    , keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['box_id']))
dataset = IC15(f'{root}/test_images', f'{root}/test_labels', transform)

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4, collate_fn=IC15.collate, shuffle=False, num_workers=4)

a = TextNet('FSNet_M', mid=True)
ckpt = torch.load('./pretrained/MixNet_FSNet_M_160_icdar_art.pth')
a.load_state_dict(ckpt['model'], strict=True)
a.cuda()
a.eval()

with torch.no_grad():
    for image, polygons, words, paths in loader:
        print('>>', image.shape)
        pred, ind = a(image.cuda())
        print(pred.shape, ind)

        for b, im in enumerate(image):
            img = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            poly = pred[ind == b].cpu().numpy().astype(np.int32).copy()
            print(img.shape, img.dtype)
            print(poly.shape, poly.dtype)

            cv2.polylines(img, poly, True, (0, 255, 0), 2)

            plt.imshow(img)
            plt.show()
