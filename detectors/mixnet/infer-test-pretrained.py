from model.textnet import TextNet
from torchsummary import summary
import torch

from torch.utils.data import DataLoader, Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from text_detection.utils.misc import set_seed
from text_detection.utils.images import unnormalize_image_tensor

import matplotlib.pyplot as plt

import torch

import time
import os


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        super().__init__()
        self.l = l
        self.transform = transform

    def __getitem__(self, idx):
        path = self.l[idx]

        image = cv2.imread(str(path))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        if self.transform is not None:
            augmented = transform(image=image)
            image = augmented['image']

        return image, str(path)

    def __len__(self):
        return len(self.l)


if __name__ == '__main__':

    set_seed(42)

    size = 1280
    transform = A.Compose(
        [
            A.LongestMaxSize(size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
            ToTensorV2()
        ])

    root = '/mldisk2/ms/datasets/OCR/AOS_OCR/AOS_Test'

    images = [p for p in Path(root).iterdir() if p.suffix == '.jpg']

    dataset = ListDataset(images, transform)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    """
    # Total-Text
    python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name Totaltext_mid --checkepoch 622 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
    # CTW1500
    python3 eval_mixNet.py --net FSNet_hor --scale 1 --exp_name Ctw1500 --checkepoch 925 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85
    # MSRA-TD500
    python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name TD500HUST_mid --checkepoch 284 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
    # ArT
    python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name ArT_mid --checkepoch 160 --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.8 --mid True
    """

    ckpts = ['./pretrained/MixNet_FSNet_M_160_icdar_art.pth',
             './pretrained/MixNet_FSNet_M_284_msra_td500.pth',
             './pretrained/MixNet_FSNet_M_622_total_text.pth',
             './pretrained/MixNet_FSNet_hor_925_ctw1500.pth']

    models = [TextNet('FSNet_M', mid=True, dist_threshold=.4, cls_threshold=0.8),
              TextNet('FSNet_M', mid=True, embed=False, dist_threshold=0.3, cls_threshold=0.85),
              TextNet('FSNet_M', mid=True, embed=False, dist_threshold=0.3, cls_threshold=0.85),
              TextNet('FSNet_hor', mid=False, pos=False, embed=False, dist_threshold=0.3, cls_threshold=0.85)]

    vis_dir = './result-pretrained/AOS_test'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for model, ckpt_path in zip(models, ckpts):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'], strict=True)
        model.cuda()
        model.eval()

    t = 0
    with torch.no_grad():
        for image, paths in loader:

            vis_images = []
            for i, model in enumerate(models):
                start = time.time()
                polys0, polys, midlines = model(image.cuda())
                end = time.time()

                if midlines is not None:

                    for b, (im, poly0, poly, mid) in enumerate(zip(image, polys0, polys, midlines)):
                        vis = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(
                            np.uint8).copy()

                        poly0 = poly0.cpu().numpy().astype(np.int32).copy()
                        poly = poly.cpu().numpy().astype(np.int32).copy()
                        mid = mid.cpu().numpy().astype(np.int32).copy()

                        cv2.polylines(vis, poly0, True, (255, 0, 0), 1)
                        cv2.polylines(vis, poly, True, (0, 255, 0), 1)
                        cv2.polylines(vis, mid, True, (0, 0, 255), 1)

                        vis_images.append(vis)

                else:
                    for b, (im, poly0, poly) in enumerate(zip(image, polys0, polys)):
                        vis = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(
                            np.uint8).copy()

                        poly0 = poly0.cpu().numpy().astype(np.int32).copy()
                        poly = poly.cpu().numpy().astype(np.int32).copy()

                        cv2.polylines(vis, poly0, True, (255, 0, 0), 1)
                        cv2.polylines(vis, poly, True, (0, 255, 0), 1)
                        vis_images.append(vis)

                t += end - start

                vis = cv2.hconcat(vis_images)
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(vis_dir, os.path.basename(paths[0])), vis)

    print(t, t / len(dataset))
