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

from utils.misc import set_seed
from utils.images import unnormalize_image_tensor

import matplotlib.pyplot as plt

import torch

import time


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        super().__init__()
        self.l = l
        self.transform = transform

    def __getitem__(self, idx):
        path = self.l[idx]

        image = cv2.imread(str(path))
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

    size = 640
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

    model = TextNet('FSNet_M', mid=True)
    ckpt = torch.load('./pretrained/MixNet_FSNet_M_160_icdar_art.pth')

    # model = TextNet('FSNet_M', mid=True, embed=False)
    # ckpt = torch.load('./pretrained/MixNet_FSNet_M_284_msra_td500.pth')

    # model = TextNet('FSNet_M', mid=True, embed=False)
    # ckpt = torch.load('./pretrained/MixNet_FSNet_M_622_total_text.pth')

    # model = TextNet('FSNet_hor', mid=False, pos=False, embed=False)
    # ckpt = torch.load('./pretrained/MixNet_FSNet_hor_925_ctw1500.pth')
    model.load_state_dict(ckpt['model'], strict=True)
    model.cuda()
    model.eval()

    t = 0
    with torch.no_grad():
        for image, paths in loader:

            start = time.time()
            polys0, polys, midlines = model(image.cuda())
            end = time.time()

            if midlines is not None:

                for b, (im, poly0, poly, mid) in enumerate(zip(image, polys0, polys, midlines)):
                    img = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

                    poly0 = poly0.cpu().numpy().astype(np.int32).copy()
                    poly = poly.cpu().numpy().astype(np.int32).copy()
                    mid = mid.cpu().numpy().astype(np.int32).copy()

                    cv2.polylines(img, poly0, True, (255, 0, 0), 1)
                    cv2.polylines(img, poly, True, (0, 255, 0), 1)
                    cv2.polylines(img, mid, True, (0, 0, 255), 1)
                    if len(poly):
                        plt.imshow(img)
                        plt.show()
            else:
                for b, (im, poly0, poly) in enumerate(zip(image, polys0, polys)):
                    img = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

                    poly0 = poly0.cpu().numpy().astype(np.int32).copy()
                    poly = poly.cpu().numpy().astype(np.int32).copy()

                    cv2.polylines(img, poly0, True, (255, 0, 0), 1)
                    cv2.polylines(img, poly, True, (0, 255, 0), 1)

                    if len(poly):
                        plt.imshow(img)
                        plt.show()
            t += end - start

    print(t, t / len(dataset))
