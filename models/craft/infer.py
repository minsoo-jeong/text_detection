from model import CRAFT, postprocess
from torchsummary import summary
import torch

from torch.utils.data import DataLoader, Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from utils.misc import set_seed, remove_module_prefix
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

    images = [p for p in Path(root).iterdir() if p.suffix == '.jpg'][-10:]

    dataset = ListDataset(images, transform)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = CRAFT()
    # ckpt = torch.load('./pretrained/CRAFT_clr_amp_29500.pth')
    # model.load_state_dict(remove_module_prefix(ckpt['craft']), strict=True)
    ckpt = torch.load('/workspace/text_detection/models/craft/checkpoints/20231221-175354/checkpoints-13.pth')
    model.load_state_dict(remove_module_prefix(ckpt), strict=True)
    model.cuda()
    model.eval()

    t = 0
    with torch.no_grad():
        for image, paths in loader:
            start = time.time()
            outputs, _ = model(image.cuda())

            pred_boxes = []
            for output in outputs:
                boxes = postprocess(output.unsqueeze(0),
                                    image.shape[2:],
                                    char_threshold=.6,
                                    link_threshold=.3,
                                    word_threshold=.7,
                                    )
                pred_boxes.append(boxes)
            end = time.time()

            for b, (im, bboxes) in enumerate(zip(image, pred_boxes)):
                img = (unnormalize_image_tensor(im).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                bboxes = bboxes.astype(np.int32)

                for box in bboxes:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                if len(bboxes):
                    plt.imshow(img)
                    plt.show()
            t += end - start

    print(t, t / len(dataset))
