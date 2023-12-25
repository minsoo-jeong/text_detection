
from torch.utils.data import DataLoader, Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from utils.misc import set_seed, filter_word
from utils.bbox import points2xyxy, check_bbox_validity

import matplotlib.pyplot as plt


class IC15(Dataset):
    def __init__(self, image_root, label_root, transform):

        self.items = self.load_icdar_data(image_root, label_root)
        self.transform = transform

    def __getitem__(self, index):
        path, ann_path = self.items[index]

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape

        bboxes, words = self.load_icdar_label(ann_path, w, h)

        try:
            augmented = self.transform(image=image, bboxes=bboxes, words=words)
        except Exception as e:
            print(f'>> Execption: {e}')
            # print(path, bboxes, words)
            return self.__getitem__(index + 1 % len(self.items))

        image = augmented['image']
        bboxes = augmented['bboxes']
        words = augmented['words']

        return image, np.array(bboxes), words, str(path)

    def __len__(self):
        return len(self.items)

    def load_icdar_data(self, image_root, label_root):
        image_root = Path(image_root)
        label_root = Path(label_root)

        images = {p.stem: p for p in image_root.iterdir() if p.suffix == '.jpg'}
        annotations = {p.stem.replace('gt_', ''): p for p in label_root.iterdir()}

        items = []
        for k in tqdm(sorted(images.keys())):
            if annotations.get(k):
                items.append((images[k], annotations[k]))

        return items

    def load_icdar_label(self, path, width, height):
        bboxes = []
        words = []
        with open(path, 'r') as f:
            for line in f.readlines():
                label = line.strip().split(',')
                points = np.array(list(map(int, label[:8]))).reshape(4, 2)
                bbox = points2xyxy(points)

                word = ','.join(label[8:])
                filtered = filter_word(word)

                # check bbox validity
                if not check_bbox_validity(bbox, (width, height)) or len(filtered) == 0:
                    continue

                bboxes.append(bbox)
                word = word
                words.append(word)

        return np.array(bboxes), words

    @staticmethod
    def collate(batch):
        image, bboxes, words, path = zip(*batch)

        return image, bboxes, words, path




if __name__ == '__main__':
    from matplotlib import pyplot as plt

    set_seed(42)

    root = '/data/OCR/023.OCR 데이터(공공)/01-1.정식개방데이터/sample01'
    root = '/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_1'
    root='/workspace/data/aos-invoice/sample01'
    size = 1536

    transform = A.Compose([
        A.LongestMaxSize(size),
        A.RandomScale(scale_limit=(1., 2.), interpolation=cv2.INTER_CUBIC),
        A.RandomResizedCrop(size, size, scale=(0.03, 1.), interpolation=cv2.INTER_CUBIC),
        A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2, always_apply=False, p=0.5),
        A.SafeRotate(20, border_mode=cv2.BORDER_CONSTANT),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),

    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['words']))

    dataset = IC15(f'{root}/test_images', f'{root}/test_labels', transform)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, collate_fn=IC15.collate, shuffle=False, num_workers=4)


    samples= []
    n_samples = 11
    while len(samples)< n_samples:
        batch=next(iter(loader),None)
        items = zip(*batch)
        samples.extend(items)
    print(len(samples))

    for i,batch in enumerate(loader):
        items = zip(*batch)
        samples.extend(items)
        print(i)
    print(len(samples))

    while len(samples) < n_samples:
        batch= next(iter(loader),None)
        if batch is None:
            break
        samples.extend(zip(*batch))
    samples = samples[:n_samples]


    exit()



    for image, bboxes, words, path in tqdm(loader):
        continue
        # print(image.shape, bboxes.shape, words)

        # im = image.permute(1, 2, 0).numpy()
        # bboxes = bboxes.astype(np.int32)
        # for box in bboxes:
        #     cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #
        # plt.imshow(im)
        # plt.show()

    exit()
