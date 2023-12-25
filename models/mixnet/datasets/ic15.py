from torch.utils.data import DataLoader, Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from utils.misc import set_seed, filter_word
from utils.bbox import points2xyxy, check_bbox_validity, check_validity

import matplotlib.pyplot as plt
from collections import defaultdict

from shapely.geometry import Polygon, LineString
from shapely.ops import clip_by_rect

from scipy import ndimage as ndimg
import torch


def sampling_points(polygon, num_points=20):
    exterior = polygon.exterior
    points = []
    for distance in np.arange(0, exterior.length, exterior.length / num_points):
        point = exterior.interpolate(distance)
        points.append(point)

    return points


def proposal_points(polygon, num_points=20):
    pass


def make_text_region(image_tensor, polygons, words):
    control_points = [sampling_points(poly, num_points=20) for poly in polygons]
    _, th, tw = image_tensor.shape
    mask = np.zeros((th, tw), dtype=np.uint8)
    ##
    for poly in polygons:
        points = np.array(poly.exterior.coords).astype(int)

        cv2.fillPoly(mask, [points], 1)

    dmp = ndimg.distance_transform_edt(mask)
    max_dmp = np.max(dmp)

    a = dmp / (max_dmp + 1e-6) > 0.35
    contours, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        poly = Polygon(np.squeeze(contour))
        ctrl_points = sampling_points(poly, num_points=20)

        offset = np.random.rand()
        print(poly)
    print(len(polygons), polygons)

    im = foo(image_tensor).numpy().transpose(1, 2, 0) * 255.
    im = im.astype(np.uint8)

    a = a.astype(np.uint8)
    plt.imshow(a)
    plt.show()

    plt.imshow(255 * dmp / (max_dmp + 1e-6))
    plt.show()

    plt.imshow(mask)
    plt.show()

    plt.imshow(im)
    plt.show()
    exit()


from functools import partial


def default_transform(size):
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

    return transform


class IC15(Dataset):
    def __init__(self, image_root, label_root=None, transform=None):

        self.items = self.load_icdar_data(image_root, label_root)
        self.transform = transform

    def __getitem__(self, index):
        path, ann_path = self.items[index]

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        polygons, words = self.load_icdar_label(ann_path, width, height)

        image_tensor, polygons, words = self.apply_transform(image, polygons, words)

        return image_tensor, polygons, words, str(path)

    def __len__(self):
        return len(self.items)

    def apply_transform(self, image, polygons, words):

        polygon_id = [pid for pid, p in enumerate(polygons) for _ in p]
        transformed = self.transform(image=image, keypoints=polygons.reshape(-1, 2), box_id=polygon_id)
        keypoints = defaultdict(list)
        for pid, point in zip(transformed['box_id'], transformed['keypoints']):
            keypoints[pid].append(point)

        _, height, width = transformed['image'].shape

        cliped_polygons = []
        cliped_words = []

        for pid, points in keypoints.items():
            polygon = Polygon(np.array(points).reshape(-1, 2).astype(int))
            cliped = clip_by_rect(polygon, 0, 0, width - 1, height - 1)

            if cliped.geom_type == 'Polygon' and not cliped.is_empty:
                # cliped_polygons.append(np.array(cliped.exterior.coords).astype(int))
                cliped_polygons.append(cliped)
                cliped_words.append(words[pid])

        return transformed['image'], cliped_polygons, cliped_words

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
        polygons = []
        words = []
        with open(path, 'r') as f:
            for line in f.readlines():
                label = line.strip().split(',')
                points = np.array(list(map(int, label[:8]))).reshape(4, 2)
                word = ','.join(label[8:])
                filtered = filter_word(word)

                # check bbox validity
                if not check_validity(points, (width, height), bbox_format='xy') or len(filtered) == 0:
                    continue

                polygons.append(points)
                word = word
                words.append(word)
        return np.array(polygons), words

    @staticmethod
    def collate(batch):
        image, polygons, words, path = zip(*batch)

        return torch.stack(image), polygons, words, path


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    set_seed(88)

    root = '/data/OCR/023.OCR 데이터(공공)/01-1.정식개방데이터/sample01'
    root = '/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_1'
    root = '/workspace/data/aos-invoice/sample01'
    root = '/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01'
    size = 1536

    transform = A.Compose(
        [
            A.LongestMaxSize(size),
            A.RandomScale(scale_limit=(1., 2.), interpolation=cv2.INTER_CUBIC),
            A.SafeRotate(45, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.RandomResizedCrop(size, size, scale=(0.03, 0.1), interpolation=cv2.INTER_CUBIC),
            # A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2, always_apply=False, p=0.5),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
            ToTensorV2()
        ]  # , keypoint_params=A.KeypointParams(format='xy', label_fields=['words','pidx']))
        , keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['box_id']))
    dataset = IC15(f'{root}/test_images', f'{root}/test_labels', transform)

    if False:
        for image, polygons, words, path in dataset:
            im = foo(image).numpy().transpose(1, 2, 0) * 255.
            im = im.astype(np.uint8)

            polys = [np.array(poly.exterior.coords).astype(int) for poly in polygons]

            cv2.polylines(im, polys, True, (0, 255, 0), 2)

            plt.imshow(im)
            plt.show()

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, collate_fn=IC15.collate, shuffle=False, num_workers=4)

    for image, polygons, words, path in loader:

        for img in image:
            im = foo(img).numpy().transpose(1, 2, 0) * 255.
            im = im.astype(np.uint8)

        polys = [np.array(poly.exterior.coords).astype(int) for poly in polygons]

        cv2.polylines(im, polys, True, (0, 255, 0), 2)

        plt.imshow(im)
        plt.show()

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
