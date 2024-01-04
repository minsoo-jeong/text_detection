from torch.utils.data import Dataset

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from collections import defaultdict
from shapely.geometry import Polygon
from shapely.ops import clip_by_rect

import torch

from text_detection.utils import filter_word, check_polygon_validity


class ICDAR15(Dataset):
    def __init__(self, image_root, label_root=None, label_format='polygon', transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.label_format = label_format
        self.label_min_area = 0.5

        self.items = self.load_icdar_data()
        self.transform = transform if transform is not None else ICDAR15.default_transform(640)

    def __getitem__(self, index):
        item = self.items[index]

        if self.label_root:
            path, ann_path = item

            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            polygons, words = self.load_icdar_label(ann_path)
            image_tensor, polygons, words = self.apply_transform_with_polygons(image, polygons, words)

            if self.label_format == 'bbox':
                polygons = [np.array(Polygon(p).bounds).astype(np.int32) for p in polygons]
            else:
                polygons = [np.array(p.exterior.coords).astype(np.int32) for p in polygons]

            return image_tensor, polygons, words, str(path)
        else:
            path = item
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_tensor = self.transform(image=image)['image']

            return image_tensor, str(path)

    def __len__(self):
        return len(self.items)

    def apply_transform_with_polygons(self, image, polygons, words):
        polygon_id = [pid for pid, p in enumerate(polygons) for _ in p]
        _poly = np.concatenate(polygons) if len(polygons) else np.empty((0, 2))

        transformed = self.transform(image=image, keypoints=_poly, polygon_id=polygon_id)

        keypoints = defaultdict(list)
        for pid, point in zip(transformed['polygon_id'], transformed['keypoints']):
            keypoints[pid].append(point)
        _, height, width = transformed['image'].shape

        cliped_polygons = []
        cliped_words = []

        for pid, points in keypoints.items():
            polygon = Polygon(np.array(points).reshape(-1, 2).astype(int))
            cliped = clip_by_rect(polygon, 0, 0, width - 1, height - 1)

            if (cliped.geom_type == 'Polygon' and not cliped.is_empty and cliped.area / (
                    polygon.area + 1e-6) > self.label_min_area):
                cliped_polygons.append(cliped)
                cliped_words.append(words[pid])

        return transformed['image'], cliped_polygons, cliped_words

    def load_icdar_data(self):

        images = {p.stem: p for p in Path(self.image_root).iterdir() if p.suffix == '.jpg'}

        if self.label_root is None:
            return list(images.values())

        items = []
        annotations = {p.stem.replace('gt_', ''): p for p in Path(self.label_root).iterdir()}

        for k in tqdm(sorted(images.keys())):
            if annotations.get(k):
                items.append((images[k], annotations[k]))

        return items

    def load_icdar_label(self, path):
        polygons = []
        words = []
        with open(path, 'r') as f:
            for line in f.readlines():
                label = line.strip().split(',')
                points = np.array(list(map(int, label[:8]))).reshape(4, 2)
                word = ','.join(label[8:])
                filtered = filter_word(word)

                if not check_polygon_validity(points) or len(filtered) == 0:
                    continue

                polygons.append(points)
                word = filtered
                words.append(word)
        return polygons, words

    @staticmethod
    def collate(batch):
        image, *elem = zip(*batch)

        return torch.stack(image), *elem

    @staticmethod
    def default_transform(size, label=False):
        transform = A.Compose([
            A.LongestMaxSize(size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
            ToTensorV2()],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False,
                                             label_fields=['polygon_id']) if label else None)

        return transform

    @staticmethod
    def train_transform(size):
        transform = A.Compose(
            [
                A.LongestMaxSize(size),
                A.RandomScale(scale_limit=(1., 2.), interpolation=cv2.INTER_CUBIC),
                A.SafeRotate(15, border_mode=cv2.BORDER_CONSTANT),
                A.RandomResizedCrop(size, size, scale=(0.03, 0.1), interpolation=cv2.INTER_CUBIC),
                A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2, always_apply=False, p=0.5),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['polygon_id']))

        return transform
