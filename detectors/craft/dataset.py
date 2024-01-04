from shapely.geometry import Polygon, LineString
from easydict import EasyDict
import numpy as np
import torch
import cv2

from model.craft import CRAFT
from pseudo_label.make_charbox import PseudoCharBoxBuilder, generate_target, generate_affinity
from text_detection.datasets.icdar15 import ICDAR15


class craft_dataset(ICDAR15):

    def __init__(self, image_root, label_root=None, label_model=None, label_format='polygon', transform=None):
        super().__init__(image_root, label_root, 'polygon', transform)

        self.label_min_area = 0.5
        self.char_builder = PseudoCharBoxBuilder(
            watershed_param=EasyDict(version='skimage', sure_fg_th=0.75, sure_bg_th=0.05))

        self.label_model = label_model
        self._label_format = label_format

    def __getitem__(self, idx):
        image_tensor, *elem = super().__getitem__(idx)

        if self.label_root:
            polygons, words, path = elem

            weight_character, weight_affinity = self.generate_pseudo_charbox(image_tensor, polygons, words)

            if self._label_format == 'bbox':
                polygons = [np.array(Polygon(p).bounds).astype(np.int32) for p in polygons]

            return image_tensor, weight_character, weight_affinity, polygons, words, path

        return image_tensor, *elem

    def generate_pseudo_charbox(self, tensor, polygons, words):
        h, w = tensor.shape[1:]
        if len(words) == 0:
            weight_character = np.zeros((h // 2, w // 2), dtype=np.float32)
            weight_affinity = np.zeros((h // 2, w // 2), dtype=np.float32)

        else:
            image = tensor.numpy().transpose(1, 2, 0)

            char_boxes, _, _, words_count = self.char_builder.build_char_box2(self.label_model,
                                                                              image.copy(),
                                                                              polygons,
                                                                              words)

            small_char_boxes = char_boxes * .5
            small_char_boxes = np.transpose(small_char_boxes, (2, 1, 0))
            small_image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            weight_character = generate_target(small_image.shape, small_char_boxes.copy())
            weight_affinity, _ = generate_affinity(small_image.shape, small_char_boxes.copy(), words, words_count)

            weight_character = weight_character.astype(np.float32)
            weight_affinity = weight_affinity.astype(np.float32)

        return weight_character, weight_affinity

    @staticmethod
    def collate(batch):
        image, weight_character, weight_affinity, polygons, words, path = zip(*batch)

        return (torch.stack(image), torch.from_numpy(np.stack(weight_character)),
                torch.from_numpy(np.stack(weight_affinity)), polygons, words, path)
