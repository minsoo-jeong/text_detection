from shapely.geometry import Polygon
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

        self.label_min_area = 0.8
        self.char_builder = PseudoCharBoxBuilder(watershed_param=EasyDict(version='skimage'))
        self.label_model = label_model
        self._label_format = label_format

    def __getitem__(self, idx):
        image_tensor, *elem = super().__getitem__(idx)

        if self.label_root:
            polygons, words, path = elem

            weight_character, weight_affinity = self.generate_pseudo_charbox(image_tensor, polygons, words, path)

            if self._label_format == 'bbox':
                polygons = [np.array(Polygon(p).bounds).astype(np.int32) for p in polygons]

            return image_tensor, weight_character, weight_affinity, polygons, words, path

        return image_tensor, *elem

    def generate_pseudo_charbox(self, tensor, polygons, words, path):
        h, w = tensor.shape[1:]
        if len(words) == 0:
            weight_character = np.zeros((h // 2, w // 2), dtype=np.float32)
            weight_affinity = np.zeros((h // 2, w // 2), dtype=np.float32)

        else:
            # image = tensor_to_image(tensor).astype(np.float32)
            image = tensor.numpy().transpose(1, 2, 0).astype(np.float32)
            word_boxes = []
            for p in polygons:
                # x1, y1, x2, y2 = np.array(Polygon(p).minimum_rotated_rectangle.bounds)
                # word_boxes.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                points = np.array(Polygon(p).minimum_rotated_rectangle.exterior.coords)[:-1]
                word_boxes.append(points)

            char_boxes, _, _, words_count = self.char_builder.build_char_box(self.label_model,
                                                                             image,
                                                                             word_boxes,
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


if __name__ == '__main__':
    import torch
    from text_detection.utils import remove_module_prefix
    import cv2

    ckpt = 'pretrained/CRAFT_clr_amp_29500.pth'
    state_dict = 'craft'
    device = 0
    # device = 'cpu'
    label_model = CRAFT().to(device)
    checkpoint = torch.load(ckpt)[state_dict]
    label_model.load_state_dict(remove_module_prefix(checkpoint), strict=True)
    label_model.eval()

    from matplotlib import pyplot as plt
    from text_detection.utils import set_seed, tensor_to_image

    set_seed(88)

    root = '/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_1'

    size = 1536

    dataset = craft_dataset(f'{root}/test_images',
                            f'{root}/test_labels',
                            label_model=label_model,
                            label_format='polygon',  # 'bbox',
                            transform=ICDAR15.train_transform(size))

    for image, char, affinity, polygons, words, path in dataset:
        im = tensor_to_image(image).copy()

        if dataset._label_format == 'bbox':
            for rec in polygons:
                cv2.rectangle(im, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
        else:
            cv2.polylines(im, polygons, True, (0, 255, 0), 2)

        plt.imshow(im)
        plt.show()

        break

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate, shuffle=False, num_workers=0)

    for batch in loader:

        for image, weight_character, weight_affinity, polygons, words, path in zip(*batch):

            im = tensor_to_image(image).copy()
            if loader.dataset._label_format == 'bbox':
                for rec in polygons:
                    cv2.rectangle(im, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
            else:
                cv2.polylines(im, polygons, True, (0, 255, 0), 2)

            plt.imshow(im)
            plt.show()

            plt.imshow(weight_character.numpy())
            plt.show()
            plt.imshow(weight_affinity.numpy())
            plt.show()
        break
