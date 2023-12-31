from PIL import Image
import torch

from text_detection.detectors import MixNet, CRAFT
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Union
from pathlib import Path
import numpy as np
import math
import cv2


def check_runtime(print_elapsed=False, return_elapsed=True):
    def wrapper(func):
        def inner(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            if print_elapsed:
                print(f"Elapsed time: {elapsed_time * 1000 :.2f} msec")
            if return_elapsed:
                return elapsed_time, result
            return result

        return inner

    return wrapper


@check_runtime(print_elapsed=False, return_elapsed=True)
@torch.no_grad()
def textDetector(model: torch.nn.Module, img: Image, device: torch.device, max_size: int = 1024, **kwargs):
    '''
    [INPUT]
     - model: load_state_dict -> .to(device) -> eval모드까지 완료된 모델
     - img: PIL
     - device: cuda

    [OUTPUT]
     - pred_text: array([[x1,y1,x2,y2], [x1,y1,x2,y2], ..., [x1,y1,x2,y2]]) bbox좌표
    '''

    w, h = img.size
    size = min(math.ceil(max(w, h) / 32) * 32, max_size)

    transform = A.Compose([
        A.LongestMaxSize(size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(),
        ToTensorV2()
    ])
    tensor = transform(image=np.array(img))['image']
    th, tw = tensor.shape[-2:]

    predict = predict_mixnet if isinstance(model, MixNet) else predict_craft
    pred_text = predict(model, tensor, device, **kwargs)

    # scale bboxes
    ratio = min(tw / w, th / h)
    pad = ((tw - w * ratio) // 2, (th - h * ratio) // 2)

    for i in range(len(pred_text)):
        pred_text[i][0] = int(((pred_text[i][0] - pad[0]) / ratio))
        pred_text[i][1] = int(((pred_text[i][1] - pad[1]) / ratio))
        pred_text[i][2] = int(((pred_text[i][2] - pad[0]) / ratio))
        pred_text[i][3] = int(((pred_text[i][3] - pad[1]) / ratio))

    return pred_text


def setup_mixnet_model(ckpt_path: Union[str, Path], device: Union[int, torch.device]):
    '''
    [INPUT]
     - ckpt_path: pt 경로
     - device: cuda

    [OUTPUT]
     - net: load_state_dict -> .to(device) -> eval모드까지 완료된 모델
    '''

    # initialize
    net = MixNet('FSNet_M', mid=True, embed=False, dist_threshold=.3, cls_threshold=0.85)
    net.load_state_dict(torch.load(ckpt_path)['model'])
    net = net.to(device)
    net.eval()

    return net


def setup_craft_model(ckpt_path: Union[str, Path], device: Union[int, torch.device]):
    net = CRAFT()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.to(device)
    net.eval()

    return net


@torch.no_grad()
def predict_mixnet(model, tensor, device, **kwargs):
    tensor = tensor.unsqueeze(0).to(device)

    polys0, polys, midlines, confs = model(tensor)

    if not len(polys[0]):
        return []

    confs = confs[0].cpu().numpy()
    polys = polys[0].cpu().numpy()

    # polygon -> bbox
    pred_text = []
    for poly, conf in zip(polys, confs):
        x1, y1, x2, y2 = np.min(poly[..., 0]), np.min(poly[..., 1]), np.max(poly[..., 0]), np.max(poly[..., 1])
        pred_text.append([x1, y1, x2, y2, conf])

    pred_text = np.array(pred_text)
    return pred_text


@torch.no_grad()
def predict_craft(net, tensor, device, **kwargs):
    tensor = tensor.unsqueeze(0).to(device)

    preds, _ = net(tensor)

    bboxes, confs = net.postprocess(preds,
                                    char_threshold=kwargs.get('char_threshold', 0.6),
                                    link_threshold=kwargs.get('link_threshold', 0.3),
                                    word_threshold=kwargs.get('word_threshold', 0.7),
                                    poly=False)

    bboxes, confs = bboxes[0], confs[0]

    # polygon -> bbox
    pred_text = []
    for poly, conf in zip(bboxes, confs):
        x1, y1, x2, y2 = np.min(poly[..., 0]), np.min(poly[..., 1]), np.max(poly[..., 0]), np.max(poly[..., 1])
        pred_text.append([x1, y1, x2, y2, conf])

    pred_text = np.array(pred_text)
    return pred_text


if __name__ == '__main__':

    from pathlib import Path
    import time

    visualize = True
    vis_path = Path('examples/mixnet/origin')
    if visualize and not vis_path.exists():
        vis_path.mkdir(parents=True)

    root = Path('/mldisk2/ms/datasets/OCR/AOS_OCR/AOS_Test')
    images = root.rglob('*.jpg')

    net = setup_mixnet_model('./detectors/mixnet/pretrained/MixNet_FSNet_M_622_total_text.pth', 0)
    # net = setup_craft_model('./detectors/craft/pretrained/checkpoints-25.pth', 0)
    images = list(root.rglob('*.jpg'))

    elapsed = 0.
    for image in images:
        im = cv2.imread(image.as_posix())
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        t, preds = textDetector(net, im, 0, 1280)
        elapsed += t
        if visualize:  # visualize
            vis = np.array(im)
            for *bbox, score in preds:
                bbox = list(map(int, bbox))
                cv2.rectangle(vis, bbox[:2], bbox[2:], (255, 0, 0), 2)
                cv2.putText(vis, f'{score:.2f}', bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path.joinpath(image.name).as_posix(), vis)

    print(elapsed, elapsed / len(images))
