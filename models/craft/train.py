
import torch.multiprocessing as mp

from datetime import datetime
import argparse

import wandb


from tqdm import tqdm
from easydict import EasyDict
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import numpy as np
import cv2


from albumentations.pytorch import ToTensorV2
import albumentations as A


from datasets.ic15 import IC15
from pseudo_label.make_charbox import PseudoCharBoxBuilder, generate_pseudo_charbox


from utils.metrics import DetectionIoUEvaluator, AverageMeter,FscoreMeter
from utils.images import hconcat_images,vconcat_images,draw_text
from utils.bbox import xyxy2points
from utils.misc import set_seed, remove_module_prefix
from utils.distributed import *


from model import CRAFT, postprocess
from losses import CraftLoss


def generate_pseudo_charbox_batch(builder, model, images, bboxes, words):
    batch_weight_character, batch_weight_affinity = [], []

    transform = A.Compose([A.Normalize(), ToTensorV2()])

    image_tensor = []
    for idx, (image, boxes, word) in enumerate(zip(images, bboxes, words)):
        weight_character, weight_affinity = generate_pseudo_charbox(builder, model, image, boxes, word)

        image_tensor.append(transform(image=image)['image'])
        batch_weight_character.append(weight_character)
        batch_weight_affinity.append(weight_affinity)

    return (torch.from_numpy(np.stack(image_tensor)),
            torch.from_numpy(np.stack(batch_weight_character)),
            torch.from_numpy(np.stack(batch_weight_affinity)))


def train_one_epoch(model, loader, label_model, builder, criterion, optimizer, scheduler, epoch, scaler, args):
    losses = AverageMeter()
    if is_main_process():
        input_sample = sampling_input(model, loader, builder, label_model, 4, args)
        wandb.log({'Train/input': wandb.Image(input_sample)},step=epoch)
    model.train()
    label_model.eval()


    progress = tqdm(loader, ncols=180) if is_main_process() else None
    if args.distributed:
        loader.sampler.set_epoch(epoch)


    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        images, bboxes, words, paths = batch

        with torch.no_grad():
            image_tensor, weight_character, weight_affinity = generate_pseudo_charbox_batch(builder,
                                                                                            label_model,
                                                                                            images,
                                                                                            bboxes,
                                                                                            words)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs, _ = model(image_tensor.to(args.device))
            loss = criterion(outputs, weight_character.to(args.device), weight_affinity.to(args.device))

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        losses.update(loss.item())

        if is_main_process():
            progress.update()
            progress.set_description(f'[Train {epoch:>2}] '
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                                     f'LR: {optimizer.param_groups[0]["lr"]:.4e}')

    if is_main_process():
        progress.close()
        wandb.log({"Train/loss": losses.avg, "Train/LR": optimizer.param_groups[0]["lr"]},step=epoch)

    return {'train_loss': losses.avg, 'LR': optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def test_aos(model, loader, label_model, builder, criterion, epoch, args):
    losses = AverageMeter()
    metrics = FscoreMeter()
    evalutator = DetectionIoUEvaluator()

    if is_main_process():
        input_sample = sampling_input(model, loader, builder, label_model, 4, args)
        wandb.log({'Test-aos/input': wandb.Image(input_sample)},step=epoch)
    model.eval()
    label_model.eval()

    progress = tqdm(loader, ncols=180) if is_main_process() else None
    for step, batch in enumerate(loader):
        images, bboxes, words, paths = batch

        image_tensor, weight_character, weight_affinity = generate_pseudo_charbox_batch(builder,
                                                                                        label_model,
                                                                                        images,
                                                                                        bboxes,
                                                                                        words)

        outputs, _ = model(image_tensor.to(args.device))
        loss = criterion(outputs, weight_character.to(args.device), weight_affinity.to(args.device))
        losses.update(loss.item())

        batch_results = []
        for i, (output, gt_boxes) in enumerate(zip(outputs, bboxes)):
            pred_boxes = postprocess(output.unsqueeze(0),
                                    image_tensor.shape[2:],
                                    char_threshold=args.char_threshold,
                                    link_threshold=args.link_threshold,
                                    word_threshold=args.word_threshold,
                                    )


            result = evalutator.evaluate_image(
                [dict(points=point, text=w, ignore=w == '###') for point, w in zip(xyxy2points(gt_boxes), words[i])],
                [dict(points=point, text='###', ignore=False) for point in xyxy2points(pred_boxes)],
            )
            batch_results.append(result)

        if args.distributed:
            batch_results = gather_list(batch_results, args.rank, args.world_size, dst=0)

        if is_main_process():
            batch_metric = EasyDict(evalutator.combine_results(batch_results))

            metrics.update(batch_metric.detMatched,
                           batch_metric.detCare - batch_metric.detMatched,
                           batch_metric.gtCare - batch_metric.detMatched,
                           len(batch_results)
                           )
            progress.update()
            progress.set_description(f'[Test-aos {epoch:>3}] '
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                                     f'F1: {batch_metric.hmean:.4f}({metrics.f1:.4f}) '
                                     f'Prec: {batch_metric.precision:.4f}({metrics.prec:.4f}) '
                                     f'Rec: {batch_metric.recall:.4f}({metrics.rec:.4f})')

    if is_main_process():
        progress.close()
        wandb.log({"Test-aos/loss": losses.avg,
                   "Test-aos/fscore": metrics.f1,
                   'Test-aos/precision': metrics.prec,
                   'Test-aos/recall': metrics.rec
                   },step=epoch)

    return {'loss': losses.avg, 'fscore': metrics.f1, 'precision': metrics.prec, 'recall': metrics.rec}


@torch.no_grad()
def test(model, loader, label_model, builder, criterion, epoch, args):
    losses = AverageMeter()
    metrics = FscoreMeter()
    evalutator = DetectionIoUEvaluator()

    if is_main_process():
        input_sample = sampling_input(model, loader, builder, label_model, 4, args)
        wandb.log({'Test/input': wandb.Image(input_sample)}, step=epoch)

    model.eval()
    label_model.eval()

    progress = tqdm(loader, ncols=180) if is_main_process() else None
    for step, batch in enumerate(loader):
        images, bboxes, words, paths = batch

        image_tensor, weight_character, weight_affinity = generate_pseudo_charbox_batch(builder,
                                                                                        label_model,
                                                                                        images,
                                                                                        bboxes,
                                                                                        words)

        outputs, _ = model(image_tensor.to(args.device))
        loss = criterion(outputs, weight_character.to(args.device), weight_affinity.to(args.device))
        losses.update(loss.item())

        batch_results = []
        for i, (output, gt_boxes) in enumerate(zip(outputs, bboxes)):
            pred_boxes = postprocess(output.unsqueeze(0),
                                    image_tensor.shape[2:],
                                    char_threshold=args.char_threshold,
                                    link_threshold=args.link_threshold,
                                    word_threshold=args.word_threshold,
                                    )
            result = evalutator.evaluate_image(
                [dict(points=point, text=w, ignore=w == '###') for point, w in zip(xyxy2points(gt_boxes), words[i])],
                [dict(points=point, text='###', ignore=False) for point in xyxy2points(pred_boxes)],
            )
            batch_results.append(result)

        if args.distributed:
            batch_results = gather_list(batch_results, args.rank, args.world_size, dst=0)

        if is_main_process():
            batch_metric = EasyDict(evalutator.combine_results(batch_results))

            metrics.update(batch_metric.detMatched,
                           batch_metric.detCare - batch_metric.detMatched,
                           batch_metric.gtCare - batch_metric.detMatched,
                           len(batch_results)
                           )
            progress.update()
            progress.set_description(f'[Test {epoch:>3}] '
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                                     f'F1: {batch_metric.hmean:.4f}({metrics.f1:.4f}) '
                                     f'Prec: {batch_metric.precision:.4f}({metrics.prec:.4f}) '
                                     f'Rec: {batch_metric.recall:.4f}({metrics.rec:.4f})')

    if is_main_process():
        progress.close()
        wandb.log({"Test/loss": losses.avg,
                   "Test/fscore": metrics.f1,
                   'Test/precision': metrics.prec,
                   'Test/recall': metrics.rec
                   },step=epoch)

    return {'loss': losses.avg, 'fscore': metrics.f1, 'precision': metrics.prec, 'recall': metrics.rec}


@torch.no_grad()
def sampling_input(model, loader, builder, label_model, n_samples, args):
    model.eval()
    label_model.eval()
    evalutator = DetectionIoUEvaluator()

    samples = []
    while len(samples) < n_samples:
        batch= next(iter(loader),None)
        if batch is None:
            break
        samples.extend(zip(*batch))
    samples = samples[:n_samples]

    vis_images = []
    for image, boxes, words, paths in samples:
        image_tensor, weight_character, weight_affinity = generate_pseudo_charbox_batch(builder, label_model, [image], [boxes], [words])

        outputs, _ = model(image_tensor.to(args.device))

        pred_boxes = postprocess(outputs,
                                image_tensor.shape[2:],
                                char_threshold=args.char_threshold,
                                link_threshold=args.link_threshold,
                                word_threshold=args.word_threshold,
                                )

        result = EasyDict(evalutator.evaluate_image(
                [dict(points=point, text=w, ignore=w == '###') for point, w in zip(xyxy2points(boxes), words)],
                [dict(points=point, text='###', ignore=False) for point in xyxy2points(pred_boxes)],
            ))

        text = f'H-mean: {result.hmean:.2f}, '
        text += f'Precision: {result.precision:.2f}, '
        text += f'Recall: {result.recall:.2f}, '
        text += f'TP: {result.detMatched}, '
        text += f'FP: {result.detCare-result.detMatched}, '
        text += f'FN: {result.gtCare-result.detMatched}'

        vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for point in xyxy2points(boxes):
            vis=cv2.polylines(vis, [point.astype(np.int32)], True, (0, 255, 0), 2)

        for point in xyxy2points(pred_boxes):
            vis = cv2.polylines(vis, [point.astype(np.int32)], True, (0, 0, 255), 2)

        vis = hconcat_images([vis, weight_character[0], weight_affinity[0]], height=480)
        vis = draw_text(vis, text)
        vis_images.append(vis)

    vis_image= vconcat_images(vis_images)
    vis_image=cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB)
    return vis_image


def main_worker(gpu, args):
    args.local_rank = gpu
    init_distributed_mode(args)
    setup_for_distributed(is_main_process())
    set_seed(args.seed)
    print(args)

    # Model
    model = CRAFT().to(args.device)
    label_model = CRAFT().to(args.device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        if args.state_dict:
            checkpoint = checkpoint[args.state_dict]
        model.load_state_dict(remove_module_prefix(checkpoint), strict=False)
        label_model.load_state_dict(remove_module_prefix(checkpoint), strict=False)
        print(f'>> Load pretrained {args.ckpt}')

    if args.distributed:
        model_without_ddp = model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Criterion
    criterion = CraftLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Dataset
    label_builder = PseudoCharBoxBuilder(EasyDict(version='skimage', sure_fg_th=args.watershed_fg, sure_bg_th=args.watershed_bg))
    train_transform = A.Compose([
        A.LongestMaxSize(args.image_size),
        A.RandomScale(scale_limit=(1., 2.), interpolation=cv2.INTER_CUBIC),
        A.RandomResizedCrop(args.image_size, args.image_size, scale=(0.03, 1.), interpolation=cv2.INTER_CUBIC),
        A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2, always_apply=False, p=0.5),
        A.SafeRotate(20, border_mode=cv2.BORDER_CONSTANT),
        A.PadIfNeeded(args.image_size, args.image_size, border_mode=cv2.BORDER_CONSTANT, value=0),

    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['words'], min_visibility=0.8))

    test_transform = A.Compose([
        A.LongestMaxSize(args.image_size_test),
        A.PadIfNeeded(args.image_size_test, args.image_size_test, border_mode=cv2.BORDER_CONSTANT, value=0),

    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['words']))

    train_dataset = IC15(f'{args.train_data_root}/train_images',
                         f'{args.train_data_root}/train_labels',
                         train_transform)
    test_dataset = IC15(f'{args.test_data_root}/test_images',
                        f'{args.test_data_root}/test_labels',
                        test_transform)

    test_dataset2 = IC15(f'/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01/test_images',
                         f'/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01/test_labels',
                         test_transform)

    train_sampler, test_sampler, test_sampler2 = None, None, None
    if args.distributed:
        # args.total_train_batch = args.train_batch * args.world_size
        # args.data_worker_per_node = args.worker * args.proc_per_node
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        test_sampler2 = torch.utils.data.distributed.DistributedSampler(test_dataset2, shuffle=False, drop_last=False)


    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              sampler=train_sampler,
                              batch_size=args.batch,
                              num_workers=args.worker,
                              shuffle=True if train_sampler is None else False,
                              )

    test_loader = DataLoader(test_dataset,
                             collate_fn=test_dataset.collate,
                             sampler=test_sampler,
                             batch_size=args.batch_test,
                             num_workers=args.worker,
                             )
    test_loader2 = DataLoader(test_dataset2,
                              collate_fn=test_dataset2.collate,
                              sampler=test_sampler2,
                              batch_size=args.batch_test,
                              num_workers=args.worker,
                             )

    if is_main_process():
        wandb.init(project=args.project, entity="ms-jeong", name=args.exp)
        wandb.config.update(args)

    if args.init_eval:
        test_aos(model, test_loader2, label_model, label_builder, criterion, 0, args)
        test(model, test_loader, label_model, label_builder, criterion, 0, args)


    for epoch in range(1, args.epoch):

        train_one_epoch(model, train_loader, label_model, label_builder, criterion, optimizer, scheduler, epoch, scaler,
                        args)

        test(model, test_loader, label_model, label_builder, criterion, epoch, args)

        test_aos(model, test_loader2, label_model, label_builder, criterion, epoch, args)


        ckpt = remove_module_prefix(model.state_dict())
        ckpt_dir = Path(f'checkpoints/{args.exp}')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(ckpt, ckpt_dir.joinpath(f'checkpoints-{epoch:02d}.pth').as_posix())
        label_model.load_state_dict(ckpt)

    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--image_size', default=1536, type=int, help='image size')
    parser.add_argument('--image_size_test', default=1536, type=int, help='image size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')


    parser.add_argument('--train_data_root', default='/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_2', type=str, help='data path')
    parser.add_argument('--test_data_root', default='/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_2', type=str, help='data path')

    parser.add_argument('--ckpt', default='pretrained/CRAFT_clr_amp_29500.pth', type=str, help='pretrained model')
    parser.add_argument('--state_dict', default='craft', type=str, help='')
    # parser.add_argument('--ckpt', default=None, type=str, help='pretrained model')
    # parser.add_argument('--state_dict', default='craft', type=str, help='')

    parser.add_argument('--batch', default=8, type=int, help='batch size for train')
    parser.add_argument('--batch_test', default=8, type=int, help='batch size for train')
    parser.add_argument('--worker', default=4, type=int, help='number of workers for train')

    parser.add_argument('--init-eval', action='store_true', help='init eval')

    # watershed parameters for generate char label mask
    parser.add_argument('--watershed-fg', default=0.75, type=float, help='sure_fg_th for pseudo label')
    parser.add_argument('--watershed-bg', default=0.05, type=float, help='sure_bg_th for pseudo label')

    # craft postprocess threshlod
    parser.add_argument('--char_threshold', default=0.6, type=float, help='character threshold')
    parser.add_argument('--link_threshold', default=0.3, type=float, help='character threshold')
    parser.add_argument('--word_threshold', default=0.7, type=float, help='character threshold')

    # auto mix precision
    parser.add_argument('--amp', default=True, action='store_true', help='use amp')

    # Distributed training parameters
    parser.add_argument('-n', '--node', default=1, type=int, help='number of distributed node')
    parser.add_argument('-r', '--node_rank', default=0, type=int, help='node rank')
    parser.add_argument('-p', '--proc_per_node', default=2, type=int,
                        help='number of distributed processes per each node')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--project', default='aos-craft-train', type=str, help='project name')
    parser.add_argument('--exp', default=None, type=str, help='experiment name')

    args = parser.parse_args()

    if args.exp is None:
        args.exp = datetime.now().strftime("%Y%m%d-%H%M%S")

    n_gpu = torch.cuda.device_count()
    if n_gpu< args.proc_per_node:
        args.proc_per_node = n_gpu

    args.world_size = args.node * args.proc_per_node
    args.distributed = args.world_size > 1

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.proc_per_node, args=(args,))
    else:
        main_worker(0, args)
