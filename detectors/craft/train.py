import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch

from datetime import datetime
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import wandb
import cv2
import os

from model.craft import CRAFT
from dataset import craft_dataset
from loss import craft_loss

from text_detection.utils import init_distributed_mode, setup_for_distributed, set_seed, is_main_process, \
    remove_module_prefix, AverageMeter, FscoreMeter, DetectionIoUEvaluator, gather_list, tensor_to_image, \
    hconcat_images, vconcat_images, draw_text

warnings.filterwarnings(action='ignore', category=UserWarning)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, scaler, args):
    losses = AverageMeter()

    if is_main_process() and args.wandb_entity:
        input_sample = visualize_samples(model, loader, 4, args)
        wandb.log({'Train/input': wandb.Image(input_sample)}, step=epoch)

    model.train()
    loader.dataset.label_model.eval()
    progress = tqdm(loader, ncols=180) if is_main_process() else None
    if args.distributed:
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        image_tensor, weight_characters, weight_affinities, bboxes, words, paths = batch

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs, _ = model(image_tensor.to(args.device))
            loss = criterion(outputs, weight_characters.to(args.device), weight_affinities.to(args.device))

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
        if args.wandb_entity:
            wandb.log({"Train/loss": losses.avg, "Train/LR": optimizer.param_groups[0]["lr"]}, step=epoch)

    return {'train_loss': losses.avg, 'LR': optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def test(model, loader, criterion, epoch, args, label='Test'):
    losses = AverageMeter()
    metrics = FscoreMeter()
    evaluator = DetectionIoUEvaluator()

    if is_main_process() and args.wandb_entity:
        input_sample = visualize_samples(model, loader, 4, args)

        wandb.log({f'{label}/input': wandb.Image(input_sample)}, step=epoch)

    model.eval()
    loader.dataset.label_model.eval()
    progress = tqdm(loader, ncols=180) if is_main_process() else None
    for step, batch in enumerate(loader):
        image_tensor, weight_characters, weight_affinities, gt_bboxes, words, paths = batch

        outputs, _ = model(image_tensor.to(args.device))
        loss = criterion(outputs, weight_characters.to(args.device), weight_affinities.to(args.device))
        losses.update(loss.item())

        batch_results = []

        postprocess = model.postprocess if not args.distributed else model.module.postprocess
        pred_bboxes, _ = postprocess(outputs,
                                     char_threshold=args.char_threshold,
                                     link_threshold=args.link_threshold,
                                     word_threshold=args.word_threshold)

        for bidx, (pred, gt) in enumerate(zip(pred_bboxes, gt_bboxes)):
            result = evaluator.evaluate_image(
                [dict(points=point, text=word, ignore=word == '###') for point, word in zip(gt, words[bidx])],
                [dict(points=point, text='###', ignore=False) for point in pred])
            batch_results.append(result)

        if args.distributed:
            batch_results = gather_list(batch_results, args.rank, args.world_size, dst=0)

        if is_main_process():
            batch_metric = EasyDict(evaluator.combine_results(batch_results))

            metrics.update(batch_metric.detMatched,
                           batch_metric.detCare - batch_metric.detMatched,
                           batch_metric.gtCare - batch_metric.detMatched,
                           len(batch_results)
                           )
            progress.update()
            progress.set_description(f'[{label} {epoch:>3}] '
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                                     f'F1: {batch_metric.hmean:.4f}({metrics.f1:.4f}) '
                                     f'Prec: {batch_metric.precision:.4f}({metrics.prec:.4f}) '
                                     f'Rec: {batch_metric.recall:.4f}({metrics.rec:.4f})')

    if is_main_process():
        progress.close()
        if args.wandb_entity:
            wandb.log({f"{label}/loss": losses.avg,
                       f"{label}/fscore": metrics.f1,
                       f'{label}/precision': metrics.prec,
                       f'{label}/recall': metrics.rec
                       }, step=epoch)

    return {'loss': losses.avg, 'fscore': metrics.f1, 'precision': metrics.prec, 'recall': metrics.rec}


@torch.no_grad()
def visualize_samples(model, loader, num_samples, args):
    model.eval()
    loader.dataset.label_model.eval()
    evaluator = DetectionIoUEvaluator()

    samples = []
    while len(samples) < num_samples:
        batch = next(iter(loader), None)
        if batch is None:
            break
        samples.extend(zip(*batch))
    samples = samples[:num_samples]

    canvas = []
    for image_tensor, weight_character, weight_affinity, gt_bboxes, words, paths in samples:
        outputs, _ = model(image_tensor.to(args.device).unsqueeze(0))

        postprocess = model.postprocess if not args.distributed else model.module.postprocess
        pred_bboxes, _ = postprocess(outputs,
                                     char_threshold=args.char_threshold,
                                     link_threshold=args.link_threshold,
                                     word_threshold=args.word_threshold)

        result = EasyDict(evaluator.evaluate_image(
            [dict(points=point, text=word, ignore=word == '###') for point, word in zip(gt_bboxes, words)],
            [dict(points=point, text='###', ignore=False) for point in pred_bboxes[0]]))

        text = f'H-mean: {result.hmean:.2f}, '
        text += f'Precision: {result.precision:.2f}, '
        text += f'Recall: {result.recall:.2f}, '
        text += f'TP: {result.detMatched}, '
        text += f'FP: {result.detCare - result.detMatched}, '
        text += f'FN: {result.gtCare - result.detMatched}'

        vis = tensor_to_image(image_tensor).copy()
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        vis = cv2.polylines(vis, gt_bboxes, True, (0, 255, 0), 2)
        vis = cv2.polylines(vis, [p.astype(np.int32) for p in pred_bboxes[0]], True, (0, 0, 255), 2)

        vis = hconcat_images([vis, weight_character, weight_affinity], height=480)
        vis = draw_text(vis, text)
        canvas.append(vis)

    canvas = vconcat_images(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def main_worker(gpu, args):
    args.local_rank = gpu
    init_distributed_mode(args)
    setup_for_distributed(is_main_process())
    set_seed(args.seed)
    print(args)

    # Model
    model = CRAFT().to(args.device)
    label_model = CRAFT().to(args.device).eval()
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        if args.state_dict:
            checkpoint = checkpoint[args.state_dict]
        model.load_state_dict(remove_module_prefix(checkpoint))
        label_model.load_state_dict(model.state_dict())
        print(f'>> Load pretrained {args.ckpt}')

    if args.distributed:
        model_without_ddp = model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # criterion
    criterion = craft_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # datasets

    train_dataset = craft_dataset(os.path.join(f'{args.train_data_root}', 'train_images'),
                                  os.path.join(f'{args.train_data_root}', 'train_labels'),
                                  label_model,
                                  label_format='polygon',
                                  transform=craft_dataset.train_transform(args.image_size)
                                  )

    test_dataset = craft_dataset(os.path.join(f'{args.test_data_root}', 'test_images'),
                                 os.path.join(f'{args.test_data_root}', 'test_labels'),
                                 label_model,
                                 label_format='polygon',
                                 transform=craft_dataset.default_transform(args.image_size_test, label=True))

    test_dataset2 = craft_dataset(f'/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01/test_images',
                                  f'/mldisk2/ms/datasets/OCR/AOS_OCR/parts_invoice/sample01/test_labels',
                                  label_model,
                                  label_format='polygon',
                                  transform=craft_dataset.default_transform(args.image_size_test, label=True))

    train_sampler, test_sampler, test_sampler2 = None, None, None
    if args.distributed:
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

    if is_main_process() and args.wandb_entity:
        wandb.init(project=args.project, entity=args.wandb_entity, name=args.exp)
        wandb.config.update(args)

    if args.init_eval:
        test(model, test_loader2, criterion, 0, args, label='AOS')
        test(model, test_loader, criterion, 0, args)

    for epoch in range(1, args.epoch):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, scaler, args)
        test(model, test_loader2, criterion, epoch, args, label='AOS')
        test(model, test_loader, criterion, epoch, args)

        ckpt = model.module.state_dict() if args.distributed else model.state_dict()
        label_model.load_state_dict(ckpt)

        if is_main_process():
            ckpt_dir = Path(f'checkpoints/{args.exp}')
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ckpt, ckpt_dir.joinpath(f'checkpoints-{epoch:02d}.pth').as_posix())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--image_size', default=1280, type=int, help='image size')
    parser.add_argument('--image_size_test', default=1280, type=int, help='image size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--train_data_root',
                        default='/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_2',
                        type=str,
                        help='data path')
    parser.add_argument('--test_data_root',
                        default='/mldisk2/ms/datasets/OCR/055.금융업 특화 문서 OCR 데이터/toy_2',
                        type=str,
                        help='data path')

    parser.add_argument('--ckpt', default='pretrained/CRAFT_clr_amp_29500.pth', type=str, help='pretrained model')
    parser.add_argument('--state_dict', default='craft', type=str, help='')
    # parser.add_argument('--ckpt', default=None, type=str, help='pretrained model')
    # parser.add_argument('--state_dict', default='craft', type=str, help='')

    parser.add_argument('--batch', default=4, type=int, help='batch size for train')
    parser.add_argument('--batch_test', default=8, type=int, help='batch size for train')
    parser.add_argument('--worker', default=0, type=int, help='number of workers for train')

    parser.add_argument('--init-eval', action='store_true', help='init eval')

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
    parser.add_argument('--wandb-entity', type=str, required=False, )
    parser.add_argument('--project', default='aos-textdet-doc-craft', type=str, help='project name')
    parser.add_argument('--exp', default=None, type=str, help='experiment name')

    args = parser.parse_args()

    if args.exp is None:
        args.exp = datetime.now().strftime("%Y%m%d-%H%M%S")

    n_gpu = torch.cuda.device_count()
    if n_gpu < args.proc_per_node:
        args.proc_per_node = n_gpu

    args.world_size = args.node * args.proc_per_node
    args.distributed = args.world_size > 1

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.proc_per_node, args=(args,))
    else:
        main_worker(0, args)
