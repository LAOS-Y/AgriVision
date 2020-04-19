import argparse
import os
import time
from tqdm import tqdm
import shutil
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torchvision.transforms import ToPILImage

import segmentation_models_pytorch as smp
from catalyst.contrib.nn import DiceLoss, IoULoss

from config import cfg
from utils import *
from dataset import AgriValDataset
from model.deeplab import DeepLab
from model.loss import ComposedLossWithLogits

from pydensecrf.utils import unary_from_softmax
import pydensecrf.densecrf as dcrf

torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )

    parser.add_argument(
        "--local_rank",
        default=0,
        type=int
    )

    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    args.distributed = False

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.world_size = 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    # print(args.world_size, args.local_rank, args.distributed)

    cfg.merge_from_file(args.cfg)

    cfg.DIR = os.path.join(cfg.DIR,
                           args.cfg.split('/')[-1].split('.')[0] +
                           datetime.now().strftime('-%Y-%m-%d-%a-%H:%M:%S:%f'))

    # Output directory
    # if not os.path.isdir(cfg.DIR):
    if args.local_rank == 0:
        os.makedirs(cfg.DIR, exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR, 'weight'), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR, 'history'), exist_ok=True)
        shutil.copy(args.cfg, cfg.DIR)

    if os.path.exists(os.path.join(cfg.DIR, 'log.txt')):
        os.remove(os.path.join(cfg.DIR, 'log.txt'))
    logger = setup_logger(distributed_rank=args.local_rank,
                          filename=os.path.join(cfg.DIR, 'log.txt'))
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))


    if cfg.MODEL.arch == 'deeplab':
        model = DeepLab(num_classes=cfg.DATASET.num_class,
                        backbone=cfg.MODEL.backbone,                  # resnet101
                        output_stride=cfg.MODEL.os,
                        ibn_mode=cfg.MODEL.ibn_mode,
                        freeze_bn=False)
    elif cfg.MODEL.arch == 'smp-deeplab':
        model = smp.DeepLabV3(encoder_name='resnet101', classes=7)
    elif cfg.MODEL.arch == 'FPN':
        model = smp.FPN(encoder_name='resnet101',classes=7)
    elif cfg.MODEL.arch == 'Unet':
        model = smp.Unet(encoder_name='resnet101',classes=7)

    if cfg.DATASET.val_channels[0] == 'rgbn':
        convert_model(model, 4)

    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    model = amp.initialize(model, opt_level="O1")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if cfg.VAL.checkpoint != "":
        if args.local_rank == 0:
            logger.info("Loading weight from {}".format(
                cfg.VAL.checkpoint))

        weight = torch.load(cfg.VAL.checkpoint,
                            map_location=lambda storage, loc: storage.cuda(args.local_rank))

        if not args.distributed:
            weight = {k[7:]: v for k, v in weight.items()}

        model.load_state_dict(weight)

    dataset_val = AgriValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        channels=cfg.DATASET.val_channels[0])

    val_sampler = None

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            num_replicas=args.world_size,
            rank=args.local_rank
        )

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size_per_gpu,
        shuffle=False,  # we do not use this param
        num_workers=cfg.VAL.batch_size_per_gpu,
        drop_last=True,
        pin_memory=True,
        sampler=val_sampler
    )

    cfg.VAL.epoch_iters = len(loader_val)

    cfg.VAL.log_fmt = 'Mean IoU: {:.4f}\n'

    logger.info("World Size: {}".format(args.world_size))

    logger.info("VAL.epoch_iters: {}".format(cfg.VAL.epoch_iters))
    logger.info("VAL.sum_bs: {}".format(cfg.VAL.batch_size_per_gpu *
                                        args.world_size))

    os.makedirs(cfg.VAL.visualized_pred, exist_ok=True)
    os.makedirs(cfg.VAL.visualized_label, exist_ok=True)

    val(loader_val, model, args, logger)

def val(loader_val, model, args, logger):
    avg_sum_loss = AverageMeter()
    avg_losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.eval()

    # main loop
    tic = time.time()

    if args.local_rank == 0:
        loader_val = tqdm(loader_val, total=cfg.VAL.epoch_iters)

    with torch.no_grad():
        for img, mask, label, info in loader_val:
            img = img.cuda()
            mask = mask.cuda()
            label = label.cuda()

            # label *= mask.unsqueeze(1)

            pred = model(img)

            save_result(info, pred, label, mask)

            intersection, union = intersectionAndUnion(pred.data, label.data, 0.5)
            intersection_meter.update(intersection)
            union_meter.update(union)

    # import ipdb; ipdb.set_trace()

    if args.distributed:
        reduced_inter = reduce_tensor(
            intersection_meter.sum,
            args.world_size).data.cpu()

        reduced_union = reduce_tensor(
            union_meter.sum,
            args.world_size).data.cpu()
    else:
        reduced_inter = intersection_meter.sum.cpu()
        reduced_union = union_meter.sum.cpu()

    iou = reduced_inter / (reduced_union + 1e-10)

    if args.local_rank == 0:
        for i, _iou in enumerate(iou):
            logger.info('class [{}], IoU: {:.4f}'.format(i, _iou))

        logger.info('[Eval Summary]:')
        logger.info(cfg.VAL.log_fmt.format(iou.mean()))

def save_result(info, pred, label, mask):
    # import ipdb; ipdb.set_trace()

    classes = pred.argmax(dim=1, keepdim=True)
    label = label.argmax(dim=1, keepdim=True)

    classes[~mask.unsqueeze(1).expand_as(classes).bool()] = 7
    label[~mask.unsqueeze(1).expand_as(label).bool()] = 7

    # classes = colorEncode(classes)
    # label = colorEncode(label)

    classes = classes.cpu()
    label = label.cpu()

    for i in range(classes.shape[0]):
        result_png = colorEncode(classes[i])
        label_png = colorEncode(label[i])

        result_png = ToPILImage()(result_png.float() / 255.)
        label_png = ToPILImage()(label_png.float() / 255.)
        img_name = info[i]

        # print(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

        result_png.save(os.path.join(cfg.VAL.visualized_pred, img_name.replace('.jpg', '.png')))
        label_png.save(os.path.join(cfg.VAL.visualized_label, img_name.replace('.jpg', '.png')))

if __name__ == "__main__":
    main()
