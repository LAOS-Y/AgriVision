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

import segmentation_models_pytorch as smp

from config import cfg
from utils import *
from dataset import AgriValDataset
from model.deeplab import DeepLab

import model.backbone as B

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

    parser.add_argument(
        "--ibn",
        default='none',
        type=str
    )

    parser.add_argument(
        "--weight", "-w",
        type=str
    )

    parser.add_argument(
        "--channels", "-c",
        type=str,
    )

    parser.add_argument(
        "--out", "-o",
        type=str
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

    logger.info("{}".format(args))

    if cfg.MODEL.arch == 'deeplab':
        model = DeepLab(num_classes=cfg.DATASET.num_class,
                        backbone=cfg.MODEL.backbone,                  # resnet101
                        output_stride=cfg.MODEL.os,
                        ibn_mode=args.ibn,
                        freeze_bn=False)
    else:
        raise NotImplementedError

    if args.local_rank == 0:
        logger.info("Loading weight from {}".format(
            args.weight))

    weight = torch.load(args.weight,
                        map_location=lambda storage, loc: storage.cuda(args.local_rank))

    if not args.distributed:
        weight = {k[7:]: v for k, v in weight.items()}

    model.load_state_dict(weight)

    model = model.backbone
    B.resnet.TRACK_FEAT = True

    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    model = amp.initialize(model, opt_level="O1")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    dataset_val = AgriValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        ret_rgb_img=False,
        channels=args.channels)

    val_sampler = None

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
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler
    )

    cfg.VAL.epoch_iters = len(loader_val)

    cfg.VAL.log_fmt = 'Mean IoU: {:.4f}\n'

    logger.info("World Size: {}".format(args.world_size))

    logger.info("VAL.epoch_iters: {}".format(cfg.VAL.epoch_iters))
    logger.info("VAL.sum_bs: {}".format(cfg.VAL.batch_size_per_gpu *
                                        args.world_size))

    means, vars = val(loader_val, model, args, logger)

    # print(vars)

    torch.save({'means': means, 'vars': vars}, args.out)
def val(loader_val, model, args, logger):
    channels = [[] for i in range(34)]

    model.eval()

    # main loop
    tic = time.time()

    if args.local_rank == 0:
        loader_val = tqdm(loader_val, total=cfg.VAL.epoch_iters)

    with torch.no_grad():
        for img, mask, _, _, in loader_val:
            img = img.cuda()
            mask = mask.cuda()
            
            last, _ = model(img)

            lst = [feat.data.float() for feat in B.resnet.SHARED_LIST]

            # import ipdb; ipdb.set_trace()

            for i, feat in enumerate(lst):
                channels[i].append(feat.mean(dim=[0, 2, 3]))

        # import ipdb; ipdb.set_trace()

        for i in range(len(channels)):
            channels[i] = torch.stack(channels[i], dim=0)

        means = [feat.mean(dim=0).cpu() for feat in channels]
        vars = [feat.var(dim=0).cpu() for feat in channels]

    return means, vars


# def val(loader_val, model, args, logger):
#     channel_meters = [AverageMeter() for i in range(34)]
#     channel_square_meters = [AverageMeter() for i in range(34)]

#     model.eval()

#     # main loop
#     tic = time.time()

#     if args.local_rank == 0:
#         loader_val = tqdm(loader_val, total=cfg.VAL.epoch_iters)

#     all_feat = None

#     with torch.no_grad():
#         for img, mask, _, _, in loader_val:
#             img = img.cuda()
#             mask = mask.cuda()
            
#             last, _ = model(img)

#             lst = [feat.data.float() for feat in B.resnet.SHARED_LIST]

#             # import ipdb; ipdb.set_trace()

#             for i, feat in enumerate(lst):
#                 num = feat.flatten().shape[0] // feat.shape[1]
#                 channel_meters[i].update(feat.mean(dim=[0, 2, 3]), weight=num)
#                 channel_square_meters[i].update((feat ** 2).mean(dim=[0, 2, 3]), weight=num)

#             # if all_feat is None:
#             #     all_feat = [[feat] for feat in lst]
#             # else:
#             #     for i, feat in enumerate(lst):
#             #         all_feat[i].append(feat)

#         # import ipdb; ipdb.set_trace()

#         # for i in range(len(all_feat)):
#         #     all_feat[i] = torch.cat(all_feat[i], dim=0)

#         # means = [feat.mean(dim=[0,2,3]) for feat in all_feat]
#         # vars = [feat.var(dim=[0,2,3]) for feat in all_feat]

#         means = []
#         vars = []

#         for c_meter, c_square_meter in zip(channel_meters,
#                                            channel_square_meters):

#             means.append(c_meter.average().cpu())

#             count = c_meter.count
#             var = (c_square_meter.sum - (c_meter.sum ** 2) / count) / (count - 1)
#             var = var.cpu()
#             vars.append(var)

#     return means, vars


if __name__ == "__main__":
    main()
