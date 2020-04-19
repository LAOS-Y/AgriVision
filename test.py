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
import ttach as tta

from config import cfg
from utils import *
from dataset import AgriTestDataset
from model.deeplab import DeepLab


def save_result(info, pred):
    classes = pred.argmax(dim=1, keepdim=True).cpu()

    for i in range(classes.shape[0]):
        result_png = ToPILImage()(classes[i].float() / 255.)
        img_name = info[i]

        # print(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

        result_png.save(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


def test(loader_test, model, args, logger):
    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")
    model.eval()

    if args.local_rank == 0:
        loader_test = tqdm(loader_test, total=cfg.TEST.epoch_iters)

    with torch.no_grad():
        for img, mask, info in loader_test:
            img = img.cuda()
            mask = mask.cuda()

            # import ipdb; ipdb.set_trace()

            pred = model(img)

            save_result(info, pred)


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
                        freeze_bn=False,
                        num_low_level_feat=cfg.MODEL.num_low_level_feat)
    elif cfg.MODEL.arch == 'smp-deeplab':
        model = smp.DeepLabV3(encoder_name='resnet101', classes=7)
    elif cfg.MODEL.arch == 'FPN':
        model = smp.FPN(encoder_name='resnet101',classes=7)
    elif cfg.MODEL.arch == 'Unet':
        model = smp.Unet(encoder_name='resnet101',classes=7)

    convert_model(model, 4)
    from pytorch_model_summary import summary
    print(summary(model, torch.zeros((1, 4, 512, 512)), show_input=True))
    return
    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    model = amp.initialize(model, opt_level="O1")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if cfg.TEST.checkpoint != "":
        if args.local_rank == 0:
            logger.info("Loading weight from {}".format(
                cfg.TEST.checkpoint))

        weight = torch.load(cfg.TEST.checkpoint,
                            map_location=lambda storage, loc: storage.cuda(args.local_rank))

        if not args.distributed:
            weight = {k[7:]: v for k, v in weight.items()}

        model.load_state_dict(weight)

    dataset_test = AgriTestDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_test,
        cfg.DATASET)

    test_sampler = None

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=args.world_size,
            rank=args.local_rank
        )

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size_per_gpu,
        shuffle=False,  # we do not use this param
        drop_last=False,
        pin_memory=True,
        sampler=test_sampler
    )

    cfg.TEST.epoch_iters = len(loader_test)

    logger.info("World Size: {}".format(args.world_size))
    logger.info("TEST.epoch_iters: {}".format(cfg.TEST.epoch_iters))
    logger.info("TEST.sum_bs: {}".format(cfg.TEST.batch_size_per_gpu *
                                         args.world_size))

    test(loader_test, model, args, logger)

if __name__ == '__main__':
    main()
