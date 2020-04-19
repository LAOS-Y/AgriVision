import argparse
import os
import time
from tqdm import tqdm
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import segmentation_models_pytorch as smp
from model.dinknet import DinkNet34, DinkNet50, DinkNet101

from config import cfg
from utils import *
from dataset import AgriTrainDataset, AgriValDataset
from model.deeplab import DeepLab
from model.loss import ComposedLossWithLogits

torch.manual_seed(42)
np.random.seed(42)
amp.register_float_function(torch, 'sigmoid')

INF_FP16 = 2 ** 15

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
                           args.cfg.split('/')[-1].rstrip('.yaml') +
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
                        num_low_level_feat=cfg.MODEL.num_low_level_feat,
                        interpolate_before_lastconv=cfg.MODEL.interpolate_before_lastconv)
    elif cfg.MODEL.arch == 'smp-deeplab':
        model = smp.DeepLabV3(encoder_name='resnet101', classes=7)
    elif cfg.MODEL.arch == 'FPN':
        model = smp.FPN(encoder_name='resnet101',classes=7)
    elif cfg.MODEL.arch == 'Unet':
        model = smp.Unet(encoder_name='resnet101',classes=7)
    elif cfg.MODEL.arch == 'Dinknet':
        if cfg.MODEL.backbone == 'resnet34':
            assert cfg.MODEL.ibn_mode == 'none'
            model = DinkNet34(num_classes=7)
            # weight = torch.load('pretrained/dinknet34.pth')
            # weight['finalconv3.weight'] = torch.Tensor(model.finalconv3.weight)
            # weight['finalconv3.bias'] = torch.Tensor(model.finalconv3.bias)
            # # weight.pop('finalconv3.weight')
            # # weight.pop('finalconv3.bias')
            # model.load_state_dict(weight)
        elif cfg.MODEL.backbone == 'resnet50':
            model = DinkNet50(num_classes=7, ibn_mode=cfg.MODEL.ibn_mode)
        elif cfg.MODEL.backbone == 'resnet101':
            model = DinkNet101(num_classes=7, ibn_mode=cfg.MODEL.ibn_mode)

    if cfg.DATASET.train_channels in ['rgbn', 'rgbr']:
        convert_model(model, 4)

    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()

    loss_fn = ComposedLossWithLogits(dict(cfg.LOSS)).cuda()

    assert cfg.TRAIN.optim in ['SGD', 'Adam']

    if cfg.TRAIN.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.TRAIN.lr,
                                    weight_decay=cfg.TRAIN.weight_decay,
                                    momentum=cfg.TRAIN.beta1)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.TRAIN.lr)

    if cfg.MODEL.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if cfg.TRAIN.resume_checkpoint != "":
        if args.local_rank == 0:
            logger.info("Loading weight from {}".format(
                cfg.TRAIN.resume_checkpoint))

        weight = torch.load(cfg.TRAIN.resume_checkpoint,
                            map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(weight)

    dataset_train = AgriTrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        channels=cfg.DATASET.train_channels)

    dataset_mixup = None

    if cfg.TRAIN.mixup_alpha > 0:
        dataset_mixup = AgriTrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            channels=cfg.DATASET.train_channels,
            reverse=True)

    dataset_vals = []

    for channels in cfg.DATASET.val_channels:
        dataset_vals.append(AgriValDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET,
            channels=channels))

    # train_sampler, val_sampler = None, None

    # if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=args.world_size,
        rank=args.local_rank
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.batch_size_per_gpu,
        shuffle=False,  # we do not use this param
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler
    )

    loader_mixup = None

    if cfg.TRAIN.mixup_alpha > 0:
        mixup_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_mixup,
            num_replicas=args.world_size,
            rank=args.local_rank
        )

        loader_mixup = torch.utils.data.DataLoader(
            dataset_mixup,
            batch_size=cfg.TRAIN.batch_size_per_gpu,
            shuffle=False,  # we do not use this param
            num_workers=cfg.TRAIN.workers,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler
        )

    loader_vals = []

    for dataset_val in dataset_vals:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            num_replicas=args.world_size,
            rank=args.local_rank
        )

        loader_vals.append(torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.VAL.batch_size_per_gpu,
            shuffle=False,  # we do not use this param
            num_workers=cfg.VAL.batch_size_per_gpu,
            drop_last=True,
            pin_memory=True,
            sampler=val_sampler
        ))

    cfg.TRAIN.epoch_iters = len(loader_train)
    cfg.VAL.epoch_iters = len(loader_vals[0])

    cfg.TRAIN.running_lr = cfg.TRAIN.lr
    # if cfg.TRAIN.lr_pow > 0:

    cfg.TRAIN.num_epoch = (cfg.TRAIN.iter_warmup + cfg.TRAIN.iter_static + cfg.TRAIN.iter_decay) \
                          // cfg.TRAIN.epoch_iters

    cfg.TRAIN.log_fmt = 'TRAIN >> Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, ' \
                        'lr: {:.6f}, Loss: {:.6f}'

    cfg.VAL.log_fmt = 'Mean IoU: {:.4f}\nMean Loss: {:.6f}'

    for name in loss_fn.names:
        cfg.TRAIN.log_fmt += ', {}_Loss: '.format(name) + '{:.6f}'
        cfg.VAL.log_fmt += '\nMean {} Loss: '.format(name) + '{:.6f}'

    # print(cfg.TRAIN.log_fmt)
    # print(cfg.VAL.log_fmt)

    logger.info("World Size: {}".format(args.world_size))
    logger.info("TRAIN.epoch_iters: {}".format(cfg.TRAIN.epoch_iters))
    logger.info("TRAIN.sum_bs: {}".format(cfg.TRAIN.batch_size_per_gpu *
                                          args.world_size))

    logger.info("VAL.epoch_iters: {}".format(cfg.VAL.epoch_iters))
    logger.info("VAL.sum_bs: {}".format(cfg.VAL.batch_size_per_gpu *
                                        args.world_size))

    logger.info("TRAIN.num_epoch: {}".format(cfg.TRAIN.num_epoch))

    history = init_history(cfg)

    for i in range(cfg.TRAIN.start_epoch,
                   cfg.TRAIN.start_epoch + cfg.TRAIN.num_epoch):
        # print(i, args.local_rank)
        train(i + 1, loader_train, loader_mixup, model, loss_fn, optimizer,
              history, args, logger)

        for loader_val in loader_vals:
            val(i + 1, loader_val, model, loss_fn,
                history, args, logger)

        if args.local_rank == 0:    
            checkpoint(model, history, cfg, i + 1, args, logger)


def init_history(cfg):
    from copy import deepcopy

    losses_dict = {name: [] for name, _ in dict(cfg.LOSS).items() if _ != 0}
    val_dict = {'epoch': [], 'sum_loss': [], 'losses': losses_dict, 'mean_iou': []}

    history = {'train': {'epoch': [], 'sum_loss': [], 'losses': losses_dict, 'lr': []},
               'val': {}}

    for channels in cfg.DATASET.val_channels:
        history['val'][channels] = deepcopy(val_dict)

    return history


def adjust_learning_rate(optimizer, cur_iter, cfg):
    if cur_iter < cfg.TRAIN.iter_warmup:
        scale_running_lr = (cur_iter + 1) / cfg.TRAIN.iter_warmup
    elif cur_iter < (cfg.TRAIN.iter_warmup + cfg.TRAIN.iter_static):
        scale_running_lr = 1
    else:
        cur_iter -= cfg.TRAIN.iter_warmup + cfg.TRAIN.iter_static
        scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.iter_decay) ** cfg.TRAIN.lr_pow)

    cfg.TRAIN.running_lr = cfg.TRAIN.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr


def train(epoch, loader_train, loader_mixup, model, loss_fn, optimizer, history, args, logger):
    iter_time = AverageMeter()
    data_time = AverageMeter()
    # ave_total_loss = AverageMeter()
    # ave_acc = AverageMeter()

    model.train(not cfg.TRAIN.fix_bn)

    if args.distributed:
        loader_train.sampler.set_epoch(epoch - 1)

    if cfg.TRAIN.mixup_alpha > 0:
        loader_mixup = iter(loader_mixup)

    # main loop
    tic = time.time()

    for i, (img, mask, label, _) in enumerate(loader_train):
        img = img.cuda()
        mask = mask.cuda()
        label = label.cuda()

        label *= mask.unsqueeze(1)

        optimizer.zero_grad()

        if cfg.TRAIN.shuffle_R_and_N > 0 and loader_train.dataset.channels == 'rgbn':
            if torch.rand(1) < cfg.TRAIN.shuffle_R_and_N:
                r = img[:, :1, :, :]
                gb = img[:, 1:3, :, :]
                n = img[:, 3:, :, :]

                img = torch.cat([n, gb, r], dim=1)

        # pred = nn.functional.interpolate(
        #     pred,
        #     scale_factor=1 / cfg.MODEL.pred_downsampling_rate,
        #     mode='bilinear')

        # if cfg.TRAIN.lr_pow > 0:
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, cfg)

        if cfg.TRAIN.mixup_alpha > 0:
            img_, mask_b, label_b, _ = next(loader_mixup)
            img_ = img_.cuda()
            mask_b = mask_b.cuda()
            label_b = label_b.cuda()

            label_b *= mask_b.unsqueeze(1)

            mixed_img, lam = mixupImg(img, img_, alpha=cfg.TRAIN.mixup_alpha)

            data_time.update(time.time() - tic)

            pred = model(mixed_img)

            pred_a = pred.clone()
            pred_a[~mask.unsqueeze(1).expand_as(pred_a).bool()] = -INF_FP16
            # label_a *= mask_a.unsqueeze(1)
            sum_loss_a, losses_a = loss_fn(pred_a, label)

            pred_b = pred.clone()
            pred_b[~mask_b.unsqueeze(1).expand_as(pred_b).bool()] = -INF_FP16
            # label_b *= mask_b.unsqueeze(1)
            sum_loss_b, losses_b = loss_fn(pred_b, label_b)

            sum_loss = lam * sum_loss_a + (1 - lam) * sum_loss_b
            losses = lam * losses_a + (1 - lam) * losses_b
        else:
            # print('No MIXUP')

            data_time.update(time.time() - tic)

            pred = model(img)
            # print(type(pred))
            pred[~mask.unsqueeze(1).expand_as(pred).bool()] = -INF_FP16
            # import ipdb; ipdb.set_trace()
            sum_loss, losses = loss_fn(pred, label)

        if cfg.MODEL.fp16:
            with amp.scale_loss(sum_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            sum_loss.backward()
        optimizer.step()

        if args.distributed:
            reduced_sum_loss = reduce_tensor(sum_loss.data, args.world_size).item()
            reduced_losses = reduce_tensor(losses.data, args.world_size).tolist()
        else:
            reduced_sum_loss = sum_loss.item()
            reduced_losses = losses.tolist()

        iter_time.update(time.time() - tic)
        tic = time.time()

        # calculate accuracy, and display
        if args.local_rank == 0 and i % cfg.TRAIN.disp_iter == 0:
            logger.info(cfg.TRAIN.log_fmt
                        .format(epoch, i, cfg.TRAIN.epoch_iters,
                                iter_time.average(), data_time.average(),
                                # cfg.TRAIN.lr,
                                cfg.TRAIN.running_lr,
                                reduced_sum_loss, *reduced_losses))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters

            history['train']['epoch'].append(fractional_epoch)
            history['train']['sum_loss'].append(reduced_sum_loss)
            history['train']['lr'].append(cfg.TRAIN.running_lr)

            for i, name in enumerate(loss_fn.names):
                history['train']['losses'][name].append(reduced_losses[i])


def val(epoch, loader_val, model, loss_fn, history, args, logger):
    avg_sum_loss = AverageMeter()
    avg_losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.eval()

    # main loop
    tic = time.time()

    channels = loader_val.dataset.channels

    if args.distributed:
        loader_val.sampler.set_epoch(epoch - 1)

    if args.local_rank == 0:
        loader_val = tqdm(loader_val, total=cfg.VAL.epoch_iters)

    with torch.no_grad():
        for img, mask, label, _ in loader_val:
            img = img.cuda()
            mask = mask.cuda()
            label = label.cuda()

            label *= mask.unsqueeze(1)

            pred = model(img)
            # pred = nn.functional.interpolate(
            #     pred,
            #     scale_factor=1 / cfg.MODEL.pred_downsampling_rate,
            #     mode='bilinear')
            # pred *= mask.unsqueeze(1)
            pred[~mask.unsqueeze(1).expand_as(pred).bool()] = -INF_FP16

            sum_loss, losses = loss_fn(pred, label)

            if args.distributed:
                reduced_sum_loss = reduce_tensor(sum_loss.data, args.world_size).item()
                reduced_losses = reduce_tensor(losses.data, args.world_size)
            else:
                reduced_sum_loss = sum_loss.item()
                reduced_losses = losses.data

            avg_sum_loss.update(reduced_sum_loss)
            avg_losses.update(reduced_losses)

            intersection, union = intersectionAndUnion(pred.data, label.data, 0)
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
        losses = avg_losses.average().tolist()

        for i, _iou in enumerate(iou):
            logger.info('class [{}], IoU: {:.4f}'.format(i, _iou))

        logger.info('[Eval Summary][Channels: {}]:'.format(channels))
        logger.info(cfg.VAL.log_fmt.format(
            iou.mean(), avg_sum_loss.average(), *losses))

        history['val'][channels]['epoch'].append(epoch)
        history['val'][channels]['sum_loss'].append(avg_sum_loss.average())
        history['val'][channels]['mean_iou'].append(iou.mean().item())

        for i, name in enumerate(loss_fn.names):
            history['val'][channels]['losses'][name].append(losses[i])


def checkpoint(model, history, cfg, epoch, args, logger):
    logger.info("Saving checkpoints to '{}'".format(cfg.DIR))

    dict_model = model.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(os.path.join(cfg.DIR, 'history'), epoch))
    torch.save(
        dict_model,
        '{}/weight_epoch_{}.pth'.format(os.path.join(cfg.DIR, 'weight'), epoch))

    plt.plot(history['train']['epoch'], history['train']['sum_loss'], color='r', label='TRN LOSS')
    # plt.plot(history['val']['epoch'], history['val']['sum_loss'], color='g', label='VAL LOSS')
    plt.title('sum_loss')
    plt.legend()
    plt.savefig('{}/sum_loss.png'.format(cfg.DIR), dpi=200)
    plt.close('all')

    for name in history['train']['losses'].keys():
        plt.plot(history['train']['epoch'], history['train']['losses'][name], color='r', label='TRN LOSS')
        # plt.plot(history['val']['epoch'], history['val']['losses'][name], color='g', label='VAL LOSS')
        plt.title(name)
        plt.legend()
        plt.savefig('{}/{}.png'.format(cfg.DIR, name), dpi=200)
        plt.close('all')

    for channels in history['val'].keys():
        plt.plot(history['val'][channels]['epoch'], history['val'][channels]['mean_iou'], color='g', label='VAL MEAN IOU')
        plt.title('iou')
        plt.legend()
        plt.savefig('{}/iou_{}.png'.format(cfg.DIR, channels), dpi=200)
        plt.close('all')

    plt.plot(history['train']['epoch'], history['train']['lr'], color='b', label='Learning Rate')
    plt.title('lr')
    plt.legend()
    plt.savefig('{}/lr.png'.format(cfg.DIR), dpi=200)
    plt.close('all')


if __name__ == "__main__":
    main()
