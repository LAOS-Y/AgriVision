import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    file_hdlr = logging.FileHandler(filename)
    file_hdlr.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_hdlr)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


# def intersectionAndUnion_old(imPred, imLab, numClass):
    # imPred = np.asarray(imPred).copy()
    # imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    # imPred = imPred * (imLab > 0)

    # # Compute area intersection:
    # intersection = imPred * (imPred == imLab)
    # (area_intersection, _) = np.histogram(
    #     intersection, bins=numClass, range=(1, numClass))

    # # Compute area union:
    # (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    # (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    # area_union = area_pred + area_lab - area_intersection

    # return (area_intersection, area_union)

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def intersectionAndUnion(pr, gt, threshold=None):
    """Calculate Intersection and Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)

    intersection = torch.sum(gt * pr, dim=[0, 2, 3])
    union = torch.sum(gt, dim=[0, 2, 3]) + torch.sum(pr, dim=[0, 2, 3]) - intersection

    return intersection, union


def convert_model(model, in_channels):
    while not isinstance(model, nn.Conv2d):
        model = next(model.children())

    model.in_channels = in_channels

    model.weight = nn.Parameter(torch.cat((model.weight,
                                           model.weight[:, 0: 1, :, :]),
                                dim=1))


def pixel_acc(pred, label, valid_mask):
    # import ipdb;
    # ipdb.set_trace()
    with torch.no_grad():
        pred = pred.detach()

        pred = pred.permute(0, 2, 3, 1)  # (*, W, H, C)
        label = label.permute(0, 2, 3, 1)  # (*, W, H, C)
        valid_mask = valid_mask.squeeze(1)  # (*, W, H)

        true_mask = label.gather(
            -1,
            pred.argmax(dim=3, keepdim=True)
        ).squeeze(-1) > 0

        true_mask = true_mask.float()
        true_mask = valid_mask * true_mask

        acc_sum = true_mask.sum(dim=(1, 2))
        pixel_sum = valid_mask.sum(dim=(1, 2))

        acc = acc_sum / pixel_sum

    return acc


def reduce_tensor(tensor, world_size):
    if isinstance(tensor, int) or isinstance(tensor, float):
        tensor = torch.Tensor([tensor]).cuda()
    elif not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(tensor).cuda()

    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def mixupData(img, label, mask, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = img.size()[0]
    index = torch.randperm(batch_size).to(img.device)

    mixed_img = lam * img + (1 - lam) * img[index, :]
    label_a, label_b = label, label[index]
    mask_a, mask_b = mask, mask[index]

    return mixed_img, label_a, label_b, mask_a, mask_b, lam


def mixupImg(img_a, img_b, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_img = lam * img_a + (1 - lam) * img_b

    return mixed_img, lam


colors = torch.Tensor([[120, 120, 120],
                       [180, 120, 120],
                       [  6, 230, 230],
                       [ 80,  50,  50],
                       [  4, 200,   3],
                       [120, 120,  80],
                       [140, 140, 140],
                       [  0,   0,   0]])

# colors = torch.Tensor([[  0,   0,   0],
#                        [120, 150, 120],
#                        [180, 120, 120],
#                        [  6, 230, 230],
#                        [ 80,  50,  50],
#                        [  4, 200,   3],
#                        [120, 120,  80],
#                        [  0,   0,   0]])

def colorEncode(classes: torch.Tensor):
    # classes.shape (1, W, H)

    colored_img = torch.zeros(3, classes.shape[1], classes.shape[2])

    for c in classes.unique():
        color = colors[c].unsqueeze(1).unsqueeze(2)
        colored_img += color.expand(3, classes.shape[1], classes.shape[2]) * (classes == c)

    return colored_img
