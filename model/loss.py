import torch
import torch.nn as nn
from catalyst.contrib.nn import \
    DiceLoss, IoULoss, LovaszLossMultiLabel, FocalLossBinary

str2Loss = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "iou": IoULoss,
    "lovasz": LovaszLossMultiLabel,
    "focal" : FocalLossBinary
}


class ComposedLossWithLogits(nn.Module):

    def __init__(self, names_and_weights):
        super().__init__()

        assert type(names_and_weights) in (dict, list)

        if isinstance(names_and_weights, dict):
            names_and_weights = names_and_weights.items()

        self.names = []
        self.loss_fns = []
        weights = []

        for name, weight in names_and_weights:
            if weight == 0:
                continue
            self.names.append(name)
            self.loss_fns.append(str2Loss[name]())
            weights.append(weight)

        self.loss_fns = nn.ModuleList(self.loss_fns)

        self.register_buffer('weights', torch.Tensor(weights))
        self.weights /= self.weights.sum()

    def forward(self, logit, target):
        losses = torch.stack([loss_fn(logit, target)
                              for loss_fn in self.loss_fns])

        sumed_loss = (losses * self.weights).sum()

        return sumed_loss, losses
