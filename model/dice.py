from functools import partial

import torch.nn as nn

# from catalyst.utils import criterion as metrics

import torch

from catalyst.utils import get_activation_fn


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
    alpha:float = 1.0
):
    """
    Computes the dice metric

    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs, dim=[0, 2, 3])
    union = torch.sum(targets, dim=[0, 2, 3]) + torch.sum(outputs, dim=[0, 2, 3])
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice = 2 * (intersection + eps * (union == 0)) / (union + eps)

    dice = dice ** alpha

    return dice.mean()

class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        alpha:float = 1.0
    ):
        super().__init__()

        self.loss_fn = partial(
            dice, eps=eps, threshold=threshold, activation=activation, alpha=alpha
        )

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice
