# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class BoundaryLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, bd_pre: Tensor, bd_gt: Tensor, weight=None, ignore_index=None, **kwargs) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): Predictions of the boundary head.
            bd_gt (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """

        # ✅ 忽略 ignore_index，避免错误
        if ignore_index is not None:
            mask = bd_gt != ignore_index
            bd_pre = bd_pre * mask
            bd_gt = bd_gt * mask

        #print(f"bd_gt.shape: {bd_gt.shape}")
        #print(f"bd_pre: {bd_pre.shape}")

        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = bd_gt.view(1, -1).float()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        loss_weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        loss_weight[pos_index] = neg_num * 1.0 / sum_num
        loss_weight[neg_index] = pos_num * 1.0 / sum_num

        if weight is not None:
            loss_weight = loss_weight * weight.view(1, -1)

        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, loss_weight, reduction='mean')

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_
