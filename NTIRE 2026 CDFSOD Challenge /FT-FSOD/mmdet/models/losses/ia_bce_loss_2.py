# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weight_reduce_loss

def py_ia_bce_loss(pred,
                   target,
                   weight=None,
                   alpha=0.25,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    """PyTorch version of Instance-Aware Binary Cross Entropy Loss.
    
    Unlike Varifocal Loss which uses IoU directly as positive weight,
    IA-BCE Loss combines prediction confidence and IoU for more balanced
    instance-aware weighting.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        alpha (float, optional): Balance factor for combining prediction 
            confidence and IoU in positive samples. Defaults to 0.25.
        gamma (float, optional): The gamma for calculating the modulating
            factor for negative samples. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to use instance-aware weighting
            that combines prediction confidence and IoU. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    
    if iou_weighted:
        # 计算正样本权重
        # 根据target和pred_sigmoid的大小关系选择使用的值
        # 当target > pred_sigmoid时使用pred_sigmoid，否则使用target
        
        # selected_val = torch.where(target > pred_sigmoid, pred_sigmoid, target)

        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        target_01 = torch.where(target > 0, 1, 0).type_as(pred)
        pt = (1 - pred_sigmoid) * target_01 + pred_sigmoid * (1 - target_01)

        pos_weights = alpha * target_01
        neg_weights = (1 - alpha) * (1 - target_01)

        focal_weight = (pos_weights + neg_weights) * pt.pow(gamma)
        
        loss = ce_loss * focal_weight
        
    else:
        # Fallback to simpler focal-like weighting without instance awareness
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    
    
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class IABCELoss2(nn.Module):
    """Instance-Aware Binary Cross Entropy Loss.
    
    This loss function applies instance-aware weighting that combines prediction
    confidence and IoU for positive samples, and uses focal mechanism for negative
    samples. Unlike Varifocal Loss that directly uses IoU as positive weight,
    IA-BCE Loss provides more balanced weighting through prob^alpha * iou^(1-alpha).
    """

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.25,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """Instance-Aware Binary Cross Entropy Loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): Balance factor for combining prediction
                confidence and IoU in positive samples. Defaults to 0.25.
            gamma (float, optional): The gamma for calculating the modulating
                factor for negative samples. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to use instance-aware weighting
                that combines prediction confidence and IoU. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(IABCELoss2, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid IA-BCE loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if self.use_sigmoid:
            if not self.activated:
                # Handle different target formats
                if pred.dim() != target.dim():
                    # Convert target to one-hot format if needed
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
            
            loss_cls = self.loss_weight * py_ia_bce_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        
        return loss_cls