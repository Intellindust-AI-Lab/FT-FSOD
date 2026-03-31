# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet.registry import MODELS
from .focal_loss import py_sigmoid_focal_loss
from .utils import weight_reduce_loss


@MODELS.register_module()
class AdvancedFSODFocalLoss(nn.Module):
    """Advanced FSOD Focal Loss designed for AdvancedFSODAssigner.
    
    This loss function extends the standard Focal Loss to work with the
    four-class labeling system of AdvancedFSODAssigner:
    - Positive samples (label=1): Matched queries
    - Background samples (label=0): Background queries  
    - Unlabeled positive samples (label=-1): Potential unmatched positive samples
    - Uncertain samples (label=-2): Samples with high uncertainty
    
    Key innovations:
    1. Confidence-aware weighting: Adjusts loss weights based on prediction confidence
    2. Uncertainty-guided loss: Special handling for uncertain samples
    3. Progressive loss scheduling: Gradually increases focus on unlabeled positives
    4. Consistency regularization: Enforces consistency between similar samples
    5. Adaptive focal parameters: Dynamically adjusts gamma and alpha
    6. Pseudo-label refinement: Improves quality of unlabeled positive assignments
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 # Advanced FSOD specific parameters
                 unlabeled_weight=0.5,
                 uncertain_weight=0.3,
                 confidence_threshold=0.7,
                 uncertainty_threshold=0.8,
                 progressive_schedule=True,
                 consistency_weight=0.1,
                 adaptive_focal=True,
                 pseudo_label_refinement=True,
                 # Progressive scheduling parameters
                 warmup_epochs=5,
                 max_unlabeled_weight=1.0,
                 # Adaptive focal parameters
                 min_gamma=1.0,
                 max_gamma=3.0,
                 min_alpha=0.1,
                 max_alpha=0.5):
        """Initialize AdvancedFSODFocalLoss.
        
        Args:
            use_sigmoid (bool): Whether to use sigmoid activation. Defaults to True.
            gamma (float): Focal loss gamma parameter. Defaults to 2.0.
            alpha (float): Focal loss alpha parameter. Defaults to 0.25.
            reduction (str): Loss reduction method. Defaults to 'mean'.
            loss_weight (float): Overall loss weight. Defaults to 1.0.
            unlabeled_weight (float): Weight for unlabeled positive samples. Defaults to 0.5.
            uncertain_weight (float): Weight for uncertain samples. Defaults to 0.3.
            confidence_threshold (float): Threshold for confidence-aware weighting. Defaults to 0.7.
            uncertainty_threshold (float): Threshold for uncertainty handling. Defaults to 0.8.
            progressive_schedule (bool): Whether to use progressive scheduling. Defaults to True.
            consistency_weight (float): Weight for consistency regularization. Defaults to 0.1.
            adaptive_focal (bool): Whether to use adaptive focal parameters. Defaults to True.
            pseudo_label_refinement (bool): Whether to refine pseudo labels. Defaults to True.
            warmup_epochs (int): Number of warmup epochs for progressive scheduling. Defaults to 5.
            max_unlabeled_weight (float): Maximum weight for unlabeled samples. Defaults to 1.0.
            min_gamma (float): Minimum gamma for adaptive focal. Defaults to 1.0.
            max_gamma (float): Maximum gamma for adaptive focal. Defaults to 3.0.
            min_alpha (float): Minimum alpha for adaptive focal. Defaults to 0.1.
            max_alpha (float): Maximum alpha for adaptive focal. Defaults to 0.5.
        """
        super(AdvancedFSODFocalLoss, self).__init__()
        assert use_sigmoid, 'Only sigmoid focal loss supported for FSOD.'
        
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        # Advanced FSOD parameters
        self.unlabeled_weight = unlabeled_weight
        self.uncertain_weight = uncertain_weight
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.progressive_schedule = progressive_schedule
        self.consistency_weight = consistency_weight
        self.adaptive_focal = adaptive_focal
        self.pseudo_label_refinement = pseudo_label_refinement
        
        # Progressive scheduling
        self.warmup_epochs = warmup_epochs
        self.max_unlabeled_weight = max_unlabeled_weight
        
        # Adaptive focal parameters
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 100  # Will be updated during training
        
    def update_epoch(self, epoch, total_epochs=None):
        """Update current training epoch for progressive scheduling."""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
            
    def get_progressive_weights(self):
        """Calculate progressive weights based on current epoch."""
        if not self.progressive_schedule:
            return self.unlabeled_weight, self.uncertain_weight
            
        # Progressive scheduling: gradually increase unlabeled weight
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            unlabeled_w = self.unlabeled_weight * progress
            uncertain_w = self.uncertain_weight * progress
        else:
            # After warmup, gradually increase to max weight
            remaining_epochs = self.total_epochs - self.warmup_epochs
            if remaining_epochs > 0:
                progress = min(1.0, (self.current_epoch - self.warmup_epochs) / remaining_epochs)
                unlabeled_w = self.unlabeled_weight + (self.max_unlabeled_weight - self.unlabeled_weight) * progress
                uncertain_w = self.uncertain_weight * (1.0 + progress * 0.5)
            else:
                unlabeled_w = self.max_unlabeled_weight
                uncertain_w = self.uncertain_weight
                
        return unlabeled_w, uncertain_w
        
    def get_adaptive_focal_params(self, pred, target):
        """Calculate adaptive focal parameters based on prediction statistics."""
        if not self.adaptive_focal:
            return self.gamma, self.alpha
            
        with torch.no_grad():
            # Calculate prediction confidence statistics
            pred_sigmoid = pred.sigmoid()
            pos_mask = target == 1
            neg_mask = target == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_conf = pred_sigmoid[pos_mask].mean()
                neg_conf = (1 - pred_sigmoid[neg_mask]).mean()
                avg_conf = (pos_conf + neg_conf) / 2
                
                # Adjust gamma: higher gamma for easier samples (high confidence)
                gamma_factor = avg_conf.item()
                adaptive_gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * gamma_factor
                
                # Adjust alpha: balance based on positive/negative ratio
                pos_ratio = pos_mask.sum().float() / target.numel()
                adaptive_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (1 - pos_ratio)
            else:
                adaptive_gamma = self.gamma
                adaptive_alpha = self.alpha
                
        return adaptive_gamma, adaptive_alpha
        
    def refine_pseudo_labels(self, pred, target, confidence_scores=None):
        """Refine pseudo labels for unlabeled positive samples."""
        if not self.pseudo_label_refinement:
            return target
            
        refined_target = target.clone()
        unlabeled_mask = target == -1
        
        if unlabeled_mask.sum() > 0:
            pred_sigmoid = pred.sigmoid()
            
            # Use confidence scores if available, otherwise use prediction confidence
            if confidence_scores is not None:
                conf_scores = confidence_scores[unlabeled_mask]
            else:
                conf_scores = pred_sigmoid[unlabeled_mask].max(dim=1)[0]
                
            # Refine labels based on high confidence predictions
            high_conf_mask = conf_scores > self.confidence_threshold
            if high_conf_mask.sum() > 0:
                unlabeled_indices = torch.where(unlabeled_mask)[0]
                high_conf_indices = unlabeled_indices[high_conf_mask]
                
                # Convert high-confidence unlabeled samples to positive
                refined_target[high_conf_indices] = 1
                
        return refined_target
        
    def compute_consistency_loss(self, pred, target, similarity_matrix=None):
        """Compute consistency regularization loss."""
        if similarity_matrix is None:
            return torch.tensor(0.0, device=pred.device)
            
        # Find similar sample pairs
        similarity_threshold = 0.8
        similar_pairs = similarity_matrix > similarity_threshold
        
        if similar_pairs.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute prediction consistency for similar pairs
        pred_sigmoid = pred.sigmoid()
        consistency_loss = 0.0
        num_pairs = 0
        
        for i in range(similar_pairs.size(0)):
            for j in range(i + 1, similar_pairs.size(1)):
                if similar_pairs[i, j]:
                    # L2 distance between predictions
                    consistency_loss += F.mse_loss(pred_sigmoid[i], pred_sigmoid[j], reduction='mean')
                    num_pairs += 1
                    
        if num_pairs > 0:
            consistency_loss /= num_pairs
            
        return consistency_loss
        
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                confidence_scores=None,
                uncertainty_scores=None,
                similarity_matrix=None):
        """Forward function.
        
        Args:
            pred (torch.Tensor): Predictions with shape (N, C).
            target (torch.Tensor): Targets with shape (N,). Values:
                1: positive samples, 0: background, -1: unlabeled positive, -2: uncertain
            weight (torch.Tensor, optional): Sample weights.
            avg_factor (int, optional): Average factor for loss normalization.
            reduction_override (str, optional): Reduction method override.
            confidence_scores (torch.Tensor, optional): Confidence scores from assigner.
            uncertainty_scores (torch.Tensor, optional): Uncertainty scores from assigner.
            similarity_matrix (torch.Tensor, optional): Sample similarity matrix for consistency.
            
        Returns:
            torch.Tensor: Computed loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Get progressive weights
        unlabeled_w, uncertain_w = self.get_progressive_weights()
        
        # Refine pseudo labels
        refined_target = self.refine_pseudo_labels(pred, target, confidence_scores)
        
        # Separate different types of samples
        pos_mask = refined_target == 1
        neg_mask = refined_target == 0
        unlabeled_mask = refined_target == -1
        uncertain_mask = refined_target == -2
        
        total_loss = 0.0
        
        # 1. Standard focal loss for positive and negative samples
        if pos_mask.sum() > 0 or neg_mask.sum() > 0:
            standard_mask = pos_mask | neg_mask
            standard_pred = pred[standard_mask]
            standard_target = refined_target[standard_mask]
            
            # Convert to binary format for focal loss
            num_classes = pred.size(1)
            binary_target = F.one_hot(standard_target, num_classes=num_classes + 1)
            binary_target = binary_target[:, :num_classes].float()
            
            # Get adaptive focal parameters
            adaptive_gamma, adaptive_alpha = self.get_adaptive_focal_params(standard_pred, standard_target)
            
            # Compute standard focal loss
            standard_weight = weight[standard_mask] if weight is not None else None
            standard_loss = py_sigmoid_focal_loss(
                standard_pred, binary_target, standard_weight,
                gamma=adaptive_gamma, alpha=adaptive_alpha,
                reduction=reduction, avg_factor=avg_factor
            )
            total_loss += standard_loss
            
        # 2. Weighted loss for unlabeled positive samples
        if unlabeled_mask.sum() > 0:
            unlabeled_pred = pred[unlabeled_mask]
            
            # Treat unlabeled as positive with reduced weight
            num_classes = pred.size(1)
            unlabeled_target = torch.ones(unlabeled_pred.size(0), dtype=torch.long, device=pred.device)
            binary_unlabeled = F.one_hot(unlabeled_target, num_classes=num_classes + 1)
            binary_unlabeled = binary_unlabeled[:, :num_classes].float()
            
            # Apply confidence-aware weighting
            if confidence_scores is not None:
                conf_weight = confidence_scores[unlabeled_mask]
                conf_weight = torch.clamp(conf_weight, min=0.1, max=1.0)
            else:
                conf_weight = torch.ones(unlabeled_pred.size(0), device=pred.device)
                
            unlabeled_weight_tensor = weight[unlabeled_mask] if weight is not None else None
            if unlabeled_weight_tensor is not None:
                unlabeled_weight_tensor = unlabeled_weight_tensor * conf_weight * unlabeled_w
            else:
                unlabeled_weight_tensor = conf_weight * unlabeled_w
                
            unlabeled_loss = py_sigmoid_focal_loss(
                unlabeled_pred, binary_unlabeled, unlabeled_weight_tensor,
                gamma=self.gamma, alpha=self.alpha,
                reduction=reduction, avg_factor=avg_factor
            )
            total_loss += unlabeled_loss
            
        # 3. Uncertainty-guided loss for uncertain samples
        if uncertain_mask.sum() > 0:
            uncertain_pred = pred[uncertain_mask]
            
            # For uncertain samples, use soft targets based on uncertainty
            if uncertainty_scores is not None:
                uncertainty_vals = uncertainty_scores[uncertain_mask]
                # Higher uncertainty -> softer targets (closer to 0.5)
                soft_targets = 0.5 + 0.3 * (1 - uncertainty_vals)  # Range: [0.2, 0.8]
            else:
                soft_targets = torch.full((uncertain_pred.size(0),), 0.5, device=pred.device)
                
            # Convert to binary format
            num_classes = pred.size(1)
            if num_classes == 1:
                binary_uncertain = soft_targets.unsqueeze(1)
            else:
                # For multi-class, distribute soft target across classes
                binary_uncertain = torch.zeros(uncertain_pred.size(0), num_classes, device=pred.device)
                binary_uncertain[:, 0] = soft_targets  # Assume first class for simplicity
                
            uncertain_weight_tensor = weight[uncertain_mask] if weight is not None else None
            if uncertain_weight_tensor is not None:
                uncertain_weight_tensor = uncertain_weight_tensor * uncertain_w
            else:
                uncertain_weight_tensor = torch.full((uncertain_pred.size(0),), uncertain_w, device=pred.device)
                
            # Use BCE loss for soft targets
            uncertain_loss = F.binary_cross_entropy_with_logits(
                uncertain_pred, binary_uncertain, weight=uncertain_weight_tensor.unsqueeze(1),
                reduction=reduction
            )
            total_loss += uncertain_loss
            
        # 4. Consistency regularization
        if self.consistency_weight > 0:
            consistency_loss = self.compute_consistency_loss(pred, refined_target, similarity_matrix)
            total_loss += self.consistency_weight * consistency_loss
            
        return self.loss_weight * total_loss


@MODELS.register_module()
class AdvancedFSODFocalLossV2(AdvancedFSODFocalLoss):
    """Enhanced version of AdvancedFSODFocalLoss with additional features.
    
    Additional features:
    1. Multi-scale loss computation
    2. Hard negative mining
    3. Class-balanced loss weighting
    4. Temporal consistency (for video sequences)
    """
    
    def __init__(self, 
                 multi_scale_loss=True,
                 hard_negative_mining=True,
                 class_balanced_loss=True,
                 temporal_consistency=False,
                 hard_negative_ratio=3.0,
                 class_balance_beta=0.9999,
                 temporal_weight=0.05,
                 **kwargs):
        """Initialize AdvancedFSODFocalLossV2.
        
        Args:
            multi_scale_loss (bool): Whether to use multi-scale loss. Defaults to True.
            hard_negative_mining (bool): Whether to use hard negative mining. Defaults to True.
            class_balanced_loss (bool): Whether to use class-balanced loss. Defaults to True.
            temporal_consistency (bool): Whether to use temporal consistency. Defaults to False.
            hard_negative_ratio (float): Ratio for hard negative mining. Defaults to 3.0.
            class_balance_beta (float): Beta for class-balanced loss. Defaults to 0.9999.
            temporal_weight (float): Weight for temporal consistency. Defaults to 0.05.
            **kwargs: Additional arguments for parent class.
        """
        super().__init__(**kwargs)
        
        self.multi_scale_loss = multi_scale_loss
        self.hard_negative_mining = hard_negative_mining
        self.class_balanced_loss = class_balanced_loss
        self.temporal_consistency = temporal_consistency
        self.hard_negative_ratio = hard_negative_ratio
        self.class_balance_beta = class_balance_beta
        self.temporal_weight = temporal_weight
        
        # Class frequency tracking for balanced loss
        self.class_frequencies = None
        
    def update_class_frequencies(self, target):
        """Update class frequency statistics for balanced loss."""
        if not self.class_balanced_loss:
            return
            
        unique_classes, counts = torch.unique(target, return_counts=True)
        
        if self.class_frequencies is None:
            self.class_frequencies = {}
            
        for cls, count in zip(unique_classes.cpu().numpy(), counts.cpu().numpy()):
            if cls in self.class_frequencies:
                self.class_frequencies[cls] = self.class_frequencies[cls] * self.class_balance_beta + count * (1 - self.class_balance_beta)
            else:
                self.class_frequencies[cls] = float(count)
                
    def get_class_balanced_weights(self, target):
        """Calculate class-balanced weights."""
        if not self.class_balanced_loss or self.class_frequencies is None:
            return None
            
        weights = torch.ones_like(target, dtype=torch.float)
        
        for cls, freq in self.class_frequencies.items():
            if cls in target:
                mask = target == cls
                # Inverse frequency weighting
                weights[mask] = 1.0 / (freq + 1e-8)
                
        # Normalize weights
        weights = weights / weights.mean()
        return weights
        
    def hard_negative_mining_loss(self, pred, target, weight=None):
        """Apply hard negative mining to focus on difficult negatives."""
        if not self.hard_negative_mining:
            return None
            
        neg_mask = target == 0
        pos_mask = target == 1
        
        if not neg_mask.any() or not pos_mask.any():
            return None
            
        # Calculate loss for all negative samples
        neg_pred = pred[neg_mask]
        neg_target = target[neg_mask]
        
        num_classes = pred.size(1)
        binary_neg_target = F.one_hot(neg_target, num_classes=num_classes + 1)
        binary_neg_target = binary_neg_target[:, :num_classes].float()
        
        # Compute per-sample loss
        neg_losses = py_sigmoid_focal_loss(
            neg_pred, binary_neg_target, None,
            gamma=self.gamma, alpha=self.alpha,
            reduction='none', avg_factor=None
        )
        
        # Select hard negatives
        num_pos = pos_mask.sum().item()
        num_hard_neg = min(int(num_pos * self.hard_negative_ratio), neg_losses.size(0))
        
        if num_hard_neg > 0:
            _, hard_neg_indices = torch.topk(neg_losses.mean(dim=1), num_hard_neg)
            hard_neg_loss = neg_losses[hard_neg_indices].mean()
            return hard_neg_loss
            
        return None
        
    def compute_temporal_consistency(self, pred, prev_pred, target):
        """Compute temporal consistency loss for video sequences."""
        if not self.temporal_consistency or prev_pred is None:
            return torch.tensor(0.0, device=pred.device)
            
        # Only apply temporal consistency to stable samples (not uncertain)
        stable_mask = (target != -2) & (target != -1)
        
        if not stable_mask.any():
            return torch.tensor(0.0, device=pred.device)
            
        stable_pred = pred[stable_mask]
        stable_prev_pred = prev_pred[stable_mask]
        
        # L2 consistency loss
        temporal_loss = F.mse_loss(stable_pred.sigmoid(), stable_prev_pred.sigmoid())
        return temporal_loss
        
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                confidence_scores=None,
                uncertainty_scores=None,
                similarity_matrix=None,
                prev_pred=None,
                multi_scale_preds=None):
        """Enhanced forward function with additional features.
        
        Args:
            pred (torch.Tensor): Main predictions.
            target (torch.Tensor): Target labels.
            weight (torch.Tensor, optional): Sample weights.
            avg_factor (int, optional): Average factor.
            reduction_override (str, optional): Reduction override.
            confidence_scores (torch.Tensor, optional): Confidence scores.
            uncertainty_scores (torch.Tensor, optional): Uncertainty scores.
            similarity_matrix (torch.Tensor, optional): Similarity matrix.
            prev_pred (torch.Tensor, optional): Previous predictions for temporal consistency.
            multi_scale_preds (list, optional): Multi-scale predictions.
            
        Returns:
            torch.Tensor: Total computed loss.
        """
        # Update class frequencies
        self.update_class_frequencies(target)
        
        # Get class-balanced weights
        class_weights = self.get_class_balanced_weights(target)
        if class_weights is not None:
            if weight is not None:
                weight = weight * class_weights
            else:
                weight = class_weights
                
        # Compute main loss
        main_loss = super().forward(
            pred, target, weight, avg_factor, reduction_override,
            confidence_scores, uncertainty_scores, similarity_matrix
        )
        
        total_loss = main_loss
        
        # Add hard negative mining loss
        if self.hard_negative_mining:
            hnm_loss = self.hard_negative_mining_loss(pred, target, weight)
            if hnm_loss is not None:
                total_loss += 0.5 * hnm_loss
                
        # Add multi-scale loss
        if self.multi_scale_loss and multi_scale_preds is not None:
            for scale_pred in multi_scale_preds:
                if scale_pred.size(0) == pred.size(0):  # Same batch size
                    scale_loss = super().forward(
                        scale_pred, target, weight, avg_factor, reduction_override,
                        confidence_scores, uncertainty_scores, similarity_matrix
                    )
                    total_loss += 0.3 * scale_loss
                    
        # Add temporal consistency loss
        if self.temporal_consistency and prev_pred is not None:
            temporal_loss = self.compute_temporal_consistency(pred, prev_pred, target)
            total_loss += self.temporal_weight * temporal_loss
            
        return total_loss