# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class PriorAwareFSODAssigner_v2_2(BaseAssigner):
    """Prior-Aware Assigner for Few-Shot Object Detection.
    
    This assigner performs standard Hungarian matching identical to HungarianAssigner,
    while additionally identifying potential unlabeled positives:
    1. Standard 1-to-1 Hungarian matching for labeled positive samples
    2. Identify potential unlabeled positives based on confidence and IoU constraints
    3. Use a three-class labeling system: 1 (matched queries), 0 (background), -1 (potential unlabeled positive)
    4. Assign potential unlabeled positives based on: unmatched + high confidence + low IoU with GT
    
    The assignment process includes:
     1. Standard Hungarian matching for labeled GT-prediction pairs (assigned label 1)
     2. Detection of potential unlabeled positives: unmatched queries with confidence higher than
        max confidence of matched queries AND IoU with GT lower than min IoU of matched queries
     3. All other unmatched queries are assigned as background (label 0)
    
    Args:
        match_costs (Union[List[Union[dict, ConfigDict]], dict, ConfigDict]):
            Match cost configs for standard Hungarian matching.
        enable_unlabeled_detection (bool): Whether to enable unlabeled positive
            detection. Default: True.
        unlabeled_label_value (int): Label value for potential unlabeled positives.
            These are unmatched queries with high confidence but low IoU with GT. Default: -1.
        background_label_value (int): Label value for background queries.
            Default: 0.
        positive_label_value (int): Label value for matched queries to optimize.
            Default: 1.
        max_unlabeled_positives (int): Maximum number of unlabeled positives
            to detect per image. Default: 10.
        spatial_nms_threshold (float): Spatial NMS threshold for removing
            overlapping unlabeled detections. Default: 0.5.
        use_nms (bool): Whether to apply spatial NMS to unlabeled positives.
            Default: True.
    """

    def __init__(
        self,
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        enable_unlabeled_detection: bool = True,
        unlabeled_label_value: int = -1,
        background_label_value: int = 0,
        positive_label_value: int = 1,
        max_unlabeled_positives: int = 10,
        spatial_nms_threshold: float = 0.5,
        use_nms: bool = True,
    ) -> None:
        
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        
        self.enable_unlabeled_detection = enable_unlabeled_detection
        self.unlabeled_label_value = unlabeled_label_value
        self.background_label_value = background_label_value
        self.positive_label_value = positive_label_value
        self.max_unlabeled_positives = max_unlabeled_positives
        self.spatial_nms_threshold = spatial_nms_threshold
        self.use_nms = use_nms

    def _compute_bbox_iou(self, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """Compute IoU between two sets of bounding boxes.
        
        Args:
            bboxes1 (Tensor): Bounding boxes, shape (N, 4).
            bboxes2 (Tensor): Bounding boxes, shape (M, 4).
            
        Returns:
            Tensor: IoU matrix, shape (N, M).
        """
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # (N, M, 2)
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # (N, M, 2)
        
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        
        return iou

    def _spatial_nms(self, bboxes: Tensor, scores: Tensor, threshold: float, labels: Tensor = None) -> Tensor:
        """Apply spatial NMS to remove overlapping detections.
        
        Args:
            bboxes (Tensor): Bounding boxes, shape (N, 4).
            scores (Tensor): Confidence scores, shape (N,).
            threshold (float): IoU threshold for NMS.
            labels (Tensor, optional): Class labels, shape (N,). If provided, NMS is applied per class.
            
        Returns:
            Tensor: Indices of kept boxes.
        """
        if len(bboxes) == 0:
            return torch.empty(0, dtype=torch.long, device=bboxes.device)
        
        if labels is None:
            # Original spatial NMS without class separation
            # Sort by scores in descending order
            sorted_indices = torch.argsort(scores, descending=True)
            
            keep = []
            while len(sorted_indices) > 0:
                # Keep the highest scoring box
                current = sorted_indices[0]
                keep.append(current)
                
                if len(sorted_indices) == 1:
                    break
                    
                # Compute IoU with remaining boxes
                current_box = bboxes[current:current+1]
                remaining_boxes = bboxes[sorted_indices[1:]]
                ious = self._compute_bbox_iou(current_box, remaining_boxes)[0]
                
                # Keep boxes with IoU below threshold
                keep_mask = ious < threshold
                sorted_indices = sorted_indices[1:][keep_mask]
            
            return torch.tensor(keep, dtype=torch.long, device=bboxes.device)
        else:
            # Class-aware NMS: apply NMS separately for each class
            keep = []
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                # Get indices for current class
                class_mask = labels == label
                class_indices = torch.where(class_mask)[0]
                
                if len(class_indices) == 0:
                    continue
                
                # Get boxes and scores for current class
                class_bboxes = bboxes[class_indices]
                class_scores = scores[class_indices]
                
                # Sort by scores in descending order
                sorted_indices = torch.argsort(class_scores, descending=True)
                class_sorted_indices = class_indices[sorted_indices]
                
                # Apply NMS within this class
                class_keep = []
                while len(class_sorted_indices) > 0:
                    # Keep the highest scoring box
                    current = class_sorted_indices[0]
                    class_keep.append(current)
                    
                    if len(class_sorted_indices) == 1:
                        break
                        
                    # Compute IoU with remaining boxes in the same class
                    current_box = bboxes[current:current+1]
                    remaining_boxes = bboxes[class_sorted_indices[1:]]
                    ious = self._compute_bbox_iou(current_box, remaining_boxes)[0]
                    
                    # Keep boxes with IoU below threshold
                    keep_mask = ious < threshold
                    class_sorted_indices = class_sorted_indices[1:][keep_mask]
                
                keep.extend(class_keep)
            
            # Sort final results by original indices
            if len(keep) > 0:
                keep = torch.tensor(keep, dtype=torch.long, device=bboxes.device)
                keep = torch.sort(keep)[0]  # Sort to maintain order
                return keep
            else:
                return torch.empty(0, dtype=torch.long, device=bboxes.device)



    def _detect_unlabeled_positives(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        assigned_gt_inds: Tensor,
        img_meta: Optional[dict] = None
    ) -> Tensor:
        """Detect potential unlabeled positive samples.
        
        This method identifies unmatched queries that have:
        1. High confidence (higher than min confidence of matched queries)
        2. Low IoU with GT (lower than min IoU of matched queries)
        These are potential unlabeled positives that should be assigned label -1.
        
        Args:
            pred_instances (InstanceData): Predicted instances.
            gt_instances (InstanceData): Ground truth instances.
            assigned_gt_inds (Tensor): Current GT assignments.
            img_meta (dict, optional): Image meta information.
            
        Returns:
            Tensor: Indices of detected unlabeled positives.
        """
        if not self.enable_unlabeled_detection:
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
            
        if not hasattr(pred_instances, 'scores'):
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
            
        pred_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        gt_bboxes = gt_instances.bboxes
        
        # Handle different score formats
        if pred_scores.dim() > 1:
            # For GroundingDINO: scores shape is (num_queries, max_text_len)
            # Use max score across text tokens as confidence
            pred_scores = torch.max(pred_scores, dim=-1)[0]  # (num_queries,)
        
        # Find matched predictions
        matched_mask = assigned_gt_inds > 0
        if not matched_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        # Compute thresholds based on matched queries
        matched_indices = torch.where(matched_mask)[0]
        matched_scores = pred_scores[matched_indices]
        matched_bboxes = pred_bboxes[matched_indices]
        
        # min_matched_confidence = torch.min(matched_scores).item()
        # Get maximum confidence of matched queries
        min_matched_confidence = torch.max(matched_scores).item()
        
        # Compute IoU between matched predictions and their assigned GT
        matched_gt_indices = assigned_gt_inds[matched_indices] - 1  # Convert to 0-based
        matched_gt_bboxes = gt_bboxes[matched_gt_indices]
        matched_ious = self._compute_bbox_iou(matched_bboxes, matched_gt_bboxes)
        # Get IoU for each matched query with its assigned GT
        matched_ious_diag = torch.diag(matched_ious)
        min_matched_iou = torch.min(matched_ious_diag).item()
        
        # Find unassigned predictions
        unassigned_mask = assigned_gt_inds == 0
        if not unassigned_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        unassigned_indices = torch.where(unassigned_mask)[0]
        unassigned_bboxes = pred_bboxes[unassigned_indices]
        unassigned_scores = pred_scores[unassigned_indices]
        
        # Filter by confidence: higher than min confidence of matched queries
        high_conf_mask = unassigned_scores >= min_matched_confidence
        if not high_conf_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        candidate_indices = unassigned_indices[high_conf_mask]
        candidate_bboxes = unassigned_bboxes[high_conf_mask]
        
        # Compute IoU between candidates and all GT boxes
        ious = self._compute_bbox_iou(candidate_bboxes, gt_bboxes)  # (N_candidates, N_gt)
        max_ious_with_gt = torch.max(ious, dim=1)[0]  # (N_candidates,)
        
        # Filter by IoU: lower than min IoU of matched queries
        low_iou_mask = max_ious_with_gt < min_matched_iou
        if not low_iou_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        final_candidate_indices = candidate_indices[low_iou_mask]
        
        # Apply spatial NMS to remove overlapping candidates
        if len(final_candidate_indices) > 0:
            final_candidate_bboxes = pred_bboxes[final_candidate_indices]
            final_candidate_scores = pred_scores[final_candidate_indices]
            
            # Get predicted labels if available for class-aware NMS
            final_candidate_labels = None
            if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
                candidate_scores = pred_instances.scores[final_candidate_indices]
                # If scores are multi-dimensional (per-class scores), use argmax to get predicted labels
                if candidate_scores.dim() > 1 and candidate_scores.size(-1) > 1:
                    final_candidate_labels = torch.argmax(candidate_scores, dim=-1)
                # If pred_instances has explicit labels, use them
                elif hasattr(pred_instances, 'labels') and pred_instances.labels is not None:
                    final_candidate_labels = pred_instances.labels[final_candidate_indices]
            
            if self.use_nms:
                keep_indices = self._spatial_nms(
                    final_candidate_bboxes, final_candidate_scores, self.spatial_nms_threshold, final_candidate_labels
                )
                final_candidate_indices = final_candidate_indices[keep_indices]
        
        # Limit the number of unlabeled positives
        if len(final_candidate_indices) > self.max_unlabeled_positives:
            # Sort by confidence and keep the best ones
            final_scores = pred_scores[final_candidate_indices]
            sorted_indices = torch.argsort(final_scores, descending=True)
            final_candidate_indices = final_candidate_indices[sorted_indices[:self.max_unlabeled_positives]]
        
        return final_candidate_indices





    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None,
        **kwargs
    ) -> AssignResult:
        """Assign gt to predictions using standard Hungarian matching.
        
        This method performs:
        1. Standard Hungarian matching for labeled GT-prediction pairs
        2. Detection of potential unlabeled positives: high confidence
        3. Three-class labeling: 1 (matched), 0 (background), -1 (unlabeled positive)
        
        Args:
            pred_instances (InstanceData): Predicted instances.
            gt_instances (InstanceData): Ground truth instances.
            img_meta (Optional[dict]): Image meta information.
            **kwargs: Additional keyword arguments.
            
        Returns:
            AssignResult: Assignment result with three-class labels.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. Initialize assignments
        assigned_gt_inds = torch.full(
            (num_preds,), 0, dtype=torch.long, device=device
        )
        assigned_labels = torch.full(
            (num_preds,), self.background_label_value, dtype=torch.long, device=device
        )

        if num_gts == 0 or num_preds == 0:
            # No ground truth or predictions
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels
            )

        # 2. Compute standard matching costs
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta
            )
            cost_list.append(cost)
        standard_cost = torch.stack(cost_list).sum(dim=0)  # (num_preds, num_gts)

        # 3. Hungarian matching for labeled GT-prediction pairs
        cost_cpu = standard_cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 5. Assign matched pairs as positive samples to optimize
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = self.positive_label_value

        # 4. Detect potential unlabeled positives
        # These are unmatched queries with high confidence
        unlabeled_indices = self._detect_unlabeled_positives(
            pred_instances, gt_instances, assigned_gt_inds, img_meta
        )
        if len(unlabeled_indices) > 0:
            # Assign potential unlabeled positives to label -1
            assigned_labels[unlabeled_indices] = self.unlabeled_label_value
            # Note: assigned_gt_inds remains 0 for unlabeled positives

        # 7. All other unmatched predictions remain as background (label 0, already initialized)
        
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels
        )
    
    def get_unlabeled_positives(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None
    ) -> Tensor:
        """Get potential unlabeled positive samples for optimization.
        
        This function can be used separately to identify potential unlabeled
        positives (queries with high confidence but low IoU with GT) for 
        specialized optimization strategies.
        
        Args:
            pred_instances (InstanceData): Predicted instances.
            gt_instances (InstanceData): Ground truth instances.
            img_meta (dict, optional): Image meta information.
            
        Returns:
            Tensor: Indices of potential unlabeled positives.
        """
        # First perform a dummy assignment to get matched queries
        assign_result = self.assign(pred_instances, gt_instances, img_meta)
        
        # Return indices where labels are unlabeled_label_value (-1)
        unlabeled_mask = assign_result.labels == self.unlabeled_label_value
        unlabeled_indices = torch.where(unlabeled_mask)[0]
        
        return unlabeled_indices