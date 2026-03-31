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
class SpatialDiversityAssigner(BaseAssigner):
    """Spatial Diversity Assigner for Few-Shot Object Detection.
    
    This assigner extends the Hungarian matching algorithm by incorporating
    spatial diversity constraints to encourage selection of spatially diverse
    positive samples. This is particularly useful in few-shot object detection
    scenarios where unlabeled positive samples may exist in different regions
    of the image.
    
    The assignment process includes:
    1. Standard cost computation (classification, regression, IoU)
    2. Spatial diversity cost to encourage spatial distribution
    3. Hungarian matching with combined costs
    4. Optional confidence-based additional positive selection
    
    Args:
        match_costs (Union[List[Union[dict, ConfigDict]], dict, ConfigDict]):
            Match cost configs for standard Hungarian matching.
        spatial_diversity_weight (float): Weight for spatial diversity cost.
            Default: 1.0.
        min_spatial_distance (float): Minimum spatial distance between
            selected positive samples (normalized by image diagonal).
            Default: 0.1.
        confidence_threshold (float): Confidence threshold for additional
            positive selection. Default: 0.7.
        max_additional_positives (int): Maximum number of additional positive
            samples to select based on confidence. Default: 5.
        enable_additional_selection (bool): Whether to enable additional
            positive selection beyond Hungarian matching. Default: True.
        spatial_cost_type (str): Type of spatial cost computation.
            Options: 'euclidean', 'manhattan', 'chebyshev'. Default: 'euclidean'.
    """

    def __init__(
        self,
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        spatial_diversity_weight: float = 1.0,
        min_spatial_distance: float = 0.1,
        confidence_threshold: float = 0.7,
        max_additional_positives: int = 5,
        enable_additional_selection: bool = True,
        spatial_cost_type: str = 'euclidean'
    ) -> None:
        
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        
        self.spatial_diversity_weight = spatial_diversity_weight
        self.min_spatial_distance = min_spatial_distance
        self.confidence_threshold = confidence_threshold
        self.max_additional_positives = max_additional_positives
        self.enable_additional_selection = enable_additional_selection
        self.spatial_cost_type = spatial_cost_type
        
        assert spatial_cost_type in ['euclidean', 'manhattan', 'chebyshev'], \
            f"spatial_cost_type must be one of ['euclidean', 'manhattan', 'chebyshev'], "\
            f"got {spatial_cost_type}"

    def _compute_spatial_diversity_cost(
        self,
        pred_centers: Tensor,
        img_meta: Optional[dict] = None
    ) -> Tensor:
        """Compute spatial diversity cost matrix.
        
        Args:
            pred_centers (Tensor): Centers of predicted boxes, shape (N, 2).
            img_meta (dict, optional): Image meta information.
            
        Returns:
            Tensor: Spatial diversity cost matrix, shape (N, N).
        """
        N = pred_centers.shape[0]
        device = pred_centers.device
        
        if N <= 1:
            return torch.zeros((N, N), device=device)
        
        # Normalize coordinates by image diagonal for scale invariance
        if img_meta is not None and 'img_shape' in img_meta:
            h, w = img_meta['img_shape'][:2]
            img_diagonal = torch.sqrt(torch.tensor(h**2 + w**2, device=device))
            pred_centers = pred_centers / img_diagonal
        
        # Compute pairwise distances
        if self.spatial_cost_type == 'euclidean':
            # Euclidean distance
            diff = pred_centers.unsqueeze(1) - pred_centers.unsqueeze(0)  # (N, N, 2)
            distances = torch.norm(diff, dim=2)  # (N, N)
        elif self.spatial_cost_type == 'manhattan':
            # Manhattan distance
            diff = pred_centers.unsqueeze(1) - pred_centers.unsqueeze(0)  # (N, N, 2)
            distances = torch.sum(torch.abs(diff), dim=2)  # (N, N)
        elif self.spatial_cost_type == 'chebyshev':
            # Chebyshev distance
            diff = pred_centers.unsqueeze(1) - pred_centers.unsqueeze(0)  # (N, N, 2)
            distances = torch.max(torch.abs(diff), dim=2)[0]  # (N, N)
        
        # Convert distances to costs (closer = higher cost)
        # Use exponential decay to penalize close predictions
        spatial_costs = torch.exp(-distances / self.min_spatial_distance)
        
        # Set diagonal to 0 (no cost for self-pairing)
        spatial_costs.fill_diagonal_(0)
        
        return spatial_costs

    def _get_bbox_centers(self, bboxes: Tensor) -> Tensor:
        """Get centers of bounding boxes.
        
        Args:
            bboxes (Tensor): Bounding boxes, shape (N, 4) in (x1, y1, x2, y2) format.
            
        Returns:
            Tensor: Centers of bounding boxes, shape (N, 2).
        """
        centers = torch.stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,  # center_x
            (bboxes[:, 1] + bboxes[:, 3]) / 2   # center_y
        ], dim=1)
        return centers

    def _select_additional_positives(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        assigned_gt_inds: Tensor,
        assigned_labels: Tensor,
        img_meta: Optional[dict] = None
    ) -> tuple:
        """Select additional positive samples based on confidence and spatial diversity.
        
        Args:
            pred_instances (InstanceData): Predicted instances.
            gt_instances (InstanceData): Ground truth instances.
            assigned_gt_inds (Tensor): Current GT assignments.
            assigned_labels (Tensor): Current label assignments.
            img_meta (dict, optional): Image meta information.
            
        Returns:
            tuple: Updated (assigned_gt_inds, assigned_labels).
        """
        if not self.enable_additional_selection:
            return assigned_gt_inds, assigned_labels
            
        # Get unassigned predictions with high confidence
        unassigned_mask = assigned_gt_inds == 0
        if not unassigned_mask.any():
            return assigned_gt_inds, assigned_labels
            
        # Check if predictions have scores/confidence
        if not hasattr(pred_instances, 'scores'):
            return assigned_gt_inds, assigned_labels
            
        pred_scores = pred_instances.scores
        high_conf_mask = pred_scores > self.confidence_threshold
        candidate_mask = unassigned_mask & high_conf_mask
        
        if not candidate_mask.any():
            return assigned_gt_inds, assigned_labels
            
        # Get centers of currently assigned positive predictions
        positive_mask = assigned_gt_inds > 0
        if not positive_mask.any():
            return assigned_gt_inds, assigned_labels
            
        pred_bboxes = pred_instances.bboxes
        assigned_centers = self._get_bbox_centers(pred_bboxes[positive_mask])
        candidate_centers = self._get_bbox_centers(pred_bboxes[candidate_mask])
        
        # Normalize coordinates
        if img_meta is not None and 'img_shape' in img_meta:
            h, w = img_meta['img_shape'][:2]
            img_diagonal = torch.sqrt(torch.tensor(h**2 + w**2, device=assigned_centers.device))
            assigned_centers = assigned_centers / img_diagonal
            candidate_centers = candidate_centers / img_diagonal
        
        # Select candidates that are spatially diverse
        candidate_indices = torch.where(candidate_mask)[0]
        selected_additional = []
        
        for i, candidate_idx in enumerate(candidate_indices):
            if len(selected_additional) >= self.max_additional_positives:
                break
                
            candidate_center = candidate_centers[i:i+1]  # (1, 2)
            
            # Compute distances to all assigned positives
            if self.spatial_cost_type == 'euclidean':
                distances = torch.norm(assigned_centers - candidate_center, dim=1)
            elif self.spatial_cost_type == 'manhattan':
                distances = torch.sum(torch.abs(assigned_centers - candidate_center), dim=1)
            elif self.spatial_cost_type == 'chebyshev':
                distances = torch.max(torch.abs(assigned_centers - candidate_center), dim=1)[0]
            
            # Check if candidate is spatially diverse enough
            min_distance = torch.min(distances)
            if min_distance > self.min_spatial_distance:
                selected_additional.append(candidate_idx)
                # Add this candidate to assigned_centers for next iterations
                assigned_centers = torch.cat([assigned_centers, candidate_center], dim=0)
        
        # Assign additional positives to the nearest GT
        if selected_additional:
            additional_indices = torch.tensor(selected_additional, device=assigned_gt_inds.device)
            
            # For simplicity, assign to GT with highest IoU or first GT
            # In practice, you might want more sophisticated assignment
            num_gts = len(gt_instances)
            if num_gts > 0:
                # Assign to first GT (can be improved with IoU-based assignment)
                assigned_gt_inds[additional_indices] = 1
                assigned_labels[additional_indices] = gt_instances.labels[0]
        
        return assigned_gt_inds, assigned_labels

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None,
        **kwargs
    ) -> AssignResult:
        """Assign predictions to ground truth with spatial diversity constraints.
        
        Args:
            pred_instances (InstanceData): Predicted instances.
            gt_instances (InstanceData): Ground truth instances.
            img_meta (dict, optional): Image meta information.
            
        Returns:
            AssignResult: Assignment result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. Initialize assignments
        assigned_gt_inds = torch.full(
            (num_preds,), -1, dtype=torch.long, device=device
        )
        assigned_labels = torch.full(
            (num_preds,), -1, dtype=torch.long, device=device
        )

        if num_gts == 0 or num_preds == 0:
            # No ground truth or predictions
            if num_gts == 0:
                assigned_gt_inds[:] = 0
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

        # 3. Compute spatial diversity cost
        if self.spatial_diversity_weight > 0 and hasattr(pred_instances, 'bboxes'):
            pred_centers = self._get_bbox_centers(pred_instances.bboxes)
            spatial_cost_matrix = self._compute_spatial_diversity_cost(
                pred_centers, img_meta
            )  # (num_preds, num_preds)
            
            # Expand spatial cost to match standard cost dimensions
            # For each GT, add spatial diversity penalty
            spatial_cost_expanded = spatial_cost_matrix.unsqueeze(2).expand(
                num_preds, num_preds, num_gts
            )  # (num_preds, num_preds, num_gts)
            
            # Sum spatial costs for each prediction-GT pair
            # This encourages selecting predictions that are far from each other
            spatial_cost_per_gt = spatial_cost_expanded.sum(dim=1)  # (num_preds, num_gts)
            
            # Combine costs
            total_cost = standard_cost + self.spatial_diversity_weight * spatial_cost_per_gt
        else:
            total_cost = standard_cost

        # 4. Hungarian matching
        total_cost = total_cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(total_cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 5. Assign based on Hungarian matching
        assigned_gt_inds[:] = 0  # Background first
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # 6. Select additional positives based on confidence and spatial diversity
        assigned_gt_inds, assigned_labels = self._select_additional_positives(
            pred_instances, gt_instances, assigned_gt_inds, assigned_labels, img_meta
        )

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels
        )