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
class FSODHungarianAssignerV2(BaseAssigner):
    """Computes two-phase matching between predictions and ground truth.

    This class performs a two-phase matching process:
    1. Phase 1: Standard Hungarian matching using match_costs to find positive queries
    2. Phase 2: For each GT, find K stubborn queries using stubborn_match_costs
    
    The costs are weighted sum of some components. For DETR the costs are weighted
    sum of classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more predictions
    than targets. After the two-phase matching, the un-matched are treated as
    backgrounds. Thus each query prediction will be assigned with `0` or a positive
    integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs for positive query matching.
        stubborn_match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]], optional): Match cost configs for stubborn query matching.
            If None, use match_costs. Default: None.
        k (int): Number of stubborn queries per GT. Default: 1.
    """

    def __init__(
        self, 
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        stubborn_match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict] = None,
        k: int = 1
    ) -> None:
        """
        Args:
            match_costs: Match cost configs for positive query matching.
            stubborn_match_costs: Match cost configs for stubborn query matching. If None, use match_costs.
            k: Number of stubborn queries per GT. Default: 1.
        """
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        
        # Handle stubborn match costs
        if stubborn_match_costs is None:
            self.stubborn_match_costs = self.match_costs
        else:
            if isinstance(stubborn_match_costs, dict):
                stubborn_match_costs = [stubborn_match_costs]
            elif isinstance(stubborn_match_costs, list):
                assert len(stubborn_match_costs) > 0, \
                    'stubborn_match_costs must not be a empty list.'
            
            self.stubborn_match_costs = [
                TASK_UTILS.build(match_cost) for match_cost in stubborn_match_costs
            ]
        
        self.k = k

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes two-phase matching based on the weighted costs.

        This method performs two-phase matching:
        1. Phase 1: Standard Hungarian matching using match_costs to find positive queries
        2. Phase 2: For each GT, find K stubborn queries using stubborn_match_costs
        
        Stubborn queries are marked but remain as background for classification loss.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result with stubborn_queries marking.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device
        
        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)
        
        # Initialize stubborn queries mask (False means not stubborn)
        stubborn_queries = torch.zeros(num_preds, dtype=torch.bool, device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)
            assign_result.set_extra_property('stubborn_queries', stubborn_queries)
            return assign_result

        # 2. compute weighted cost for positive query matching
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        positive_cost = torch.stack(cost_list).sum(dim=0)
        
        # 3. compute weighted cost for stubborn query matching
        stubborn_cost_list = []
        for match_cost in self.stubborn_match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            stubborn_cost_list.append(cost)
        stubborn_cost = torch.stack(stubborn_cost_list).sum(dim=0)

        # Track which queries have been matched
        matched_query_mask = torch.zeros(num_preds, dtype=torch.bool, device=device)
        
        # Store GT to first-round positive query mapping
        gt_to_positive_query = torch.full((num_gts, ), -1, dtype=torch.long, device=device)
        
        # Store stubborn query to corresponding positive query mapping for regularization loss
        stubborn_to_positive_query = torch.full((num_preds, ), -1, dtype=torch.long, device=device)
        
        # Phase 1: Standard Hungarian matching for positive queries
        # Get all available queries and GTs
        available_query_inds = torch.arange(num_preds, device=device)
        available_gt_inds = torch.arange(num_gts, device=device)
        
        if len(available_query_inds) > 0 and len(available_gt_inds) > 0:
            # Extract cost matrix for all queries and all gts
            round_cost = positive_cost[available_query_inds][:, available_gt_inds]
            
            # 4. do Hungarian matching on CPU using linear_sum_assignment
            round_cost_cpu = round_cost.detach().cpu()
            if linear_sum_assignment is None:
                raise ImportError('Please run "pip install scipy" '
                                  'to install scipy first.')

            matched_row_inds, matched_col_inds = linear_sum_assignment(round_cost_cpu)
            
            if len(matched_row_inds) > 0:
                # Convert back to original indices
                matched_query_inds = available_query_inds[matched_row_inds]
                matched_gt_inds = available_gt_inds[matched_col_inds]
                
                # Phase 1: normal positive assignment
                assigned_gt_inds[matched_query_inds] = matched_gt_inds + 1
                assigned_labels[matched_query_inds] = gt_labels[matched_gt_inds]
                # Store GT to positive query mapping
                gt_to_positive_query[matched_gt_inds] = matched_query_inds
                
                # Update matched query mask
                matched_query_mask[matched_query_inds] = True
        
        # Set unmatched queries as background
        assigned_gt_inds[~matched_query_mask] = 0
        
        # Phase 2: Find K stubborn queries for each GT using stubborn_match_costs
        if self.k > 0:
            # Get remaining unmatched queries
            remaining_query_inds = torch.where(~matched_query_mask)[0]
            # print('remaining_query_inds.shape', remaining_query_inds.shape) #  torch.Size([899])
            
            if len(remaining_query_inds) > 0:
                for gt_idx in range(num_gts):
                    # Extract cost for remaining queries to this specific GT
                    gt_costs = stubborn_cost[remaining_query_inds, gt_idx]
                    
                    # Get top-K queries with lowest cost for this GT
                    if len(gt_costs) > 0:
                        k_actual = min(self.k, len(gt_costs))
                        _, top_k_indices = torch.topk(gt_costs, k_actual, largest=False)
                        
                        # Convert to original query indices
                        stubborn_query_inds = remaining_query_inds[top_k_indices]
                        
                        # Mark as stubborn queries
                        stubborn_queries[stubborn_query_inds] = True
                        # print('stubborn_queries.shape', stubborn_queries.shape) # torch.Size([900])
                        
                        # Store the corresponding positive query indices for stubborn queries
                        positive_query_idx = gt_to_positive_query[gt_idx]
                        if positive_query_idx != -1:  # If this GT was matched in phase 1
                            stubborn_to_positive_query[stubborn_query_inds] = positive_query_idx
        
        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
        
        # Add stubborn queries information
        assign_result.set_extra_property('stubborn_queries', stubborn_queries)
        assign_result.set_extra_property('stubborn_to_positive_query', stubborn_to_positive_query)
        
        return assign_result
