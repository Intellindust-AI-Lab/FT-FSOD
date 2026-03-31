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
class FSODHungarianAssigner(BaseAssigner):
    """Computes multi-round one-to-one matching between predictions and ground truth.

    This class performs k rounds of Hungarian matching. In the first round, it works
    like the standard Hungarian assigner. In subsequent rounds, it excludes already
    matched queries and ground truths, and performs matching on the remaining ones.
    Queries matched in rounds 2 to k are marked as "stubborn queries".
    
    The costs are weighted sum of some components. For DETR the costs are weighted
    sum of classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more predictions
    than targets. After the multi-round matching, the un-matched are treated as
    backgrounds. Thus each query prediction will be assigned with `0` or a positive
    integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
        k (int): Number of Hungarian matching rounds. Default: 1.
    """

    def __init__(
        self, 
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        k: int = 1
    ) -> None:
        """
        Args:
            match_costs: Match cost configs.
            k: Number of Hungarian matching iterations. Default: 1.
        """
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.k = k

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes multi-round one-to-one matching based on the weighted costs.

        This method performs k rounds of Hungarian matching. In each round after
        the first, matched queries from previous rounds are excluded. Queries
        matched in rounds 2 to k are marked as "stubborn queries".

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

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        original_cost = torch.stack(cost_list).sum(dim=0)

        # Track which queries have been matched
        matched_query_mask = torch.zeros(num_preds, dtype=torch.bool, device=device)
        
        # Store stubborn query to GT mapping for regularization loss
        stubborn_gt_inds = torch.full((num_preds, ), -1, dtype=torch.long, device=device)
        
        # Perform k rounds of Hungarian matching
        for round_idx in range(self.k):
            # Get available queries for this round (unmatched queries)
            available_query_inds = torch.where(~matched_query_mask)[0]
            # All GTs are available for each round
            available_gt_inds = torch.arange(num_gts, device=device)
            
            if len(available_query_inds) == 0 or len(available_gt_inds) == 0:
                print(f'Round {round_idx}: no available query or gt. '
                      f'Available queries: {len(available_query_inds)}, '
                      f'Available GTs: {len(available_gt_inds)}, '
                      f'Total queries: {num_preds}, Total GTs: {num_gts}')
                break
                
            # Extract cost matrix for available queries and all gts
            round_cost = original_cost[available_query_inds][:, available_gt_inds]
            
            # 3. do Hungarian matching on CPU using linear_sum_assignment
            round_cost_cpu = round_cost.detach().cpu()
            if linear_sum_assignment is None:
                raise ImportError('Please run "pip install scipy" '
                                  'to install scipy first.')

            matched_row_inds, matched_col_inds = linear_sum_assignment(round_cost_cpu)
            
            if len(matched_row_inds) == 0:
                break
                
            # Convert back to original indices
            matched_query_inds = available_query_inds[matched_row_inds]
            matched_gt_inds = available_gt_inds[matched_col_inds]
            
            # Update assignments
            if round_idx == 0:
                # First round: normal positive assignment
                assigned_gt_inds[matched_query_inds] = matched_gt_inds + 1
                assigned_labels[matched_query_inds] = gt_labels[matched_gt_inds]
            else:
                 # Subsequent rounds: mark as stubborn but keep as background for classification
                 # These queries will be treated as negative samples in classification loss
                 # but will have regularization loss to push them away from matched GTs
                 stubborn_queries[matched_query_inds] = True
                 # Store the matched GT indices for stubborn queries (for regularization loss)
                 stubborn_gt_inds[matched_query_inds] = matched_gt_inds
                 # Keep assigned_gt_inds as -1 for background classification
            
            # Update matched query mask
            matched_query_mask[matched_query_inds] = True

            if round_idx == 0:
                assigned_gt_inds[~matched_query_mask] = 0
        
        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
        
        # Add stubborn queries information
        assign_result.set_extra_property('stubborn_queries', stubborn_queries)
        assign_result.set_extra_property('stubborn_gt_inds', stubborn_gt_inds)
        
        return assign_result
