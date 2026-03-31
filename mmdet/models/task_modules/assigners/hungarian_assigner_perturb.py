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
class HungarianAssigner_Perturb(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth with query-based perturbations.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    
    Additionally, this assigner maintains query pools for each class and applies
    perturbations to matched positive queries to enhance training diversity while
    preserving their positive sample identity.

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
        enable_perturbation (bool): Whether to enable query perturbation computation.
        query_pool_size (int): Maximum size of query pool for each class.
        perturbation_lambda (float): L2 regularization weight for perturbations.
        distance_lambda (float): Weight for distance constraint to query pool center.
        pgd_steps (int): Number of PGD optimization steps.
        pgd_lr (float): Learning rate for PGD optimization.
        max_perturbation (float): Maximum allowed perturbation magnitude.
        min_distance_ratio (float): Minimum distance ratio to query pool center.
        max_distance_ratio (float): Maximum distance ratio to query pool center.
    """

    def __init__(
        self, 
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        enable_perturbation: bool = False,
        query_pool_size: int = 100,
        perturbation_lambda: float = 0.1,
        distance_lambda: float = 0.05,
        pgd_steps: int = 10,
        pgd_lr: float = 0.01,
        max_perturbation: float = 0.1,
        min_distance_ratio: float = 0.3,
        max_distance_ratio: float = 0.8
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        
        # Query扰动相关参数
        self.enable_perturbation = enable_perturbation
        self.query_pool_size = query_pool_size
        self.perturbation_lambda = perturbation_lambda
        self.distance_lambda = distance_lambda
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr
        self.max_perturbation = max_perturbation
        self.min_distance_ratio = min_distance_ratio
        self.max_distance_ratio = max_distance_ratio
        
        # 为每个类别和层级维护query池 {layer: {class_id: {'queries': [], 'center': tensor}}}
        self.query_pools = {}
        # 记录当前层级（用于区分不同层的匈牙利匹配）
        self.current_layer = 0
    
    def set_layer(self, layer_id: int):
        """设置当前层级ID，用于区分不同层的匈牙利匹配
        
        Args:
            layer_id (int): 层级ID
        """
        self.current_layer = layer_id
        
        # 初始化该层级的query池
        if layer_id not in self.query_pools:
            self.query_pools[layer_id] = {}

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

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
            :obj:`AssignResult`: The assigned result.
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

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        
        # 5. 更新query池（使用原始未扰动的query）
        if self.enable_perturbation and self.training:
            self._update_query_pools(
                pred_instances, matched_row_inds, matched_col_inds, gt_instances
            )
        
        # 6. 计算正样本query扰动（如果启用）
        perturbations = None
        if self.enable_perturbation and self.training:
            perturbations = self._compute_positive_query_perturbations(
                pred_instances, gt_instances, cost.to(device), 
                matched_row_inds, matched_col_inds, img_meta
            )
        
        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
        
        # 将扰动信息添加到结果中
        if perturbations is not None:
            assign_result.perturbations = perturbations
            
        return assign_result
    
    def _compute_positive_query_perturbations(self, pred_instances, gt_instances, 
                                            cost_matrix, matched_row_inds, 
                                            matched_col_inds, img_meta):
        """计算正样本query的扰动
        
        Args:
            pred_instances: 预测实例
            gt_instances: 真实标签实例
            cost_matrix: 成本矩阵 [num_preds, num_gts]
            matched_row_inds: 匹配的预测索引
            matched_col_inds: 匹配的GT索引
            img_meta: 图像元信息
            
        Returns:
            dict: 包含扰动信息的字典
        """
        device = cost_matrix.device
        
        if len(matched_row_inds) == 0:
            return None
            
        # 获取匹配的GT标签
        matched_gt_labels = gt_instances.labels[matched_col_inds]
        
        # 为每个匹配的正样本计算扰动
        perturbations_dict = {}
        
        for i, (query_idx, gt_idx, gt_label) in enumerate(zip(matched_row_inds, matched_col_inds, matched_gt_labels)):
            gt_label_item = gt_label.item()
            
            # 检查该类别在当前层级是否有query池
            current_layer_pools = self.query_pools.get(self.current_layer, {})
            if gt_label_item not in current_layer_pools or len(current_layer_pools[gt_label_item]['queries']) == 0:
                continue
                
            # 计算扰动
            perturbation = self._pgd_optimize_positive_perturbation(
                pred_instances, gt_instances, query_idx, gt_idx, 
                gt_label_item, cost_matrix, img_meta
            )
            
            if perturbation is not None:
                perturbations_dict[query_idx.item()] = {
                    'perturbation': perturbation,
                    'gt_label': gt_label_item,
                    'gt_idx': gt_idx.item()
                }
        
        return perturbations_dict if perturbations_dict else None
    
    def _pgd_optimize_positive_perturbation(self, pred_instances, gt_instances, 
                                          query_idx, gt_idx, gt_label, cost_matrix, img_meta):
        """使用PGD优化单个正样本query的扰动
        
        Args:
            pred_instances: 预测实例
            gt_instances: 真实标签实例  
            query_idx: 正样本query索引
            gt_idx: 对应的GT索引
            gt_label: GT标签
            cost_matrix: 成本矩阵
            img_meta: 图像元信息
            
        Returns:
            torch.Tensor: 优化后的扰动 [feature_dim] 或 None
        """
        device = pred_instances.bboxes.device
        
        # 获取query特征
        if hasattr(pred_instances, 'query_feats'):
            feature_dim = pred_instances.query_feats.shape[-1]
            original_feature = pred_instances.query_feats[query_idx].clone()
        else:
            feature_dim = 4  # bbox的4个坐标
            original_feature = pred_instances.bboxes[query_idx].clone()
        
        # 获取当前层级的query池中心
        current_layer_pools = self.query_pools.get(self.current_layer, {})
        query_pool = current_layer_pools[gt_label]
        pool_center = query_pool['center']
        
        # 计算到池中心的距离
        center_distance = torch.norm(original_feature - pool_center, p=2)
        
        # 设置目标距离范围
        min_target_distance = center_distance * self.min_distance_ratio
        max_target_distance = center_distance * self.max_distance_ratio
        
        # 初始化扰动
        perturbation = torch.zeros_like(original_feature, requires_grad=True)
        
        # PGD优化
        optimizer = torch.optim.SGD([perturbation], lr=self.pgd_lr)
        
        for step in range(self.pgd_steps):
            optimizer.zero_grad()
            
            # 应用扰动
            perturbed_feature = original_feature + perturbation
            
            # 创建扰动后的pred_instances副本
            perturbed_pred_instances = self._create_single_perturbed_instance(
                pred_instances, query_idx, perturbed_feature
            )
            
            # 计算扰动后的cost矩阵
            perturbed_cost_matrix = self._compute_full_cost_matrix(
                perturbed_pred_instances, gt_instances, img_meta
            )
            
            # 确保该query对其匹配GT的cost仍然是最低的
            query_costs = perturbed_cost_matrix[query_idx]  # 该query对所有GT的cost
            matched_cost = query_costs[gt_idx]  # 对匹配GT的cost
            min_cost = query_costs.min()  # 对所有GT的最小cost
            
            # 计算到池中心的距离
            current_distance = torch.norm(perturbed_feature - pool_center, p=2)
            
            # 损失函数组成：
            # 1. 保持正样本身份：确保对匹配GT的cost是最低的
            positive_identity_loss = torch.relu(matched_cost - min_cost + 1e-6)
            
            # 2. 距离约束：保持与池中心的适当距离
            if current_distance < min_target_distance:
                distance_loss = (min_target_distance - current_distance) ** 2
            elif current_distance > max_target_distance:
                distance_loss = (current_distance - max_target_distance) ** 2
            else:
                distance_loss = torch.tensor(0.0, device=device)
            
            # 3. L2正则化
            reg_loss = self.perturbation_lambda * torch.norm(perturbation, p=2)
            
            # 总损失
            total_loss = positive_identity_loss + self.distance_lambda * distance_loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 限制扰动大小
            with torch.no_grad():
                perturbation.data = torch.clamp(
                    perturbation.data, 
                    -self.max_perturbation, 
                    self.max_perturbation
                )
        
        return perturbation.detach()
    
    def _update_query_pools(self, pred_instances, matched_row_inds, matched_col_inds, gt_instances):
        """更新每个类别的query池"""
        if len(matched_row_inds) == 0:
            return
            
        # 获取query特征
        if hasattr(pred_instances, 'query_feats'):
            query_features = pred_instances.query_feats[matched_row_inds]
        else:
            query_features = pred_instances.bboxes[matched_row_inds]
            
        matched_gt_labels = gt_instances.labels[matched_col_inds]
        
        for query_feat, gt_label in zip(query_features, matched_gt_labels):
            gt_label_item = gt_label.item()
            
            # 确保当前层级的query池存在
            if self.current_layer not in self.query_pools:
                self.query_pools[self.current_layer] = {}
            
            current_layer_pools = self.query_pools[self.current_layer]
            
            # 初始化该类别的query池
            if gt_label_item not in current_layer_pools:
                current_layer_pools[gt_label_item] = {
                    'queries': [],
                    'center': query_feat.clone().detach()
                }
            else:
                # 添加新query到池中
                pool = current_layer_pools[gt_label_item]
                pool['queries'].append(query_feat.clone().detach())
                
                # 限制池大小
                if len(pool['queries']) > self.query_pool_size:
                    pool['queries'].pop(0)  # 移除最旧的query
                
                # 更新池中心（移动平均）
                if len(pool['queries']) > 0:
                    pool_tensor = torch.stack(pool['queries'])
                    pool['center'] = pool_tensor.mean(dim=0)
    
    def _create_single_perturbed_instance(self, pred_instances, query_idx, perturbed_feature):
        """创建单个query扰动后的预测实例"""
        # 创建pred_instances的副本
        perturbed_instances = InstanceData()
        
        # 复制所有属性
        for key, value in pred_instances.items():
            if isinstance(value, torch.Tensor):
                perturbed_instances[key] = value.clone()
            else:
                perturbed_instances[key] = value
        
        # 应用扰动到指定query
        if hasattr(pred_instances, 'query_feats'):
            perturbed_instances.query_feats[query_idx] = perturbed_feature
        else:
            perturbed_instances.bboxes[query_idx] = perturbed_feature
            
        return perturbed_instances
    
    def _compute_full_cost_matrix(self, pred_instances, gt_instances, img_meta):
        """计算完整的cost矩阵"""
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta
            )
            cost_list.append(cost)
        
        total_cost = torch.stack(cost_list).sum(dim=0)
        return total_cost
