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
class HungarianAssigner_Perturb_Simple(BaseAssigner):
    """简化版本的匈牙利分配器，支持query扰动。

    这个类计算预测和真实标签之间的一对一匹配，并支持对匹配的正样本query进行扰动。
    扰动约束确保：
    1. 扰动后的query仍然匹配相同的GT
    2. 扰动后的query对其匹配GT的cost仍然是最低的

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): 匹配成本配置。
        enable_perturbation (bool): 是否启用query扰动计算。
        pgd_steps (int): PGD优化步数。
        pgd_lr (float): PGD优化学习率。
        max_perturbation (float): 最大允许扰动幅度。
        cost_margin (float): cost的边际值，确保扰动后cost仍然最低。
    """

    def __init__(
        self, 
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        enable_perturbation: bool = False,
        pgd_steps: int = 5,
        pgd_lr: float = 0.01,
        max_perturbation: float = 0.1,
        # cost_margin: float = 0.1
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
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr
        self.max_perturbation = max_perturbation
        # self.cost_margin = cost_margin
    
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """计算基于加权成本的一对一匹配。

        Args:
            pred_instances (:obj:`InstanceData`): 模型预测实例。
            gt_instances (:obj:`InstanceData`): 真实标签实例。
            img_meta (dict): 图像元信息。

        Returns:
            :obj:`AssignResult`: 分配结果。
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. 默认分配-1
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # 没有真实标签或预测框，返回空分配
            if num_gts == 0:
                # 没有真实标签，全部分配为背景
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. 计算加权成本
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. 在CPU上使用linear_sum_assignment进行匈牙利匹配
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. 分配背景和前景
        # 首先将所有索引分配为背景
        assigned_gt_inds[:] = 0
        # 基于匹配结果分配前景
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        
        # 5. 计算正样本query扰动（如果启用）
        perturbations = None
        if self.enable_perturbation:
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
            
        # 为每个匹配的正样本计算扰动
        perturbations_dict = {}
        
        for i, (query_idx, gt_idx) in enumerate(zip(matched_row_inds, matched_col_inds)):
            # 计算扰动
            perturbation = self._pgd_optimize_positive_perturbation(
                pred_instances, gt_instances, query_idx, gt_idx, 
                cost_matrix, img_meta
            )
            
            if perturbation is not None:
                perturbations_dict[query_idx.item()] = {
                    'perturbation': perturbation,
                    'gt_idx': gt_idx.item()
                }
        
        return perturbations_dict if perturbations_dict else None
    
    def _pgd_optimize_positive_perturbation(self, pred_instances, gt_instances, 
                                          query_idx, gt_idx, cost_matrix, img_meta):
        """使用PGD优化单个正样本query的扰动
        
        Args:
            pred_instances: 预测实例
            gt_instances: 真实标签实例  
            query_idx: 正样本query索引
            gt_idx: 对应的GT索引
            cost_matrix: 成本矩阵
            img_meta: 图像元信息
            
        Returns:
            torch.Tensor: 优化后的扰动 [feature_dim] 或 None
        """
        device = cost_matrix.device
        
        # 获取query特征
        if hasattr(pred_instances, 'query_feats'):
            feature_dim = pred_instances.query_feats.shape[-1]
            original_feature = pred_instances.query_feats[query_idx].clone()
        else:
            feature_dim = 4  # bbox的4个坐标
            original_feature = pred_instances.bboxes[query_idx].clone()
        
        # 获取当前GT的原始cost
        original_cost = cost_matrix[query_idx, gt_idx]
        
        # 动态计算cost_margin：找到该GT对应的最低cost和第二低cost
        gt_costs = cost_matrix[:, gt_idx]
        
        # 找到该GT对应的最低cost和第二低cost（排除当前query）
        other_gt_costs = torch.cat([gt_costs[:query_idx], gt_costs[query_idx+1:]])
        if len(other_gt_costs) > 0:
            # 找到第二低的cost
            second_lowest_cost = other_gt_costs.min()
            # cost_margin为最低cost和第二低cost差值的一半
            cost_margin = (second_lowest_cost - original_cost) * 0.5
            # 确保cost_margin是张量且需要梯度
            cost_margin = cost_margin.detach().clone().requires_grad_(False)
            print('cost_margin', cost_margin)
        else:
            # 如果没有其他query，使用默认值
            cost_margin = torch.tensor(0.1, device=device, dtype=original_cost.dtype, requires_grad=False)
        
        # 训练 perturbation，所以需要梯度吧
        with torch.requires_grad():
            # 初始化扰动
            perturbation = torch.zeros_like(original_feature, requires_grad=True)
            
            # PGD优化
            optimizer = torch.optim.SGD([perturbation], lr=self.pgd_lr)
            
            for _ in range(self.pgd_steps):
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
                
                # 获取扰动后该query对其匹配GT的cost
                perturbed_cost = perturbed_cost_matrix[query_idx, gt_idx]
                
                # 获取扰动后该GT对应的所有query的cost
                perturbed_gt_costs = perturbed_cost_matrix[:, gt_idx]
                
                # 找到该GT对应的第二低的cost（排除当前query）
                other_perturbed_gt_costs = torch.cat([perturbed_gt_costs[:query_idx], perturbed_gt_costs[query_idx+1:]])
                if len(other_perturbed_gt_costs) > 0:
                    second_lowest_perturbed_cost = other_perturbed_gt_costs.min()
                else:
                    # 使用一个很大的值，但保持为张量
                    second_lowest_perturbed_cost = torch.tensor(1e6, device=device, dtype=perturbed_cost.dtype, requires_grad=False)
                
                # 损失函数组成：
                # 1. 保持正样本身份：确保对匹配GT的cost仍然是最低的
                # 需要比第二低的cost至少低cost_margin
                identity_loss = torch.relu(perturbed_cost - second_lowest_perturbed_cost + cost_margin)
                
                # 2. 保持cost不要过高：扰动后的cost不应该比原始cost高太多
                # 使用detach()确保original_cost不参与梯度计算，只作为参考值
                cost_increase_loss = torch.relu(perturbed_cost - original_cost.detach() - cost_margin)
                
                # 3. L2正则化
                reg_loss = 0.01 * torch.norm(perturbation, p=2)
                
                # 总损失
                total_loss = identity_loss + cost_increase_loss + reg_loss
                
                # 检查损失是否包含梯度信息
                if total_loss.requires_grad and total_loss.grad_fn is not None:
                    print('total_loss', total_loss)
                    total_loss.backward()
                    optimizer.step()
                else:
                    # 如果损失没有梯度信息，跳过这次迭代
                    continue
        
        return perturbation.detach()
    
    def _create_single_perturbed_instance(self, pred_instances, query_idx, perturbed_feature):
        """创建单个query扰动后的预测实例"""
        # 创建pred_instances的副本
        perturbed_instances = InstanceData()
        
        # 复制所有属性
        for key, value in pred_instances.items():
            if isinstance(value, torch.Tensor):
                # 对于张量，需要保持梯度信息
                if key in ['query_feats', 'bboxes']:
                    # 对于关键特征，创建新的张量以保持梯度
                    # 不要使用detach()，这会丢失梯度信息
                    perturbed_instances[key] = value.clone()
                else:
                    perturbed_instances[key] = value.clone()
            else:
                perturbed_instances[key] = value
        
        # 应用扰动到指定query，确保梯度信息正确传递
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