# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class FSODAssigner_v7_5(BaseAssigner):
    """FSOD Assigner v7.5 - 添加类别平衡的未标注检测机制
    
    基于v7.4版本，新增功能：
    1. 继承v7.1的动态置信度校准
    2. 继承v7.2的语义-空间一致性检查
    3. 继承v7.3的不确定性感知分配
    4. 继承v7.4的多阶段渐进分配
    5. 类别平衡的未标注检测：
       - 每个类别独立的未标注样本配额
       - 类别感知的置信度校准
       - 类别间的平衡策略
    6. 时序一致性跟踪（可选）
    
    验证目标：证明类别平衡机制能够避免某些类别的未标注样本被忽略
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
        # v7.1: 动态置信度校准参数
        enable_confidence_calibration: bool = True,
        confidence_ema_alpha: float = 0.9,
        adaptive_threshold_factor: float = 0.8,
        # v7.2: 语义-空间一致性检查参数
        enable_semantic_spatial_consistency: bool = True,
        consistency_iou_threshold: float = 0.3,
        consistency_feature_threshold: float = 0.5,
        feature_similarity_weight: float = 0.3,
        # v7.3: 不确定性感知分配参数
        enable_uncertainty_estimation: bool = True,
        uncertain_label_value: int = -2,
        uncertainty_threshold: float = 0.7,
        uncertainty_method: str = 'entropy',
        max_uncertain_samples: int = 5,
        # v7.4: 多阶段渐进分配参数
        enable_progressive_assignment: bool = True,
        stage1_confidence_threshold: float = 0.8,
        stage2_confidence_threshold: float = 0.5,
        stage3_confidence_threshold: float = 0.3,
        progressive_cost_weight: float = 0.7,
        max_stage2_assignments: int = 15,
        max_stage3_assignments: int = 20,
        # v7.5: 类别平衡的未标注检测参数
        enable_class_balanced_detection: bool = True,
        max_unlabeled_per_class: int = 3,        # 每个类别最大未标注样本数
        class_balance_strategy: str = 'adaptive', # 'fixed', 'adaptive', 'proportional'
        min_class_confidence: float = 0.1,       # 类别最小置信度阈值
        enable_temporal_consistency: bool = False, # 时序一致性跟踪
        temporal_momentum: float = 0.8,          # 时序动量
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
        
        # v7.1: 动态置信度校准参数
        self.enable_confidence_calibration = enable_confidence_calibration
        self.confidence_ema_alpha = confidence_ema_alpha
        self.adaptive_threshold_factor = adaptive_threshold_factor
        
        # v7.2: 语义-空间一致性检查参数
        self.enable_semantic_spatial_consistency = enable_semantic_spatial_consistency
        self.consistency_iou_threshold = consistency_iou_threshold
        self.consistency_feature_threshold = consistency_feature_threshold
        self.feature_similarity_weight = feature_similarity_weight
        
        # v7.3: 不确定性感知分配参数
        self.enable_uncertainty_estimation = enable_uncertainty_estimation
        self.uncertain_label_value = uncertain_label_value
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_method = uncertainty_method
        self.max_uncertain_samples = max_uncertain_samples
        
        # v7.4: 多阶段渐进分配参数
        self.enable_progressive_assignment = enable_progressive_assignment
        self.stage1_confidence_threshold = stage1_confidence_threshold
        self.stage2_confidence_threshold = stage2_confidence_threshold
        self.stage3_confidence_threshold = stage3_confidence_threshold
        self.progressive_cost_weight = progressive_cost_weight
        self.max_stage2_assignments = max_stage2_assignments
        self.max_stage3_assignments = max_stage3_assignments
        
        # v7.5: 类别平衡的未标注检测参数
        self.enable_class_balanced_detection = enable_class_balanced_detection
        self.max_unlabeled_per_class = max_unlabeled_per_class
        self.class_balance_strategy = class_balance_strategy
        self.min_class_confidence = min_class_confidence
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_momentum = temporal_momentum
        
        # 初始化置信度统计
        self.register_buffer('confidence_mean', torch.tensor(0.5))
        self.register_buffer('confidence_std', torch.tensor(0.2))
        self.register_buffer('update_count', torch.tensor(0))
        
        # 初始化类别统计（动态创建）
        self.class_confidence_stats = {}
        self.class_assignment_history = {}

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区，确保参数在正确设备上"""
        setattr(self, name, tensor)

    def _update_class_statistics(
        self, 
        pred_instances: InstanceData, 
        gt_instances: InstanceData,
        assigned_labels: Tensor
    ) -> None:
        """更新类别统计信息
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            assigned_labels (Tensor): 分配的标签
        """
        if not self.enable_class_balanced_detection:
            return
        
        # 获取预测分数和类别
        pred_scores = pred_instances.scores
        if pred_scores.dim() > 1:
            pred_class_scores = pred_scores
            pred_classes = torch.argmax(pred_scores, dim=-1)
        else:
            # 单一分数，假设为二分类
            pred_classes = (pred_scores > 0.5).long()
            pred_class_scores = pred_scores.unsqueeze(-1)
        
        # 获取GT类别
        gt_classes = gt_instances.labels
        unique_gt_classes = torch.unique(gt_classes)
        
        # 更新每个类别的统计信息
        for gt_class in unique_gt_classes:
            gt_class_item = gt_class.item()
            
            # 找到该类别的正样本
            positive_mask = (assigned_labels == self.positive_label_value)
            if positive_mask.any():
                positive_indices = torch.where(positive_mask)[0]
                
                # 计算该类别的置信度统计
                if len(positive_indices) > 0:
                    if pred_class_scores.size(-1) > gt_class_item:
                        class_scores = pred_class_scores[positive_indices, gt_class_item]
                    else:
                        class_scores = pred_scores[positive_indices]
                    
                    class_mean = class_scores.mean().item()
                    class_std = class_scores.std().item() if len(class_scores) > 1 else 0.1
                    
                    # 使用EMA更新类别统计
                    if gt_class_item in self.class_confidence_stats:
                        old_mean, old_std = self.class_confidence_stats[gt_class_item]
                        new_mean = self.confidence_ema_alpha * old_mean + (1 - self.confidence_ema_alpha) * class_mean
                        new_std = self.confidence_ema_alpha * old_std + (1 - self.confidence_ema_alpha) * class_std
                        self.class_confidence_stats[gt_class_item] = (new_mean, new_std)
                    else:
                        self.class_confidence_stats[gt_class_item] = (class_mean, class_std)
            
            # 初始化类别分配历史
            if gt_class_item not in self.class_assignment_history:
                self.class_assignment_history[gt_class_item] = []

    def _get_class_adaptive_thresholds(self, pred_instances: InstanceData, gt_instances: InstanceData) -> dict:
        """获取类别自适应阈值
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            
        Returns:
            dict: 每个类别的自适应阈值
        """
        if not self.enable_class_balanced_detection:
            return {}
        
        gt_classes = gt_instances.labels
        unique_gt_classes = torch.unique(gt_classes)
        class_thresholds = {}
        
        for gt_class in unique_gt_classes:
            gt_class_item = gt_class.item()
            
            if gt_class_item in self.class_confidence_stats:
                class_mean, class_std = self.class_confidence_stats[gt_class_item]
                
                if self.class_balance_strategy == 'adaptive':
                    # 基于类别统计的自适应阈值
                    threshold = max(class_mean - 0.5 * class_std, self.min_class_confidence)
                elif self.class_balance_strategy == 'proportional':
                    # 基于全局统计的比例阈值
                    global_mean = self.confidence_mean.item()
                    global_std = self.confidence_std.item()
                    ratio = class_mean / max(global_mean, 0.1)
                    threshold = max(ratio * global_mean - 0.3 * global_std, self.min_class_confidence)
                else:  # 'fixed'
                    threshold = self.stage3_confidence_threshold
            else:
                # 新类别使用默认阈值
                threshold = self.stage3_confidence_threshold
            
            class_thresholds[gt_class_item] = threshold
        
        return class_thresholds

    def _class_balanced_selection(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        candidate_indices: Tensor
    ) -> Tensor:
        """类别平衡的候选样本选择
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            candidate_indices (Tensor): 候选预测索引
            
        Returns:
            Tensor: 经过类别平衡选择的索引
        """
        if not self.enable_class_balanced_detection or len(candidate_indices) == 0:
            return candidate_indices
        
        # 获取预测分数和类别
        pred_scores = pred_instances.scores
        if pred_scores.dim() > 1:
            candidate_class_scores = pred_scores[candidate_indices]
            candidate_classes = torch.argmax(candidate_class_scores, dim=-1)
            candidate_confidences = torch.max(candidate_class_scores, dim=-1)[0]
        else:
            candidate_confidences = pred_scores[candidate_indices]
            candidate_classes = (candidate_confidences > 0.5).long()
        
        # 获取类别自适应阈值
        class_thresholds = self._get_class_adaptive_thresholds(pred_instances, gt_instances)
        
        # 按类别分组选择
        selected_indices = []
        unique_classes = torch.unique(candidate_classes)
        
        for class_id in unique_classes:
            class_id_item = class_id.item()
            class_mask = candidate_classes == class_id
            class_indices = candidate_indices[class_mask]
            class_confidences = candidate_confidences[class_mask]
            
            # 应用类别特定阈值
            if class_id_item in class_thresholds:
                threshold = class_thresholds[class_id_item]
                valid_mask = class_confidences >= threshold
                class_indices = class_indices[valid_mask]
                class_confidences = class_confidences[valid_mask]
            
            # 限制每个类别的样本数量
            if len(class_indices) > self.max_unlabeled_per_class:
                sorted_indices = torch.argsort(class_confidences, descending=True)
                class_indices = class_indices[sorted_indices[:self.max_unlabeled_per_class]]
            
            if len(class_indices) > 0:
                selected_indices.append(class_indices)
        
        if len(selected_indices) > 0:
            selected_indices = torch.cat(selected_indices)
        else:
            selected_indices = torch.empty(0, dtype=torch.long, device=candidate_indices.device)
        
        return selected_indices

    def _temporal_consistency_filter(
        self,
        pred_instances: InstanceData,
        candidate_indices: Tensor,
        img_meta: Optional[dict] = None
    ) -> Tensor:
        """时序一致性过滤
        
        Args:
            pred_instances (InstanceData): 预测实例
            candidate_indices (Tensor): 候选预测索引
            img_meta (Optional[dict]): 图像元信息
            
        Returns:
            Tensor: 经过时序一致性过滤的索引
        """
        if not self.enable_temporal_consistency or img_meta is None:
            return candidate_indices
        
        # 这里可以实现基于时序信息的一致性检查
        # 例如：检查候选框在连续帧中的稳定性
        # 由于这需要额外的时序信息，这里提供一个简化的实现
        
        # 简化实现：基于候选框的空间稳定性
        if len(candidate_indices) <= 1:
            return candidate_indices
        
        candidate_bboxes = pred_instances.bboxes[candidate_indices]
        candidate_scores = pred_instances.scores[candidate_indices]
        
        if candidate_scores.dim() > 1:
            candidate_scores = torch.max(candidate_scores, dim=-1)[0]
        
        # 计算候选框之间的IoU
        ious = self._compute_bbox_iou(candidate_bboxes, candidate_bboxes)
        
        # 找到具有高IoU的候选框组
        consistent_mask = torch.zeros(len(candidate_indices), dtype=torch.bool, device=candidate_indices.device)
        
        for i in range(len(candidate_indices)):
            # 找到与当前候选框IoU > 0.3的其他候选框
            similar_mask = ious[i] > 0.3
            similar_count = similar_mask.sum().item()
            
            # 如果有多个相似的候选框，认为是时序一致的
            if similar_count > 1:
                consistent_mask[i] = True
        
        # 如果没有找到一致的候选框，返回置信度最高的几个
        if not consistent_mask.any():
            sorted_indices = torch.argsort(candidate_scores, descending=True)
            top_k = min(len(candidate_indices), 3)
            consistent_mask[sorted_indices[:top_k]] = True
        
        return candidate_indices[consistent_mask]

    def _get_progressive_thresholds(self, pred_scores: Tensor) -> tuple[float, float, float]:
        """获取渐进式阈值（继承自v7.4）"""
        if not self.enable_progressive_assignment:
            return self.stage1_confidence_threshold, self.stage2_confidence_threshold, self.stage3_confidence_threshold
        
        score_mean = pred_scores.mean().item()
        score_std = pred_scores.std().item()
        score_max = pred_scores.max().item()
        score_min = pred_scores.min().item()
        
        if self.enable_confidence_calibration and self.update_count > 0:
            calibrated_mean = self.confidence_mean.item()
            calibrated_std = self.confidence_std.item()
            
            stage1_threshold = min(calibrated_mean + 1.5 * calibrated_std, score_max * 0.95)
            stage2_threshold = min(calibrated_mean + 0.5 * calibrated_std, stage1_threshold * 0.8)
            stage3_threshold = max(calibrated_mean - 0.5 * calibrated_std, score_min * 1.1)
        else:
            stage1_threshold = self.stage1_confidence_threshold
            stage2_threshold = self.stage2_confidence_threshold
            stage3_threshold = self.stage3_confidence_threshold
        
        stage1_threshold = max(stage1_threshold, stage2_threshold + 0.1)
        stage2_threshold = max(stage2_threshold, stage3_threshold + 0.1)
        
        return stage1_threshold, stage2_threshold, stage3_threshold

    def _progressive_assignment_stage1(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        cost_matrix: Tensor,
        pred_scores: Tensor,
        img_meta: Optional[dict] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """阶段1：高置信度匈牙利匹配（继承自v7.4）"""
        num_preds = len(pred_instances)
        device = pred_scores.device
        
        assigned_gt_inds = torch.zeros(num_preds, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_preds,), self.background_label_value, dtype=torch.long, device=device)
        
        stage1_threshold, _, _ = self._get_progressive_thresholds(pred_scores)
        
        high_conf_mask = pred_scores >= stage1_threshold
        if not high_conf_mask.any():
            return assigned_gt_inds, assigned_labels, torch.empty(0, dtype=torch.long, device=device)
        
        high_conf_indices = torch.where(high_conf_mask)[0]
        high_conf_cost = cost_matrix[high_conf_indices]
        
        if len(high_conf_indices) > 0 and len(gt_instances) > 0:
            cost_cpu = high_conf_cost.detach().cpu().numpy()
            matched_pred_inds, matched_gt_inds = linear_sum_assignment(cost_cpu)
            
            matched_pred_inds = torch.from_numpy(matched_pred_inds).to(device)
            matched_gt_inds = torch.from_numpy(matched_gt_inds).to(device)
            
            original_pred_inds = high_conf_indices[matched_pred_inds]
            
            assigned_gt_inds[original_pred_inds] = matched_gt_inds + 1
            assigned_labels[original_pred_inds] = self.positive_label_value
            
            used_pred_indices = original_pred_inds
        else:
            used_pred_indices = torch.empty(0, dtype=torch.long, device=device)
        
        return assigned_gt_inds, assigned_labels, used_pred_indices

    def _progressive_assignment_stage2(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        cost_matrix: Tensor,
        pred_scores: Tensor,
        used_pred_indices: Tensor,
        used_gt_indices: Tensor,
        img_meta: Optional[dict] = None
    ) -> tuple[Tensor, Tensor]:
        """阶段2：中等置信度自适应分配（继承自v7.4）"""
        device = pred_scores.device
        
        _, stage2_threshold, _ = self._get_progressive_thresholds(pred_scores)
        
        all_pred_indices = torch.arange(len(pred_instances), device=device)
        available_pred_mask = torch.ones(len(pred_instances), dtype=torch.bool, device=device)
        if len(used_pred_indices) > 0:
            available_pred_mask[used_pred_indices] = False
        
        available_gt_mask = torch.ones(len(gt_instances), dtype=torch.bool, device=device)
        if len(used_gt_indices) > 0:
            available_gt_mask[used_gt_indices] = False
        
        available_pred_indices = all_pred_indices[available_pred_mask]
        available_pred_scores = pred_scores[available_pred_indices]
        
        medium_conf_mask = (available_pred_scores >= stage2_threshold) & (available_pred_scores < self._get_progressive_thresholds(pred_scores)[0])
        
        if not medium_conf_mask.any() or not available_gt_mask.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        medium_conf_pred_indices = available_pred_indices[medium_conf_mask]
        available_gt_indices = torch.where(available_gt_mask)[0]
        
        if len(used_pred_indices) > 0:
            matched_costs = cost_matrix[used_pred_indices]
            if matched_costs.numel() > 0:
                cost_threshold = matched_costs.mean() + matched_costs.std()
            else:
                cost_threshold = cost_matrix.mean()
        else:
            cost_threshold = cost_matrix.mean()
        
        new_pred_assignments = []
        new_gt_assignments = []
        
        for pred_idx in medium_conf_pred_indices:
            if len(available_gt_indices) == 0:
                break
            
            pred_costs = cost_matrix[pred_idx, available_gt_indices]
            min_cost, min_gt_local_idx = torch.min(pred_costs, dim=0)
            
            if min_cost < cost_threshold:
                gt_idx = available_gt_indices[min_gt_local_idx]
                new_pred_assignments.append(pred_idx)
                new_gt_assignments.append(gt_idx)
                
                available_gt_indices = available_gt_indices[available_gt_indices != gt_idx]
                
                if len(new_pred_assignments) >= self.max_stage2_assignments:
                    break
        
        if len(new_pred_assignments) > 0:
            new_pred_assignments = torch.tensor(new_pred_assignments, dtype=torch.long, device=device)
            new_gt_assignments = torch.tensor(new_gt_assignments, dtype=torch.long, device=device)
        else:
            new_pred_assignments = torch.empty(0, dtype=torch.long, device=device)
            new_gt_assignments = torch.empty(0, dtype=torch.long, device=device)
        
        return new_pred_assignments, new_gt_assignments

    def _progressive_assignment_stage3(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        pred_scores: Tensor,
        used_pred_indices: Tensor,
        img_meta: Optional[dict] = None
    ) -> tuple[Tensor, Tensor]:
        """阶段3：低置信度伪标签生成（改进版，加入类别平衡）"""
        device = pred_scores.device
        
        _, _, stage3_threshold = self._get_progressive_thresholds(pred_scores)
        
        all_pred_indices = torch.arange(len(pred_instances), device=device)
        available_pred_mask = torch.ones(len(pred_instances), dtype=torch.bool, device=device)
        if len(used_pred_indices) > 0:
            available_pred_mask[used_pred_indices] = False
        
        available_pred_indices = all_pred_indices[available_pred_mask]
        available_pred_scores = pred_scores[available_pred_indices]
        
        low_conf_mask = available_pred_scores >= stage3_threshold
        
        if not low_conf_mask.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        candidate_indices = available_pred_indices[low_conf_mask]
        
        # 应用语义-空间一致性过滤
        if self.enable_semantic_spatial_consistency:
            candidate_indices = self._semantic_spatial_consistency_filter(
                pred_instances, gt_instances, candidate_indices
            )
        
        if len(candidate_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        # v7.5新增：类别平衡选择
        if self.enable_class_balanced_detection:
            candidate_indices = self._class_balanced_selection(
                pred_instances, gt_instances, candidate_indices
            )
        
        # 应用时序一致性过滤
        if self.enable_temporal_consistency:
            candidate_indices = self._temporal_consistency_filter(
                pred_instances, candidate_indices, img_meta
            )
        
        if len(candidate_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        # 限制阶段3分配数量
        if len(candidate_indices) > self.max_stage3_assignments:
            candidate_scores = pred_scores[candidate_indices]
            sorted_indices = torch.argsort(candidate_scores, descending=True)
            candidate_indices = candidate_indices[sorted_indices[:self.max_stage3_assignments]]
        
        # 应用不确定性感知分离
        if self.enable_uncertainty_estimation:
            unlabeled_positive_indices, uncertain_indices = self._uncertainty_based_filter(
                pred_instances, candidate_indices
            )
        else:
            unlabeled_positive_indices = candidate_indices
            uncertain_indices = torch.empty(0, dtype=torch.long, device=device)
        
        return unlabeled_positive_indices, uncertain_indices

    # 继承其他方法（为了简洁，这里省略了重复的方法实现）
    def _estimate_uncertainty(self, pred_instances: InstanceData) -> Tensor:
        """估计预测的不确定性（继承自v7.3）"""
        if not self.enable_uncertainty_estimation:
            return torch.zeros(len(pred_instances), device=pred_instances.bboxes.device)
        
        scores = pred_instances.scores
        
        if scores.dim() == 1:
            uncertainty = 1.0 - scores
        else:
            if self.uncertainty_method == 'entropy':
                probs = F.softmax(scores, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                max_entropy = torch.log(torch.tensor(scores.size(-1), dtype=torch.float))
                uncertainty = entropy / max_entropy
            elif self.uncertainty_method == 'variance':
                probs = F.softmax(scores, dim=-1)
                variance = torch.var(probs, dim=-1)
                max_variance = 0.25
                uncertainty = torch.clamp(variance / max_variance, 0, 1)
            elif self.uncertainty_method == 'combined':
                probs = F.softmax(scores, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                max_entropy = torch.log(torch.tensor(scores.size(-1), dtype=torch.float))
                norm_entropy = entropy / max_entropy
                variance = torch.var(probs, dim=-1)
                max_variance = 0.25
                norm_variance = torch.clamp(variance / max_variance, 0, 1)
                uncertainty = 0.7 * norm_entropy + 0.3 * norm_variance
            else:
                max_scores = torch.max(scores, dim=-1)[0]
                uncertainty = 1.0 - max_scores
        
        return uncertainty

    def _uncertainty_based_filter(
        self,
        pred_instances: InstanceData,
        candidate_indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """基于不确定性的过滤（继承自v7.3）"""
        if not self.enable_uncertainty_estimation or len(candidate_indices) == 0:
            return candidate_indices, torch.empty(0, dtype=torch.long, device=candidate_indices.device)
        
        uncertainties = self._estimate_uncertainty(pred_instances)
        candidate_uncertainties = uncertainties[candidate_indices]
        
        high_uncertainty_mask = candidate_uncertainties > self.uncertainty_threshold
        low_uncertainty_mask = ~high_uncertainty_mask
        
        uncertain_indices = candidate_indices[high_uncertainty_mask]
        unlabeled_positive_indices = candidate_indices[low_uncertainty_mask]
        
        if len(uncertain_indices) > self.max_uncertain_samples:
            uncertain_scores = candidate_uncertainties[high_uncertainty_mask]
            sorted_indices = torch.argsort(uncertain_scores, descending=True)
            uncertain_indices = uncertain_indices[sorted_indices[:self.max_uncertain_samples]]
        
        return unlabeled_positive_indices, uncertain_indices

    def _compute_feature_similarity(
        self, 
        pred_instances: InstanceData, 
        gt_instances: InstanceData, 
        candidate_indices: Tensor
    ) -> Tensor:
        """计算预测特征与GT特征之间的相似性（继承自v7.2）"""
        if not (hasattr(pred_instances, 'features') and hasattr(gt_instances, 'features')):
            return torch.zeros(
                len(candidate_indices), len(gt_instances),
                device=candidate_indices.device
            )
        
        pred_features = pred_instances.features[candidate_indices]
        gt_features = gt_instances.features
        
        pred_features = F.normalize(pred_features, p=2, dim=-1)
        gt_features = F.normalize(gt_features, p=2, dim=-1)
        
        similarity = torch.mm(pred_features, gt_features.t())
        
        return similarity

    def _semantic_spatial_consistency_filter(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        candidate_indices: Tensor
    ) -> Tensor:
        """语义-空间一致性过滤（继承自v7.2）"""
        if not self.enable_semantic_spatial_consistency or len(candidate_indices) == 0:
            return candidate_indices
        
        candidate_bboxes = pred_instances.bboxes[candidate_indices]
        gt_bboxes = gt_instances.bboxes
        
        ious = self._compute_bbox_iou(candidate_bboxes, gt_bboxes)
        max_ious = torch.max(ious, dim=1)[0]
        iou_consistency = max_ious > self.consistency_iou_threshold
        
        feature_similarities = self._compute_feature_similarity(
            pred_instances, gt_instances, candidate_indices
        )
        
        if feature_similarities.numel() > 0:
            max_similarities = torch.max(feature_similarities, dim=1)[0]
            feature_scores = max_similarities
            iou_scores = max_ious
            
            combined_scores = (
                (1 - self.feature_similarity_weight) * iou_scores + 
                self.feature_similarity_weight * feature_scores
            )
            
            combined_threshold = (
                (1 - self.feature_similarity_weight) * self.consistency_iou_threshold + 
                self.feature_similarity_weight * self.consistency_feature_threshold
            )
            combined_consistency = combined_scores > combined_threshold
        else:
            combined_consistency = iou_consistency
        
        return candidate_indices[combined_consistency]

    def _update_confidence_calibration(self, pred_scores: Tensor) -> None:
        """更新动态置信度校准参数（继承自v7.1）"""
        if not self.enable_confidence_calibration or len(pred_scores) == 0:
            return
            
        current_mean = pred_scores.mean()
        current_std = pred_scores.std()
        
        if self.update_count == 0:
            self.confidence_mean = current_mean
            self.confidence_std = current_std
        else:
            self.confidence_mean = (self.confidence_ema_alpha * self.confidence_mean + 
                                   (1 - self.confidence_ema_alpha) * current_mean)
            self.confidence_std = (self.confidence_ema_alpha * self.confidence_std + 
                                  (1 - self.confidence_ema_alpha) * current_std)
        
        self.update_count += 1

    def _compute_bbox_iou(self, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """计算两组边界框之间的IoU"""
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        
        return iou

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None,
        **kwargs
    ) -> AssignResult:
        """使用类别平衡的多阶段渐进分配方法"""
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 初始化分配
        assigned_gt_inds = torch.full(
            (num_preds,), 0, dtype=torch.long, device=device
        )
        assigned_labels = torch.full(
            (num_preds,), self.background_label_value, dtype=torch.long, device=device
        )

        if num_gts == 0 or num_preds == 0:
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels
            )

        # 计算匹配成本
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta
            )
            cost_list.append(cost)
        cost_matrix = torch.stack(cost_list).sum(dim=0)
        
        # 获取预测分数
        pred_scores = pred_instances.scores
        if pred_scores.dim() > 1:
            pred_scores = torch.max(pred_scores, dim=-1)[0]
        
        # 更新动态置信度校准
        self._update_confidence_calibration(pred_scores)

        if not self.enable_progressive_assignment:
            # 使用标准匈牙利匹配
            cost_cpu = cost_matrix.detach().cpu()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(device)
            
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = self.positive_label_value
            
            # 检测未标注正样本和不确定样本
            if self.enable_unlabeled_detection:
                unlabeled_indices, uncertain_indices = self._progressive_assignment_stage3(
                    pred_instances, gt_instances, pred_scores, matched_row_inds, img_meta
                )
                
                if len(unlabeled_indices) > 0:
                    assigned_labels[unlabeled_indices] = self.unlabeled_label_value
                
                if len(uncertain_indices) > 0:
                    assigned_labels[uncertain_indices] = self.uncertain_label_value
        else:
            # 多阶段渐进分配
            
            # 阶段1：高置信度匈牙利匹配
            stage1_gt_inds, stage1_labels, used_pred_indices = self._progressive_assignment_stage1(
                pred_instances, gt_instances, cost_matrix, pred_scores, img_meta
            )
            
            # 更新分配结果
            assigned_gt_inds = stage1_gt_inds
            assigned_labels = stage1_labels
            
            # 获取已使用的GT索引
            used_gt_indices = torch.unique(assigned_gt_inds[assigned_gt_inds > 0]) - 1
            
            # 阶段2：中等置信度自适应分配
            if self.enable_unlabeled_detection:
                stage2_pred_indices, stage2_gt_indices = self._progressive_assignment_stage2(
                    pred_instances, gt_instances, cost_matrix, pred_scores, 
                    used_pred_indices, used_gt_indices, img_meta
                )
                
                if len(stage2_pred_indices) > 0:
                    assigned_gt_inds[stage2_pred_indices] = stage2_gt_indices + 1
                    assigned_labels[stage2_pred_indices] = self.positive_label_value
                    used_pred_indices = torch.cat([used_pred_indices, stage2_pred_indices])
                
                # 阶段3：低置信度伪标签生成（类别平衡）
                unlabeled_indices, uncertain_indices = self._progressive_assignment_stage3(
                    pred_instances, gt_instances, pred_scores, used_pred_indices, img_meta
                )
                
                if len(unlabeled_indices) > 0:
                    assigned_labels[unlabeled_indices] = self.unlabeled_label_value
                
                if len(uncertain_indices) > 0:
                    assigned_labels[uncertain_indices] = self.uncertain_label_value
        
        # 更新类别统计信息
        self._update_class_statistics(pred_instances, gt_instances, assigned_labels)
        
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels
        )
    
    def get_assignment_statistics(self, assign_result: AssignResult) -> dict:
        """获取分配统计信息"""
        labels = assign_result.labels
        
        stats = {
            'num_positive': (labels == self.positive_label_value).sum().item(),
            'num_background': (labels == self.background_label_value).sum().item(),
            'num_unlabeled_positive': (labels == self.unlabeled_label_value).sum().item(),
            'num_uncertain': (labels == self.uncertain_label_value).sum().item(),
            'total_predictions': len(labels),
        }
        
        stats['positive_ratio'] = stats['num_positive'] / stats['total_predictions']
        stats['unlabeled_ratio'] = stats['num_unlabeled_positive'] / stats['total_predictions']
        stats['uncertain_ratio'] = stats['num_uncertain'] / stats['total_predictions']
        
        return stats
    
    def get_class_balanced_stats(self) -> dict:
        """获取类别平衡统计信息"""
        return {
            'class_balanced_detection_enabled': self.enable_class_balanced_detection,
            'max_unlabeled_per_class': self.max_unlabeled_per_class,
            'class_balance_strategy': self.class_balance_strategy,
            'min_class_confidence': self.min_class_confidence,
            'temporal_consistency_enabled': self.enable_temporal_consistency,
            'temporal_momentum': self.temporal_momentum,
            'class_confidence_stats': self.class_confidence_stats,
            'num_tracked_classes': len(self.class_confidence_stats),
        }