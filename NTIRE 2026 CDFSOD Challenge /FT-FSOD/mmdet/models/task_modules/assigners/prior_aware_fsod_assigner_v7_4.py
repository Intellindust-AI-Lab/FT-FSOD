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
class FSODAssigner_v7_4(BaseAssigner):
    """FSOD Assigner v7.4 - 添加多阶段渐进分配机制
    
    基于v7.3版本，新增功能：
    1. 继承v7.1的动态置信度校准
    2. 继承v7.2的语义-空间一致性检查
    3. 继承v7.3的不确定性感知分配
    4. 多阶段渐进分配：
       - 阶段1：高置信度匈牙利匹配
       - 阶段2：中等置信度自适应分配
       - 阶段3：低置信度伪标签生成
    5. 渐进式阈值调整
    
    验证目标：证明多阶段渐进分配能够更充分地利用预测信息
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
        stage1_confidence_threshold: float = 0.8,  # 高置信度阈值
        stage2_confidence_threshold: float = 0.5,  # 中等置信度阈值
        stage3_confidence_threshold: float = 0.3,  # 低置信度阈值
        progressive_cost_weight: float = 0.7,      # 渐进成本权重
        max_stage2_assignments: int = 15,          # 阶段2最大分配数
        max_stage3_assignments: int = 20,          # 阶段3最大分配数
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
        
        # 初始化置信度统计
        self.register_buffer('confidence_mean', torch.tensor(0.5))
        self.register_buffer('confidence_std', torch.tensor(0.2))
        self.register_buffer('update_count', torch.tensor(0))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区，确保参数在正确设备上"""
        setattr(self, name, tensor)

    def _get_progressive_thresholds(self, pred_scores: Tensor) -> tuple[float, float, float]:
        """获取渐进式阈值
        
        根据当前预测分数分布动态调整三个阶段的置信度阈值
        
        Args:
            pred_scores (Tensor): 预测分数
            
        Returns:
            tuple[float, float, float]: (阶段1阈值, 阶段2阈值, 阶段3阈值)
        """
        if not self.enable_progressive_assignment:
            return self.stage1_confidence_threshold, self.stage2_confidence_threshold, self.stage3_confidence_threshold
        
        # 基于分数分布调整阈值
        score_mean = pred_scores.mean().item()
        score_std = pred_scores.std().item()
        score_max = pred_scores.max().item()
        score_min = pred_scores.min().item()
        
        # 动态调整阈值
        if self.enable_confidence_calibration and self.update_count > 0:
            # 使用校准后的统计信息
            calibrated_mean = self.confidence_mean.item()
            calibrated_std = self.confidence_std.item()
            
            # 阶段1：高置信度（均值 + 1.5 * 标准差）
            stage1_threshold = min(calibrated_mean + 1.5 * calibrated_std, score_max * 0.95)
            
            # 阶段2：中等置信度（均值 + 0.5 * 标准差）
            stage2_threshold = min(calibrated_mean + 0.5 * calibrated_std, stage1_threshold * 0.8)
            
            # 阶段3：低置信度（均值 - 0.5 * 标准差）
            stage3_threshold = max(calibrated_mean - 0.5 * calibrated_std, score_min * 1.1)
        else:
            # 使用固定阈值
            stage1_threshold = self.stage1_confidence_threshold
            stage2_threshold = self.stage2_confidence_threshold
            stage3_threshold = self.stage3_confidence_threshold
        
        # 确保阈值顺序正确
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
        """阶段1：高置信度匈牙利匹配
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            cost_matrix (Tensor): 成本矩阵
            pred_scores (Tensor): 预测分数
            img_meta (Optional[dict]): 图像元信息
            
        Returns:
            tuple[Tensor, Tensor, Tensor]: (分配的GT索引, 分配的标签, 已使用的预测索引)
        """
        num_preds = len(pred_instances)
        device = pred_scores.device
        
        # 初始化分配结果
        assigned_gt_inds = torch.zeros(num_preds, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_preds,), self.background_label_value, dtype=torch.long, device=device)
        
        # 获取渐进式阈值
        stage1_threshold, _, _ = self._get_progressive_thresholds(pred_scores)
        
        # 筛选高置信度预测
        high_conf_mask = pred_scores >= stage1_threshold
        if not high_conf_mask.any():
            return assigned_gt_inds, assigned_labels, torch.empty(0, dtype=torch.long, device=device)
        
        high_conf_indices = torch.where(high_conf_mask)[0]
        
        # 构建高置信度成本矩阵
        high_conf_cost = cost_matrix[high_conf_indices]
        
        # 匈牙利匹配
        if len(high_conf_indices) > 0 and len(gt_instances) > 0:
            cost_cpu = high_conf_cost.detach().cpu().numpy()
            matched_pred_inds, matched_gt_inds = linear_sum_assignment(cost_cpu)
            
            # 转换回tensor
            matched_pred_inds = torch.from_numpy(matched_pred_inds).to(device)
            matched_gt_inds = torch.from_numpy(matched_gt_inds).to(device)
            
            # 映射回原始索引
            original_pred_inds = high_conf_indices[matched_pred_inds]
            
            # 分配结果
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
        """阶段2：中等置信度自适应分配
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            cost_matrix (Tensor): 成本矩阵
            pred_scores (Tensor): 预测分数
            used_pred_indices (Tensor): 已使用的预测索引
            used_gt_indices (Tensor): 已使用的GT索引
            img_meta (Optional[dict]): 图像元信息
            
        Returns:
            tuple[Tensor, Tensor]: (新分配的预测索引, 对应的GT索引)
        """
        device = pred_scores.device
        
        # 获取渐进式阈值
        _, stage2_threshold, _ = self._get_progressive_thresholds(pred_scores)
        
        # 找到可用的预测和GT
        all_pred_indices = torch.arange(len(pred_instances), device=device)
        available_pred_mask = torch.ones(len(pred_instances), dtype=torch.bool, device=device)
        if len(used_pred_indices) > 0:
            available_pred_mask[used_pred_indices] = False
        
        available_gt_mask = torch.ones(len(gt_instances), dtype=torch.bool, device=device)
        if len(used_gt_indices) > 0:
            available_gt_mask[used_gt_indices] = False
        
        # 筛选中等置信度的可用预测
        available_pred_indices = all_pred_indices[available_pred_mask]
        available_pred_scores = pred_scores[available_pred_indices]
        
        medium_conf_mask = (available_pred_scores >= stage2_threshold) & (available_pred_scores < self._get_progressive_thresholds(pred_scores)[0])
        
        if not medium_conf_mask.any() or not available_gt_mask.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        medium_conf_pred_indices = available_pred_indices[medium_conf_mask]
        available_gt_indices = torch.where(available_gt_mask)[0]
        
        # 计算自适应成本阈值
        if len(used_pred_indices) > 0:
            # 基于已匹配样本的成本统计
            matched_costs = cost_matrix[used_pred_indices]
            if matched_costs.numel() > 0:
                cost_threshold = matched_costs.mean() + matched_costs.std()
            else:
                cost_threshold = cost_matrix.mean()
        else:
            cost_threshold = cost_matrix.mean()
        
        # 为每个中等置信度预测找到最佳GT匹配
        new_pred_assignments = []
        new_gt_assignments = []
        
        for pred_idx in medium_conf_pred_indices:
            if len(available_gt_indices) == 0:
                break
            
            # 计算与可用GT的成本
            pred_costs = cost_matrix[pred_idx, available_gt_indices]
            
            # 找到最低成本的GT
            min_cost, min_gt_local_idx = torch.min(pred_costs, dim=0)
            
            # 检查成本是否低于阈值
            if min_cost < cost_threshold:
                gt_idx = available_gt_indices[min_gt_local_idx]
                new_pred_assignments.append(pred_idx)
                new_gt_assignments.append(gt_idx)
                
                # 移除已使用的GT
                available_gt_indices = available_gt_indices[available_gt_indices != gt_idx]
                
                # 限制阶段2分配数量
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
        """阶段3：低置信度伪标签生成
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            pred_scores (Tensor): 预测分数
            used_pred_indices (Tensor): 已使用的预测索引
            img_meta (Optional[dict]): 图像元信息
            
        Returns:
            tuple[Tensor, Tensor]: (未标注正样本索引, 不确定样本索引)
        """
        device = pred_scores.device
        
        # 获取渐进式阈值
        _, _, stage3_threshold = self._get_progressive_thresholds(pred_scores)
        
        # 找到可用的预测
        all_pred_indices = torch.arange(len(pred_instances), device=device)
        available_pred_mask = torch.ones(len(pred_instances), dtype=torch.bool, device=device)
        if len(used_pred_indices) > 0:
            available_pred_mask[used_pred_indices] = False
        
        available_pred_indices = all_pred_indices[available_pred_mask]
        available_pred_scores = pred_scores[available_pred_indices]
        
        # 筛选低置信度预测
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

    def _spatial_nms(self, bboxes: Tensor, scores: Tensor, threshold: float, labels: Tensor = None) -> Tensor:
        """应用空间NMS去除重叠检测"""
        if len(bboxes) == 0:
            return torch.empty(0, dtype=torch.long, device=bboxes.device)
        
        if labels is None:
            sorted_indices = torch.argsort(scores, descending=True)
            
            keep = []
            while len(sorted_indices) > 0:
                current = sorted_indices[0]
                keep.append(current)
                
                if len(sorted_indices) == 1:
                    break
                    
                current_box = bboxes[current:current+1]
                remaining_boxes = bboxes[sorted_indices[1:]]
                ious = self._compute_bbox_iou(current_box, remaining_boxes)[0]
                
                keep_mask = ious < threshold
                sorted_indices = sorted_indices[1:][keep_mask]
            
            return torch.tensor(keep, dtype=torch.long, device=bboxes.device)
        else:
            keep = []
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                class_mask = labels == label
                class_indices = torch.where(class_mask)[0]
                
                if len(class_indices) == 0:
                    continue
                
                class_bboxes = bboxes[class_indices]
                class_scores = scores[class_indices]
                
                sorted_indices = torch.argsort(class_scores, descending=True)
                class_sorted_indices = class_indices[sorted_indices]
                
                class_keep = []
                while len(class_sorted_indices) > 0:
                    current = class_sorted_indices[0]
                    class_keep.append(current)
                    
                    if len(class_sorted_indices) == 1:
                        break
                        
                    current_box = bboxes[current:current+1]
                    remaining_boxes = bboxes[class_sorted_indices[1:]]
                    ious = self._compute_bbox_iou(current_box, remaining_boxes)[0]
                    
                    keep_mask = ious < threshold
                    class_sorted_indices = class_sorted_indices[1:][keep_mask]
                
                keep.extend(class_keep)
            
            if len(keep) > 0:
                keep = torch.tensor(keep, dtype=torch.long, device=bboxes.device)
                keep = torch.sort(keep)[0]
                return keep
            else:
                return torch.empty(0, dtype=torch.long, device=bboxes.device)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None,
        **kwargs
    ) -> AssignResult:
        """使用多阶段渐进分配的方法"""
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
                
                # 阶段3：低置信度伪标签生成
                unlabeled_indices, uncertain_indices = self._progressive_assignment_stage3(
                    pred_instances, gt_instances, pred_scores, used_pred_indices, img_meta
                )
                
                if len(unlabeled_indices) > 0:
                    assigned_labels[unlabeled_indices] = self.unlabeled_label_value
                
                if len(uncertain_indices) > 0:
                    assigned_labels[uncertain_indices] = self.uncertain_label_value
        
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
    
    def get_progressive_stats(self) -> dict:
        """获取渐进分配统计信息"""
        return {
            'progressive_assignment_enabled': self.enable_progressive_assignment,
            'stage1_threshold': self.stage1_confidence_threshold,
            'stage2_threshold': self.stage2_confidence_threshold,
            'stage3_threshold': self.stage3_confidence_threshold,
            'max_stage2_assignments': self.max_stage2_assignments,
            'max_stage3_assignments': self.max_stage3_assignments,
            'confidence_mean': self.confidence_mean.item(),
            'confidence_std': self.confidence_std.item(),
            'update_count': self.update_count.item(),
        }