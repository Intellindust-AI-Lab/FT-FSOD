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
class FSODAssigner_v7_2(BaseAssigner):
    """FSOD Assigner v7.2 - 添加语义-空间一致性检查功能
    
    基于v7.1版本，新增功能：
    1. 继承v7.1的动态置信度校准
    2. 语义-空间一致性检查：结合IoU和特征相似性
    3. 特征相似性计算（余弦相似度）
    4. 多模态过滤机制
    
    验证目标：证明语义-空间一致性检查能够提高未标注正样本的质量
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
        
        # 初始化置信度统计
        self.register_buffer('confidence_mean', torch.tensor(0.5))
        self.register_buffer('confidence_std', torch.tensor(0.2))
        self.register_buffer('update_count', torch.tensor(0))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区，确保参数在正确设备上"""
        setattr(self, name, tensor)

    def _compute_feature_similarity(
        self, 
        pred_instances: InstanceData, 
        gt_instances: InstanceData, 
        candidate_indices: Tensor
    ) -> Tensor:
        """计算预测特征与GT特征之间的相似性
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            candidate_indices (Tensor): 候选预测索引
            
        Returns:
            Tensor: 特征相似性矩阵，形状为 (N_candidates, N_gt)
        """
        if not (hasattr(pred_instances, 'features') and hasattr(gt_instances, 'features')):
            # 如果没有特征，返回全零矩阵（不影响IoU过滤）
            return torch.zeros(
                len(candidate_indices), len(gt_instances),
                device=candidate_indices.device
            )
        
        pred_features = pred_instances.features[candidate_indices]  # (N_candidates, D)
        gt_features = gt_instances.features  # (N_gt, D)
        
        # L2归一化
        pred_features = F.normalize(pred_features, p=2, dim=-1)
        gt_features = F.normalize(gt_features, p=2, dim=-1)
        
        # 计算余弦相似度
        similarity = torch.mm(pred_features, gt_features.t())  # (N_candidates, N_gt)
        
        return similarity

    def _semantic_spatial_consistency_filter(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        candidate_indices: Tensor
    ) -> Tensor:
        """语义-空间一致性过滤
        
        结合IoU和特征相似性进行过滤，确保候选样本在空间和语义上都与GT一致
        
        Args:
            pred_instances (InstanceData): 预测实例
            gt_instances (InstanceData): GT实例
            candidate_indices (Tensor): 候选预测索引
            
        Returns:
            Tensor: 过滤后的候选索引
        """
        if not self.enable_semantic_spatial_consistency or len(candidate_indices) == 0:
            return candidate_indices
        
        candidate_bboxes = pred_instances.bboxes[candidate_indices]
        gt_bboxes = gt_instances.bboxes
        
        # 计算IoU一致性
        ious = self._compute_bbox_iou(candidate_bboxes, gt_bboxes)  # (N_candidates, N_gt)
        max_ious = torch.max(ious, dim=1)[0]  # (N_candidates,)
        iou_consistency = max_ious > self.consistency_iou_threshold
        
        # 计算特征相似性一致性
        feature_similarities = self._compute_feature_similarity(
            pred_instances, gt_instances, candidate_indices
        )  # (N_candidates, N_gt)
        
        if feature_similarities.numel() > 0:
            max_similarities = torch.max(feature_similarities, dim=1)[0]  # (N_candidates,)
            feature_consistency = max_similarities > self.consistency_feature_threshold
            
            # 结合IoU和特征相似性
            # 策略1: 两者都满足（严格模式）
            # combined_consistency = iou_consistency & feature_consistency
            
            # 策略2: 加权组合（更灵活）
            iou_scores = max_ious
            feature_scores = max_similarities
            combined_scores = (
                (1 - self.feature_similarity_weight) * iou_scores + 
                self.feature_similarity_weight * feature_scores
            )
            
            # 使用动态阈值
            combined_threshold = (
                (1 - self.feature_similarity_weight) * self.consistency_iou_threshold + 
                self.feature_similarity_weight * self.consistency_feature_threshold
            )
            combined_consistency = combined_scores > combined_threshold
        else:
            # 如果没有特征，只使用IoU
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

    def _get_adaptive_confidence_threshold(self, matched_scores: Tensor) -> float:
        """获取自适应置信度阈值（继承自v7.1）"""
        if not self.enable_confidence_calibration:
            return torch.max(matched_scores).item()
        
        base_threshold = self.confidence_mean + self.adaptive_threshold_factor * self.confidence_std
        matched_max = torch.max(matched_scores).item()
        matched_mean = torch.mean(matched_scores).item()
        
        adaptive_threshold = 0.6 * base_threshold.item() + 0.4 * matched_mean
        min_matched = torch.min(matched_scores).item()
        adaptive_threshold = max(adaptive_threshold, min_matched)
        
        return adaptive_threshold

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

    def _detect_unlabeled_positives(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        assigned_gt_inds: Tensor,
        img_meta: Optional[dict] = None
    ) -> Tensor:
        """检测潜在的未标注正样本（使用语义-空间一致性检查）
        
        改进点：
        1. 继承v7.1的动态置信度校准
        2. 新增语义-空间一致性过滤
        3. 结合IoU和特征相似性进行质量控制
        """
        if not self.enable_unlabeled_detection:
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
            
        if not hasattr(pred_instances, 'scores'):
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
            
        pred_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        gt_bboxes = gt_instances.bboxes
        
        if pred_scores.dim() > 1:
            pred_scores = torch.max(pred_scores, dim=-1)[0]
        
        # 找到已匹配的预测
        matched_mask = assigned_gt_inds > 0
        if not matched_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        matched_indices = torch.where(matched_mask)[0]
        matched_scores = pred_scores[matched_indices]
        matched_bboxes = pred_bboxes[matched_indices]
        
        # 使用动态置信度校准获取自适应阈值
        adaptive_confidence_threshold = self._get_adaptive_confidence_threshold(matched_scores)
        
        # 计算IoU阈值
        matched_gt_indices = assigned_gt_inds[matched_indices] - 1
        matched_gt_bboxes = gt_bboxes[matched_gt_indices]
        matched_ious = self._compute_bbox_iou(matched_bboxes, matched_gt_bboxes)
        matched_ious_diag = torch.diag(matched_ious)
        min_matched_iou = torch.min(matched_ious_diag).item()
        
        # 找到未分配的预测
        unassigned_mask = assigned_gt_inds == 0
        if not unassigned_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        unassigned_indices = torch.where(unassigned_mask)[0]
        unassigned_bboxes = pred_bboxes[unassigned_indices]
        unassigned_scores = pred_scores[unassigned_indices]
        
        # 使用自适应置信度阈值进行过滤
        high_conf_mask = unassigned_scores >= adaptive_confidence_threshold
        if not high_conf_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        candidate_indices = unassigned_indices[high_conf_mask]
        candidate_bboxes = unassigned_bboxes[high_conf_mask]
        
        # 计算与GT的IoU并过滤
        ious = self._compute_bbox_iou(candidate_bboxes, gt_bboxes)
        max_ious_with_gt = torch.max(ious, dim=1)[0]
        
        low_iou_mask = max_ious_with_gt < min_matched_iou
        if not low_iou_mask.any():
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        candidate_indices = candidate_indices[low_iou_mask]
        
        # v7.2新增：语义-空间一致性过滤
        candidate_indices = self._semantic_spatial_consistency_filter(
            pred_instances, gt_instances, candidate_indices
        )
        
        if len(candidate_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=assigned_gt_inds.device)
        
        # 应用空间NMS
        final_candidate_bboxes = pred_bboxes[candidate_indices]
        final_candidate_scores = pred_scores[candidate_indices]
        
        final_candidate_labels = None
        if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
            candidate_scores = pred_instances.scores[candidate_indices]
            if candidate_scores.dim() > 1 and candidate_scores.size(-1) > 1:
                final_candidate_labels = torch.argmax(candidate_scores, dim=-1)
            elif hasattr(pred_instances, 'labels') and pred_instances.labels is not None:
                final_candidate_labels = pred_instances.labels[candidate_indices]
        
        if self.use_nms:
            keep_indices = self._spatial_nms(
                final_candidate_bboxes, final_candidate_scores, self.spatial_nms_threshold, final_candidate_labels
            )
            candidate_indices = candidate_indices[keep_indices]
        
        # 限制未标注正样本数量
        if len(candidate_indices) > self.max_unlabeled_positives:
            final_scores = pred_scores[candidate_indices]
            sorted_indices = torch.argsort(final_scores, descending=True)
            candidate_indices = candidate_indices[sorted_indices[:self.max_unlabeled_positives]]
        
        return candidate_indices

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        img_meta: Optional[dict] = None,
        **kwargs
    ) -> AssignResult:
        """使用语义-空间一致性检查的分配方法"""
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
        standard_cost = torch.stack(cost_list).sum(dim=0)
        
        # 更新动态置信度校准
        pred_scores = pred_instances.scores
        if pred_scores.dim() > 1:
            pred_scores = torch.max(pred_scores, dim=-1)[0]
        self._update_confidence_calibration(pred_scores)

        # 匈牙利匹配
        cost_cpu = standard_cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 分配匹配的正样本
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = self.positive_label_value

        # 检测未标注正样本（使用语义-空间一致性检查）
        unlabeled_indices = self._detect_unlabeled_positives(
            pred_instances, gt_instances, assigned_gt_inds, img_meta
        )
        if len(unlabeled_indices) > 0:
            assigned_labels[unlabeled_indices] = self.unlabeled_label_value
        
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
        """获取潜在的未标注正样本"""
        assign_result = self.assign(pred_instances, gt_instances, img_meta)
        unlabeled_mask = assign_result.labels == self.unlabeled_label_value
        unlabeled_indices = torch.where(unlabeled_mask)[0]
        return unlabeled_indices
    
    def get_calibration_stats(self) -> dict:
        """获取置信度校准统计信息"""
        return {
            'confidence_mean': self.confidence_mean.item(),
            'confidence_std': self.confidence_std.item(),
            'update_count': self.update_count.item(),
            'calibration_enabled': self.enable_confidence_calibration,
            'semantic_spatial_consistency_enabled': self.enable_semantic_spatial_consistency
        }