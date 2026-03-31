# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from collections import defaultdict
from typing import Dict, List

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class FSODAssigner_v7_1(BaseAssigner):
    """FSOD Assigner v7.1 with Feature Cache-based Semantic-Spatial Consistency Check.
    
    This assigner performs standard Hungarian matching identical to HungarianAssigner,
    while additionally identifying potential unlabeled positives with semantic-spatial consistency:
    1. Standard 1-to-1 Hungarian matching for labeled positive samples
    2. Identify potential unlabeled positives based on confidence and IoU constraints
    3. Use a three-class labeling system: 1 (matched queries), 0 (background), -1 (potential unlabeled positive)
    4. Assign potential unlabeled positives based on: unmatched + high confidence + low IoU with GT
    5. NEW: Feature cache-based semantic-spatial consistency check combining IoU and feature similarity
    
    Key Features:
    - Feature Cache Pool: Maintains cached features from matched queries per class
    - FIFO Cache Limitation: Each class caches at most 10 features, automatically removing oldest when exceeded
    - Semantic Similarity: Uses cached features instead of GT features for similarity comparison
    - Spatial Consistency: Ensures low IoU with GT to avoid overlap with labeled samples
    - Automatic Memory Management: FIFO mechanism controls cache size to prevent unlimited memory growth
    
    The assignment process includes:
     1. Standard Hungarian matching for labeled GT-prediction pairs (assigned label 1)
     2. Feature cache update with matched query features for each class
     3. Detection of potential unlabeled positives: unmatched queries with confidence higher than
        max confidence of matched queries AND IoU with GT lower than min IoU of matched queries
     4. Semantic-spatial consistency filtering using cached features and IoU constraints
     5. All other unmatched queries are assigned as background (label 0)
    
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
        enable_semantic_spatial_check (bool): Whether to enable semantic-spatial
            consistency check using feature cache. Default: True.
        feature_similarity_threshold (float): Threshold for feature similarity
            in consistency check. Default: 0.5.
        semantic_spatial_weight (float): Weight for combining IoU and feature
            similarity in consistency score. Default: 0.5.
        max_cached_features_per_class (int): Maximum number of cached features
            per class in the feature cache pool. Default: 10.
        feature_cache_update_strategy (str): Strategy for updating feature cache
            ('fifo' or 'replace'). Default: 'fifo'.
        enable_feature_cache (bool): Whether to enable feature caching mechanism.
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
        # 语义-空间一致性检查参数
        enable_semantic_spatial_check: bool = True,
        feature_similarity_threshold: float = 0.5,
        semantic_spatial_weight: float = 0.5,
        # 特征缓存池参数
        max_cached_features_per_class: int = 10,
        feature_cache_update_strategy: str = 'fifo',  # 'fifo' or 'replace'
        enable_feature_cache: bool = True,
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
        
        # 语义-空间一致性检查参数
        self.enable_semantic_spatial_check = enable_semantic_spatial_check
        self.feature_similarity_threshold = feature_similarity_threshold
        self.semantic_spatial_weight = semantic_spatial_weight
        
        # 特征缓存池参数
        self.max_cached_features_per_class = max_cached_features_per_class
        self.feature_cache_update_strategy = feature_cache_update_strategy
        self.enable_feature_cache = enable_feature_cache
        
        # 初始化特征缓存池：每个类别维护一个特征列表
        self.feature_cache: Dict[int, List[Tensor]] = defaultdict(list)

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

    def _compute_feature_similarity(self, features1: Tensor, features2: Tensor) -> Tensor:
        """计算特征相似性。
        
        Args:
            features1 (Tensor): 第一组特征，形状为 [N, D]
            features2 (Tensor): 第二组特征，形状为 [M, D]
            
        Returns:
            Tensor: 特征相似性矩阵，形状为 [N, M]
        """
        # L2归一化
        features1_norm = F.normalize(features1, p=2, dim=-1)
        features2_norm = F.normalize(features2, p=2, dim=-1)
        
        # 计算余弦相似性
        similarity = torch.mm(features1_norm, features2_norm.t())
        
        return similarity

    def _update_feature_cache(self, features: Tensor, labels: Tensor) -> None:
        """批量更新特征缓存池。
        
        Args:
            features (Tensor): 特征向量，形状为 [N, D]
            labels (Tensor): 类别标签，形状为 [N]
        """
        if not self.enable_feature_cache:
            return
            
        # 批量更新每个类别的特征缓存
        for feature, label in zip(features, labels):
            label_item = label.item()
            
            if self.feature_cache_update_strategy == 'fifo':
                # FIFO策略：先进先出
                if len(self.feature_cache[label_item]) >= self.max_cached_features_per_class:
                    self.feature_cache[label_item].pop(0)  # 移除最旧的特征
                self.feature_cache[label_item].append(feature)
            elif self.feature_cache_update_strategy == 'replace':
                # 替换策略：完全替换当前epoch的特征
                if len(self.feature_cache[label_item]) >= self.max_cached_features_per_class:
                    self.feature_cache[label_item] = [feature]
                else:
                    self.feature_cache[label_item].append(feature)
    
    def _get_cached_features_for_classes(self, class_labels: Tensor) -> Tensor:
        """获取指定类别的缓存特征。
        
        Args:
            class_labels (Tensor): 需要获取特征的类别标签
            
        Returns:
            Tensor: 缓存的特征，形状为 [N_cached, D]，如果没有缓存则返回None
        """
        if not self.enable_feature_cache:
            return None
            
        all_cached_features = []
        unique_labels = torch.unique(class_labels)
        
        for label in unique_labels:
            label_item = label.item()
            if label_item in self.feature_cache and len(self.feature_cache[label_item]) > 0:
                # 将该类别的所有缓存特征添加到列表中
                all_cached_features.extend(self.feature_cache[label_item])
        
        if len(all_cached_features) == 0:
            return None
            
        # 将所有特征堆叠成一个张量
        return torch.stack(all_cached_features, dim=0)
    
    def clear_feature_cache(self) -> None:
        """清空特征缓存池（用于新epoch开始时）。"""
        self.feature_cache.clear()
    
    def get_cache_stats(self) -> Dict[int, int]:
        """获取缓存统计信息。
        
        Returns:
            Dict[int, int]: 每个类别的缓存特征数量
        """
        return {label: len(features) for label, features in self.feature_cache.items()}

    def _semantic_spatial_consistency_filter(self, 
                                           unmatched_indices: Tensor,
                                           pred_bboxes: Tensor,
                                           pred_scores: Tensor,
                                           pred_features: Optional[Tensor],
                                           gt_bboxes: Tensor,
                                           gt_labels: Tensor,
                                           matched_pred_indices: Tensor) -> Tensor:
        """基于缓存特征池的语义-空间一致性过滤。
        
        针对未标注正样本检测，使用缓存的匹配查询特征进行相似性比较：
        - IoU较小：避免与已标注GT产生空间重叠
        - 特征相似性较高：与历史匹配查询特征相似（语义相关）
        - 一致性分数 = feature_similarity - α * IoU
        - 使用缓存特征池替代GT特征，更符合实际FSOD场景
        
        Args:
            unmatched_indices (Tensor): 未匹配的预测索引
            pred_bboxes (Tensor): 预测边界框
            pred_scores (Tensor): 预测置信度
            pred_features (Tensor, optional): 预测特征
            gt_bboxes (Tensor): GT边界框
            gt_labels (Tensor): GT标签
            matched_pred_indices (Tensor): 已匹配的预测索引
            
        Returns:
            Tensor: 通过一致性检查的未匹配索引
        """
        if len(unmatched_indices) == 0:
            return unmatched_indices
            
        # 如果没有特征或未启用语义-空间检查，直接返回
        if (not self.enable_semantic_spatial_check or pred_features is None):
            return unmatched_indices
        
        # 获取缓存的特征池
        cached_features = self._get_cached_features_for_classes(gt_labels)
        if cached_features is None or len(cached_features) == 0:
            # 如果没有缓存特征，跳过语义检查，只返回原始候选
            return unmatched_indices
            
        # 计算未匹配预测与GT的IoU
        # unmatched_bboxes = pred_bboxes[unmatched_indices]
        # ious = self._compute_bbox_iou(unmatched_bboxes, gt_bboxes)  # [num_unmatched, num_gt]
        
        # 计算未匹配查询与缓存特征的相似性
        unmatched_features = pred_features[unmatched_indices]
        feature_sim_with_cache = self._compute_feature_similarity(unmatched_features, cached_features)  # [num_unmatched, num_cached]
        
        # 对每个未匹配查询，取与缓存特征的最大相似性作为语义相关性指标
        max_feature_sim, _ = feature_sim_with_cache.max(dim=1)  # [num_unmatched]
        
        # 对每个未匹配查询，取与GT的最大IoU作为空间重叠指标
        # max_ious_with_gt, _ = ious.max(dim=1)  # [num_unmatched]
        
        # 结合IoU和特征相似性进行未标注正样本检测
        # 对于未标注正样本：IoU应该较小（避免与GT重叠），特征相似性应该较高（语义相关）
        # 一致性分数 = feature_similarity - α * IoU （IoU越小，特征相似性越高，分数越高）
        # alpha = self.semantic_spatial_weight
        # consistency_scores = max_feature_sim - alpha * max_ious_with_gt
        # consistency_scores = max_feature_sim
        
        # 过滤：保留一致性分数高于阈值的预测
        # 这意味着与缓存特征相似性高且IoU相对较小的预测会被保留
        # consistency_mask = consistency_scores > self.feature_similarity_threshold
        consistency_mask = max_feature_sim > self.feature_similarity_threshold
        
        filtered_indices = unmatched_indices[consistency_mask]
        
        return filtered_indices

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
        
        # Apply semantic-spatial consistency filtering
        if self.enable_semantic_spatial_check and len(final_candidate_indices) > 0:
            # Extract features if available
            pred_features = getattr(pred_instances, 'features', None)
            
            final_candidate_indices = self._semantic_spatial_consistency_filter(
                final_candidate_indices,
                pred_bboxes,
                pred_scores,
                pred_features,
                gt_bboxes,
                gt_instances.labels,
                matched_indices
            )
        
        # Apply spatial NMS to remove overlapping candidates
        print('len(final_candidate_indices)', len(final_candidate_indices))
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
        
        # 6. Update feature cache with matched query features (if enabled)
        if (self.enable_feature_cache and len(matched_row_inds) > 0 and 
            hasattr(pred_instances, 'features') and pred_instances.features is not None):
            matched_features = pred_instances.features[matched_row_inds]  # [num_matched, feature_dim]
            matched_gt_labels = gt_labels[matched_col_inds]  # [num_matched]
            
            # Update cache with matched features and labels
            self._update_feature_cache(matched_features, matched_gt_labels)

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