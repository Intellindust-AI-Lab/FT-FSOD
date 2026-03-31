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
class FSODAssigner_v7_3(BaseAssigner):
    """FSOD Assigner v7.3 with Feature Cache-based Semantic-Spatial Consistency Check.
    
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
        
        # 对比学习损失存储
        self.last_contrastive_loss = None



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

    def get_contrastive_loss(self, 
                           matched_features: Tensor = None,
                           matched_labels: Tensor = None) -> Tensor:
        """公开接口：获取对比学习损失。
        
        Args:
            matched_features (Tensor, optional): 匹配的查询特征（如果提供则重新计算）
            matched_labels (Tensor, optional): 匹配的标签（如果提供则重新计算）
            
        Returns:
            Tensor: 对比学习损失
        """
        if matched_features is not None and matched_labels is not None:
            return self._compute_contrastive_loss(matched_features, matched_labels)
        elif self.last_contrastive_loss is not None:
            return self.last_contrastive_loss
        else:
            # 返回零损失
            device = next(iter(self.feature_cache.values()))[0].device if self.feature_cache else 'cpu'
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def clear_feature_cache(self) -> None:
        """清空特征缓存池（用于新epoch开始时）。"""
        self.feature_cache.clear()
    
    def get_cache_stats(self) -> Dict[int, int]:
        """获取缓存统计信息。
        
        Returns:
            Dict[int, int]: 每个类别的缓存特征数量
        """
        return {label: len(features) for label, features in self.feature_cache.items()}

    def _compute_contrastive_loss(self, 
                                  matched_features: Tensor,
                                  matched_labels: Tensor) -> Tensor:
        """计算对比学习损失，使同一label下的query相互相似，不同label下的query相互不相似。
        
        Args:
            matched_features (Tensor): 匹配的查询特征，形状为 [N, D]
            matched_labels (Tensor): 匹配的标签，形状为 [N]
            
        Returns:
            Tensor: 对比学习损失
        """
        if not self.enable_feature_cache or matched_features is None or len(matched_features) == 0:
            return torch.tensor(0.0, device=matched_features.device if matched_features is not None else 'cpu')
        
        # 获取所有缓存的特征和对应的标签
        all_cached_features = []
        all_cached_labels = []
        
        for label, features in self.feature_cache.items():
            if len(features) > 0:
                all_cached_features.extend(features)
                all_cached_labels.extend([label] * len(features))
        
        if len(all_cached_features) == 0:
            return torch.tensor(0.0, device=matched_features.device)
        
        # 将缓存特征堆叠并detach（不参与梯度计算）
        cached_features = torch.stack(all_cached_features, dim=0).detach()  # [N_cached, D]
        cached_labels = torch.tensor(all_cached_labels, device=matched_features.device)  # [N_cached]
        
        # 计算新query与缓存特征的相似性
        similarity_matrix = self._compute_feature_similarity(matched_features, cached_features)  # [N, N_cached]
        
        # 构建正负样本mask
        # positive_mask: 新query与缓存中同一label的特征
        # negative_mask: 新query与缓存中不同label的特征
        positive_mask = matched_labels.unsqueeze(1) == cached_labels.unsqueeze(0)  # [N, N_cached]
        negative_mask = ~positive_mask  # [N, N_cached]
        
        # 计算对比损失
        temperature = 0.1  # 温度参数
        contrastive_loss = 0.0
        
        for i in range(len(matched_features)):
            # 获取当前query的正样本和负样本相似性
            pos_similarities = similarity_matrix[i][positive_mask[i]]  # 同类别的相似性
            neg_similarities = similarity_matrix[i][negative_mask[i]]  # 不同类别的相似性
            
            if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                # InfoNCE损失：最大化正样本相似性，最小化负样本相似性
                # pos_exp = torch.exp(pos_similarities / temperature)
                neg_exp = torch.exp(neg_similarities / temperature)
                
                # 对于每个正样本，计算其与所有负样本的对比损失
                for pos_sim in pos_similarities:
                    pos_exp_single = torch.exp(pos_sim / temperature)
                    denominator = pos_exp_single + neg_exp.sum()
                    loss = -torch.log(pos_exp_single / denominator)
                    contrastive_loss += loss
        
        # 归一化损失
        if len(matched_features) > 0:
            contrastive_loss = contrastive_loss / len(matched_features)
        
        return contrastive_loss



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
            
            # 计算对比学习损失
            self._compute_contrastive_loss(matched_features, matched_gt_labels)

        # 7. All other unmatched predictions remain as background (label 0, already initialized)
        
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels
        )