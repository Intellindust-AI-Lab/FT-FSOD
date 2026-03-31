# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union, Dict, List

from .utils import MLP, coordinate_to_encoding, inverse_sigmoid

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed
import random

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class GroundingDinoTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text


class GroundingDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


class GroundingDinoTransformerDecoder_parallel_Choice1(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]

        if reference_points.shape[-1] == 4:
            reference_points_input = \
                reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = \
                reference_points[:, :, None] * valid_ratios[:, None]
            
        query_sine_embed = coordinate_to_encoding(
            reference_points_input[:, :, 0, :])
        query_pos = self.ref_point_head(query_sine_embed)
        
        for lid, layer in enumerate(self.layers):

            query_output = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query_output)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                # reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query_output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 保存第一层（lid=0）的输出，用于后续parallel层
                query_after_layer0 = query.clone()
                reference_points_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 不进行任何dropout

                parallel_query = layer(
                    query_after_layer0,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer0[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer0[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer0.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer0, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15_dropout(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def _apply_negative_query_dropout(self, query: Tensor, dn_meta: Dict = None, 
                                     assigned_gt_inds: Tensor = None, dropout_p: float = 0.1) -> Tensor:
        """对负样本query进行dropout
        
        Args:
            query (Tensor): Input queries with shape (bs, num_queries, dim)
            dn_meta (Dict, optional): DN meta information
            assigned_gt_inds (Tensor, optional): Assignment results with shape (bs, num_queries),
                                               where 0 means negative, >0 means positive
            dropout_p (float): Dropout probability for negative queries
            
        Returns:
            Tensor: Query after applying selective dropout to negative queries
        """
        if not self.training:
            return query
            
        bs, num_queries, dim = query.shape
        num_dn_queries = dn_meta.get('num_denoising_queries', 0) if dn_meta is not None else 0
        
        # 如果没有assignment信息，回退到简化实现（对所有matching queries进行dropout）
        if assigned_gt_inds is None:
            if num_dn_queries > 0:
                dn_queries = query[:, :num_dn_queries]
                matching_queries = query[:, num_dn_queries:]
                # 对所有matching queries进行dropout（包含正样本和负样本）
                matching_queries = torch.nn.functional.dropout(matching_queries, p=dropout_p, training=self.training)
                return torch.cat([dn_queries, matching_queries], dim=1)
            else:
                # 没有DN queries时，对所有queries进行dropout
                return torch.nn.functional.dropout(query, p=dropout_p, training=self.training)
        
        # 有assignment信息时，精确地只对负样本进行dropout
        result_query = query.clone()
        
        for batch_idx in range(bs):
            if num_dn_queries > 0:
                # 对于DN queries：不进行dropout
                # 对于matching queries：只对负样本（assigned_gt_inds == 0）进行dropout
                matching_assigned = assigned_gt_inds[batch_idx, num_dn_queries:]
                negative_matching_indices = torch.where(matching_assigned == 0)[0] + num_dn_queries
                
                if len(negative_matching_indices) > 0:
                    # 对负样本进行dropout
                    noise = torch.rand(len(negative_matching_indices), device=query.device)
                    dropout_mask = noise < dropout_p
                    
                    if dropout_mask.any():
                        dropout_indices = negative_matching_indices[dropout_mask]
                        result_query[batch_idx, dropout_indices] = 0.0
            else:
                # 没有DN queries时，只对负样本进行dropout
                negative_indices = torch.where(assigned_gt_inds[batch_idx] == 0)[0]
                
                if len(negative_indices) > 0:
                    noise = torch.rand(len(negative_indices), device=query.device)
                    dropout_mask = noise < dropout_p
                    
                    if dropout_mask.any():
                        dropout_indices = negative_indices[dropout_mask]
                        result_query[batch_idx, dropout_indices] = 0.0
                        
        return result_query

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                dn_meta: Dict = None, assigned_gt_inds: Tensor = None, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            dn_meta (Dict, optional): DN meta information.
            assigned_gt_inds (Tensor, optional): Assignment results with shape (bs, num_queries),
                                               where 0 means negative, >0 means positive.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 保存第一层（lid=0）的输出，用于后续parallel层
                query_after_layer0 = query.clone()
                reference_points_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 对负样本query进行selective dropout
                dropout_query = self._apply_negative_query_dropout(
                    query_after_layer0, dn_meta=dn_meta, assigned_gt_inds=assigned_gt_inds, dropout_p=0.1)

                parallel_query = layer(
                    dropout_query,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer0[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer0[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer0.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer0, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice2(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query_output = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query_output)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query_output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15_DNQueryDropout(DinoTransformerDecoder):
    """
    Parallel decoder with variable DN queries per layer.
    - Front 3 layers: Hierarchy mode
    - Back 3 layers: Parallel mode with same matching query but different DN queries
    """

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def _split_dn_and_matching_queries(self, query: Tensor, dn_meta: Dict) -> tuple:
        """Split query into DN queries and matching queries."""
        if dn_meta is None:
            return None, query, 0
        
        num_dn_queries = dn_meta.get('num_denoising_queries', 0)
        if num_dn_queries == 0:
            return None, query, 0
        
        dn_queries = query[:, :num_dn_queries]  # [bs, num_dn, dim]
        matching_queries = query[:, num_dn_queries:]  # [bs, num_matching, dim]
        
        return dn_queries, matching_queries, num_dn_queries

    def _generate_varied_dn_queries(self, dn_queries: Tensor) -> Tensor:
        """Generate varied DN queries for different layers."""
        if dn_queries is None:
            return None
        
        # 策略1: 添加不同的噪声
        noise_scale = 0.1
        noise = torch.randn_like(dn_queries) * noise_scale
        varied_dn = dn_queries + noise
        
        # 策略2: 可以添加层特定的变换
        # varied_dn = varied_dn * (1.0 + 0.05 * layer_id)
        
        return varied_dn

    def _generate_varied_dn_reference_points(self, dn_ref_points: Tensor) -> Tensor:
        """Generate varied DN reference points for different layers."""
        if dn_ref_points is None:
            return None
            
        # 为不同层的reference points添加小的扰动
        noise_scale = 0.02
        noise = torch.randn_like(dn_ref_points) * noise_scale
        # 确保reference points仍在合理范围内
        varied_ref = torch.clamp(dn_ref_points + noise, 0.01, 0.99)
        
        return varied_ref
    

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                dn_meta: Dict = None, additional_dn_items: List = None, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            dn_meta (Dict): Meta information for denoising queries.
            additional_dn_items (List): Pre-generated DN queries for each layer.
                Contains [dn_label_querys, dn_bbox_querys, dn_masks, dn_metas].

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 分离DN queries和matching queries
        original_dn_queries, matching_queries, num_dn_queries = \
            self._split_dn_and_matching_queries(query, dn_meta)
        
        # 分离DN reference points和matching reference points
        if num_dn_queries > 0:
            dn_reference_points = reference_points[:, :num_dn_queries]
            matching_reference_points = reference_points[:, num_dn_queries:]
        else:
            dn_reference_points = None
            matching_reference_points = reference_points
        
        # 解析additional_dn_items
        if additional_dn_items is not None:
            dn_label_querys, dn_bbox_querys, dn_masks, dn_metas = additional_dn_items
        else:
            dn_label_querys = dn_bbox_querys = dn_masks = dn_metas = None
        
        # 保存第1层的输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None
        matching_query_after_layer0 = None
        matching_ref_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 分离matching query和DN query，保存第1层的输出
                if num_dn_queries > 0:
                    matching_query_after_layer0 = query[:, num_dn_queries:].clone()
                    matching_ref_after_layer0 = reference_points[:, num_dn_queries:].clone()
                else:
                    matching_query_after_layer0 = query.clone()
                    matching_ref_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    
            else:
                # 后5层：Parallel模式 - 使用相同的matching query，但使用预生成的不同DN query
                
                if self.training and num_dn_queries > 0 and dn_label_querys is not None:
                    # 训练时：使用预生成的DN query（每层都有自己独特的DN query）
                    layer_dn_query = dn_label_querys[lid-1]  # 获取当前层的DN query
                    layer_dn_ref = dn_bbox_querys[lid-1]     # 获取当前层的DN reference points
                    
                    # 组合当前层的DN query和固定的matching query
                    matching_query_after_layer0 = torch.nn.Dropout(0.1)(matching_query_after_layer0)
                    parallel_query_input = torch.cat([
                        layer_dn_query, matching_query_after_layer0
                    ], dim=1)
                    parallel_ref_input = torch.cat([
                        layer_dn_ref, matching_ref_after_layer0
                    ], dim=1)
                elif self.training and num_dn_queries > 0:
                    # 训练时：如果没有预生成的DN queries，回退到之前的方法
                    varied_dn_queries = self._generate_varied_dn_queries(
                        original_dn_queries)
                    varied_dn_ref = self._generate_varied_dn_reference_points(
                        dn_reference_points)
                    
                    # 组合变化的DN query和固定的matching query
                    matching_query_after_layer0 = torch.nn.Dropout(0.1)(matching_query_after_layer0)
                    parallel_query_input = torch.cat([
                        varied_dn_queries, matching_query_after_layer0
                    ], dim=1)
                    parallel_ref_input = torch.cat([
                        varied_dn_ref, matching_ref_after_layer0
                    ], dim=1)
                else:
                    # 推理时 或 没有DN queries时：直接使用matching queries
                    parallel_query_input = matching_query_after_layer0
                    parallel_ref_input = matching_ref_after_layer0

                # 计算position encoding
                if parallel_ref_input.shape[-1] == 4:
                    parallel_ref_points_input = \
                        parallel_ref_input[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None]
                else:
                    parallel_ref_points_input = \
                        parallel_ref_input[:, :, None] * valid_ratios[:, None]

                parallel_query_sine_embed = coordinate_to_encoding(
                    parallel_ref_points_input[:, :, 0, :])
                parallel_query_pos = self.ref_point_head(parallel_query_sine_embed)

                parallel_query = layer(
                    parallel_query_input,
                    query_pos=parallel_query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=parallel_ref_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert parallel_ref_input.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        parallel_ref_input, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        
        # 返回最后一层的结果
        return parallel_query if 'parallel_query' in locals() else query, \
               new_reference_points if 'new_reference_points' in locals() else reference_points
    

class GroundingDinoTransformerDecoder_parallel_Choice1_06(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第3层的query输出，用于后续parallel层
        query_after_layer2 = None
        reference_points_after_layer2 = None

        query_after_layer2 = query.clone()
        reference_points_after_layer2 = reference_points.clone()

        for lid, layer in enumerate(self.layers):

            # query_after_layer2 = query.clone()
            # reference_points_after_layer2 = reference_points.clone()

            # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
            # 使用第3层的query和reference_points作为输入，实现真正的parallel
            # if self.training:
            #     dropout_query = torch.nn.Dropout(0.1)(query_after_layer2)
            # else:
            #     dropout_query = query_after_layer2

            parallel_query = layer(
                query_after_layer2,  # 使用第一层的query输出
                query_pos=self.ref_point_head(coordinate_to_encoding(
                    (reference_points_after_layer2[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=(reference_points_after_layer2[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1)[:, None]),
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](parallel_query)
                assert reference_points_after_layer2.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points_after_layer2, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                # parallel模式下不更新全局reference_points

            if self.return_intermediate:
                intermediate.append(self.norm(parallel_query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        #暂不支持
        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15_DNQuery_randDNQ(DinoTransformerDecoder):
    """
    Parallel decoder with variable DN queries per layer.
    - Front 3 layers: Hierarchy mode
    - Back 3 layers: Parallel mode with same matching query but different DN queries
    """

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def _split_dn_and_matching_queries(self, query: Tensor, dn_meta: Dict) -> tuple:
        """Split query into DN queries and matching queries."""
        if dn_meta is None:
            return None, query, 0
        
        num_dn_queries = dn_meta.get('num_denoising_queries', 0)
        if num_dn_queries == 0:
            return None, query, 0
        
        dn_queries = query[:, :num_dn_queries]  # [bs, num_dn, dim]
        matching_queries = query[:, num_dn_queries:]  # [bs, num_matching, dim]
        
        return dn_queries, matching_queries, num_dn_queries

    def _generate_varied_dn_queries(self, dn_queries: Tensor) -> Tensor:
        """Generate varied DN queries for different layers."""
        if dn_queries is None:
            return None
        
        # 策略1: 添加不同的噪声
        noise_scale = 0.1
        noise = torch.randn_like(dn_queries) * noise_scale
        varied_dn = dn_queries + noise
        
        # 策略2: 可以添加层特定的变换
        # varied_dn = varied_dn * (1.0 + 0.05 * layer_id)
        
        return varied_dn

    def _generate_varied_dn_reference_points(self, dn_ref_points: Tensor) -> Tensor:
        """Generate varied DN reference points for different layers."""
        if dn_ref_points is None:
            return None
            
        # 为不同层的reference points添加小的扰动
        noise_scale = 0.02
        noise = torch.randn_like(dn_ref_points) * noise_scale
        # 确保reference points仍在合理范围内
        varied_ref = torch.clamp(dn_ref_points + noise, 0.01, 0.99)
        
        return varied_ref
    

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                dn_meta: Dict = None, additional_dn_items: List = None, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            dn_meta (Dict): Meta information for denoising queries.
            additional_dn_items (List): Pre-generated DN queries for each layer.
                Contains [dn_label_querys, dn_bbox_querys, dn_masks, dn_metas].

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 分离DN queries和matching queries
        original_dn_queries, matching_queries, num_dn_queries = \
            self._split_dn_and_matching_queries(query, dn_meta)
        
        # 分离DN reference points和matching reference points
        if num_dn_queries > 0:
            dn_reference_points = reference_points[:, :num_dn_queries]
            matching_reference_points = reference_points[:, num_dn_queries:]
        else:
            dn_reference_points = None
            matching_reference_points = reference_points
        
        # 解析additional_dn_items
        if additional_dn_items is not None:
            dn_label_querys, dn_bbox_querys, dn_masks, dn_metas = additional_dn_items
        else:
            dn_label_querys = dn_bbox_querys = dn_masks = dn_metas = None
        
        # 保存第1层的输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None
        matching_query_after_layer0 = None
        matching_ref_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 分离matching query和DN query，保存第1层的输出
                if num_dn_queries > 0:
                    matching_query_after_layer0 = query[:, num_dn_queries:].clone()
                    matching_ref_after_layer0 = reference_points[:, num_dn_queries:].clone()
                else:
                    matching_query_after_layer0 = query.clone()
                    matching_ref_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    
            else:
                # 后5层：Parallel模式 - 使用相同的matching query，但使用预生成的不同DN query
                
                if self.training and num_dn_queries > 0 and dn_label_querys is not None:
                    # 训练时：使用预生成的DN query（每层都有自己独特的DN query）
                    layer_dn_query = dn_label_querys[lid-1]  # 获取当前层的DN query
                    layer_dn_ref = dn_bbox_querys[lid-1]     # 获取当前层的DN reference points
                    
                    # 组合当前层的DN query和固定的matching query
                    if random.random() < 0.5:
                        parallel_query_input = torch.nn.Dropout(0.1)(query)
                        parallel_ref_input = reference_points
                    else:
                        parallel_query_input = torch.cat([
                            layer_dn_query, matching_query_after_layer0
                        ], dim=1)
                        parallel_ref_input = torch.cat([
                            layer_dn_ref, matching_ref_after_layer0
                        ], dim=1)

                elif self.training and num_dn_queries > 0:
                    # 训练时：如果没有预生成的DN queries，回退到之前的方法
                    varied_dn_queries = self._generate_varied_dn_queries(
                        original_dn_queries)
                    varied_dn_ref = self._generate_varied_dn_reference_points(
                        dn_reference_points)
                    
                    # 组合变化的DN query和固定的matching query
                    parallel_query_input = torch.cat([
                        varied_dn_queries, matching_query_after_layer0
                    ], dim=1)
                    parallel_ref_input = torch.cat([
                        varied_dn_ref, matching_ref_after_layer0
                    ], dim=1)
                else:
                    # 推理时 或 没有DN queries时：直接使用matching queries
                    parallel_query_input = matching_query_after_layer0
                    parallel_ref_input = matching_ref_after_layer0

                # 计算position encoding
                if parallel_ref_input.shape[-1] == 4:
                    parallel_ref_points_input = \
                        parallel_ref_input[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None]
                else:
                    parallel_ref_points_input = \
                        parallel_ref_input[:, :, None] * valid_ratios[:, None]

                parallel_query_sine_embed = coordinate_to_encoding(
                    parallel_ref_points_input[:, :, 0, :])
                parallel_query_pos = self.ref_point_head(parallel_query_sine_embed)

                parallel_query = layer(
                    parallel_query_input,
                    query_pos=parallel_query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=parallel_ref_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert parallel_ref_input.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        parallel_ref_input, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        
        # 返回最后一层的结果
        return parallel_query if 'parallel_query' in locals() else query, \
               new_reference_points if 'new_reference_points' in locals() else reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15_dropout_negpos(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def _apply_matching_query_dropout(self, query: Tensor, dn_meta: Dict = None, dropout_p: float = 0.1) -> Tensor:
        """对所有非DN query（matching queries）进行dropout
        
        Args:
            query (Tensor): Input queries with shape (bs, num_queries, dim)
            dn_meta (Dict, optional): DN meta information
            dropout_p (float): Dropout probability for matching queries
            
        Returns:
            Tensor: Query after applying dropout to all matching queries (non-DN queries)
        """
        if not self.training:
            return query
            
        bs, num_queries, dim = query.shape
        num_dn_queries = dn_meta.get('num_denoising_queries', 0) if dn_meta is not None else 0
        
        if num_dn_queries > 0:
            # 分离DN queries和matching queries
            dn_queries = query[:, :num_dn_queries]  # [bs, num_dn, dim]
            matching_queries = query[:, num_dn_queries:]  # [bs, num_matching, dim]
            
            # 对所有matching queries进行dropout（包含正样本和负样本）
            matching_queries = torch.nn.functional.dropout(matching_queries, p=dropout_p, training=self.training)
            
            # 重新组合：DN queries保持不变，matching queries应用dropout
            return torch.cat([dn_queries, matching_queries], dim=1)
        else:
            # 没有DN queries时，对所有queries进行dropout
            return torch.nn.functional.dropout(query, p=dropout_p, training=self.training)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                dn_meta: Dict = None, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            dn_meta (Dict, optional): DN meta information.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 保存第一层（lid=0）的输出，用于后续parallel层
                query_after_layer0 = query.clone()
                reference_points_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 对所有非DN query进行dropout
                dropout_query = self._apply_matching_query_dropout(
                    query_after_layer0, dn_meta=dn_meta, dropout_p=0.1)

                parallel_query = layer(
                    dropout_query,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer0[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer0[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer0.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer0, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_15_dropout_fordnquery(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def _apply_dn_query_dropout(self, query: Tensor, dn_meta: Dict = None, dropout_p: float = 0.1) -> Tensor:
        """只对DN query进行dropout
        
        Args:
            query (Tensor): Input queries with shape (bs, num_queries, dim)
            dn_meta (Dict, optional): DN meta information
            dropout_p (float): Dropout probability for DN queries
            
        Returns:
            Tensor: Query after applying dropout to DN queries only
        """
        if not self.training:
            return query
            
        bs, num_queries, dim = query.shape
        num_dn_queries = dn_meta.get('num_denoising_queries', 0) if dn_meta is not None else 0
        
        if num_dn_queries > 0:
            # 分离DN queries和matching queries
            dn_queries = query[:, :num_dn_queries]  # [bs, num_dn, dim]
            matching_queries = query[:, num_dn_queries:]  # [bs, num_matching, dim]
            
            # 只对DN queries进行dropout
            dn_queries = torch.nn.functional.dropout(dn_queries, p=dropout_p, training=self.training)
            
            # 重新组合：DN queries应用dropout，matching queries保持不变
            return torch.cat([dn_queries, matching_queries], dim=1)
        else:
            # 没有DN queries时，不进行任何dropout
            return query

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                dn_meta: Dict = None, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            dn_meta (Dict, optional): DN meta information.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 保存第一层（lid=0）的输出，用于后续parallel层
                query_after_layer0 = query.clone()
                reference_points_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 只对DN query进行dropout
                dropout_query = self._apply_dn_query_dropout(
                    query_after_layer0, dn_meta=dn_meta, dropout_p=0.1)

                parallel_query = layer(
                    dropout_query,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer0[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer0[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer0.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer0, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points


class GroundingDinoTransformerDecoder_parallel_Choice1_1N(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer0 = None
        reference_points_after_layer0 = None

        # 在训练时从后5层中随机选择3层进行梯度更新，其他层detach
        if self.training:
            import random
            # 从层索引1-5中随机选择3个进行梯度更新
            gradient_layers = random.sample(range(1, 6), 3)
            # 保持第0层 + 选择的3层进行梯度更新
            gradient_active_layers = [0] + sorted(gradient_layers)
        else:
            # 推理时所有层都进行梯度更新
            gradient_active_layers = list(range(len(self.layers)))

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid == 0:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # 保存第一层（lid=0）的输出，用于后续parallel层
                query_after_layer0 = query.clone()
                reference_points_after_layer0 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.
            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 不进行任何dropout

                parallel_query = layer(
                    query_after_layer0,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer0[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer0[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer0.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer0, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points














class GroundingDinoTransformerDecoder_parallel_Choice1_24(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer1 = None
        reference_points_after_layer1 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid < 2:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # # 保存第一层（lid=0）的输出，用于后续parallel层
                # query_after_layer0 = query.clone()
                # reference_points_after_layer0 = reference_points.clone()

                # if self.return_intermediate:
                #     intermediate.append(self.norm(query))
                #     intermediate_reference_points.append(new_reference_points)
                #     # NOTE this is for the "Look Forward Twice" module,
                #     # in the DeformDETR, reference_points was appended.

                # 保存第二层（lid=1）的输出
                if lid == 1:
                    query_after_layer1 = query.clone()
                    reference_points_after_layer1 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)

            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 不进行任何dropout

                parallel_query = layer(
                    query_after_layer1,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer1[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer1[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer1.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer1, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points







class GroundingDinoTransformerDecoder_parallel_Choice1_33(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer2 = None
        reference_points_after_layer2 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid < 3:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # # 保存第一层（lid=0）的输出，用于后续parallel层
                # query_after_layer0 = query.clone()
                # reference_points_after_layer0 = reference_points.clone()

                # if self.return_intermediate:
                #     intermediate.append(self.norm(query))
                #     intermediate_reference_points.append(new_reference_points)
                #     # NOTE this is for the "Look Forward Twice" module,
                #     # in the DeformDETR, reference_points was appended.

                # 保存第二层（lid=1）的输出
                if lid == 2:
                    query_after_layer2 = query.clone()
                    reference_points_after_layer2 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)

            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 不进行任何dropout

                parallel_query = layer(
                    query_after_layer2,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer2[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer2[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer2.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer2, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points








class GroundingDinoTransformerDecoder_parallel_Choice1_42(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        # 保存第1层的query输出，用于后续parallel层
        query_after_layer3 = None
        reference_points_after_layer3 = None

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if lid < 4:
                # 第一层：Hierarchy模式 - 逐层更新reference_points和query
                query = layer(
                    query,
                    query_pos=query_pos,
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](query)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()  # 更新reference_points

                # # 保存第一层（lid=0）的输出，用于后续parallel层
                # query_after_layer0 = query.clone()
                # reference_points_after_layer0 = reference_points.clone()

                # if self.return_intermediate:
                #     intermediate.append(self.norm(query))
                #     intermediate_reference_points.append(new_reference_points)
                #     # NOTE this is for the "Look Forward Twice" module,
                #     # in the DeformDETR, reference_points was appended.

                # 保存第二层（lid=1）的输出
                if lid == 3:
                    query_after_layer3 = query.clone()
                    reference_points_after_layer3 = reference_points.clone()

                if self.return_intermediate:
                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)

            else:
                # 第二层及以后：真正的Parallel模式 - 都基于第一层的相同输出
                # 不进行任何dropout

                parallel_query = layer(
                    query_after_layer3,  # 使用第一层的query输出
                    query_pos=self.ref_point_head(coordinate_to_encoding(
                        (reference_points_after_layer3[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None])[:, :, 0, :])),
                    value=value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=(reference_points_after_layer3[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]),
                    **kwargs)

                if reg_branches is not None:
                    tmp = reg_branches[lid](parallel_query)
                    assert reference_points_after_layer3.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points_after_layer3, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    # parallel模式下不更新全局reference_points

                if self.return_intermediate:
                    intermediate.append(self.norm(parallel_query))
                    intermediate_reference_points.append(new_reference_points)
                    # NOTE this is for the "Look Forward Twice" module,
                    # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        # 暂不支持
        return query, reference_points
