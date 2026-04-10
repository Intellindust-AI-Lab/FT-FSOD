CDFSOD_PATH = '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets'
MMGDINOB_PATH = '/data/yuxuanlong/codebase/mmdetection/checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'
auto_scale_lr = dict(base_batch_size=10, enable=False)
backend_args = None
class_name = 'ArTaxOr'
class_name_list = [
    'Araneae',
    'Coleoptera',
    'Diptera',
    'Hemiptera',
    'Hymenoptera',
    'Lepidoptera',
    'Odonata',
]
class_names = (
    'Araneae',
    'Coleoptera',
    'Diptera',
    'Hemiptera',
    'Hymenoptera',
    'Lepidoptera',
    'Odonata',
)
custom_hooks = [
    dict(
        adjust_scheduler_patience=True,
        patience_frozen=3,
        patience_unfrozen=8,
        type='BBoxHeadFirstHook6'),
]
data_root = '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=1,
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='GroundingVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
lang_model_name = 'bert-base-uncased'
launcher = 'pytorch'
load_from = '/data/yuxuanlong/codebase/mmdetection/checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
max_epochs = 100
metainfo = dict(
    classes=(
        'Araneae',
        'Coleoptera',
        'Diptera',
        'Hemiptera',
        'Hymenoptera',
        'Lepidoptera',
        'Odonata',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        frozen_stages=-1,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=True, log_scale='auto', max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=7,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead_ParallelDecoder_DN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        max_tokens=256,
        name='bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            256,
            512,
            1024,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    rand_dnquery_rate=0.5,
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO_ParallelDecoder_15_DNQuery_rand',
    use_autocast=True,
    with_box_refine=True)
num_classes = 7
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.2),
            language_model=dict(lr_mult=0.2))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        cooldown=1,
        factor=0.5,
        min_value=1e-06,
        monitor='coco/bbox_mAP',
        param_name='lr',
        patience=5,
        rule='greater',
        threshold=0.0001,
        threshold_rule='rel',
        type='ReduceOnPlateauParamScheduler',
        verbose=True),
]
pretrained = 'swin_tiny_patch4_window7_224.pth'
randomness = dict(
    deterministic=True, diff_rank_seed=False, seed=42, warn_only=True)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=
        '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr',
        metainfo=dict(
            classes=(
                'Araneae',
                'Coleoptera',
                'Diptera',
                'Hemiptera',
                'Hymenoptera',
                'Lepidoptera',
                'Odonata',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    800,
                    1333,
                ),
                type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                    'tokens_positive',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow',
        keep_ratio=True,
        scale=(
            800,
            1333,
        ),
        type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'tokens_positive',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='annotations/1_shot_converted.json',
        data_prefix=dict(img='train/'),
        data_root=
        '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'Araneae',
                'Coleoptera',
                'Diptera',
                'Hemiptera',
                'Hymenoptera',
                'Lepidoptera',
                'Odonata',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=10,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                prob=0.3,
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    800,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    960,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    600,
                                    4200,
                                ),
                                (
                                    800,
                                    4200,
                                ),
                                (
                                    960,
                                    4200,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                640,
                                960,
                            ),
                            crop_type='absolute_range',
                            type='HumanityRandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    800,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    960,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=10,
        pad_val=(
            114,
            114,
            114,
        ),
        prob=0.3,
        ratio_range=(
            1.0,
            1.0,
        ),
        type='CachedMixUp'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            800,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            960,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            600,
                            4200,
                        ),
                        (
                            800,
                            4200,
                        ),
                        (
                            960,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        640,
                        960,
                    ),
                    crop_type='absolute_range',
                    type='HumanityRandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            800,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            960,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root=
        '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr',
        metainfo=dict(
            classes=(
                'Araneae',
                'Coleoptera',
                'Diptera',
                'Hemiptera',
                'Hymenoptera',
                'Lepidoptera',
                'Odonata',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    800,
                    1333,
                ),
                type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                    'tokens_positive',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/ArTaxOr/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'exp_cdfosd_results//swinB_all_ArTaxOr_1shot'
