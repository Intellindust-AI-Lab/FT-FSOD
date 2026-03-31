_base_ = '../../grounding_dino_swin-l_pretrain_all.py'
import os
from src_path import ODINW_PATH, MMGDINOL_PATH


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
randomness = dict(
    seed=42,  
    deterministic=True,  
    diff_rank_seed=False,
    warn_only=True # should hack the implementation in mmdet，otherwise the program will encounter error
)


class_name = 'Aquarium'
data_root = f'{ODINW_PATH}/{class_name}/Aquarium_Combined.v2-raw-1024.coco'

class_name_list = [
'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'
]

class_names = tuple(class_name_list)
num_classes = len(class_names)
metainfo = dict(classes=class_names, palette=[(220, 20, 60)])

model = dict(
    type='GroundingDINO_ParallelDecoder_15_DNQuery_rand',
    bbox_head=dict(
        type='GroundingDINOHead_ParallelDecoder_DN',
        num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="CachedMixUp", img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=10, pad_val=(114, 114, 114), prob=0.3),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 1333), (768, 1333), (960, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(600, 4200), (800, 4200), (960, 4200)],
                    keep_ratio=True),
                dict(
                    type='HumanityRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(640, 960),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 1333), (768, 1333), (960, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    num_workers = 2,
    batch_size=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'train/fewshot_train_shot1_seed3.json',
        data_prefix=dict(img='train/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    num_workers = 2,
    batch_size=4,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'valid/fewshot_val_shot1_seed3.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline, 
        return_classes=True))

test_dataloader = dict(
    num_workers = 2,
    batch_size=4,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'test/annotations_without_background.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline, 
        return_classes=True))

val_evaluator = dict(
    ann_file=f'{data_root}/valid/fewshot_val_shot1_seed3.json')

test_evaluator = dict(
    ann_file=f'{data_root}/test/annotations_without_background.json')



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.2),
            'language_model': dict(lr_mult=0.2),
        }))

max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=1,
)

auto_scale_lr = dict(base_batch_size=10)

param_scheduler = [
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',
        monitor='coco/bbox_mAP',
        rule='greater',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_rule='rel',
        cooldown=1,
        min_value=1e-6,
        verbose=True)
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto')) 


custom_hooks = [
    dict(
        type='BBoxHeadFirstHook6',
        adjust_scheduler_patience=True,
        patience_frozen=3,
        patience_unfrozen=8,
    )
]
load_from = MMGDINOL_PATH
