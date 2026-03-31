# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class AugmentationSwitchHookLR_Inverse(Hook):
    """Switch data augmentation at specified epoch.

    This hook turns off specified data augmentation transforms at a given epoch.
    Unlike YOLOXModeSwitchHook, this hook works with any dataset type (including CocoDataset)
    and does not modify model loss functions.

    Args:
        lr_threshold (float): The learning rate threshold at which to switch the pipeline.
            Defaults to 5e-5.
        target_component (str): The component name to monitor for learning rate.
            Defaults to 'language_model.encoder'.
    """

    def __init__(
        self,
        lr_threshold: float = 5e-5,
        target_component: str = 'language_model.encoder',
    ) -> None:
        self.lr_threshold = lr_threshold
        self.target_component = target_component
        self._restart_dataloader = False
        self._has_switched = False
        self._original_pipeline = None

    def _get_component_lr(self, runner) -> float:
        """Get the learning rate of the target component."""
        optimizer = runner.optim_wrapper.optimizer

        # 如果仍然没有找到，返回第一个参数组的学习率作为默认值
        target_lr = optimizer.param_groups[0]['lr']
        runner.logger.warning(f'Target component {self.target_component} not found, '
                            f'using default learning rate: {target_lr}')
        
        return target_lr


    def before_train_epoch(self, runner) -> None:
        """Switch pipeline to remove specified augmentations based on learning rate."""
        train_loader = runner.train_dataloader
        
        # 获取目标组件的当前学习率
        current_lr = self._get_component_lr(runner)
        
        if current_lr <= self.lr_threshold and not self._has_switched:
            runner.logger.info(f'Learning rate {current_lr:.2e} <= threshold {self.lr_threshold:.2e}, '
                              f'switching on augmentations: PixmixAugModi3!')
            
            # Store original pipeline if not already stored
            if self._original_pipeline is None:
                self._original_pipeline = train_loader.dataset.pipeline.transforms.copy()
            
            # 在 CachedMixUp 之后添加 PixMix 增强
            pixmix_added = False
            final_transforms = []
            
            for i, transform in enumerate(self._original_pipeline):
                final_transforms.append(transform)
                
                # 在 CachedMixUp 之后插入 PixMix
                if (transform.__class__.__name__ == 'CachedMixUp' and 
                    not pixmix_added):
                    
                    # 导入 PixMix 增强
                    from mmdet.datasets.transforms import PixmixAugModi3
                    
                    pixmix_transform = PixmixAugModi3(
                        pixmix_path='/data6022/PixMixSet/fractals_and_fvis/first_layers_resized256_onevis',
                        prob=0.3
                    )
                    
                    final_transforms.append(pixmix_transform)
                    pixmix_added = True
                    runner.logger.info(f'Added PixMix augmentation after CachedMixUp with prob=0.3')
            
            # Update pipeline
            train_loader.dataset.pipeline = Compose(final_transforms)
            
            # Handle persistent workers
            if hasattr(train_loader, 'persistent_workers') and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
                self._restart_dataloader = False