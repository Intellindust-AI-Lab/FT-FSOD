# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class AugmentationSwitchHookLR(Hook):
    """Switch data augmentation at specified epoch.

    This hook turns off specified data augmentation transforms at a given epoch.
    Unlike YOLOXModeSwitchHook, this hook works with any dataset type (including CocoDataset)
    and does not modify model loss functions.

    Args:
        lr_threshold (float): The learning rate threshold at which to switch the pipeline.
            Defaults to 5e-5.
        target_component (str): The component name to monitor for learning rate.
            Defaults to 'language_model.encoder'.
        skip_type_keys (Sequence[str], optional): Sequence of type string to be
            removed from pipeline. Defaults to ('PixMix',).
    """

    def __init__(
        self,
        lr_threshold: float = 5e-5,
        target_component: str = 'language_model.encoder',
        skip_type_keys: Sequence[str] = ('PixMix',)
    ) -> None:
        self.lr_threshold = lr_threshold
        self.target_component = target_component
        self.skip_type_keys = skip_type_keys
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
                              f'switching off augmentations: {self.skip_type_keys}!')
            
            # Store original pipeline if not already stored
            if self._original_pipeline is None:
                self._original_pipeline = train_loader.dataset.pipeline.transforms.copy()
            
            # Create new pipeline without specified transforms
            new_transforms = []
            for transform in self._original_pipeline:
                transform_type = transform.__class__.__name__
                if transform_type not in self.skip_type_keys:
                    new_transforms.append(transform)
                else:
                    runner.logger.info(f'Removing {transform_type} from pipeline')
            
            # Update pipeline
            train_loader.dataset.pipeline = Compose(new_transforms)
            
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