# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class AugmentationSwitchHook(Hook):
    """Switch data augmentation at specified epoch.

    This hook turns off specified data augmentation transforms at a given epoch.
    Unlike YOLOXModeSwitchHook, this hook works with any dataset type (including CocoDataset)
    and does not modify model loss functions.

    Args:
        num_last_epochs (int): The epoch at which to switch the pipeline.
            Defaults to 5.
        skip_type_keys (Sequence[str], optional): Sequence of type string to be
            removed from pipeline. Defaults to ('CachedMixUp',).
    """

    def __init__(
        self,
        num_last_epochs: int = 5,
        skip_type_keys: Sequence[str] = ('CachedMixUp',)
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False
        self._has_switched = False
        self._original_pipeline = None

    def before_train_epoch(self, runner) -> None:
        """Switch pipeline to remove specified augmentations."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        
        if epoch >= self.num_last_epochs and not self._has_switched:
            runner.logger.info(f'Switching off augmentations: {self.skip_type_keys} at epoch {epoch}!')
            
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