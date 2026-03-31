# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DisablePixmixHook(Hook):
    """Hook to disable PixmixAug after a certain number of epochs.
    
    Args:
        disable_epoch (int): The epoch after which to disable PixmixAug.
            Defaults to 6.
    """
    
    def __init__(self, disable_epoch=6, **kwargs):
        super().__init__(**kwargs)
        self.disable_epoch = disable_epoch
        self.pixmix_disabled = False
        
    def before_train_epoch(self, runner):
        """Disable PixmixAug before training epoch if conditions are met."""
        if (runner.epoch >= self.disable_epoch and 
            not self.pixmix_disabled):
            self._disable_pixmix_aug(runner)
            self.pixmix_disabled = True
            runner.logger.info(
                f'PixmixAug disabled at epoch {runner.epoch}')
    
    def _disable_pixmix_aug(self, runner):
        """Remove PixmixAug from the training pipeline."""
        # Get the training dataloader
        train_dataloader = runner.train_dataloader
        
        # Get the dataset pipeline
        if hasattr(train_dataloader.dataset, 'pipeline'):
            pipeline = train_dataloader.dataset.pipeline.transforms
            
            # Find and remove PixmixAug transform
            new_pipeline = []
            for transform in pipeline:
                if hasattr(transform, '__class__') and transform.__class__.__name__ != 'PixmixAug':
                    new_pipeline.append(transform)
                else:
                    runner.logger.info(f'Removed {transform.__class__.__name__} from pipeline')
            
            # Update the pipeline
            train_dataloader.dataset.pipeline.transforms = new_pipeline