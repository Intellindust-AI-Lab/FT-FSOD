# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
from typing import Dict, Optional

from mmengine.hooks.ema_hook import EMAHook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from mmengine.hooks.hook import DATA_BATCH, Hook
from mmengine.runner import Runner


@HOOKS.register_module()
class EpochEMAHook(EMAHook):
    """A Hook to apply Exponential Moving Average (EMA) on the model during
    training, but updates EMA parameters only at the end of each epoch.

    This hook inherits from EMAHook but overrides the update behavior to
    perform EMA updates once per epoch instead of once per iteration.

    Note:
        - EpochEMAHook takes priority over CheckpointHook.
        - The original model parameters are actually saved in ema field after
          train.
        - ``begin_iter`` and ``begin_epoch`` cannot be set at the same time.

    Args:
        ema_type (str): The type of EMA strategy to use. You can find the
            supported strategies in :mod:`mmengine.model.averaged_model`.
            Defaults to 'ExponentialMovingAverage'.
        strict_load (bool): Whether to strictly enforce that the keys of
            ``state_dict`` in checkpoint match the keys returned by
            ``self.module.state_dict``. Defaults to False.
            Changed in v0.3.0.
        begin_iter (int): The number of iteration to enable ``EpochEMAHook``.
            Defaults to 0.
        begin_epoch (int): The number of epoch to enable ``EpochEMAHook``.
            Defaults to 0.
        **kwargs: Keyword arguments passed to subclasses of
            :obj:`BaseAveragedModel`
    """

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Override the original after_train_iter to disable per-iteration updates.
        
        This method intentionally does nothing to prevent EMA updates
        after each iteration.
        """
        pass

    def after_train_epoch(self, runner: Runner) -> None:
        """Update ema parameter at the end of each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self._ema_started(runner):
            # Perform EMA update once per epoch
            self.ema_model.update_parameters(self.src_model)
            runner.logger.info(f'EMA model updated at epoch {runner.epoch}')
        else:
            # Initialize EMA parameters with current model parameters
            ema_params = self.ema_model.module.state_dict()
            src_params = self.src_model.state_dict()
            for k, p in ema_params.items():
                p.data.copy_(src_params[k].data)
            runner.logger.info(f'EMA model initialized at epoch {runner.epoch}')