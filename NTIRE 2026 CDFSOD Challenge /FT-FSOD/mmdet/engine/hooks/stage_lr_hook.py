

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

@HOOKS.register_module()
class BBoxHeadFirstHook6(Hook):
    """Hook for implementing two-stage training.

    Stage 1: Only train bbox_head, language_model and related components, 
             freeze all other components.
    Stage 2: Unfreeze all components and set specific learning rates for 
             different components.
        - Backbone learning rate matches the final learning rate of language_model.
        - Other components (e.g., neck) learning rate matches the final 
          learning rate of bbox_head.

    Args:
        adjust_scheduler_patience (bool): Whether to adjust scheduler patience 
            when unfreezing backbone.
        patience_frozen (int): Patience value when backbone is frozen.
        patience_unfrozen (int): Patience value after backbone is unfrozen.
    
    Note:
        This Hook relies on the configuration file setting different parameter 
        groups for different parts of the model (e.g., 'backbone', 'language_model') 
        through `paramwise_cfg`.
    """

    def __init__(self,
                 adjust_scheduler_patience: bool = False,
                 patience_frozen: int = 3,
                 patience_unfrozen: int = 5):
        self.adjust_scheduler_patience = adjust_scheduler_patience
        self.patience_frozen = patience_frozen
        self.patience_unfrozen = patience_unfrozen
        self._stage2_started = False
        
        # Save original learning rates for each parameter group
        self._original_lrs = {}

        # Store parameter group indices corresponding to different components
        self._head_groups = []
        self._lang_model_groups = []
        self._backbone_groups = []
        self._other_groups = []

    def before_train(self, runner) -> None:
        """Identify parameter groups and set Stage 1 learning rates before training starts."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # 1. Identify parameter groups corresponding to different components in the model
        self._identify_param_groups(runner, model)

        # 2. Save original learning rates for all groups
        optimizer = runner.optim_wrapper.optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            self._original_lrs[i] = param_group['lr']
        
        # 3. Set learning rates for Stage 1
        self._set_stage1_lr(runner)
        
        # 4. If patience adjustment is enabled, set initial patience to frozen state value
        if self.adjust_scheduler_patience:
            self._adjust_scheduler_patience(runner, self.patience_frozen)

    def before_train_epoch(self, runner) -> None:
        """Check learning rate changes, unfreeze all components when language model's lr drops to half of original lr."""
        if self._stage2_started:
            return
            
        # Check if should enter Stage 2
        if self._should_start_stage2(runner):
            runner.logger.info(
                f'Epoch {runner.epoch}: Language model LR has dropped to half of original LR. '
                'Starting Stage 2.')
            # Set learning rates for Stage 2
            self._set_stage2_lr(runner)
            self._stage2_started = True

    def _identify_param_groups(self, runner, model: nn.Module):
        """Identify parameter groups for different components in optimizer by parameter names."""
        param_id_to_name = {id(p): n for n, p in model.named_parameters()}
        optimizer = runner.optim_wrapper.optimizer

        for i, param_group in enumerate(optimizer.param_groups):
            if not param_group['params']:
                continue
            
            # Determine the type of the entire group by the name of the first parameter in the group
            first_param_id = id(param_group['params'][0])
            param_name = param_id_to_name.get(first_param_id, '')

            if param_name.startswith('backbone'):
                self._backbone_groups.append(i)
            elif param_name.startswith('language_model'):
                self._lang_model_groups.append(i)
            elif (param_name.startswith('bbox_head') or
                  param_name.startswith('text_feat_map') or
                  param_name.startswith('neck') or
                  param_name.startswith('decoder') or
                  param_name.startswith('dn_query_generator')):
                self._head_groups.append(i)
            else:
                self._other_groups.append(i)
        
        runner.logger.info(f'Identified param groups: Backbone={self._backbone_groups}, '
                        f'LanguageModel={self._lang_model_groups}, Head={self._head_groups}, '
                        f'Other={self._other_groups}')

    def _should_start_stage2(self, runner) -> bool:
        """Determine if Stage 2 should start: check if language model's lr has dropped to half of original lr."""
        if not self._lang_model_groups:
            return False
            
        optimizer = runner.optim_wrapper.optimizer
        group_idx = self._lang_model_groups[0]
        current_lr = optimizer.param_groups[group_idx]['lr']
        original_lr = self._original_lrs.get(group_idx, 0.0)
        
        # Start Stage 2 when current_lr <= original_lr * 0.5
        return current_lr <= original_lr * 0.5

    def _set_stage2_lr(self, runner):
        """Stage 2: Unfreeze all components and set learning rates according to specific rules."""
        runner.logger.info('Setting LRs for Stage 2...')
        optimizer = runner.optim_wrapper.optimizer

        # 1. Get current learning rates of language_model and bbox head
        current_lang_lr = 0.0
        if self._lang_model_groups:
            group_idx = self._lang_model_groups[0]
            current_lang_lr = optimizer.param_groups[group_idx]['lr']
            # No longer modify language model's lr, keep current value
            runner.logger.info(f'  - Current LanguageModel LR: {current_lang_lr:.8f}')
        else:
            runner.logger.warning('Language model group not found for Stage 2 LR setting!')

        current_head_lr = 0.0
        if self._head_groups:
            group_idx = self._head_groups[0]
            current_head_lr = optimizer.param_groups[group_idx]['lr']
            # No longer modify bbox head's lr, keep current value
            runner.logger.info(f'  - Current BboxHead LR: {current_head_lr:.8f}')
        else:
            runner.logger.warning('Bbox head group not found for Stage 2 LR setting!')

        # Other components follow bbox head's learning rate
        for group_idx in self._other_groups:
            optimizer.param_groups[group_idx]['lr'] = current_head_lr
        runner.logger.info(f'  - Other groups LR set to match BboxHead: {current_head_lr:.8f}')

        # 3. If patience adjustment is enabled, change patience to unfrozen state value
        if self.adjust_scheduler_patience:
            self._adjust_scheduler_patience(runner, self.patience_unfrozen)
            runner.logger.info(f'  - Scheduler patience adjusted to {self.patience_unfrozen} (unfrozen state)')

        # 4. Log final learning rate settings
        runner.logger.info('Stage 2 LR setting completed:')
        runner.logger.info(f'  - LanguageModel & Backbone: {current_lang_lr:.8f}')
        runner.logger.info(f'  - BboxHead & Other: {current_head_lr:.8f}')

    def _adjust_scheduler_patience(self, runner, patience: int):
        """Adjust the patience parameter of the learning rate scheduler."""
        if not hasattr(runner, 'scheduler'):
            # Try to get scheduler from different places
            if hasattr(runner, 'optim_wrapper') and hasattr(runner.optim_wrapper, 'scheduler'):
                scheduler = runner.optim_wrapper.scheduler
            elif hasattr(runner, 'param_schedulers'):
                # If there are multiple schedulers, find the first ReduceOnPlateauParamScheduler
                for scheduler in runner.param_schedulers:
                    if hasattr(scheduler, 'patience'):
                        scheduler.patience = patience
                        runner.logger.info(f'Adjusted scheduler patience to {patience}')
                        return
                return
            else:
                runner.logger.warning('Could not find scheduler to adjust patience')
                return
        else:
            scheduler = runner.scheduler

        # Adjust patience
        if hasattr(scheduler, 'patience'):
            scheduler.patience = patience
            runner.logger.info(f'Adjusted scheduler patience to {patience}')
        else:
            runner.logger.warning('Scheduler does not have patience attribute')


    def _set_stage1_lr(self, runner):
        """Stage 1: bbox_head and language_model use specified learning rates, other components are frozen."""
        runner.logger.info('Setting LRs for Stage 1...')
        optimizer = runner.optim_wrapper.optimizer
        
        # Set learning rates for trainable groups
        trainable_groups = self._backbone_groups + self._head_groups + self._lang_model_groups
        for group_idx in trainable_groups:
            base_lr = self._original_lrs[group_idx]
            optimizer.param_groups[group_idx]['lr'] = base_lr
            # runner.logger.info(
                # f'  - Group {group_idx} (Trainable) LR set to {base_lr:.8f} ')

        # Freeze other groups
        frozen_groups = self._other_groups
        for group_idx in frozen_groups:
            base_lr = self._original_lrs[group_idx]
            optimizer.param_groups[group_idx]['lr'] = 0.0
            # runner.logger.info(
                # f'  - Group {group_idx} (Frozen) LR set to 0.0 '
                # f'(base: {base_lr:.6f})')

    def after_train_epoch(self, runner) -> None:
        """Print current learning rate status once after each epoch."""
        optimizer = runner.optim_wrapper.optimizer
        lr_status = [f'Epoch [{runner.epoch}] Current LRs:']
        
        group_map = {
            'Head': self._head_groups,
            'Language Model': self._lang_model_groups,
            'Backbone': self._backbone_groups,
            'Other': self._other_groups
        }

        for name, indices in group_map.items():
            if indices:
                lr = optimizer.param_groups[indices[0]]['lr']
                lr_status.append(f'  - {name} (groups {indices}): {lr:.8f}')
        
        runner.logger.info(' | '.join(lr_status))


@HOOKS.register_module()
class StageWiseFreezeHook(Hook):
    """Hook for implementing explicit epoch-based two-stage training.

    This hook allows explicit control over which components are trainable in
    each stage, with stage transitions based on specific epoch numbers.

    Stage 1 (epoch 0 to stage1_epochs):

    Stage 2 (epoch stage1_epochs to max_epochs):
        - Unfreeze all components

    Args:
        stage1_epochs (int): Number of epochs for Stage 1 training.

    Note:
        This hook follows the same parameter grouping logic as BBoxHeadFirstHook6:
        - backbone: parameters starting with 'backbone'
        - language_model: parameters starting with 'language_model'
        - head: parameters starting with 'bbox_head', 'text_feat_map', 'neck',
          'decoder', 'dn_query_generator'
        - other: all other parameters

        Freezing is implemented by setting lr=0 only (no requires_grad=False),
        same as BBoxHeadFirstHook6, to avoid DDP "unused parameters" errors.
    """

    def __init__(self, stage1_epochs: int = 50):
        self.stage1_epochs = stage1_epochs
        self._stage2_started = False

        # Save original learning rates for each parameter group
        self._original_lrs = {}

        # Store parameter group indices corresponding to different components
        self._head_groups = []
        self._lang_model_groups = []
        self._backbone_groups = []
        self._other_groups = []

    def before_train(self, runner) -> None:
        """Identify parameter groups and set Stage 1 configuration."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # 1. Identify parameter groups corresponding to different components in the model
        self._identify_param_groups(runner, model)

        # 2. Save original learning rates for all groups
        optimizer = runner.optim_wrapper.optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            self._original_lrs[i] = param_group['lr']

        # 3. Set learning rates for Stage 1
        self._set_stage1_lr(runner)

    def before_train_epoch(self, runner) -> None:
        """Check if we should transition to Stage 2 at the beginning of each epoch."""
        if self._stage2_started:
            return

        # Check if we should start Stage 2
        if runner.epoch >= self.stage1_epochs:
            runner.logger.info(
                f'Epoch {runner.epoch}: Transitioning to Stage 2. '
                f'Stage 1 completed at epoch {self.stage1_epochs}.')
            self._set_stage2_lr(runner)
            self._stage2_started = True

    def _identify_param_groups(self, runner, model: nn.Module):
        """Identify parameter groups for different components in optimizer by parameter names."""
        param_id_to_name = {id(p): n for n, p in model.named_parameters()}
        optimizer = runner.optim_wrapper.optimizer

        # Reset group lists
        self._head_groups = []
        self._lang_model_groups = []
        self._backbone_groups = []
        self._other_groups = []

        for i, param_group in enumerate(optimizer.param_groups):
            if not param_group['params']:
                continue

            # Determine the type of the entire group by the name of the first parameter in the group
            first_param_id = id(param_group['params'][0])
            param_name = param_id_to_name.get(first_param_id, '')

            if param_name.startswith('backbone'):
                self._backbone_groups.append(i)
            elif param_name.startswith('language_model'):
                self._lang_model_groups.append(i)
            elif (param_name.startswith('bbox_head') or
                  param_name.startswith('text_feat_map') or
                  param_name.startswith('neck') or
                  param_name.startswith('decoder') or
                  param_name.startswith('dn_query_generator')):
                self._head_groups.append(i)
            else:
                self._other_groups.append(i)

        runner.logger.info(f'Identified param groups: Backbone={self._backbone_groups}, '
                        f'LanguageModel={self._lang_model_groups}, Head={self._head_groups}, '
                        f'Other={self._other_groups}')

    def _set_stage1_lr(self, runner):
        """Stage 1: backbone, decoder and language_model use base lr; only other is frozen via lr=0.

        Aligned with BBoxHeadFirstHook6: only 'other' has lr=0 in Stage 1.
        """
        runner.logger.info('Setting LRs for Stage 1...')
        optimizer = runner.optim_wrapper.optimizer

        # Trainable: backbone, decoder, language_model (same as BBoxHeadFirstHook6)
        trainable_groups = self._backbone_groups + self._head_groups + self._lang_model_groups
        for group_idx in trainable_groups:
            base_lr = self._original_lrs[group_idx]
            optimizer.param_groups[group_idx]['lr'] = base_lr

        # Freeze other groups only (lr=0)
        for group_idx in self._other_groups:
            base_lr = self._original_lrs[group_idx]
            optimizer.param_groups[group_idx]['lr'] = 0.0

        runner.logger.info('Stage 1 configuration completed: backbone, decoder, language_model trainable; only other frozen (lr=0)')

    def _set_stage2_lr(self, runner):
        """Stage 2: Unfreeze all components and set learning rates according to specific rules."""
        runner.logger.info('Setting LRs for Stage 2...')
        optimizer = runner.optim_wrapper.optimizer

        # 1. Get current learning rates of language_model and bbox head
        current_lang_lr = 0.0
        if self._lang_model_groups:
            group_idx = self._lang_model_groups[0]
            current_lang_lr = optimizer.param_groups[group_idx]['lr']
            runner.logger.info(f'  - Current LanguageModel LR: {current_lang_lr:.8f}')
        else:
            runner.logger.warning('Language model group not found for Stage 2 LR setting!')

        current_head_lr = 0.0
        if self._head_groups:
            group_idx = self._head_groups[0]
            current_head_lr = optimizer.param_groups[group_idx]['lr']
            runner.logger.info(f'  - Current BboxHead LR: {current_head_lr:.8f}')
        else:
            runner.logger.warning('Bbox head group not found for Stage 2 LR setting!')

        # 2. Other components follow bbox head's learning rate (same as BBoxHeadFirstHook6; backbone/decoder/lang_model keep scheduler lr)
        for group_idx in self._other_groups:
            optimizer.param_groups[group_idx]['lr'] = current_head_lr
        runner.logger.info(f'  - Other groups LR set to match BboxHead: {current_head_lr:.8f}')

        # 3. Log final learning rate settings
        runner.logger.info('Stage 2 LR setting completed:')
        runner.logger.info(f'  - LanguageModel & Backbone: {current_lang_lr:.8f}')
        runner.logger.info(f'  - BboxHead & Other: {current_head_lr:.8f}')

    def after_train_epoch(self, runner) -> None:
        """Print current training stage and frozen status after each epoch.

        'Frozen' = params in groups with lr=0; 'Trainable' = params in groups with lr>0.
        """
        stage = 'Stage 1' if not self._stage2_started else 'Stage 2'
        optimizer = runner.optim_wrapper.optimizer

        # Count trainable (lr>0) vs frozen (lr=0) parameters by param_group lr
        trainable_count = 0
        frozen_count = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            for param in param_group['params']:
                n = param.numel()
                if lr is not None and lr > 1e-10:
                    trainable_count += n
                else:
                    frozen_count += n

        total_count = trainable_count + frozen_count
        trainable_percent = (trainable_count / total_count * 100) if total_count > 0 else 0

        runner.logger.info(
            f'Epoch [{runner.epoch}] {stage} - '
            f'Trainable params: {trainable_count:,} ({trainable_percent:.2f}%), '
            f'Frozen params (lr=0): {frozen_count:,}')
