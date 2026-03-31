from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import os


@HOOKS.register_module()
class AutoLoadBestCheckpointHook(Hook):
    """Hook to automatically load the best checkpoint when learning rate is reduced.
    
    This hook monitors the learning rate and automatically loads the best saved
    checkpoint when the learning rate is reduced by a scheduler like ReduceOnPlateauParamScheduler.
    
    Args:
        monitor_param (str): Parameter to monitor for changes. Defaults to 'lr'.
    """
    
    def __init__(self, monitor_param='lr', **kwargs):
        super().__init__(**kwargs)
        self.monitor_param = monitor_param
        self.last_lr = None
        
    def after_val_epoch(self, runner, metrics=None):
        """Check if learning rate has decreased and load best checkpoint if so."""
        # 检查学习率是否发生变化
        current_lr = runner.optim_wrapper.optimizer.param_groups[0][self.monitor_param]
        
        if self.last_lr is not None and current_lr < self.last_lr:
            # 学习率降低了，加载最佳检查点
            best_ckpt_path = self._find_best_checkpoint(runner)
            if best_ckpt_path:
                runner.logger.info(f'Learning rate reduced from {self.last_lr} to {current_lr}, '
                                  f'loading best checkpoint: {best_ckpt_path}')
                runner.load_checkpoint(best_ckpt_path)
                
        self.last_lr = current_lr
    
    def _find_best_checkpoint(self, runner):
        """Find the best checkpoint file in the work directory."""
        work_dir = runner.work_dir
        
        # 首先尝试查找以iter为后缀的最佳检查点文件
        import glob
        
        # 查找所有可能的最佳检查点文件模式
        patterns = [
            'best_coco_bbox_mAP_iter_*.pth',
            'best_bbox_mAP_iter_*.pth',
            'best_mAP_iter_*.pth',
            'best_coco_bbox_mAP_epoch_*.pth',
            'best_bbox_mAP_epoch_*.pth',
            'best_mAP_epoch_*.pth'
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(work_dir, pattern))
            if files:
                # 如果找到多个文件，选择最新的（按文件名中的数字排序）
                files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                return files[0]
        
        # 如果没有找到带数字后缀的，尝试固定名称的文件
        fixed_names = [
            'best_coco_bbox_mAP_epoch.pth',
            'best_bbox_mAP_epoch.pth', 
            'best_mAP_epoch.pth',
            'best_epoch.pth'
        ]
        
        for name in fixed_names:
            path = os.path.join(work_dir, name)
            if os.path.exists(path):
                return path
                
        runner.logger.warning('No best checkpoint found in work directory')
        return None