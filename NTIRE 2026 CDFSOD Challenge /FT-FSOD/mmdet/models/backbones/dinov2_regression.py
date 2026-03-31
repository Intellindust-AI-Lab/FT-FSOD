# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
import os
import sys
from PIL import Image

from mmdet.registry import MODELS


def get_dinov2_transform(image_size=(518, 518)):
    """Returns the preprocessing transform for DINOv2."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def resize_with_aspect_ratio(img_pil, target_long_side=1024, patch_size=16):
    """
    Resize a PIL image to have a specific long side, maintaining aspect ratio,
    and ensure new dimensions are multiples of the patch size.
    Uses BICUBIC filter for resampling.

    Args:
        img_pil (PIL.Image): Input image.
        target_long_side (int): Desired size of the longer side.
        patch_size (int): Size of the patches, new dimensions must be multiples of this.

    Returns:
        PIL.Image: Resized image with dimensions as multiples of patch_size.
    """
    orig_width, orig_height = img_pil.size
    aspect_ratio = orig_width / orig_height

    # Calculate initial resized dimensions based on long side
    if orig_width >= orig_height:
        new_width = target_long_side
        new_height = int(target_long_side / aspect_ratio)
    else:
        new_height = target_long_side
        new_width = int(target_long_side * aspect_ratio)

    # Ensure dimensions are multiples of patch_size
    # Using floor division to guarantee we don't exceed target_long_side
    new_width = max((new_width // patch_size), 1) * patch_size
    new_height = max((new_height // patch_size), 1) * patch_size

    return img_pil.resize((new_width, new_height), resample=Image.BICUBIC)


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # tensor shape: [3, H, W] or [B, 3, H, W]
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    # Denormalize from [-1, 1] or [0, 1] to [0, 255]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # From [-1, 1] to [0, 1]
    
    tensor = (tensor * 255).clamp(0, 255).byte()
    tensor = tensor.cpu()
    
    # Convert to PIL
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(tensor)


def pil_to_tensor(pil_img, device='cpu'):
    """Convert PIL Image to tensor"""
    from torchvision.transforms.functional import to_tensor
    return to_tensor(pil_img).to(device)


@MODELS.register_module()
class DINOv2Regression(BaseModule):
    """真实的DINOv2回归模型，集成Facebook的DINOv2
    
    Args:
        dinov2_model_name (str): DINOv2模型名称，如'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'
        output_dim (int): 输出特征维度，默认384
        repo_or_dir (str): DINOv2仓库路径
        pretrained (str): 预训练权重路径
        image_size (tuple): 输入图像尺寸
        freeze_backbone (bool): 是否冻结DINOv2骨干网络
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(self,
                 dinov2_model_name='dinov2_vits14',
                 repo_or_dir="/mnt/data2/FSOD_FoundationModels-main/dinov2",
                 pretrained="/mnt/data2/FSOD_FoundationModels-main/checkpoints/dinov2_vits14_pretrain.pth",
                 image_size=(518, 518),
                 target_long_side=630,
                 patch_size=14,
                 freeze_backbone=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.dinov2_model_name = dinov2_model_name
        self.repo_or_dir = repo_or_dir
        self.pretrained = pretrained
        self.image_size = image_size
        self.target_long_side = target_long_side
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        
        # DINOv2模型的特征维度映射
        self.model_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        
        self.feature_dim = self.model_dims.get(dinov2_model_name, 384)
        
        # 获取DINOv2预处理变换
        self.dinov2_transform = get_dinov2_transform(image_size)
        
        # 加载DINOv2模型
        self.dinov2_model = self._load_dinov2_model()
        
        # 注意：feature_projection现在移到GroundingDINO_DINOV2中，以确保在优化器参数组中
        
        # 如果需要冻结backbone
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def _load_dinov2_model(self):
        """加载DINOv2模型"""
        try:
            print(f"正在加载DINOv2模型: {self.dinov2_model_name}")
            print(f"仓库路径: {self.repo_or_dir}")
            print(f"权重路径: {self.pretrained}")
            
            # 加载DINOv2模型
            if os.path.exists(self.repo_or_dir):
                # 从本地路径加载
                dinov2_model = torch.hub.load(
                    repo_or_dir=self.repo_or_dir,
                    model=self.dinov2_model_name,
                    source='local',
                    pretrained=False  # 先不加载权重，稍后手动加载
                )
                
                # 手动加载预训练权重
                if self.pretrained and os.path.exists(self.pretrained):
                    print(f"加载预训练权重: {self.pretrained}")
                    state_dict = torch.load(self.pretrained, map_location='cpu')
                    dinov2_model.load_state_dict(state_dict, strict=False)
                
            else:
                # 从在线加载
                print("本地路径不存在，从在线加载DINOv2模型")
                dinov2_model = torch.hub.load('facebookresearch/dinov2', self.dinov2_model_name, pretrained=True)
            
            print("✅ DINOv2模型加载成功")
            return dinov2_model
            
        except Exception as e:
            print(f"❌ 无法加载真实的DINOv2模型: {e}")
            exit()
    
    def get_dinov2_features(self, dinov2_model, input_tensor, device='cpu'):
        """
        提取DINOv2特征，使用中间层特征
        
        Args:
            dinov2_model: DINOv2模型
            input_tensor: 输入张量 [B, 3, H, W]
            device: 设备
            
        Returns:
            torch.Tensor: 特征图 [B, C, H_feat, W_feat]
        """
        # 直接进行前向传播，让优化器的lr_mult=0.0控制参数更新
        # 使用get_intermediate_layers提取中间层特征
        output = dinov2_model.get_intermediate_layers(
            input_tensor, 
            n=1, 
            reshape=True, 
            return_class_token=True, 
            norm=True
        )
        # output = output[0]
        # print('len(output)', len(output)) # 1
        # print('output.shape', len(output)) # 2
        # for out in output:
        # print('out.shape', output[0].shape) # torch.Size([1, 384, 22, 45])
        # print('len(out)', len(output)) # 2
        # print('out[1].shape', output[1].shape) # torch.Size([1, 384])
        output = torch.stack([out[0] for out in output], dim=0).sum(dim=0)
        print('output.shape', output.shape)
        # 堆叠并求和多层特征
        return output  # Shape: (B, C, H_feat, W_feat)
    
    def _freeze_backbone(self):
        """冻结DINOv2 backbone参数"""
        if hasattr(self.dinov2_model, 'parameters'):
            for param in self.dinov2_model.parameters():
                param.requires_grad = False
            print("✅ DINOv2 backbone已冻结")
    
    def _preprocess_for_dinov2_with_resize(self, x):
        """为DINOv2预处理输入图像，包含适当的resize
        
        Args:
            x (Tensor): 来自GroundingDINO的输入 [B, 3, H, W]，已经过DetDataPreprocessor处理
            
        Returns:
            Tensor: 适合DINOv2的输入 [B, 3, H, W]，经过正确resize和归一化
        """
        B, C, orig_H, orig_W = x.shape
        
        # 先反归一化GroundingDINO的预处理得到[0, 255]范围的RGB图像
        gdino_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(x.device)
        gdino_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(x.device)
        x_denorm = x * gdino_std + gdino_mean
        
        # 转换到[0, 1]范围
        x_01 = x_denorm / 255.0
        
        # 为每个batch元素进行resize
        processed_tensors = []
        for i in range(B):
            # 转换为PIL图像
            pil_img = tensor_to_pil(x_01[i])
            
            # 使用aspect ratio resize
            pil_resized = resize_with_aspect_ratio(
                pil_img, 
                target_long_side=self.target_long_side, 
                patch_size=self.patch_size
            )
            
            # 转换回tensor并应用DINOv2归一化
            tensor_resized = pil_to_tensor(pil_resized, device=x.device)
            
            # 应用DINOv2归一化
            dinov2_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
            dinov2_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)
            tensor_normalized = (tensor_resized - dinov2_mean) / dinov2_std
            
            processed_tensors.append(tensor_normalized)
        
        # 堆叠回batch
        return torch.stack(processed_tensors, dim=0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入图像 [B, 3, H, W]，来自GroundingDINO的预处理
            
        Returns:
            Tensor: 特征图 [B, output_dim, H', W']，尺寸与输入图像相同
        """
        # 保存原始尺寸用于后续resize
        B, C, orig_H, orig_W = x.shape
        orig_H = orig_H//2
        orig_W = orig_W//2
        
        # 为DINOv2重新预处理输入（包含resize）
        x_processed = self._preprocess_for_dinov2_with_resize(x)
        
        # DINOv2前向传播
        # 直接进行前向传播，让优化器的lr_mult=0.0来控制参数更新
        if hasattr(self.dinov2_model, 'get_intermediate_layers'):
            # 使用中间层特征提取
            features = self.get_dinov2_features(self.dinov2_model, x_processed, x.device)
        else:
            print('错了！')
            exit()
        
        # 返回原始特征维度[B, feature_dim, H', W']和目标尺寸
        # resize将在GroundingDINO中进行，这样可以减少显存占用：先投影降维再resize
        return features, (orig_H, orig_W)
    
    def init_weights(self):
        """初始化权重"""
        # DINOv2权重已经是预训练的，feature_projection现在在GroundingDINO中初始化
        pass
