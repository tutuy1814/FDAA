"""
检查点管理模块

包含:
1. 保存/加载检查点
2. 最佳模型管理
3. 训练状态恢复
"""

import os
import torch
import shutil
from typing import Dict, Any, Optional
from pathlib import Path


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth'
) -> str:
    """
    保存检查点

    Args:
        state: 包含模型状态、优化器状态等的字典
        is_best: 是否为最佳模型
        checkpoint_dir: 检查点目录
        filename: 文件名
    Returns:
        filepath: 保存的文件路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_path)

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        strict: 是否严格匹配模型参数
    Returns:
        checkpoint: 检查点内容
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型状态
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 加载调度器状态
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint


def save_training_state(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_metric: float,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
) -> str:
    """
    保存完整训练状态

    Args:
        epoch: 当前epoch
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        best_metric: 最佳指标值
        metrics: 当前指标
        checkpoint_dir: 检查点目录
        is_best: 是否为最佳
    Returns:
        filepath: 保存路径
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'metrics': metrics
    }

    filename = f'checkpoint_epoch_{epoch}.pth'
    filepath = save_checkpoint(state, is_best, checkpoint_dir, filename)

    # 只保留最近的几个检查点
    cleanup_old_checkpoints(checkpoint_dir, keep_last=3)

    return filepath


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """
    清理旧的检查点，只保留最新的几个

    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留的检查点数量
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 找到所有epoch检查点
    checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pth'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    # 删除旧的检查点
    for ckpt in checkpoints[keep_last:]:
        ckpt.unlink()


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新的检查点

    Args:
        checkpoint_dir: 检查点目录
    Returns:
        latest_checkpoint: 最新检查点路径，如果没有则返回None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # 查找所有检查点
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))

    if not checkpoints:
        # 查找其他可能的检查点文件
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        # 排除best模型
        checkpoints = [c for c in checkpoints if 'best' not in c.name]

    if not checkpoints:
        return None

    # 返回最新的
    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return str(latest)


def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_path: str,
    exclude_keys: list = None,
    prefix: str = ''
) -> torch.nn.Module:
    """
    加载预训练权重（支持部分加载）

    Args:
        model: 目标模型
        pretrained_path: 预训练权重路径
        exclude_keys: 要排除的键
        prefix: 权重键的前缀
    Returns:
        model: 加载了权重的模型
    """
    if exclude_keys is None:
        exclude_keys = []

    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # 处理不同格式的检查点
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    elif 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model']

    model_dict = model.state_dict()

    # 过滤并匹配权重
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        # 添加前缀
        if prefix:
            k = prefix + k

        # 检查是否应该排除
        if any(exc in k for exc in exclude_keys):
            continue

        # 检查是否在模型中存在且形状匹配
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v

    # 更新模型权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(filtered_dict)}/{len(model_dict)} pretrained weights")

    return model


class CheckpointManager:
    """
    检查点管理器
    """
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = 'auc',
        mode: str = 'max'
    ):
        """
        Args:
            checkpoint_dir: 检查点目录
            max_checkpoints: 最大保留检查点数
            metric_name: 用于判断最佳的指标名
            mode: 'max' 或 'min'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.checkpoints = []

    def is_better(self, metric: float) -> bool:
        """判断是否更好"""
        if self.mode == 'max':
            return metric > self.best_metric
        return metric < self.best_metric

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float]
    ) -> str:
        """保存检查点并管理"""
        current_metric = metrics.get(self.metric_name, 0)
        is_best = self.is_better(current_metric)

        if is_best:
            self.best_metric = current_metric

        # 保存检查点
        filepath = save_training_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=self.best_metric,
            metrics=metrics,
            checkpoint_dir=str(self.checkpoint_dir),
            is_best=is_best
        )

        return filepath

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """加载最佳模型"""
        best_path = self.checkpoint_dir / 'model_best.pth'
        if not best_path.exists():
            raise FileNotFoundError("No best model found")
        return load_checkpoint(str(best_path), model, optimizer, scheduler)


# 测试代码
if __name__ == "__main__":
    print("Testing checkpoint utilities...")

    import torch.nn as nn

    # 创建测试模型
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())

    # 测试保存
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        state = {
            'epoch': 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': 0.95
        }

        filepath = save_checkpoint(state, is_best=True, checkpoint_dir=tmpdir)
        print(f"Saved checkpoint to {filepath}")

        # 测试加载
        new_model = nn.Linear(10, 2)
        checkpoint = load_checkpoint(filepath, new_model)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        # 测试CheckpointManager
        manager = CheckpointManager(tmpdir, metric_name='auc', mode='max')
        print(f"Best metric: {manager.best_metric}")

    print("Checkpoint test passed!")
