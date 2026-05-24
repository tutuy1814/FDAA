"""
训练器模块

包含:
1. 主训练循环
2. 验证循环
3. 训练状态管理
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any, Callable
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import MetricsAccumulator
from utils.logger import AverageMeter, MetricLogger, Timer, setup_logger
from utils.checkpoint import CheckpointManager, save_training_state


class Trainer:
    """
    AIGC检测模型训练器
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        config: Optional[Dict] = None,
        output_dir: str = './outputs',
        use_amp: bool = True,
        log_interval: int = 100,
        save_interval: int = 5
    ):
        """
        Args:
            model: 检测模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            config: 配置字典
            output_dir: 输出目录
            use_amp: 是否使用混合精度训练
            log_interval: 日志间隔
            save_interval: 保存间隔
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.output_dir = output_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        self.log_interval = log_interval
        self.save_interval = save_interval

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        # 设置日志
        self.logger = setup_logger(
            'trainer',
            log_dir=os.path.join(output_dir, 'logs')
        )

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None

        # 检查点管理
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
            metric_name='auc',
            mode='max'
        )

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
        Returns:
            metrics: 训练指标
        """
        self.model.train()

        # 指标记录
        loss_meter = AverageMeter('Loss', ':.4f')
        cls_loss_meter = AverageMeter('ClsLoss', ':.4f')
        metric_logger = MetricLogger()

        # 进度条
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            # 数据移至设备
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict['total_loss']

                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict['total_loss']

                # 反向传播
                loss.backward()
                self.optimizer.step()

            # 更新指标
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            cls_loss_meter.update(loss_dict['cls_loss'].item(), batch_size)

            for k, v in loss_dict.items():
                metric_logger.update(**{k: v.item()})

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # 日志
            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.info(
                    f'Epoch [{self.current_epoch}][{batch_idx + 1}/{len(train_loader)}] '
                    f'Loss: {loss_meter.avg:.4f} '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )

            self.global_step += 1

        return metric_logger.get_avg_dict()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器
        Returns:
            metrics: 验证指标
        """
        self.model.eval()

        # 累积器
        accumulator = MetricsAccumulator()
        loss_meter = AverageMeter('Loss', ':.4f')

        pbar = tqdm(val_loader, desc='Validating')

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, labels)

            # 计算预测
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]

            # 更新累积器
            accumulator.update(
                probs.cpu(),
                labels.cpu(),
                methods=batch.get('method', batch.get('class_name', None))
            )

            loss_meter.update(loss_dict['total_loss'].item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # 计算指标
        metrics = accumulator.compute()
        metrics['loss'] = loss_meter.avg

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            start_epoch: 起始epoch
        Returns:
            history: 训练历史
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': []
        }

        timer = Timer()

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics.get('total_loss', 0))

            # 验证
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_acc'].append(val_metrics['accuracy'])

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 日志
            self.logger.info(
                f"Epoch {epoch} completed - "
                f"Train Loss: {train_metrics.get('total_loss', 0):.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # 保存检查点
            is_best = val_metrics['auc'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['auc']
                self.logger.info(f"New best model with AUC: {self.best_metric:.4f}")

            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.checkpoint_manager.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics
                )

        total_time = timer.elapsed()
        self.logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Best AUC: {self.best_metric:.4f}")

        return history

    def resume(self, checkpoint_path: str):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点路径
        """
        from utils.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler
        )

        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_metric = checkpoint.get('best_metric', 0)

        self.logger.info(f"Resumed from epoch {self.current_epoch - 1}")
        self.logger.info(f"Best metric so far: {self.best_metric:.4f}")


class EvaluatorMixin:
    """
    评估混入类
    """
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        评估模型

        Args:
            model: 模型
            test_loader: 测试数据加载器
            device: 设备
        Returns:
            results: 评估结果
        """
        model.eval()
        model = model.to(device)

        accumulator = MetricsAccumulator()

        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]

            accumulator.update(
                probs.cpu(),
                labels,
                paths=batch.get('path'),
                methods=batch.get('method', batch.get('class_name'))
            )

        # 整体指标
        overall_metrics = accumulator.compute()

        # 按方法分组指标
        per_method_metrics = accumulator.compute_per_method()

        return {
            'overall': overall_metrics,
            'per_method': per_method_metrics
        }


# 测试代码
if __name__ == "__main__":
    print("Testing trainer...")

    import torch.nn as nn

    # 简单模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, 2)

        def forward(self, x, **kwargs):
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            return {'logits': logits}

    # 简单损失
    class DummyLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss()

        def forward(self, outputs, labels, **kwargs):
            loss = self.ce(outputs['logits'], labels)
            return {'cls_loss': loss, 'total_loss': loss}

    model = DummyModel()
    criterion = DummyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device='cpu',
            output_dir=tmpdir,
            use_amp=False
        )

        print("Trainer initialized successfully")

    print("Trainer test passed!")
