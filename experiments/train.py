"""
训练脚本

Usage:
    python experiments/train.py --config configs/default.yaml
    python experiments/train.py --config configs/default.yaml --resume checkpoints/latest.pth
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AIGCDetector, AIGCDetectorLite
from models.losses import AIGCDetectionLoss
from data import AIGCDataset, FFppDataset, get_transforms, create_dataloader
from trainers import Trainer
from utils.logger import setup_logger


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> nn.Module:
    """构建模型"""
    model_config = config.get('model', {})

    if model_config.get('lite', False):
        model = AIGCDetectorLite(
            num_classes=model_config.get('num_classes', 2),
            img_size=model_config.get('img_size', 224),
            embed_dim=model_config.get('embed_dim', 768),
            num_adapter_layers=model_config.get('num_adapter_layers', 3),
            num_prototypes=model_config.get('num_prototypes', 4),
            use_hierarchical=model_config.get('use_hierarchical', True),
            dropout=model_config.get('dropout', 0.1)
        )
    else:
        model = AIGCDetector(
            backbone_name=model_config.get('backbone_name', 'ViT-L/14'),
            num_classes=model_config.get('num_classes', 2),
            img_size=model_config.get('img_size', 224),
            num_adapter_layers=model_config.get('num_adapter_layers', 3),
            num_prototypes=model_config.get('num_prototypes', 4),
            use_hierarchical=model_config.get('use_hierarchical', True),
            freeze_backbone=model_config.get('freeze_backbone', True),
            dropout=model_config.get('dropout', 0.1)
        )

    return model


def build_criterion(config: dict) -> nn.Module:
    """构建损失函数"""
    loss_config = config.get('loss', {})

    criterion = AIGCDetectionLoss(
        use_focal=loss_config.get('use_focal', True),
        use_contrastive=loss_config.get('use_contrastive', False),
        use_aux=loss_config.get('use_aux', True),
        use_localization=loss_config.get('use_localization', False),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        contrastive_weight=loss_config.get('contrastive_weight', 0.1),
        aux_weight=loss_config.get('aux_weight', 0.5),
        localization_weight=loss_config.get('localization_weight', 0.1)
    )

    return criterion


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """构建优化器"""
    train_config = config.get('training', {})

    # 分离backbone和其他参数
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'clip' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': train_config.get('learning_rate', 1e-4)},
        {'params': backbone_params, 'lr': train_config.get('learning_rate', 1e-4) * 0.1}
    ]

    optimizer_name = train_config.get('optimizer', 'AdamW')

    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int):
    """构建学习率调度器"""
    train_config = config.get('training', {})
    scheduler_name = train_config.get('scheduler', 'CosineAnnealingLR')
    num_epochs = train_config.get('num_epochs', 50)
    warmup_epochs = train_config.get('warmup_epochs', 5)

    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7
        )
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif scheduler_name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config.get('learning_rate', 1e-4),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
    else:
        scheduler = None

    return scheduler


def build_dataloaders(config: dict):
    """构建数据加载器"""
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    aug_config = config.get('augmentation', {})

    data_root = config.get('paths', {}).get('data_root', './datasets')
    batch_size = train_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 8)
    img_size = aug_config.get('resize', 224)

    # 训练集
    train_loader = create_dataloader(
        data_root=data_root,
        split='train',
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )

    # 验证集 - 如果没有val文件夹，使用test文件夹
    val_split = 'val' if (Path(data_root) / 'val').exists() else 'test'
    val_loader = create_dataloader(
        data_root=data_root,
        split=val_split,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )

    return train_loader, val_loader


def main(args):
    """主函数"""
    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    seed = config.get('seed', 42)
    set_seed(seed)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = config.get('paths', {}).get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger('train', log_dir=os.path.join(output_dir, 'logs'))
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")

    # 构建模型
    logger.info("Building model...")
    model = build_model(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 构建损失函数
    criterion = build_criterion(config)

    # 构建数据加载器
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=output_dir,
        use_amp=config.get('training', {}).get('use_amp', True),
        log_interval=config.get('logging', {}).get('log_interval', 100),
        save_interval=config.get('logging', {}).get('save_interval', 5)
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        trainer.resume(args.resume)
        start_epoch = trainer.current_epoch

    # 开始训练
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch
    )

    logger.info("Training completed!")

    # 保存训练历史
    import json
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AIGC Detection Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')

    args = parser.parse_args()
    main(args)
