"""
统一论文实验脚本 — FDAA-Net V2

一个入口脚本完成论文所需的全部实验:
1. 训练我们的模型 (多源GenImage, CLIP ViT-L/14 backbone)
2. 训练SOTA对比方法 (相同数据, 相同epoch, 相同评估)
3. 消融实验 (Baseline / +FDAA / +MGFP / Full)
4. 跨域泛化评估 (跨生成器 + 跨数据集)
5. 鲁棒性测试 (JPEG压缩, 高斯噪声, 高斯模糊)
6. 结果汇总与表格生成

使用方法:
    # 完整实验 (所有步骤)
    python experiments/run_paper_experiments.py --mode all

    # 分步执行
    python experiments/run_paper_experiments.py --mode train_ours
    python experiments/run_paper_experiments.py --mode train_sota
    python experiments/run_paper_experiments.py --mode ablation
    python experiments/run_paper_experiments.py --mode cross_domain
    python experiments/run_paper_experiments.py --mode robustness
    python experiments/run_paper_experiments.py --mode report
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import io

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import AIGCDetectorV2, AIGCDetectorV3
from models.losses.losses import AIGCDetectionLoss
from models.sota_methods import create_sota_model, SOTA_METHODS
from data.multi_source_dataset import (
    create_multi_source_dataloader,
    MultiSourceGenImageDataset,
    get_multi_source_transforms,
)
from data.genimage_dataset import GenImageDataset, get_genimage_transforms
from utils.metrics import compute_metrics, MetricsAccumulator


# =============================================================================
# 论文实验配置
# =============================================================================

DEFAULT_DATASETS_ROOT = Path(
    os.environ.get(
        'FDAA_DATASETS_ROOT',
        str(PROJECT_ROOT / 'datasets' / 'authoritative'),
    )
)

PAPER_CONFIG = {
    # 数据设置
    'genimage_root': os.environ.get(
        'FDAA_GENIMAGE_ROOT',
        str(DEFAULT_DATASETS_ROOT / 'GenImage'),
    ),
    'train_sources': ['biggan', 'adm', 'glide', 'vqdm'],
    'max_samples_per_source': 10000,

    # 训练设置
    'batch_size': 64,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 20,
    'warmup_epochs': 5,
    'num_workers': 8,
    'img_size': 224,
    'embed_dim': 1024,  # CLIP ViT-L/14 内部维度

    # 模型设置
    'backbone': 'ViT-L/14',
    'use_hierarchical': True,
    'dropout': 0.1,

    # 损失设置
    'use_focal': True,
    'use_contrastive': True,
    'contrastive_weight': 0.3,
    'aux_weight': 0.3,
    'label_smoothing': 0.05,

    # SOTA 对比方法 (与 Ours 使用完全相同的训练设置)
    'sota_methods': ['cnndetection', 'f3net', 'univfd', 'freqnet', 'npr', 'spec', 'dire', 'lare2', 'drct', 'c2pclip'],
    'sota_epochs': 20,  # 与 Ours 相同

    # 评估数据集 (跨数据集泛化)
    'eval_datasets': {
        'diffusion_forensics': os.environ.get(
            'FDAA_DIFFUSION_FORENSICS_ROOT',
            str(DEFAULT_DATASETS_ROOT / 'DiffusionForensics'),
        ),
        'cifake': os.environ.get(
            'FDAA_CIFAKE_ROOT',
            str(DEFAULT_DATASETS_ROOT / 'CIFAKE'),
        ),
        'ntire2026': os.environ.get(
            'FDAA_NTIRE2026_ROOT',
            str(DEFAULT_DATASETS_ROOT / 'NTIRE2026'),
        ),
    },

    # 跨生成器评估 (GenImage 内部)
    'cross_gen_sources': ['biggan', 'adm', 'glide', 'vqdm', 'sdv4', 'midjourney'],

    # 鲁棒性测试
    'robustness_tests': {
        'jpeg_70': {'type': 'jpeg', 'quality': 70},
        'jpeg_50': {'type': 'jpeg', 'quality': 50},
        'jpeg_30': {'type': 'jpeg', 'quality': 30},
        'blur_1.0': {'type': 'blur', 'radius': 1.0},
        'blur_2.0': {'type': 'blur', 'radius': 2.0},
        'noise_0.02': {'type': 'noise', 'std': 0.02},
        'noise_0.05': {'type': 'noise', 'std': 0.05},
    },

    # V3 设置
    'v3_epochs': 25,
    'v3_contrastive_weight': 0.5,

    # 输出
    'output_dir': str(PROJECT_ROOT / 'outputs' / 'paper_results'),
}


# =============================================================================
# 工具函数
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Device] {num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return 'cuda', num_gpus
    print("[Device] Using CPU")
    return 'cpu', 0


def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ['checkpoints', 'logs', 'results', 'reports']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir


def _json_default(obj):
    """JSON 序列化: numpy 类型 → Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[Save] {filepath}")


def fmt(metrics):
    """格式化指标"""
    parts = []
    for k in ['auc', 'ap', 'accuracy', 'eer']:
        if k in metrics:
            parts.append(f"{k.upper()}: {metrics[k]*100:.2f}%")
    return ' | '.join(parts)


# =============================================================================
# 训练核心
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    warmup_epochs=5, base_lr=5e-4, source_to_id=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # 源 ID（用于源感知对比损失）
        sources = None
        if source_to_id is not None and 'source' in batch:
            source_ids = torch.tensor(
                [source_to_id.get(s, 0) for s in batch['source']],
                dtype=torch.long, device=device
            )
            sources = source_ids

        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(loader) + batch_idx) / (warmup_epochs * len(loader))
            lr = base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group.get('_lr_scale', 1.0)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast('cuda'):
                outputs = model(images, return_features=True)
                features = outputs.get('features', None)
                loss_dict = criterion(outputs, labels, features=features, sources=sources)
                loss = loss_dict['total_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, return_features=True)
            features = outputs.get('features', None)
            loss_dict = criterion(outputs, labels, features=features, sources=sources)
            loss = loss_dict['total_loss']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {'loss': total_loss / total, 'accuracy': correct / total}


@torch.no_grad()
def evaluate_model(model, loader, device, desc='Eval'):
    model.eval()
    accumulator = MetricsAccumulator()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']

            if device == 'cuda':
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.softmax(outputs['logits'].float(), dim=1)[:, 1]
            accumulator.update(probs.cpu(), labels)

    return accumulator.compute()


def train_model(model, train_loader, val_loader, config, model_name='model',
                output_dir='./outputs', device='cuda', num_gpus=1,
                is_our_model=False, source_to_id=None):
    """完整训练流程"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # 损失函数
    criterion = AIGCDetectionLoss(
        use_focal=config.get('use_focal', True),
        use_contrastive=config.get('use_contrastive', False) if not is_our_model else config.get('use_contrastive', True),
        contrastive_weight=config.get('contrastive_weight', 0.3),
        use_aux=config.get('use_hierarchical', False) if is_our_model else False,
        aux_weight=config.get('aux_weight', 0.3),
        label_smoothing=config.get('label_smoothing', 0.05),
    )

    # 优化器 — 分层学习率（仅限我们的模型）
    raw = model.module if hasattr(model, 'module') else model
    base_lr = config['lr']

    if is_our_model and hasattr(raw, 'fdaa') and hasattr(raw, 'mgfp'):
        # SRM 参数使用极低学习率，保护手工滤波器初始化
        srm_params = list(raw.fdaa.srm.parameters()) if hasattr(raw.fdaa, 'srm') else []
        srm_param_ids = {id(p) for p in srm_params}
        fdaa_other_params = [p for p in raw.fdaa.parameters()
                             if id(p) not in srm_param_ids and p.requires_grad]

        param_groups = []
        if srm_params:
            param_groups.append({'params': srm_params, 'lr': base_lr * 0.1, '_lr_scale': 0.1})
        if fdaa_other_params:
            param_groups.append({'params': fdaa_other_params, 'lr': base_lr, '_lr_scale': 1.0})

        mgfp_params = [p for p in raw.mgfp.parameters() if p.requires_grad]
        if mgfp_params:
            param_groups.append({'params': mgfp_params, 'lr': base_lr, '_lr_scale': 1.0})

        cls_params = [p for p in raw.classifier.parameters() if p.requires_grad]
        if cls_params:
            param_groups.append({'params': cls_params, 'lr': base_lr, '_lr_scale': 1.0})

        if hasattr(raw, 'patch_norm'):
            pn_params = [p for p in raw.patch_norm.parameters() if p.requires_grad]
            if pn_params:
                param_groups.append({'params': pn_params, 'lr': base_lr, '_lr_scale': 1.0})
        if hasattr(raw, 'aux_classifier'):
            aux_params = [p for p in raw.aux_classifier.parameters() if p.requires_grad]
            if aux_params:
                param_groups.append({'params': aux_params, 'lr': base_lr, '_lr_scale': 1.0})
        optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=config['weight_decay']
        )

    # 学习率调度器
    epochs = config.get('sota_epochs', config['epochs']) if not is_our_model else config['epochs']
    warmup_epochs = config.get('warmup_epochs', 5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
    )

    scaler = GradScaler('cuda') if device == 'cuda' else None

    best_auc = 0
    best_metrics = {}
    training_log = []  # 每 epoch 记录，用于训练曲线

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, warmup_epochs=warmup_epochs, base_lr=base_lr,
            source_to_id=source_to_id,
        )

        if epoch >= warmup_epochs:
            scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device, desc=f'Val E{epoch}')

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']*100:.1f}% | "
              f"Val {fmt(val_metrics)}")

        # 记录训练曲线数据
        training_log.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_auc': val_metrics.get('auc', 0),
            'val_ap': val_metrics.get('ap', 0),
            'val_acc': val_metrics.get('accuracy', 0),
            'val_eer': val_metrics.get('eer', 0),
            'lr': optimizer.param_groups[0]['lr'],
        })

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch

            save_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'metrics': best_metrics,
            }, os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f"  -> New best AUC: {best_auc*100:.2f}%")

    # 保存训练曲线日志
    log_path = os.path.join(output_dir, 'logs', f'{model_name}_training_log.json')
    save_json(training_log, log_path)

    print(f"\n{model_name} done. Best AUC: {best_auc*100:.2f}% (epoch {best_metrics.get('epoch', -1)})")
    return best_metrics


# =============================================================================
# 数据工具
# =============================================================================

class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'path': str(path)}


class DegradedDataset(torch.utils.data.Dataset):
    """对已有数据集应用降质操作"""
    def __init__(self, base_dataset, degradation_type, degradation_param, img_size=224, seed=42):
        self.base_dataset = base_dataset
        self.deg_type = degradation_type
        self.deg_param = degradation_param
        self.img_size = img_size
        self.seed = seed
        from torchvision import transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.resize = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        path = item.get('path', '')
        label = item['label'] if isinstance(item['label'], int) else item['label'].item()

        # base_dataset (transform=None) 已经加载了 PIL Image，直接复用
        img = item.get('image')
        if img is None or not isinstance(img, Image.Image):
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"[DegradedDataset] Error loading {path}: {e}")
                img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))

        img = self.resize(img)

        # 应用降质
        if self.deg_type == 'jpeg':
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=self.deg_param)
            buf.seek(0)
            img_jpeg = Image.open(buf).convert('RGB')
            img = img_jpeg.copy()  # 复制后释放 BytesIO 引用
            buf.close()
        elif self.deg_type == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=self.deg_param))
        elif self.deg_type == 'noise':
            # 使用确定性种子：seed + idx 确保可复现
            rng = np.random.RandomState(self.seed + idx)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr + rng.normal(0, self.deg_param, arr.shape)
            arr = np.clip(arr, 0, 1) * 255
            img = Image.fromarray(arr.astype(np.uint8))

        tensor = self.normalize(self.to_tensor(img))
        return {'image': tensor, 'label': label, 'path': path}


def create_eval_loader(dataset_root, split='test', batch_size=64, img_size=224, num_workers=4):
    transform = get_multi_source_transforms(img_size, is_train=False)
    dataset_root = Path(dataset_root)
    split_dir = dataset_root / split
    if not split_dir.exists():
        split_dir = dataset_root

    if (split_dir / 'ai').exists() and (split_dir / 'nature').exists():
        dataset = GenImageDataset(
            root_dir=str(dataset_root), split=split,
            transform=transform, balance_classes=False
        )
    elif (split_dir / 'real').exists() and (split_dir / 'fake').exists():
        samples = []
        extensions = {'.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG', '.bmp', '.webp'}
        for p in (split_dir / 'real').iterdir():
            if p.suffix in extensions:
                samples.append((p, 0))
        for p in (split_dir / 'fake').iterdir():
            if p.suffix in extensions:
                samples.append((p, 1))
        dataset = SimpleImageDataset(samples, transform)
    else:
        # NTIRE2026 格式: shard_*/images/ + shard_*/labels.csv
        import csv
        samples = []
        extensions = {'.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG', '.bmp', '.webp'}
        for shard_dir in sorted(dataset_root.glob('shard_*')):
            labels_csv = shard_dir / 'labels.csv'
            images_dir = shard_dir / 'images'
            if labels_csv.exists() and images_dir.exists():
                with open(labels_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_path = images_dir / row['image_name']
                        if img_path.suffix in extensions:
                            samples.append((str(img_path), int(row['label'])))
        if samples:
            dataset = SimpleImageDataset(samples, transform)
        else:
            print(f"[Warning] Unknown dataset format at {dataset_root}")
            return None

    if len(dataset) == 0:
        return None

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# =============================================================================
# 实验 1: 训练我们的模型
# =============================================================================

def exp_train_ours(config, device, num_gpus):
    """训练 FDAA-Net V2 (主模型)"""
    output_dir = setup_output_dir(config['output_dir'])

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    model = AIGCDetectorV2(
        backbone_name=config['backbone'],
        num_classes=2,
        img_size=config['img_size'],
        embed_dim=config['embed_dim'],
        use_hierarchical=config['use_hierarchical'],
        dropout=config['dropout'],
        freeze_backbone=True,
    )

    best_metrics = train_model(
        model, train_loader, val_loader, config,
        model_name='fdaa_net_v2',
        output_dir=output_dir,
        device=device, num_gpus=num_gpus,
        is_our_model=True,
    )

    save_json({'fdaa_net_v2': best_metrics},
              os.path.join(output_dir, 'results', 'ours_train_results.json'))
    return best_metrics


# =============================================================================
# 实验 1b: 训练 V3 模型
# =============================================================================

def exp_train_ours_v3(config, device, num_gpus):
    """训练 FDAA-Net V3 (空间频率特征 + 频率引导注意力 + 源感知对比)"""
    output_dir = setup_output_dir(config['output_dir'])

    # 源到 ID 映射
    source_to_id = {s: i for i, s in enumerate(config['train_sources'])}

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
        aug_version='v3',  # V3 增强：更强频率扰动 + 社交媒体模拟
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
        aug_version='v3',
    )

    model = AIGCDetectorV3(
        backbone_name=config['backbone'],
        num_classes=2,
        img_size=config['img_size'],
        embed_dim=config['embed_dim'],
        use_hierarchical=config['use_hierarchical'],
        dropout=config['dropout'],
        freeze_backbone=True,
    )

    # V3 使用更强的对比损失权重
    v3_config = config.copy()
    v3_config['contrastive_weight'] = config.get('v3_contrastive_weight', 0.5)
    v3_config['epochs'] = config.get('v3_epochs', 25)

    best_metrics = train_model(
        model, train_loader, val_loader, v3_config,
        model_name='fdaa_net_v3',
        output_dir=output_dir,
        device=device, num_gpus=num_gpus,
        is_our_model=True,
        source_to_id=source_to_id,
    )

    save_json({'fdaa_net_v3': best_metrics},
              os.path.join(output_dir, 'results', 'ours_v3_train_results.json'))
    return best_metrics


# =============================================================================
# 实验 2: 训练 SOTA 对比方法
# =============================================================================

def exp_train_sota(config, device, num_gpus):
    """训练所有 SOTA 方法 (相同数据、相同 epoch)"""
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config.get('sota_val_samples', 5000),
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    for method_name in config.get('sota_methods', []):
        print(f"\n{'='*60}")
        print(f"Training SOTA: {method_name}")
        print(f"{'='*60}")

        # 如果 checkpoint 已存在，直接加载评估（跳过重训练）
        ckpt_path = os.path.join(checkpoint_dir, f'{method_name}_best.pth')
        if os.path.exists(ckpt_path):
            try:
                print(f"  -> Checkpoint exists, evaluating: {ckpt_path}")
                model = create_sota_model(method_name, num_classes=2)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)
                metrics = evaluate_model(model, val_loader, device, desc=f'{method_name} eval')
                metrics['epoch'] = ckpt.get('epoch', -1)
                results[method_name] = metrics
                print(f"  {method_name} (reused): AUC={metrics['auc']*100:.2f}% (epoch {metrics['epoch']})")
                del model; torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [Warning] Checkpoint reuse failed: {e}, retraining...")

        try:
            model = create_sota_model(method_name, num_classes=2)
            best_metrics = train_model(
                model, train_loader, val_loader, config,
                model_name=method_name,
                output_dir=output_dir,
                device=device, num_gpus=num_gpus,
                is_our_model=False,
            )
            results[method_name] = best_metrics
        except Exception as e:
            print(f"[Error] {method_name}: {e}")
            import traceback; traceback.print_exc()
            results[method_name] = {'error': str(e)}

    save_json(results, os.path.join(output_dir, 'results', 'sota_train_results.json'))
    return results


# =============================================================================
# 实验 3: 消融实验
# =============================================================================

def _create_ablation_model(config, variant):
    """
    创建消融变体模型（均基于 CLIP ViT-L/14，与主模型相同 backbone）

    变体:
      - 'baseline': CLIP + CLS-only 分类（无 FDAA, 无 MGFP）
      - 'baseline+fdaa': CLIP + FDAAv2 + 简单 CLS+freq 融合（无 MGFP attention）
      - 'baseline+mgfp': CLIP + MGFPv2（无 freq 输入）
      - 'full': 完整模型（FDAA + MGFP + hierarchical）
    """
    embed_dim = config['embed_dim']

    # baseline+mgfp 和 full 均使用 hierarchical，以公平评估 MGFP 贡献
    use_hier = variant in ('baseline+mgfp', 'full')
    model = AIGCDetectorV2(
        backbone_name=config['backbone'],
        num_classes=2,
        img_size=config['img_size'],
        embed_dim=embed_dim,
        use_hierarchical=use_hier,
        dropout=config['dropout'],
        freeze_backbone=True,
    )

    # 零频率输出模块（替代 FDAA）
    class ZeroFreq(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, x):
            return torch.zeros(x.shape[0], self.dim, device=x.device)

    if variant == 'baseline':
        # 替换 MGFP 为 CLS-only
        class CLSOnlyPool(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
            def forward(self, cls_token, patch_tokens, freq_feat, return_attention=False):
                out = self.norm(cls_token)
                if return_attention:
                    B, N, _ = patch_tokens.shape
                    return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
                return out

        model.fdaa = ZeroFreq(embed_dim)
        model.mgfp = CLSOnlyPool(embed_dim)

    elif variant == 'baseline+fdaa':
        # 保留 FDAAv2，替换 MGFP 为简单 CLS+freq concat
        class CLSFreqPool(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                )
            def forward(self, cls_token, patch_tokens, freq_feat, return_attention=False):
                out = self.mlp(torch.cat([cls_token, freq_feat], dim=-1))
                if return_attention:
                    B, N, _ = patch_tokens.shape
                    return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
                return out

        model.mgfp = CLSFreqPool(embed_dim)

    elif variant == 'baseline+mgfp':
        model.fdaa = ZeroFreq(embed_dim)

    # variant == 'full': 不做任何替换

    return model


def exp_ablation(config, device, num_gpus):
    """
    消融实验: Baseline / +FDAA / +MGFP / Full

    所有变体使用相同的 CLIP ViT-L/14 backbone，确保公平对比。
    仅改变 FDAA 和 MGFP 模块的有无。
    """
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    abl_config = config.copy()
    abl_epochs = min(config['epochs'], 10)  # 消融统一 10 epoch（效率 + 公平）
    abl_config['epochs'] = abl_epochs

    variants = [
        ('baseline',       'Baseline (CLIP ViT-L/14 + CLS-only)'),
        ('baseline+fdaa',  'Baseline + FDAA (CLS + freq concat)'),
        ('baseline+mgfp',  'Baseline + MGFP (no freq input)'),
        ('full',           'Full (FDAA + MGFP + hierarchical)'),
    ]

    # 检查 train_ours 的 checkpoint 和训练日志
    ours_ckpt = os.path.join(output_dir, 'checkpoints', 'fdaa_net_v2_best.pth')
    ours_log_path = os.path.join(output_dir, 'logs', 'fdaa_net_v2_training_log.json')
    ours_ckpt_exists = os.path.exists(ours_ckpt)

    for i, (variant, desc) in enumerate(variants):
        print(f"\n[Ablation {i+1}/{len(variants)}] {desc}")

        # full 变体: 复用 train_ours，但公平使用 abl_epochs 处的性能
        if variant == 'full' and ours_ckpt_exists:
            model = _create_ablation_model(config, variant)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # 优先从训练日志中提取 abl_epochs 处的指标（公平对比）
            used_log = False
            if os.path.exists(ours_log_path):
                try:
                    with open(ours_log_path) as fp:
                        ours_log = json.load(fp)
                    # 取前 abl_epochs 个 epoch 中 AUC 最高的（与其他变体一致，都取 best）
                    candidates = ours_log[:abl_epochs]
                    if candidates:
                        log_entry = max(candidates, key=lambda e: e['val_auc'])
                        results[variant] = {
                            'auc': log_entry['val_auc'],
                            'ap': log_entry['val_ap'],
                            'accuracy': log_entry['val_acc'],
                            'eer': log_entry['val_eer'],
                            'epoch': log_entry['epoch'],
                        }
                        used_log = True
                        print(f"  -> Using best of first {abl_epochs} epochs: epoch {log_entry['epoch']} (AUC={log_entry['val_auc']*100:.2f}%)")
                except Exception as e:
                    print(f"  [Warning] Failed to read training log: {e}")

            # 回退: 加载 checkpoint 评估
            if not used_log:
                print(f"  -> Reusing train_ours checkpoint (evaluate on val)")
                ckpt = torch.load(ours_ckpt, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)
                results[variant] = evaluate_model(model, val_loader, device, desc='Abl-Full Val')
                results[variant]['epoch'] = ckpt.get('epoch', -1)
                del model; torch.cuda.empty_cache()

            results[variant]['trainable_params'] = trainable
            results[variant]['trainable_params_M'] = trainable / 1e6
            print(f"  Full: {fmt(results[variant])}")
            continue

        model = _create_ablation_model(config, variant)
        # 统一使用对比损失(所有变体均产出features)，公平对比
        abl_config['use_contrastive'] = True
        abl_config['use_hierarchical'] = variant in ('baseline+mgfp', 'full')

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        metrics = train_model(
            model, train_loader, val_loader, abl_config,
            model_name=f'abl_{variant.replace("+", "_")}',
            output_dir=output_dir,
            device=device, num_gpus=num_gpus,
            is_our_model=True,  # 统一LR策略(SRM分组由hasattr自动处理)
        )
        metrics['trainable_params'] = trainable
        metrics['trainable_params_M'] = trainable / 1e6
        results[variant] = metrics

    # -----------------------------------------------
    # 3b. 损失函数消融（仅在 full 模型上测试）
    # -----------------------------------------------
    print(f"\n{'='*60}")
    print("Loss Function Ablation (on full model)")
    print(f"{'='*60}")

    loss_variants = [
        ('focal_only',       {'use_focal': True, 'use_contrastive': False, 'use_hierarchical': False}),
        ('focal+contrastive', {'use_focal': True, 'use_contrastive': True, 'use_hierarchical': False}),
        ('focal+aux',        {'use_focal': True, 'use_contrastive': False, 'use_hierarchical': True}),
        ('focal+contr+aux',  {'use_focal': True, 'use_contrastive': True, 'use_hierarchical': True}),
    ]

    loss_results = {}
    for lv_name, lv_cfg in loss_variants:
        # focal+contr+aux 复用 train_ours 的 full 结果
        if lv_name == 'focal+contr+aux' and 'full' in results:
            loss_results[lv_name] = results['full']
            print(f"  {lv_name}: reused from full variant")
            continue

        print(f"\n[Loss Ablation] {lv_name}")
        model = AIGCDetectorV2(
            backbone_name=config['backbone'], num_classes=2,
            img_size=config['img_size'], embed_dim=config['embed_dim'],
            use_hierarchical=lv_cfg['use_hierarchical'],
            dropout=config['dropout'], freeze_backbone=True,
        )

        la_config = abl_config.copy()
        la_config.update(lv_cfg)
        loss_results[lv_name] = train_model(
            model, train_loader, val_loader, la_config,
            model_name=f'abl_loss_{lv_name.replace("+", "_")}',
            output_dir=output_dir, device=device, num_gpus=num_gpus,
            is_our_model=True,  # 所有变体均为 AIGCDetectorV2(含SRM)，统一LR策略
        )

    results['_loss_ablation'] = loss_results

    save_json(results, os.path.join(output_dir, 'results', 'ablation_results.json'))
    return results


# =============================================================================
# 实验 3b: V3 消融实验
# =============================================================================

def _create_ablation_model_v3(config, variant):
    """
    创建 V3 消融变体模型

    变体:
      - 'baseline': CLIP + CLS-only（无 FDAA, 无 MGFP）
      - 'baseline+fdaa': CLIP + FDAAv3 + 简单 CLS+freq_global concat（无 MGFP）
      - 'baseline+mgfp': CLIP + MGFPv3（无 freq_tokens, 无 freq_global → fallback）
      - 'full': 完整 V3（FDAAv3 + MGFPv3 + hierarchical）
    """
    embed_dim = config['embed_dim']
    use_hier = variant in ('baseline+mgfp', 'full')

    model = AIGCDetectorV3(
        backbone_name=config['backbone'],
        num_classes=2,
        img_size=config['img_size'],
        embed_dim=embed_dim,
        use_hierarchical=use_hier,
        dropout=config['dropout'],
        freeze_backbone=True,
    )

    class ZeroFreqV3(nn.Module):
        """替代 FDAAv3，输出零频率 tokens 和 global"""
        def __init__(self, dim, num_patches):
            super().__init__()
            self.dim = dim
            self.num_patches = num_patches
        def forward(self, x):
            B = x.shape[0]
            device = x.device
            return (torch.zeros(B, self.num_patches, self.dim, device=device),
                    torch.zeros(B, self.dim, device=device))

    num_patches = (config['img_size'] // 14) ** 2  # CLIP ViT-L/14

    if variant == 'baseline':
        class CLSOnlyPoolV3(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
            def forward(self, cls_token, patch_tokens, freq_tokens=None,
                        freq_global=None, return_attention=False):
                out = self.norm(cls_token)
                if return_attention:
                    B, N, _ = patch_tokens.shape
                    return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
                return out

        model.fdaa = ZeroFreqV3(embed_dim, num_patches)
        model.mgfp = CLSOnlyPoolV3(embed_dim)

    elif variant == 'baseline+fdaa':
        class CLSFreqPoolV3(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                )
            def forward(self, cls_token, patch_tokens, freq_tokens=None,
                        freq_global=None, return_attention=False):
                if freq_global is None:
                    freq_global = torch.zeros_like(cls_token)
                out = self.mlp(torch.cat([cls_token, freq_global], dim=-1))
                if return_attention:
                    B, N, _ = patch_tokens.shape
                    return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
                return out

        model.mgfp = CLSFreqPoolV3(embed_dim)

    elif variant == 'baseline+mgfp':
        # MGFPv3 无 freq 输入 → 退化为 fallback 查询
        model.fdaa = ZeroFreqV3(embed_dim, num_patches)

    # variant == 'full': 不做替换

    return model


def exp_ablation_v3(config, device, num_gpus):
    """V3 消融实验"""
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    source_to_id = {s: i for i, s in enumerate(config['train_sources'])}

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
        aug_version='v3',
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
        aug_version='v3',
    )

    abl_config = config.copy()
    abl_epochs = min(config.get('v3_epochs', 25), 10)
    abl_config['epochs'] = abl_epochs
    abl_config['contrastive_weight'] = config.get('v3_contrastive_weight', 0.5)

    variants = [
        ('baseline',       'V3 Baseline (CLIP + CLS-only)'),
        ('baseline+fdaa',  'V3 Baseline + FDAAv3 (CLS + freq_global)'),
        ('baseline+mgfp',  'V3 Baseline + MGFPv3 (no freq input → fallback)'),
        ('full',           'V3 Full (FDAAv3 + MGFPv3 + hierarchical)'),
    ]

    # 检查 V3 checkpoint
    ours_v3_ckpt = os.path.join(output_dir, 'checkpoints', 'fdaa_net_v3_best.pth')
    ours_v3_log = os.path.join(output_dir, 'logs', 'fdaa_net_v3_training_log.json')
    ours_v3_exists = os.path.exists(ours_v3_ckpt)

    for i, (variant, desc) in enumerate(variants):
        print(f"\n[V3 Ablation {i+1}/{len(variants)}] {desc}")

        if variant == 'full' and ours_v3_exists:
            model = _create_ablation_model_v3(config, variant)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            used_log = False
            if os.path.exists(ours_v3_log):
                try:
                    with open(ours_v3_log) as fp:
                        log_data = json.load(fp)
                    target_epoch = abl_epochs - 1
                    if len(log_data) > target_epoch:
                        entry = log_data[target_epoch]
                        results[variant] = {
                            'auc': entry['val_auc'], 'ap': entry['val_ap'],
                            'accuracy': entry['val_acc'], 'eer': entry['val_eer'],
                            'epoch': entry['epoch'],
                        }
                        used_log = True
                        print(f"  -> Using V3 training log epoch {target_epoch}")
                except Exception as e:
                    print(f"  [Warning] Failed to read V3 log: {e}")

            if not used_log:
                ckpt = torch.load(ours_v3_ckpt, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)
                results[variant] = evaluate_model(model, val_loader, device, desc='V3-Abl-Full')
                results[variant]['epoch'] = ckpt.get('epoch', -1)
                del model; torch.cuda.empty_cache()

            results[variant]['trainable_params'] = trainable
            results[variant]['trainable_params_M'] = trainable / 1e6
            print(f"  Full: {fmt(results[variant])}")
            continue

        model_name = f'abl_v3_{variant.replace("+", "_")}'
        ckpt_path = os.path.join(output_dir, 'checkpoints', f'{model_name}_best.pth')
        log_path = os.path.join(output_dir, 'logs', f'{model_name}_training_log.json')

        # 跳过已完成的变体（checkpoint存在 + 训练日志完整）
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            try:
                with open(log_path) as fp:
                    log_data = json.load(fp)
                if len(log_data) >= abl_epochs:
                    entry = log_data[abl_epochs - 1]
                    model_tmp = _create_ablation_model_v3(config, variant)
                    trainable = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
                    del model_tmp
                    results[variant] = {
                        'auc': entry['val_auc'], 'ap': entry['val_ap'],
                        'accuracy': entry['val_acc'], 'eer': entry['val_eer'],
                        'epoch': entry['epoch'],
                        'trainable_params': trainable,
                        'trainable_params_M': trainable / 1e6,
                    }
                    print(f"  -> Skipping (already trained {len(log_data)} epochs): {fmt(results[variant])}")
                    continue
            except Exception as e:
                print(f"  [Warning] Failed to read log {log_path}: {e}, will retrain")

        model = _create_ablation_model_v3(config, variant)
        abl_config['use_contrastive'] = True
        abl_config['use_hierarchical'] = variant in ('baseline+mgfp', 'full')

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        metrics = train_model(
            model, train_loader, val_loader, abl_config,
            model_name=model_name,
            output_dir=output_dir, device=device, num_gpus=num_gpus,
            is_our_model=True, source_to_id=source_to_id,
        )
        metrics['trainable_params'] = trainable
        metrics['trainable_params_M'] = trainable / 1e6
        results[variant] = metrics

    save_json(results, os.path.join(output_dir, 'results', 'ablation_v3_results.json'))
    return results


# =============================================================================
# 实验 4: 跨域泛化评估
# =============================================================================

def exp_cross_domain(config, device):
    """跨域评估: 跨生成器 + 跨数据集"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results = {}

    # 收集训练好的模型
    model_configs = {}

    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if os.path.exists(ours_ckpt):
        model_configs['Ours_V2'] = {
            'checkpoint': ours_ckpt,
            'create_fn': lambda: AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    ours_v3_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v3_best.pth')
    if os.path.exists(ours_v3_ckpt):
        model_configs['Ours_V3'] = {
            'checkpoint': ours_v3_ckpt,
            'create_fn': lambda: AIGCDetectorV3(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    for method in config.get('sota_methods', []):
        ckpt = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt):
            m = method
            model_configs[method] = {
                'checkpoint': ckpt,
                'create_fn': lambda m=m: create_sota_model(m, num_classes=2)
            }

    if not model_configs:
        print("[Warning] No trained models found")
        return {}

    # 预加载所有模型（避免每个测试集重复创建/加载）
    loaded_models = {}
    for model_name, mcfg in model_configs.items():
        try:
            m = mcfg['create_fn']()
            ckpt = torch.load(mcfg['checkpoint'], map_location='cpu')
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[model_name] = m
            print(f"  [Loaded] {model_name}")
        except Exception as e:
            print(f"  [Error loading] {model_name}: {e}")

    # 4.1 跨生成器评估
    print(f"\n{'='*60}")
    print("Cross-Generator Evaluation (GenImage)")
    print(f"{'='*60}")

    train_sources = set(config['train_sources'])

    for source in config.get('cross_gen_sources', []):
        key = f"GenImage_{source}"
        is_held_out = source not in train_sources
        marker = "HELD-OUT" if is_held_out else "IN-DOMAIN"

        loader = create_multi_source_dataloader(
            genimage_root=config['genimage_root'],
            sources=[source], split='val',
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            max_samples_per_source=5000,
            num_workers=config['num_workers'],
            strong_aug=False,
        )

        if loader is None or len(loader.dataset) == 0:
            continue

        results[key] = {'_type': marker}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {key}')
                results[key][model_name] = metrics
                print(f"  {model_name} [{marker}] {source}: {fmt(metrics)}")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[key][model_name] = {'error': str(e)}

    # 4.2 跨数据集评估
    for ds_name, ds_root in config.get('eval_datasets', {}).items():
        print(f"\n--- {ds_name} ---")
        loader = create_eval_loader(ds_root, split='test', batch_size=config['batch_size'],
                                     img_size=config['img_size'], num_workers=config['num_workers'])
        if loader is None or len(loader.dataset) == 0:
            print(f"  [Warning] Empty dataset for {ds_name}, skipping")
            continue

        results[ds_name] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {ds_name}')
                results[ds_name][model_name] = metrics
                print(f"  {model_name}: {fmt(metrics)}")
            except Exception as e:
                results[ds_name][model_name] = {'error': str(e)}

    # 释放预加载模型
    del loaded_models; torch.cuda.empty_cache()

    save_json(results, os.path.join(output_dir, 'results', 'cross_domain_results.json'))
    return results


# =============================================================================
# 实验 4b: Leave-One-Out 跨生成器泛化
# =============================================================================

def exp_leave_one_out(config, device, num_gpus):
    """
    Leave-One-Out 跨生成器评估

    对每个训练源 s ∈ {biggan, adm, glide, vqdm}:
      - 在剩余 3 个源上训练
      - 在 s 的 val 集上评估
    证明模型对未见过的生成器有泛化能力。
    """
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    all_sources = list(config['train_sources'])
    loo_epochs = min(config['epochs'], 10)  # 缩短为 10 epoch 节省时间

    print(f"\n{'='*60}")
    print(f"Leave-One-Out Cross-Generator Evaluation")
    print(f"Sources: {all_sources}, Epochs per run: {loo_epochs}")
    print(f"{'='*60}")

    for held_out in all_sources:
        train_sources = [s for s in all_sources if s != held_out]
        print(f"\n--- Held-out: {held_out}, Train on: {train_sources} ---")

        # 训练数据 (不含 held-out)
        train_loader = create_multi_source_dataloader(
            genimage_root=config['genimage_root'],
            sources=train_sources,
            split='train',
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            max_samples_per_source=config['max_samples_per_source'],
            num_workers=config['num_workers'],
            strong_aug=True,
        )

        # 验证数据: 用训练源的 val 做 model selection（避免 held-out 信息泄露）
        val_loader = create_multi_source_dataloader(
            genimage_root=config['genimage_root'],
            sources=train_sources,
            split='val',
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            max_samples_per_source=2000,
            num_workers=config['num_workers'],
            strong_aug=False,
        )

        # 测试数据: held-out 生成器的 val 集
        test_loader = create_multi_source_dataloader(
            genimage_root=config['genimage_root'],
            sources=[held_out],
            split='val',
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            max_samples_per_source=5000,
            num_workers=config['num_workers'],
            strong_aug=False,
        )

        if test_loader is None or len(test_loader.dataset) == 0:
            print(f"  [Warning] No test data for {held_out}, skipping")
            continue

        # 创建模型
        model = AIGCDetectorV2(
            backbone_name=config['backbone'],
            num_classes=2,
            img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'],
            freeze_backbone=True,
        )

        loo_config = config.copy()
        loo_config['epochs'] = loo_epochs

        # 在训练源 val 上做 model selection
        train_model(
            model, train_loader, val_loader, loo_config,
            model_name=f'loo_held_{held_out}',
            output_dir=output_dir,
            device=device, num_gpus=num_gpus,
            is_our_model=True,
        )

        # 加载最佳 checkpoint，在 held-out 上评估
        ckpt_path = os.path.join(output_dir, 'checkpoints', f'loo_held_{held_out}_best.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            raw = model.module if hasattr(model, 'module') else model
            raw.load_state_dict(ckpt['model_state_dict'])
            raw = raw.to(device)
            test_metrics = evaluate_model(raw, test_loader, device, desc=f'LOO test {held_out}')
            del raw
        else:
            test_metrics = evaluate_model(model, test_loader, device, desc=f'LOO test {held_out}')

        results[held_out] = {
            'train_sources': train_sources,
            'held_out': held_out,
            'metrics': test_metrics,
        }
        print(f"  Held-out {held_out}: {fmt(test_metrics)}")
        del model; torch.cuda.empty_cache()

    # 计算平均
    auc_values = [r['metrics']['auc'] for r in results.values() if 'auc' in r.get('metrics', {})]
    if auc_values:
        results['_average'] = {
            'auc': np.mean(auc_values),
            'std': np.std(auc_values),
        }
        print(f"\n  Average LOO AUC: {np.mean(auc_values)*100:.2f}% ± {np.std(auc_values)*100:.2f}%")

    save_json(results, os.path.join(output_dir, 'results', 'leave_one_out_results.json'))
    return results


# =============================================================================
# 实验 4c: Leave-One-Out V3
# =============================================================================

def exp_leave_one_out_v3(config, device, num_gpus):
    """V3 Leave-One-Out 跨生成器评估"""
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    all_sources = list(config['train_sources'])
    loo_epochs = min(config.get('v3_epochs', 25), 10)

    print(f"\n{'='*60}")
    print(f"V3 Leave-One-Out Cross-Generator Evaluation")
    print(f"Sources: {all_sources}, Epochs per run: {loo_epochs}")
    print(f"{'='*60}")

    for held_out in all_sources:
        train_sources = [s for s in all_sources if s != held_out]
        source_to_id = {s: i for i, s in enumerate(train_sources)}
        model_name = f'loo_v3_held_{held_out}'
        ckpt_path = os.path.join(output_dir, 'checkpoints', f'{model_name}_best.pth')
        log_path = os.path.join(output_dir, 'logs', f'{model_name}_training_log.json')

        print(f"\n--- Held-out: {held_out}, Train on: {train_sources} ---")

        # 跳过已完成训练的 held-out（checkpoint + 完整日志都存在）
        need_train = True
        if os.path.exists(ckpt_path) and os.path.exists(log_path):
            try:
                with open(log_path) as fp:
                    log_data = json.load(fp)
                if len(log_data) >= loo_epochs:
                    print(f"  -> Already trained ({len(log_data)} epochs), loading checkpoint for test...")
                    need_train = False
            except Exception:
                pass

        test_loader = create_multi_source_dataloader(
            genimage_root=config['genimage_root'],
            sources=[held_out], split='val',
            batch_size=config['batch_size'], img_size=config['img_size'],
            max_samples_per_source=5000,
            num_workers=config['num_workers'], strong_aug=False,
        )

        if test_loader is None or len(test_loader.dataset) == 0:
            print(f"  [Warning] No test data for {held_out}, skipping")
            continue

        if need_train:
            train_loader = create_multi_source_dataloader(
                genimage_root=config['genimage_root'],
                sources=train_sources, split='train',
                batch_size=config['batch_size'], img_size=config['img_size'],
                max_samples_per_source=config['max_samples_per_source'],
                num_workers=config['num_workers'], strong_aug=True, aug_version='v3',
            )

            val_loader = create_multi_source_dataloader(
                genimage_root=config['genimage_root'],
                sources=train_sources, split='val',
                batch_size=config['batch_size'], img_size=config['img_size'],
                max_samples_per_source=2000,
                num_workers=config['num_workers'], strong_aug=False, aug_version='v3',
            )

            model = AIGCDetectorV3(
                backbone_name=config['backbone'], num_classes=2,
                img_size=config['img_size'], embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'], freeze_backbone=True,
            )

            loo_config = config.copy()
            loo_config['epochs'] = loo_epochs
            loo_config['contrastive_weight'] = config.get('v3_contrastive_weight', 0.5)

            train_model(
                model, train_loader, val_loader, loo_config,
                model_name=model_name,
                output_dir=output_dir, device=device, num_gpus=num_gpus,
                is_our_model=True, source_to_id=source_to_id,
            )
            del model; torch.cuda.empty_cache()

        # 加载 best checkpoint 进行测试
        model = AIGCDetectorV3(
            backbone_name=config['backbone'], num_classes=2,
            img_size=config['img_size'], embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'], freeze_backbone=True,
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        test_metrics = evaluate_model(model, test_loader, device, desc=f'LOO-V3 test {held_out}')

        results[held_out] = {
            'train_sources': train_sources,
            'held_out': held_out,
            'metrics': test_metrics,
        }
        print(f"  V3 Held-out {held_out}: {fmt(test_metrics)}")
        del model; torch.cuda.empty_cache()

    auc_values = [r['metrics']['auc'] for r in results.values() if 'auc' in r.get('metrics', {})]
    if auc_values:
        results['_average'] = {'auc': np.mean(auc_values), 'std': np.std(auc_values)}
        print(f"\n  V3 Average LOO AUC: {np.mean(auc_values)*100:.2f}% ± {np.std(auc_values)*100:.2f}%")

    save_json(results, os.path.join(output_dir, 'results', 'leave_one_out_v3_results.json'))
    return results


# =============================================================================
# 实验 5: 鲁棒性测试
# =============================================================================

def exp_robustness(config, device):
    """鲁棒性测试: JPEG压缩、高斯模糊、高斯噪声"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results = {}

    # 加载已训练模型
    model_configs = {}
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if os.path.exists(ours_ckpt):
        model_configs['Ours_V2'] = {
            'checkpoint': ours_ckpt,
            'create_fn': lambda: AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    ours_v3_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v3_best.pth')
    if os.path.exists(ours_v3_ckpt):
        model_configs['Ours_V3'] = {
            'checkpoint': ours_v3_ckpt,
            'create_fn': lambda: AIGCDetectorV3(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    for method in config.get('sota_methods', []):
        ckpt = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt):
            m = method
            model_configs[method] = {
                'checkpoint': ckpt,
                'create_fn': lambda m=m: create_sota_model(m, num_classes=2)
            }

    if not model_configs:
        print("[Warning] No trained models found")
        return {}

    # 预加载所有模型（避免每种降质重复创建/加载）
    loaded_models = {}
    for model_name, mcfg in model_configs.items():
        try:
            m = mcfg['create_fn']()
            ckpt = torch.load(mcfg['checkpoint'], map_location='cpu')
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[model_name] = m
            print(f"  [Loaded] {model_name}")
        except Exception as e:
            print(f"  [Error loading] {model_name}: {e}")

    # 创建基础验证数据集 (不带 transform，用于后续自定义降质)
    base_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        transform=None,  # 无 transform，手动处理
        max_samples_per_source=2000,
        balance_classes=True,
    )

    print(f"\n{'='*60}")
    print("Robustness Evaluation")
    print(f"{'='*60}")

    for test_name, test_cfg in config.get('robustness_tests', {}).items():
        deg_type = test_cfg['type']
        deg_param = test_cfg.get('quality') or test_cfg.get('radius') or test_cfg.get('std')

        print(f"\n--- {test_name} ({deg_type}={deg_param}) ---")

        degraded = DegradedDataset(base_dataset, deg_type, deg_param, config['img_size'])
        loader = DataLoader(degraded, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)

        results[test_name] = {'type': deg_type, 'param': deg_param}

        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {test_name}')
                results[test_name][model_name] = metrics
                print(f"  {model_name}: {fmt(metrics)}")
            except Exception as e:
                results[test_name][model_name] = {'error': str(e)}

    # 释放预加载模型
    del loaded_models; torch.cuda.empty_cache()

    save_json(results, os.path.join(output_dir, 'results', 'robustness_results.json'))
    return results


# =============================================================================
# 实验 6: 可视化分析 (t-SNE + 伪造注意力图)
# =============================================================================

def exp_visualization(config, device):
    """
    特征可视化:
    1. t-SNE: 不同来源 / 真假标签的特征分布
    2. 伪造注意力图: 真/假样本的注意力热力图
    """
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 加载主模型（优先 V3，回退 V2）
    ours_v3_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v3_best.pth')
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')

    if os.path.exists(ours_v3_ckpt):
        print("[Vis] Using V3 model for visualization")
        ours_ckpt = ours_v3_ckpt
        model = AIGCDetectorV3(
            backbone_name=config['backbone'],
            num_classes=2, img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'],
        )
    elif os.path.exists(ours_ckpt):
        print("[Vis] Using V2 model for visualization")
        model = AIGCDetectorV2(
            backbone_name=config['backbone'],
            num_classes=2, img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'],
        )
    else:
        print("[Warning] No trained model found for visualization")
        return
    ckpt = torch.load(ours_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 收集特征
    print(f"\n{'='*60}")
    print("Feature Visualization (t-SNE)")
    print(f"{'='*60}")

    all_features = []
    all_labels = []
    all_sources = []
    max_vis_samples = 500  # 每个源最多 500 样本

    val_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        transform=get_multi_source_transforms(config['img_size'], is_train=False),
        max_samples_per_source=max_vis_samples,
        balance_classes=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Extracting features'):
            images = batch['image'].to(device, non_blocking=True)
            with autocast('cuda'):
                outputs = model(images, return_features=True, return_attention=True)
            if 'features' not in outputs:
                print("[Warning] Model does not return 'features', skipping visualization")
                del model; torch.cuda.empty_cache()
                return
            features = outputs['features'].float().cpu().numpy()
            all_features.append(features)
            all_labels.extend(batch['label'].tolist())
            all_sources.extend(batch['source'])

    all_features = np.concatenate(all_features, axis=0)

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print(f"Running t-SNE on {len(all_features)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(all_features)

        # 按标签着色
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        labels_arr = np.array(all_labels)
        for label, name, color in [(0, 'Real', '#2196F3'), (1, 'Fake', '#F44336')]:
            mask = labels_arr == label
            axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=color, label=name, alpha=0.5, s=8)
        axes[0].set_title('t-SNE by Label (Real vs Fake)', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # 按来源着色
        sources_arr = np.array(all_sources)
        source_colors = {'biggan': '#4CAF50', 'adm': '#FF9800', 'glide': '#9C27B0', 'vqdm': '#00BCD4'}
        for src in sorted(set(all_sources)):
            mask = sources_arr == src
            axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=source_colors.get(src, '#999999'), label=src.upper(), alpha=0.5, s=8)
        axes[1].set_title('t-SNE by Generator Source', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        plt.tight_layout()
        tsne_path = os.path.join(vis_dir, 'tsne_features.png')
        plt.savefig(tsne_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Vis] t-SNE saved: {tsne_path}")

    except Exception as e:
        print(f"[Warning] t-SNE visualization failed: {e}")
        import traceback; traceback.print_exc()

    # 伪造注意力图
    print("\nGenerating forgery attention maps...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 选取样本: 每个源取 1 real + 1 fake
        sample_indices = {}
        for i, (_, label, source) in enumerate(val_dataset.samples):
            key = (source, label)
            if key not in sample_indices:
                sample_indices[key] = i
            if len(sample_indices) >= len(config['train_sources']) * 2:
                break

        fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes[np.newaxis, :]

        for row, ((source, label), idx) in enumerate(sorted(sample_indices.items())):
            sample = val_dataset[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device)

            with torch.no_grad(), autocast('cuda'):
                outputs = model(img_tensor, return_attention=True)

            # 原图
            img_np = sample['image'].permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            axes[row, 0].imshow(img_np)
            label_str = 'Fake' if label == 1 else 'Real'
            axes[row, 0].set_title(f'{source.upper()} ({label_str})', fontsize=11)
            axes[row, 0].axis('off')

            # 注意力图
            forgery_map = outputs.get('forgery_map')
            if forgery_map is None:
                forgery_map = outputs.get('attention_map')
            if forgery_map is not None:
                attn = forgery_map[0].float().cpu().numpy()
                H = W = int(np.sqrt(len(attn)))
                attn_2d = attn.reshape(H, W)
                im = axes[row, 1].imshow(attn_2d, cmap='jet', interpolation='bilinear')
                axes[row, 1].set_title('Attention Map', fontsize=11)
                axes[row, 1].axis('off')
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

                # 叠加
                import matplotlib.cm as cm
                attn_norm = attn_2d / (attn_2d.max() + 1e-8)
                attn_resized = np.array(Image.fromarray(
                    (cm.jet(attn_norm)[:, :, :3] * 255).astype(np.uint8)
                ).resize((config['img_size'], config['img_size']), Image.BILINEAR)) / 255.0
                overlay = 0.5 * img_np + 0.5 * attn_resized
                axes[row, 2].imshow(overlay)
                axes[row, 2].set_title('Overlay', fontsize=11)
                axes[row, 2].axis('off')
            else:
                axes[row, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, 1].axis('off')
                axes[row, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, 2].axis('off')

        plt.tight_layout()
        attn_path = os.path.join(vis_dir, 'forgery_attention_maps.png')
        plt.savefig(attn_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Vis] Attention maps saved: {attn_path}")

    except Exception as e:
        print(f"[Warning] Attention map visualization failed: {e}")
        import traceback; traceback.print_exc()

    del model; torch.cuda.empty_cache()

    # -----------------------------------------------
    # 3. 训练曲线（从日志文件绘制）
    # -----------------------------------------------
    print("\nGenerating training curves...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        log_dir = os.path.join(output_dir, 'logs')
        # 收集所有训练日志
        log_files = {}
        if os.path.exists(log_dir):
            for f in sorted(os.listdir(log_dir)):
                if f.endswith('_training_log.json'):
                    name = f.replace('_training_log.json', '')
                    with open(os.path.join(log_dir, f)) as fp:
                        log_files[name] = json.load(fp)

        if log_files:
            # 绘制 Ours 的训练曲线
            ours_log = log_files.get('fdaa_net_v3') or log_files.get('fdaa_net_v2')
            if ours_log:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                epochs_x = [e['epoch'] for e in ours_log]
                # Loss
                axes[0].plot(epochs_x, [e['train_loss'] for e in ours_log], 'b-o', markersize=3, label='Train Loss')
                axes[0].set_xlabel('Epoch', fontsize=12)
                axes[0].set_ylabel('Loss', fontsize=12)
                axes[0].set_title('Training Loss', fontsize=14)
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                # Val AUC
                axes[1].plot(epochs_x, [e['val_auc'] * 100 for e in ours_log], 'r-o', markersize=3, label='Val AUC')
                axes[1].set_xlabel('Epoch', fontsize=12)
                axes[1].set_ylabel('AUC (%)', fontsize=12)
                axes[1].set_title('Validation AUC', fontsize=14)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()

                # LR
                axes[2].plot(epochs_x, [e['lr'] for e in ours_log], 'g-o', markersize=3, label='Learning Rate')
                axes[2].set_xlabel('Epoch', fontsize=12)
                axes[2].set_ylabel('LR', fontsize=12)
                axes[2].set_title('Learning Rate Schedule', fontsize=14)
                axes[2].set_yscale('log')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()

                plt.tight_layout()
                curve_path = os.path.join(vis_dir, 'training_curves.png')
                plt.savefig(curve_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"[Vis] Training curves saved: {curve_path}")

            # 绘制所有方法 val AUC 对比
            if len(log_files) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
                for (name, log), color in zip(sorted(log_files.items()), colors):
                    epochs_x = [e['epoch'] for e in log]
                    auc_y = [e['val_auc'] * 100 for e in log]
                    if name == 'fdaa_net_v3':
                        label = 'FDAA-Net V3 (Ours)'
                    elif name == 'fdaa_net_v2':
                        label = 'FDAA-Net V2'
                    else:
                        label = name
                    lw = 2.5 if name in ('fdaa_net_v3', 'fdaa_net_v2') else 1.5
                    ax.plot(epochs_x, auc_y, '-o', markersize=2, label=label, color=color, linewidth=lw)

                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Val AUC (%)', fontsize=12)
                ax.set_title('Validation AUC: All Methods', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='lower right')
                plt.tight_layout()
                compare_path = os.path.join(vis_dir, 'val_auc_comparison.png')
                plt.savefig(compare_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"[Vis] AUC comparison saved: {compare_path}")

    except Exception as e:
        print(f"[Warning] Training curve visualization failed: {e}")
        import traceback; traceback.print_exc()

    print("[Vis] Visualization complete")


# =============================================================================
# 实验 7: 效率对比
# =============================================================================

def exp_efficiency(config, device):
    """计算各模型的参数量和推理速度"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results = {}

    print(f"\n{'='*60}")
    print("Efficiency Comparison")
    print(f"{'='*60}")

    # 构建模型列表
    models_to_test = []

    # Ours V2
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if os.path.exists(ours_ckpt):
        models_to_test.append(('Ours (FDAA-Net V2)', lambda: AIGCDetectorV2(
            backbone_name=config['backbone'],
            num_classes=2, img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'],
        )))

    # Ours V3
    ours_v3_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v3_best.pth')
    if os.path.exists(ours_v3_ckpt):
        models_to_test.append(('Ours (FDAA-Net V3)', lambda: AIGCDetectorV3(
            backbone_name=config['backbone'],
            num_classes=2, img_size=config['img_size'],
            embed_dim=config['embed_dim'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout'],
        )))

    # SOTA
    for method in config.get('sota_methods', []):
        ckpt = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt):
            m = method
            models_to_test.append((method, lambda m=m: create_sota_model(m, num_classes=2)))

    if not models_to_test:
        print("[Warning] No models found for efficiency test")
        return {}

    dummy_input = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
    n_warmup = 10
    n_measure = 50

    for model_name, create_fn in models_to_test:
        try:
            m = create_fn().to(device)
            m.eval()

            total_params = sum(p.numel() for p in m.parameters())
            trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

            # 推理速度（GPU Event Timing + autocast 匹配实际推理条件）
            with torch.no_grad():
                for _ in range(n_warmup):
                    with autocast('cuda'):
                        _ = m(dummy_input)
                torch.cuda.synchronize()

                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                for _ in range(n_measure):
                    with autocast('cuda'):
                        _ = m(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender) / n_measure  # ms (GPU精确计时)

            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'total_params_M': total_params / 1e6,
                'trainable_params_M': trainable_params / 1e6,
                'inference_ms': elapsed,
            }
            print(f"  {model_name}: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable), {elapsed:.1f} ms/image")
            del m; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [Error] {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    save_json(results, os.path.join(output_dir, 'results', 'efficiency_results.json'))
    return results


# =============================================================================
# 实验 8: 生成论文报告
# =============================================================================

def exp_report(config):
    """生成 Markdown 报告 + 论文级别表格"""
    output_dir = config['output_dir']
    results_dir = os.path.join(output_dir, 'results')
    report_path = os.path.join(output_dir, 'reports', 'paper_report.md')

    # 方法名显示映射
    METHOD_DISPLAY = {
        'fdaa_net_v2': 'FDAA-Net V2',
        'fdaa_net_v3': 'FDAA-Net V3 (Ours)',
        'Ours_V2': 'FDAA-Net V2',
        'Ours_V3': 'FDAA-Net V3 (Ours)',
        'Ours': 'FDAA-Net V2',
        'cnndetection': 'CNNDetection',
        'f3net': 'F3Net',
        'univfd': 'UnivFD',
        'freqnet': 'FreqNet',
        'npr': 'NPR',
        'spec': 'Spec',
    }

    def _fmt_name(name):
        return METHOD_DISPLAY.get(name, name)

    def _bold_best(values, fmt_str='{:.2f}'):
        """在列表中找最大值并加粗, 返回格式化后的字符串列表"""
        if not values:
            return []
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid:
            return ['-' for _ in values]
        best_val = max(v for _, v in valid)
        result = []
        for v in values:
            if v is None:
                result.append('-')
            elif abs(v - best_val) < 1e-6:
                result.append(f'**{fmt_str.format(v * 100)}**')
            else:
                result.append(fmt_str.format(v * 100))
        return result

    lines = [
        "# FDAA-Net V2/V3 论文实验报告",
        f"\n生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验设置",
        f"- Backbone: CLIP {config['backbone']} (frozen, width={config['embed_dim']})",
        f"- 训练数据: GenImage ({', '.join(config['train_sources'])})",
        f"- 每源样本: {config['max_samples_per_source']}",
        f"- V2 Epochs: {config['epochs']} (warmup={config['warmup_epochs']})",
        f"- V3 Epochs: {config.get('v3_epochs', 25)} (contrastive_weight={config.get('v3_contrastive_weight', 0.5)})",
        f"- Batch size: {config['batch_size']}",
        f"- LR: {config['lr']} (SRM×0.1, FDAA/MGFP/classifier×1, cosine decay)",
        f"- Loss: Focal(alpha=0.5) + Contrastive(w={config['contrastive_weight']}) + Aux(w={config['aux_weight']})",
        f"- Label smoothing: {config['label_smoothing']}",
        "",
    ]

    def _load_json(name):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    table_num = 1

    # =====================================================
    # Table: SOTA 对比 (域内)
    # =====================================================
    ours_res = _load_json('ours_train_results.json')
    sota_res = _load_json('sota_train_results.json')

    if ours_res or sota_res:
        lines.append(f"## Table {table_num}: 域内检测性能对比")
        table_num += 1
        lines.append("")

        all_methods = {}
        if ours_res:
            all_methods.update(ours_res)
        if sota_res:
            all_methods.update(sota_res)

        # 按 AUC 排序
        sorted_methods = sorted(
            [(n, m) for n, m in all_methods.items() if isinstance(m, dict) and 'error' not in m],
            key=lambda x: x[1].get('auc', 0), reverse=True
        )

        lines.append("| Method | AUC (%) | AP (%) | Accuracy (%) | EER (%) |")
        lines.append("|--------|---------|--------|--------------|---------|")

        auc_vals = [m.get('auc') for _, m in sorted_methods]
        ap_vals = [m.get('ap') for _, m in sorted_methods]
        acc_vals = [m.get('accuracy') for _, m in sorted_methods]
        # EER: lower is better, need special handling
        eer_vals = [m.get('eer') for _, m in sorted_methods]
        eer_valid = [v for v in eer_vals if v is not None]
        best_eer = min(eer_valid) if eer_valid else None

        auc_strs = _bold_best(auc_vals)
        ap_strs = _bold_best(ap_vals)
        acc_strs = _bold_best(acc_vals)

        for idx, (name, m) in enumerate(sorted_methods):
            eer_str = '-'
            if m.get('eer') is not None:
                if best_eer is not None and abs(m['eer'] - best_eer) < 1e-6:
                    eer_str = f"**{m['eer']*100:.2f}**"
                else:
                    eer_str = f"{m['eer']*100:.2f}"
            lines.append(
                f"| {_fmt_name(name)} | {auc_strs[idx]} | {ap_strs[idx]} | "
                f"{acc_strs[idx]} | {eer_str} |"
            )
        lines.append("")

    # =====================================================
    # Table: 消融实验
    # =====================================================
    abl_res = _load_json('ablation_results.json')
    if abl_res:
        lines.append(f"## Table {table_num}: 消融实验 (CLIP ViT-L/14 backbone)")
        table_num += 1
        lines.append("")
        lines.append("| Variant | FDAA | MGFP | Params (M) | AUC (%) | AP (%) | Acc (%) | ΔAUC |")
        lines.append("|---------|------|------|-----------|---------|--------|---------|------|")

        variant_info = {
            'baseline': ('✗', '✗'),
            'baseline+fdaa': ('✓', '✗'),
            'baseline+mgfp': ('✗', '✓'),
            'full': ('✓', '✓'),
        }
        variant_order = ['baseline', 'baseline+fdaa', 'baseline+mgfp', 'full']
        baseline_auc = abl_res.get('baseline', {}).get('auc', 0)

        auc_list = [abl_res.get(v, {}).get('auc') for v in variant_order if v in abl_res]
        auc_strs = _bold_best(auc_list)

        idx = 0
        for v in variant_order:
            m = abl_res.get(v)
            if m is None or not isinstance(m, dict) or 'error' in m or 'auc' not in m:
                continue
            fdaa, mgfp = variant_info.get(v, ('?', '?'))
            delta = (m['auc'] - baseline_auc) * 100
            delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}" if delta != 0 else "-"
            params_m = m.get('trainable_params_M', '-')
            params_str = f"{params_m:.1f}" if isinstance(params_m, (int, float)) else str(params_m)
            lines.append(
                f"| {v} | {fdaa} | {mgfp} | {params_str} | {auc_strs[idx]} | "
                f"{m.get('ap',0)*100:.2f} | {m.get('accuracy',0)*100:.2f} | {delta_str} |"
            )
            idx += 1
        lines.append("")

        # 损失消融子表
        loss_abl = abl_res.get('_loss_ablation', {})
        if loss_abl:
            lines.append(f"### Table {table_num}: 损失函数消融 (Full model)")
            table_num += 1
            lines.append("")
            lines.append("| Loss Configuration | Contrastive | Aux | AUC (%) | AP (%) | Acc (%) |")
            lines.append("|-------------------|-------------|-----|---------|--------|---------|")

            loss_order = ['focal_only', 'focal+contrastive', 'focal+aux', 'focal+contr+aux']
            loss_info = {
                'focal_only': ('✗', '✗'),
                'focal+contrastive': ('✓', '✗'),
                'focal+aux': ('✗', '✓'),
                'focal+contr+aux': ('✓', '✓'),
            }
            lauc_list = [loss_abl.get(v, {}).get('auc') for v in loss_order if v in loss_abl]
            lauc_strs = _bold_best(lauc_list)
            lidx = 0
            for lv in loss_order:
                lm = loss_abl.get(lv)
                if lm is None or not isinstance(lm, dict) or 'auc' not in lm:
                    continue
                contr, aux = loss_info.get(lv, ('?', '?'))
                lines.append(
                    f"| {lv} | {contr} | {aux} | {lauc_strs[lidx]} | "
                    f"{lm.get('ap',0)*100:.2f} | {lm.get('accuracy',0)*100:.2f} |"
                )
                lidx += 1
            lines.append("")

    # =====================================================
    # Table: V3 消融实验
    # =====================================================
    abl_v3_res = _load_json('ablation_v3_results.json')
    if abl_v3_res:
        lines.append(f"## Table {table_num}: V3 消融实验 (FDAAv3 + MGFPv3)")
        table_num += 1
        lines.append("")
        lines.append("| Variant | FDAAv3 | MGFPv3 | Params (M) | AUC (%) | AP (%) | Acc (%) | ΔAUC |")
        lines.append("|---------|--------|--------|-----------|---------|--------|---------|------|")

        variant_info_v3 = {
            'baseline': ('✗', '✗'),
            'baseline+fdaa': ('✓', '✗'),
            'baseline+mgfp': ('✗', '✓ (fallback)'),
            'full': ('✓', '✓ (freq-guided)'),
        }
        variant_order_v3 = ['baseline', 'baseline+fdaa', 'baseline+mgfp', 'full']
        baseline_auc_v3 = abl_v3_res.get('baseline', {}).get('auc', 0)

        auc_list_v3 = [abl_v3_res.get(v, {}).get('auc') for v in variant_order_v3 if v in abl_v3_res]
        auc_strs_v3 = _bold_best(auc_list_v3)

        idx_v3 = 0
        for v in variant_order_v3:
            m = abl_v3_res.get(v)
            if m is None or not isinstance(m, dict) or 'error' in m or 'auc' not in m:
                continue
            fdaa, mgfp = variant_info_v3.get(v, ('?', '?'))
            delta = (m['auc'] - baseline_auc_v3) * 100
            delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}" if delta != 0 else "-"
            params_m = m.get('trainable_params_M', '-')
            params_str = f"{params_m:.1f}" if isinstance(params_m, (int, float)) else str(params_m)
            lines.append(
                f"| {v} | {fdaa} | {mgfp} | {params_str} | {auc_strs_v3[idx_v3]} | "
                f"{m.get('ap',0)*100:.2f} | {m.get('accuracy',0)*100:.2f} | {delta_str} |"
            )
            idx_v3 += 1
        lines.append("")

    # =====================================================
    # Table: V3 域内性能
    # =====================================================
    ours_v3_res = _load_json('ours_v3_train_results.json')
    if ours_v3_res:
        lines.append(f"## Table {table_num}: V3 域内检测性能")
        table_num += 1
        lines.append("")
        lines.append("| Method | AUC (%) | AP (%) | Accuracy (%) | EER (%) |")
        lines.append("|--------|---------|--------|--------------|---------|")
        for name, m in ours_v3_res.items():
            if isinstance(m, dict) and 'auc' in m:
                eer_str = f"{m.get('eer',0)*100:.2f}" if m.get('eer') is not None else '-'
                lines.append(
                    f"| {_fmt_name(name)} | {m['auc']*100:.2f} | {m.get('ap',0)*100:.2f} | "
                    f"{m.get('accuracy',0)*100:.2f} | {eer_str} |"
                )
        lines.append("")

    # =====================================================
    # Table: V3 Leave-One-Out
    # =====================================================
    loo_v3_res = _load_json('leave_one_out_v3_results.json')
    if loo_v3_res:
        lines.append(f"## Table {table_num}: V3 Leave-One-Out 跨生成器泛化")
        table_num += 1
        lines.append("")
        lines.append("| Held-out Generator | Train Sources | AUC (%) | AP (%) | Acc (%) |")
        lines.append("|-------------------|---------------|---------|--------|---------|")
        for held_out, data in sorted(loo_v3_res.items()):
            if held_out.startswith('_'):
                continue
            m = data.get('metrics', {})
            if 'error' not in m and 'auc' in m:
                train_src = ', '.join(data.get('train_sources', []))
                lines.append(
                    f"| {held_out} | {train_src} | {m.get('auc',0)*100:.2f} | "
                    f"{m.get('ap',0)*100:.2f} | {m.get('accuracy',0)*100:.2f} |"
                )
        avg = loo_v3_res.get('_average', {})
        if avg:
            lines.append(f"| **Average** | - | **{avg.get('auc',0)*100:.2f} ± {avg.get('std',0)*100:.2f}** | - | - |")
        lines.append("")

    # =====================================================
    # Table: 跨域泛化 (方法为列，标准论文格式)
    # =====================================================
    cross_res = _load_json('cross_domain_results.json')
    if cross_res:
        gen_results = {k: v for k, v in cross_res.items() if k.startswith('GenImage_')}
        ext_results = {k: v for k, v in cross_res.items() if not k.startswith('GenImage_')}

        # 获取所有方法名
        all_method_names = set()
        for data in cross_res.values():
            if isinstance(data, dict):
                for k in data:
                    if k not in ('_type', 'type', 'param'):
                        all_method_names.add(k)
        method_names = sorted(all_method_names)

        if gen_results:
            lines.append(f"## Table {table_num}: 跨生成器泛化 — AUC (%)")
            table_num += 1
            lines.append("")

            header = "| Generator | Type | " + " | ".join(_fmt_name(m) for m in method_names) + " |"
            separator = "|-----------|------|" + "|".join("------" for _ in method_names) + "|"
            lines.append(header)
            lines.append(separator)

            for ds in sorted(gen_results):
                methods = gen_results[ds]
                ds_type = methods.get('_type', '-')
                gen_name = ds.replace('GenImage_', '')
                auc_vals = [methods.get(mn, {}).get('auc') if isinstance(methods.get(mn, {}), dict) else None for mn in method_names]
                auc_strs = _bold_best(auc_vals)
                row = f"| {gen_name} | {ds_type} | " + " | ".join(auc_strs) + " |"
                lines.append(row)
            lines.append("")

        if ext_results:
            lines.append(f"## Table {table_num}: 跨数据集泛化 — AUC (%)")
            table_num += 1
            lines.append("")

            header = "| Dataset | " + " | ".join(_fmt_name(m) for m in method_names) + " |"
            separator = "|---------|" + "|".join("------" for _ in method_names) + "|"
            lines.append(header)
            lines.append(separator)

            for ds in sorted(ext_results):
                methods = ext_results[ds]
                auc_vals = [methods.get(mn, {}).get('auc') if isinstance(methods.get(mn, {}), dict) else None for mn in method_names]
                auc_strs = _bold_best(auc_vals)
                row = f"| {ds} | " + " | ".join(auc_strs) + " |"
                lines.append(row)
            lines.append("")

    # =====================================================
    # Table: Leave-one-out
    # =====================================================
    loo_res = _load_json('leave_one_out_results.json')
    if loo_res:
        lines.append(f"## Table {table_num}: Leave-One-Out 跨生成器泛化")
        table_num += 1
        lines.append("")
        lines.append("| Held-out Generator | Train Sources | AUC (%) | AP (%) | Acc (%) |")
        lines.append("|-------------------|---------------|---------|--------|---------|")
        for held_out, data in sorted(loo_res.items()):
            if held_out.startswith('_'):
                continue
            m = data.get('metrics', {})
            if 'error' not in m and 'auc' in m:
                train_src = ', '.join(data.get('train_sources', []))
                lines.append(
                    f"| {held_out} | {train_src} | {m.get('auc',0)*100:.2f} | "
                    f"{m.get('ap',0)*100:.2f} | {m.get('accuracy',0)*100:.2f} |"
                )
        avg = loo_res.get('_average', {})
        if avg:
            lines.append(f"| **Average** | - | **{avg.get('auc',0)*100:.2f} ± {avg.get('std',0)*100:.2f}** | - | - |")
        lines.append("")

    # =====================================================
    # Table: 鲁棒性 (方法为列，标准论文格式)
    # =====================================================
    rob_res = _load_json('robustness_results.json')
    if rob_res:
        lines.append(f"## Table {table_num}: 鲁棒性评估 — AUC (%)")
        table_num += 1
        lines.append("")

        rob_method_names = set()
        for data in rob_res.values():
            if isinstance(data, dict):
                for k in data:
                    if k not in ('type', 'param'):
                        rob_method_names.add(k)
        rob_methods = sorted(rob_method_names)

        header = "| Degradation | " + " | ".join(_fmt_name(m) for m in rob_methods) + " |"
        separator = "|-------------|" + "|".join("------" for _ in rob_methods) + "|"
        lines.append(header)
        lines.append(separator)

        for test_name, data in sorted(rob_res.items()):
            auc_vals = [data.get(mn, {}).get('auc') if isinstance(data.get(mn, {}), dict) else None for mn in rob_methods]
            auc_strs = _bold_best(auc_vals)
            deg_label = test_name.replace('jpeg_', 'JPEG Q=').replace('blur_', 'Blur σ=').replace('noise_', 'Noise σ=')
            row = f"| {deg_label} | " + " | ".join(auc_strs) + " |"
            lines.append(row)
        lines.append("")

    # =====================================================
    # Table: 效率对比
    # =====================================================
    eff_res = _load_json('efficiency_results.json')
    if eff_res:
        lines.append(f"## Table {table_num}: 模型效率对比")
        table_num += 1
        lines.append("")
        lines.append("| Method | Total Params (M) | Trainable (M) | Inference (ms) |")
        lines.append("|--------|-----------------|---------------|----------------|")
        for name, m in sorted(eff_res.items(), key=lambda x: x[1].get('total_params_M', 0) if isinstance(x[1], dict) else 0):
            if isinstance(m, dict) and 'error' not in m:
                lines.append(
                    f"| {_fmt_name(name)} | {m.get('total_params_M',0):.1f} | "
                    f"{m.get('trainable_params_M',0):.1f} | {m.get('inference_ms',0):.1f} |"
                )
        lines.append("")

    # =====================================================
    # 可视化
    # =====================================================
    vis_dir = os.path.join(output_dir, 'visualizations')
    if os.path.exists(vis_dir) and os.listdir(vis_dir):
        lines.append(f"## 可视化")
        lines.append("")
        for f_name in sorted(os.listdir(vis_dir)):
            if f_name.endswith('.png'):
                lines.append(f"### {f_name.replace('.png', '').replace('_', ' ').title()}")
                lines.append(f"![{f_name}](visualizations/{f_name})")
                lines.append("")
            else:
                lines.append(f"- `visualizations/{f_name}`")
        lines.append("")

    report = '\n'.join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[Report] Generated: {report_path}")
    print(report[:3000])


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FDAA-Net V2/V3 Paper Experiments')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_ours', 'train_ours_v3', 'train_sota',
                                 'ablation', 'ablation_v3',
                                 'cross_domain', 'leave_one_out', 'leave_one_out_v3',
                                 'robustness', 'visualization', 'efficiency', 'report'],
                        help='Experiment mode')
    parser.add_argument('--genimage_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')

    args = parser.parse_args()

    config = PAPER_CONFIG.copy()
    if args.genimage_root:
        config['genimage_root'] = args.genimage_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    if args.max_samples:
        config['max_samples_per_source'] = args.max_samples

    # 完整种子控制，确保可复现
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device, num_gpus = get_device()
    setup_output_dir(config['output_dir'])
    save_json(config, os.path.join(config['output_dir'], 'paper_config.json'))

    print(f"\n{'#'*60}")
    print(f"# FDAA-Net V2 Paper Experiments")
    print(f"# Mode: {args.mode}")
    print(f"# Device: {device} ({num_gpus} GPUs)")
    print(f"# Output: {config['output_dir']}")
    print(f"{'#'*60}")

    start = time.time()
    results_dir = os.path.join(config['output_dir'], 'results')
    force = args.force

    def _should_skip(result_file, label):
        """检查是否可以跳过（结果已存在且非空）"""
        if force:
            return False
        fp = os.path.join(results_dir, result_file)
        if os.path.exists(fp) and os.path.getsize(fp) > 10:
            print(f"\n[Skip] {label}: {result_file} already exists (use --force to re-run)")
            return True
        return False

    if args.mode in ['all', 'train_ours']:
        if not _should_skip('ours_train_results.json', 'train_ours'):
            exp_train_ours(config, device, num_gpus)

    if args.mode in ['all', 'train_ours_v3']:
        if not _should_skip('ours_v3_train_results.json', 'train_ours_v3'):
            exp_train_ours_v3(config, device, num_gpus)

    if args.mode in ['all', 'train_sota']:
        if not _should_skip('sota_train_results.json', 'train_sota'):
            exp_train_sota(config, device, num_gpus)

    if args.mode in ['all', 'ablation']:
        if not _should_skip('ablation_results.json', 'ablation'):
            exp_ablation(config, device, num_gpus)

    if args.mode in ['all', 'ablation_v3']:
        if not _should_skip('ablation_v3_results.json', 'ablation_v3'):
            exp_ablation_v3(config, device, num_gpus)

    if args.mode in ['all', 'cross_domain']:
        if not _should_skip('cross_domain_results.json', 'cross_domain'):
            exp_cross_domain(config, device)

    if args.mode in ['all', 'leave_one_out']:
        if not _should_skip('leave_one_out_results.json', 'leave_one_out'):
            exp_leave_one_out(config, device, num_gpus)

    if args.mode in ['all', 'leave_one_out_v3']:
        if not _should_skip('leave_one_out_v3_results.json', 'leave_one_out_v3'):
            exp_leave_one_out_v3(config, device, num_gpus)

    if args.mode in ['all', 'robustness']:
        if not _should_skip('robustness_results.json', 'robustness'):
            exp_robustness(config, device)

    if args.mode in ['all', 'visualization']:
        vis_dir = os.path.join(config['output_dir'], 'visualizations')
        if force or not os.path.exists(os.path.join(vis_dir, 'tsne_features.png')):
            exp_visualization(config, device)
        else:
            print(f"\n[Skip] visualization: results already exist (use --force to re-run)")

    if args.mode in ['all', 'efficiency']:
        if not _should_skip('efficiency_results.json', 'efficiency'):
            exp_efficiency(config, device)

    if args.mode in ['all', 'report']:
        exp_report(config)  # 报告总是重新生成

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Paper experiments completed in {elapsed/3600:.2f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
