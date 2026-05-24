"""
多源数据集加载器
合并多个 GenImage 子集进行多源训练

支持的数据源 (GenImage 格式):
- BigGAN, ADM, GLIDE, VQDM, Stable Diffusion, Midjourney

目录结构:
    GenImage/
        BigGAN/imagenet_ai_0419_biggan/
            train/
                ai/       # AI生成图像
                nature/   # 真实图像
            val/
                ai/
                nature/
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# GenImage 子数据集名称 → 子目录映射
GENIMAGE_SOURCES = {
    'biggan': 'BigGAN/imagenet_ai_0419_biggan',
    'adm': 'ADM/imagenet_ai_0508_adm',
    'glide': 'glide/imagenet_glide',
    'vqdm': 'VQDM/imagenet_ai_0419_vqdm',
    'sdv4': 'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4',
    'midjourney': 'Midjourney/imagenet_midjourney',
}


class MultiSourceGenImageDataset(Dataset):
    """
    多源 GenImage 数据集

    合并多个生成模型的数据源，支持:
    - 每源最大样本数限制
    - 类别平衡采样
    - 记录每个样本的来源信息
    """
    def __init__(
        self,
        genimage_root: str,
        sources: List[str] = None,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_samples_per_source: Optional[int] = None,
        balance_classes: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            genimage_root: GenImage 数据集根目录 (包含 BigGAN/, ADM/ 等子目录)
            sources: 要加载的数据源列表，如 ['biggan', 'adm', 'glide', 'vqdm']
                     如果为 None，自动检测所有可用源
            split: 'train' 或 'val'
            transform: 图像变换
            max_samples_per_source: 每个源的最大样本数（每类）
            balance_classes: 是否在每个源内平衡真假类别
            seed: 随机种子
        """
        self.genimage_root = Path(genimage_root)
        self.split = split
        self.transform = transform
        self.seed = seed

        rng = random.Random(seed)

        # 自动检测可用源
        if sources is None:
            sources = self._detect_available_sources()

        # 收集所有样本
        self.samples = []  # List of (path, label, source_name)
        self.source_stats = {}

        image_extensions = {'.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG'}

        for source_name in sources:
            source_dir = self._get_source_dir(source_name)
            if source_dir is None:
                print(f"[MultiSource] Warning: source '{source_name}' not found, skipping")
                continue

            split_dir = source_dir / split
            ai_dir = split_dir / 'ai'
            nature_dir = split_dir / 'nature'

            if not ai_dir.exists() or not nature_dir.exists():
                print(f"[MultiSource] Warning: {split_dir} missing ai/ or nature/, skipping")
                continue

            # 收集图像
            ai_images = [p for p in ai_dir.iterdir() if p.suffix in image_extensions]
            nature_images = [p for p in nature_dir.iterdir() if p.suffix in image_extensions]

            rng.shuffle(ai_images)
            rng.shuffle(nature_images)

            # 限制每源样本数
            if max_samples_per_source:
                ai_images = ai_images[:max_samples_per_source]
                nature_images = nature_images[:max_samples_per_source]

            # 类别平衡
            if balance_classes:
                min_count = min(len(ai_images), len(nature_images))
                ai_images = ai_images[:min_count]
                nature_images = nature_images[:min_count]

            # 添加到样本列表
            for img_path in nature_images:
                self.samples.append((img_path, 0, source_name))
            for img_path in ai_images:
                self.samples.append((img_path, 1, source_name))

            self.source_stats[source_name] = {
                'real': len(nature_images),
                'fake': len(ai_images),
                'total': len(nature_images) + len(ai_images)
            }

            print(f"[MultiSource] {source_name}: {len(nature_images)} real + {len(ai_images)} fake = {len(nature_images) + len(ai_images)} total")

        # 打乱所有样本
        rng.shuffle(self.samples)

        total = len(self.samples)
        total_real = sum(1 for _, l, _ in self.samples if l == 0)
        total_fake = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"[MultiSource] Total {split}: {total_real} real + {total_fake} fake = {total} samples from {len(self.source_stats)} sources")

    def _detect_available_sources(self) -> List[str]:
        """自动检测可用的数据源"""
        available = []
        for name, subdir in GENIMAGE_SOURCES.items():
            source_path = self.genimage_root / subdir
            if source_path.exists():
                available.append(name)
        if not available:
            # 如果标准映射不匹配，尝试直接扫描子目录
            for d in self.genimage_root.iterdir():
                if d.is_dir():
                    for sub in d.iterdir():
                        if sub.is_dir() and (sub / 'train').exists():
                            available.append(d.name.lower())
                            break
        return available

    def _get_source_dir(self, source_name: str) -> Optional[Path]:
        """获取数据源目录"""
        source_name_lower = source_name.lower()

        # 标准映射
        if source_name_lower in GENIMAGE_SOURCES:
            path = self.genimage_root / GENIMAGE_SOURCES[source_name_lower]
            if path.exists():
                return path

        # 尝试直接匹配子目录名
        for d in self.genimage_root.iterdir():
            if d.is_dir() and source_name_lower in d.name.lower():
                for sub in d.iterdir():
                    if sub.is_dir() and (sub / 'train').exists():
                        return sub
                # 如果目录本身就有 train/
                if (d / 'train').exists():
                    return d

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[MultiSource] Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'path': str(img_path),
            'source': source,
        }

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于加权采样或损失加权）"""
        labels = [l for _, l, _ in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """计算每个样本的权重（用于 WeightedRandomSampler）"""
        labels = [l for _, l, _ in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([class_weights[l] for l in labels])
        return sample_weights


def get_multi_source_transforms(
    img_size: int = 224,
    is_train: bool = True,
    strong_aug: bool = True
) -> transforms.Compose:
    """
    获取多源训练的数据变换

    Args:
        img_size: 图像大小
        is_train: 是否训练集
        strong_aug: 是否使用强增强（JPEG/Blur/ColorJitter）
    """
    # CLIP ViT-L/14 标准化参数（非 ImageNet 标准化！）
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    if is_train:
        aug_list = [
            # CLIP 风格预处理：Resize 较大 → RandomCrop
            transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        if strong_aug:
            # 增强：降低频率类增强概率，保护 FDAA 学习信号
            from data.augmentations import RobustAugmentation
            robust_aug = RobustAugmentation(
                jpeg_prob=0.2,
                blur_prob=0.15,
                noise_prob=0.1,
                resize_prob=0.2,
                color_prob=0.5,
                cutout_prob=0.1,
                freq_prob=0.05
            )
            aug_list.append(robust_aug)

        aug_list.extend([
            transforms.ToTensor(),
            normalize,
        ])

        return transforms.Compose(aug_list)
    else:
        # CLIP 风格预处理：Resize → CenterCrop
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_multi_source_dataloader(
    genimage_root: str,
    sources: List[str] = None,
    split: str = 'train',
    batch_size: int = 64,
    img_size: int = 224,
    max_samples_per_source: Optional[int] = 50000,
    num_workers: int = 8,
    strong_aug: bool = True,
    use_weighted_sampler: bool = False,
    seed: int = 42,
    aug_version: str = 'v2',
) -> DataLoader:
    """
    创建多源数据加载器

    Args:
        genimage_root: GenImage 根目录
        sources: 数据源列表
        split: 'train' / 'val'
        batch_size: 批次大小
        img_size: 图像大小
        max_samples_per_source: 每源最大样本数
        num_workers: 工作线程数
        strong_aug: 是否使用强增强
        use_weighted_sampler: 是否使用加权采样器
        seed: 随机种子
        aug_version: 'v2' (标准增强) 或 'v3' (V3 增强，更强频率扰动)
    """
    is_train = (split == 'train')

    if aug_version == 'v3' and is_train:
        from data.augmentations import get_v3_transforms
        transform = get_v3_transforms(img_size, 'train' if is_train else 'val')
    else:
        transform = get_multi_source_transforms(img_size, is_train, strong_aug and is_train)

    dataset = MultiSourceGenImageDataset(
        genimage_root=genimage_root,
        sources=sources,
        split=split,
        transform=transform,
        max_samples_per_source=max_samples_per_source,
        balance_classes=True,
        seed=seed,
    )

    sampler = None
    shuffle = is_train

    if is_train and use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False  # sampler 和 shuffle 互斥

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    import sys

    # Default path for local smoke testing. Override with the first CLI argument
    # or FDAA_GENIMAGE_ROOT.
    genimage_root = os.environ.get(
        "FDAA_GENIMAGE_ROOT",
        "./datasets/authoritative/GenImage",
    )

    if len(sys.argv) > 1:
        genimage_root = sys.argv[1]

    print(f"Testing MultiSourceGenImageDataset from: {genimage_root}")
    print()

    # 测试自动检测
    dataset = MultiSourceGenImageDataset(
        genimage_root=genimage_root,
        sources=['biggan', 'adm', 'glide', 'vqdm'],
        split='train',
        max_samples_per_source=100,
    )

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Source stats: {dataset.source_stats}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Label: {sample['label']}, Source: {sample['source']}")

    # 测试 dataloader
    print("\nTesting dataloader...")
    loader = create_multi_source_dataloader(
        genimage_root=genimage_root,
        sources=['biggan', 'adm'],
        split='train',
        batch_size=4,
        max_samples_per_source=50,
        num_workers=0,
        strong_aug=False,
    )

    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels: {batch['label']}")
    print(f"Batch sources: {batch['source']}")
