"""
GenImage 本地数据集加载器

支持加载本地解压后的 GenImage 数据集
目录结构:
    dataset_root/
        train/
            ai/       # AI生成图像
            nature/   # 真实图像
        val/
            ai/
            nature/
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class GenImageDataset(Dataset):
    """GenImage 本地数据集"""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = True
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train' 或 'val'
            transform: 图像变换
            max_samples: 最大样本数 (每类)
            balance_classes: 是否平衡类别
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # 构建路径
        split_dir = self.root_dir / split
        ai_dir = split_dir / 'ai'
        nature_dir = split_dir / 'nature'

        if not ai_dir.exists() or not nature_dir.exists():
            raise ValueError(f"Dataset directories not found: {split_dir}")

        # 收集图像路径 (支持多种扩展名)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        ai_images = []
        nature_images = []
        for ext in image_extensions:
            ai_images.extend(ai_dir.glob(ext))
            nature_images.extend(nature_dir.glob(ext))

        # 随机打乱
        random.shuffle(ai_images)
        random.shuffle(nature_images)

        # 限制样本数
        if max_samples:
            ai_images = ai_images[:max_samples]
            nature_images = nature_images[:max_samples]

        # 平衡类别
        if balance_classes:
            min_count = min(len(ai_images), len(nature_images))
            ai_images = ai_images[:min_count]
            nature_images = nature_images[:min_count]

        # 构建数据列表: (path, label)
        # label: 0 = real (nature), 1 = AI-generated
        self.samples = []
        for img_path in nature_images:
            self.samples.append((img_path, 0))
        for img_path in ai_images:
            self.samples.append((img_path, 1))

        # 打乱
        random.shuffle(self.samples)

        print(f"GenImage {split}: {len(nature_images)} real + {len(ai_images)} AI = {len(self.samples)} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回随机图像
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'path': str(img_path)
        }


def get_genimage_transforms(img_size: int = 224, is_train: bool = True):
    """获取数据变换（使用 CLIP 标准化参数）"""
    # CLIP ViT-L/14 标准化参数
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            clip_normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            clip_normalize,
        ])


def create_genimage_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    img_size: int = 224,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """创建 GenImage 数据加载器"""

    is_train = (split == 'train')
    transform = get_genimage_transforms(img_size, is_train)

    dataset = GenImageDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        max_samples=max_samples,
        balance_classes=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )


if __name__ == "__main__":
    # 测试数据加载
    import sys

    # 默认路径
    data_root = "datasets/genimage_partial/imagenet_ai_0419_biggan"

    if len(sys.argv) > 1:
        data_root = sys.argv[1]

    print(f"Testing GenImage dataset from: {data_root}")

    # 测试训练集
    train_loader = create_genimage_dataloader(
        root_dir=data_root,
        split='train',
        batch_size=4,
        max_samples=100,
        num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch['image'].shape}")
    print(f"Labels: {batch['label']}")
    print(f"Sample paths: {batch['path'][:2]}")

    # 测试验证集
    val_loader = create_genimage_dataloader(
        root_dir=data_root,
        split='val',
        batch_size=4,
        max_samples=100,
        num_workers=0
    )

    batch = next(iter(val_loader))
    print(f"\nVal batch shape: {batch['image'].shape}")
    print(f"Val labels: {batch['label']}")
