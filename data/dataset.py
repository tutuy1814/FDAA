"""
数据集加载模块

支持多种数据集格式:
1. FaceForensics++ 格式
2. GenImage 格式
3. 通用图像文件夹格式
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class AIGCDataset(Dataset):
    """
    AI生成图像检测数据集

    支持的数据格式:
    1. 文件夹格式: real/ 和 fake/ 子文件夹
    2. JSON格式: 包含图像路径和标签的JSON文件
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        json_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = True
    ):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据增强
            json_path: JSON配置文件路径 (可选)
            max_samples: 最大样本数 (可选)
            balance_classes: 是否平衡正负样本
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.balance_classes = balance_classes

        # 加载数据
        if json_path and os.path.exists(json_path):
            self.samples = self._load_from_json(json_path)
        else:
            self.samples = self._load_from_folder()

        # 平衡类别
        if balance_classes and split == 'train':
            self.samples = self._balance_samples(self.samples)

        # 限制样本数
        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_from_folder(self) -> List[Dict]:
        """从文件夹结构加载数据"""
        samples = []

        # 查找 real 和 fake 文件夹
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            split_dir = self.data_root

        real_dir = split_dir / 'real'
        fake_dir = split_dir / 'fake'

        # 也支持 0_real 和 1_fake 命名
        if not real_dir.exists():
            real_dir = split_dir / '0_real'
        if not fake_dir.exists():
            fake_dir = split_dir / '1_fake'

        # 加载真实图像
        if real_dir.exists():
            for img_path in real_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append({
                        'path': str(img_path),
                        'label': 0,
                        'class_name': 'real'
                    })

        # 加载伪造图像
        if fake_dir.exists():
            for img_path in fake_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append({
                        'path': str(img_path),
                        'label': 1,
                        'class_name': 'fake'
                    })

        return samples

    def _load_from_json(self, json_path: str) -> List[Dict]:
        """从JSON文件加载数据"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            if self.split in item.get('split', self.split):
                samples.append({
                    'path': os.path.join(self.data_root, item['path']),
                    'label': item['label'],
                    'class_name': item.get('class_name', 'real' if item['label'] == 0 else 'fake')
                })

        return samples

    def _balance_samples(self, samples: List[Dict]) -> List[Dict]:
        """平衡正负样本"""
        real_samples = [s for s in samples if s['label'] == 0]
        fake_samples = [s for s in samples if s['label'] == 1]

        min_count = min(len(real_samples), len(fake_samples))

        if len(real_samples) > min_count:
            real_samples = random.sample(real_samples, min_count)
        if len(fake_samples) > min_count:
            fake_samples = random.sample(fake_samples, min_count)

        balanced = real_samples + fake_samples
        random.shuffle(balanced)

        return balanced

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # 返回一个随机样本
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': sample['label'],
            'path': sample['path'],
            'class_name': sample['class_name']
        }


class FFppDataset(Dataset):
    """
    FaceForensics++ 数据集

    目录结构:
    FaceForensics++/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/
    │           └── frames/
    └── manipulated_sequences/
        ├── Deepfakes/
        ├── Face2Face/
        ├── FaceSwap/
        └── NeuralTextures/
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        compression: str = 'c23',
        manipulation_methods: List[str] = None,
        transform: Optional[Callable] = None,
        frames_per_video: int = 32,
        split_json: Optional[str] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.compression = compression
        self.transform = transform
        self.frames_per_video = frames_per_video

        if manipulation_methods is None:
            manipulation_methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self.manipulation_methods = manipulation_methods

        # 加载视频列表
        if split_json and os.path.exists(split_json):
            self.video_list = self._load_split_json(split_json)
        else:
            self.video_list = self._create_video_list()

        # 展开为帧级别
        self.samples = self._expand_to_frames()

        print(f"FFpp {split}: {len(self.samples)} frames from {len(self.video_list)} videos")

    def _load_split_json(self, json_path: str) -> List[Dict]:
        """从JSON加载数据划分"""
        with open(json_path, 'r') as f:
            splits = json.load(f)
        return splits.get(self.split, [])

    def _create_video_list(self) -> List[Dict]:
        """创建视频列表"""
        video_list = []

        # 真实视频
        real_dir = self.data_root / 'original_sequences' / 'youtube' / self.compression / 'frames'
        if real_dir.exists():
            for video_dir in real_dir.iterdir():
                if video_dir.is_dir():
                    video_list.append({
                        'path': str(video_dir),
                        'label': 0,
                        'method': 'real'
                    })

        # 伪造视频
        for method in self.manipulation_methods:
            fake_dir = self.data_root / 'manipulated_sequences' / method / self.compression / 'frames'
            if fake_dir.exists():
                for video_dir in fake_dir.iterdir():
                    if video_dir.is_dir():
                        video_list.append({
                            'path': str(video_dir),
                            'label': 1,
                            'method': method
                        })

        return video_list

    def _expand_to_frames(self) -> List[Dict]:
        """展开为帧级别样本"""
        samples = []

        for video in self.video_list:
            video_path = Path(video['path'])
            frames = sorted(video_path.glob('*.png'))[:self.frames_per_video]

            for frame_path in frames:
                samples.append({
                    'path': str(frame_path),
                    'label': video['label'],
                    'method': video['method'],
                    'video_name': video_path.name
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image = Image.open(sample['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': sample['label'],
            'path': sample['path'],
            'method': sample['method'],
            'video_name': sample['video_name']
        }


def get_transforms(img_size: int = 224, split: str = 'train') -> transforms.Compose:
    """
    获取数据变换

    Args:
        img_size: 图像大小
        split: 数据集划分
    Returns:
        transforms.Compose
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8,
    **kwargs
) -> DataLoader:
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批大小
        img_size: 图像大小
        num_workers: 工作进程数
    Returns:
        DataLoader
    """
    transform = get_transforms(img_size, split)

    dataset = AIGCDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        **kwargs
    )

    shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


# 测试代码
if __name__ == "__main__":
    print("Testing dataset...")

    # 创建测试变换
    transform = get_transforms(224, 'train')

    # 创建临时测试数据
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建目录结构
        real_dir = Path(tmpdir) / 'train' / 'real'
        fake_dir = Path(tmpdir) / 'train' / 'fake'
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)

        # 创建测试图像
        for i in range(10):
            img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            img.save(real_dir / f'real_{i}.jpg')
            img.save(fake_dir / f'fake_{i}.jpg')

        # 测试数据集
        dataset = AIGCDataset(tmpdir, split='train', transform=transform)
        print(f"Dataset size: {len(dataset)}")

        # 测试单个样本
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")

        # 测试数据加载器
        dataloader = create_dataloader(tmpdir, split='train', batch_size=4)
        batch = next(iter(dataloader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch labels: {batch['label']}")

    print("Dataset test passed!")
