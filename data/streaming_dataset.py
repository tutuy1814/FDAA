"""
流式数据集加载模块

支持从 Hugging Face 流式读取数据，不占用本地存储空间

支持的数据集:
1. Hemg/AI-Generated-vs-Real-Images-Datasets
2. InfImagine/FakeImageDataset
3. blorg469/genimage
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from typing import Optional, Callable, Iterator
from PIL import Image
import io


class HFStreamingDataset(IterableDataset):
    """
    Hugging Face 流式数据集

    不需要下载到本地，直接从网络流式读取
    """

    def __init__(
        self,
        dataset_name: str = "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        skip_samples: int = 0,
        image_key: str = "image",
        label_key: str = "label",
        label_mapping: Optional[dict] = None
    ):
        """
        Args:
            dataset_name: Hugging Face 数据集名称
            split: 数据集划分 ('train', 'test', 'validation')
            transform: 图像变换
            max_samples: 最大样本数 (None表示全部)
            skip_samples: 跳过前N个样本（用于划分训练/验证集）
            image_key: 图像字段名
            label_key: 标签字段名
            label_mapping: 标签映射字典 (如 {'real': 0, 'fake': 1})
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.image_key = image_key
        self.label_key = label_key
        self.label_mapping = label_mapping or {}

        # 延迟加载数据集
        self._dataset = None

    def _load_dataset(self):
        """延迟加载数据集"""
        if self._dataset is None:
            from datasets import load_dataset
            print(f"Loading streaming dataset: {self.dataset_name} [{self.split}]")
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=True
            )
            # 添加 shuffle 确保类别均匀分布
            self._dataset = self._dataset.shuffle(seed=42, buffer_size=10000)
        return self._dataset

    def __iter__(self) -> Iterator[dict]:
        dataset = self._load_dataset()

        count = 0
        skipped = 0
        for sample in dataset:
            # 跳过前 skip_samples 个样本
            if skipped < self.skip_samples:
                skipped += 1
                continue

            if self.max_samples and count >= self.max_samples:
                break

            try:
                # 获取图像
                image = sample[self.image_key]
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image)).convert('RGB')
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')
                else:
                    image = image.convert('RGB')

                # 获取标签
                label = sample[self.label_key]
                if isinstance(label, str):
                    label = self.label_mapping.get(label.lower(), label)
                    if isinstance(label, str):
                        # 尝试推断标签
                        if 'real' in label.lower():
                            label = 0
                        elif 'fake' in label.lower() or 'ai' in label.lower() or 'generated' in label.lower():
                            label = 1
                        else:
                            label = int(label) if label.isdigit() else 0

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                yield {
                    'image': image,
                    'label': int(label),
                    'path': f"streaming_{count}"
                }

                count += 1

            except Exception as e:
                print(f"Error processing sample {count}: {e}")
                continue

    def __len__(self):
        # 流式数据集无法准确知道长度
        return self.max_samples if self.max_samples else 10000  # 估计值


class MultiSourceStreamingDataset(IterableDataset):
    """
    多源流式数据集

    合并多个 Hugging Face 数据集
    """

    SUPPORTED_DATASETS = {
        'ai_vs_real': {
            'name': 'Hemg/AI-Generated-vs-Real-Images-Datasets',
            'image_key': 'image',
            'label_key': 'label',
            'label_mapping': {'real': 0, 'fake': 1, 'ai': 1}
        },
        'fake_image': {
            'name': 'InfImagine/FakeImageDataset',
            'image_key': 'image',
            'label_key': 'label',
            'label_mapping': {'real': 0, 'fake': 1}
        },
        'genimage': {
            'name': 'blorg469/genimage',
            'image_key': 'image',
            'label_key': 'label',
            'label_mapping': {'real': 0, 'fake': 1}
        }
    }

    def __init__(
        self,
        datasets: list = ['ai_vs_real'],
        split: str = 'train',
        transform: Optional[Callable] = None,
        samples_per_dataset: int = 5000
    ):
        """
        Args:
            datasets: 要使用的数据集列表
            split: 数据集划分
            transform: 图像变换
            samples_per_dataset: 每个数据集的样本数
        """
        self.datasets = datasets
        self.split = split
        self.transform = transform
        self.samples_per_dataset = samples_per_dataset

    def __iter__(self):
        for ds_key in self.datasets:
            if ds_key not in self.SUPPORTED_DATASETS:
                print(f"Unknown dataset: {ds_key}")
                continue

            ds_config = self.SUPPORTED_DATASETS[ds_key]

            try:
                streaming_ds = HFStreamingDataset(
                    dataset_name=ds_config['name'],
                    split=self.split,
                    transform=self.transform,
                    max_samples=self.samples_per_dataset,
                    image_key=ds_config['image_key'],
                    label_key=ds_config['label_key'],
                    label_mapping=ds_config['label_mapping']
                )

                for sample in streaming_ds:
                    yield sample

            except Exception as e:
                print(f"Error loading dataset {ds_key}: {e}")
                continue

    def __len__(self):
        return len(self.datasets) * self.samples_per_dataset


def get_streaming_transforms(img_size: int = 224, split: str = 'train'):
    """获取流式数据集的变换"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def create_streaming_dataloader(
    dataset_name: str = "Hemg/AI-Generated-vs-Real-Images-Datasets",
    split: str = 'train',
    batch_size: int = 32,
    img_size: int = 224,
    max_samples: int = 10000,
    skip_samples: int = 0,
    num_workers: int = 0,  # 流式数据集建议用0
    is_training: bool = None  # None 则根据 split 自动判断
) -> DataLoader:
    """
    创建流式数据加载器

    Args:
        dataset_name: 数据集名称
        split: 数据集划分
        batch_size: 批大小
        img_size: 图像大小
        max_samples: 最大样本数
        skip_samples: 跳过前N个样本（用于从train划分验证集）
        num_workers: 工作进程数 (流式建议用0)
        is_training: 是否为训练模式（影响数据增强），None则根据split自动判断
    Returns:
        DataLoader
    """
    # 确定 transform 模式
    if is_training is None:
        is_training = (split == 'train' and skip_samples == 0)
    transform_split = 'train' if is_training else 'val'
    transform = get_streaming_transforms(img_size, transform_split)

    dataset = HFStreamingDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        max_samples=max_samples,
        skip_samples=skip_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# 测试代码
if __name__ == "__main__":
    print("测试流式数据集...")
    print("=" * 60)

    # 安装依赖提示
    print("\n请确保已安装 datasets 库:")
    print("  pip install datasets\n")

    try:
        # 测试流式加载
        transform = get_streaming_transforms(224, 'train')

        dataset = HFStreamingDataset(
            dataset_name="Hemg/AI-Generated-vs-Real-Images-Datasets",
            split="train",
            transform=transform,
            max_samples=10  # 只测试10个样本
        )

        print("加载数据集...")
        count = 0
        for sample in dataset:
            print(f"  样本 {count}: shape={sample['image'].shape}, label={sample['label']}")
            count += 1
            if count >= 5:
                break

        print(f"\n成功加载 {count} 个样本!")
        print("流式数据集测试通过!")

    except ImportError:
        print("请先安装 datasets 库:")
        print("  pip install datasets")
    except Exception as e:
        print(f"测试失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. 数据集不存在或格式变化")
        print("  3. 需要登录 Hugging Face")
