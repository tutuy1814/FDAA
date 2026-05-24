"""
增强的数据增强模块
用于提升模型的泛化能力和鲁棒性

包含:
1. RobustAugmentation - 鲁棒性增强（模拟各种图像降质）
2. DomainRandomization - 域随机化（减少对特定域的依赖）
3. FrequencyAugmentation - 频率域增强（干扰频率特征）
4. get_robust_transforms - 获取增强的数据变换
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import io
import math


class JPEGCompression:
    """JPEG压缩增强"""
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianBlur:
    """高斯模糊"""
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class GaussianNoise:
    """高斯噪声"""
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        img_array = np.array(img).astype(np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1) * 255
        return Image.fromarray(img_array.astype(np.uint8))


class DownsampleUpsample:
    """下采样再上采样（模拟压缩伪影）"""
    def __init__(self, scale_range=(0.25, 0.75)):
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)

        # 下采样
        img_down = img.resize((new_w, new_h), Image.BILINEAR)
        # 上采样回原始大小
        img_up = img_down.resize((w, h), Image.BILINEAR)

        return img_up


class MultiScaleResize:
    """多尺度随机缩放 - 模拟不同来源图像的分辨率差异"""
    def __init__(self, target_size=224, scale_range=(0.3, 1.5)):
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolations = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        w, h = img.size
        scale = random.uniform(*self.scale_range)
        interp = random.choice(self.interpolations)

        # 先缩放到不同尺寸
        new_w, new_h = max(int(w * scale), 16), max(int(h * scale), 16)
        img = img.resize((new_w, new_h), interp)

        # 再缩放回来
        img = img.resize((w, h), interp)
        return img


class RandomBrightness:
    """随机亮度调整"""
    def __init__(self, factor_range=(0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)


class RandomContrast:
    """随机对比度调整"""
    def __init__(self, factor_range=(0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)


class RandomSaturation:
    """随机饱和度调整"""
    def __init__(self, factor_range=(0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)


class RandomSharpness:
    """随机锐度调整"""
    def __init__(self, factor_range=(0.5, 2.0)):
        self.factor_range = factor_range

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)


class RandomCutout:
    """随机擦除（Cutout）"""
    def __init__(self, num_holes=1, max_h_size=32, max_w_size=32):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, img):
        if random.random() < 0.5:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        for _ in range(self.num_holes):
            h_size = random.randint(1, self.max_h_size)
            w_size = random.randint(1, self.max_w_size)
            y = random.randint(0, h - h_size)
            x = random.randint(0, w - w_size)

            # 用灰色填充
            img_array[y:y+h_size, x:x+w_size] = 128

        return Image.fromarray(img_array)


class RandomGridShuffle:
    """随机网格打乱"""
    def __init__(self, grid_size=4):
        self.grid_size = grid_size

    def __call__(self, img):
        if random.random() < 0.7:  # 30%概率应用
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]
        gh, gw = h // self.grid_size, w // self.grid_size

        # 分割成网格
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = img_array[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                patches.append(patch)

        # 随机打乱
        random.shuffle(patches)

        # 重组
        new_img = np.zeros_like(img_array)
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                new_img[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = patches[idx]
                idx += 1

        return Image.fromarray(new_img)


class FrequencyMask:
    """频率域遮罩（干扰特定频率）"""
    def __init__(self, mask_type='random'):
        """
        mask_type: 'low', 'high', 'band', 'random'
        """
        self.mask_type = mask_type

    def __call__(self, img):
        if random.random() < 0.7:  # 30%概率应用
            return img

        img_array = np.array(img).astype(np.float32)

        # 处理每个通道
        for c in range(3):
            channel = img_array[:, :, c]
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)

            h, w = channel.shape
            cy, cx = h // 2, w // 2

            # 创建掩码
            if self.mask_type == 'low':
                # 遮罩低频
                r = random.randint(5, 20)
                mask = np.ones((h, w))
                mask[cy-r:cy+r, cx-r:cx+r] = random.uniform(0.5, 0.9)
            elif self.mask_type == 'high':
                # 遮罩高频
                r = random.randint(h//4, h//2)
                y, x = np.ogrid[:h, :w]
                mask = np.sqrt((y - cy)**2 + (x - cx)**2) < r
                mask = mask.astype(np.float32)
                mask = mask * (1 - random.uniform(0.1, 0.3)) + random.uniform(0.1, 0.3)
            elif self.mask_type == 'band':
                # 遮罩特定频带
                r1 = random.randint(10, 30)
                r2 = random.randint(r1 + 10, r1 + 40)
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                mask = ~((dist > r1) & (dist < r2))
                mask = mask.astype(np.float32) * 0.7 + 0.3
            else:  # random
                mask = np.random.uniform(0.7, 1.0, (h, w))

            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            channel_back = np.fft.ifft2(f_ishift)
            img_array[:, :, c] = np.abs(channel_back)

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class JPEGResizeChain:
    """
    JPEG-Resize 处理链（模拟社交媒体分享场景）

    模拟真实世界中图像经过多次压缩和缩放的过程：
    1. JPEG 压缩（模拟上传平台压缩）
    2. 随机缩放（模拟平台自动 resize）
    3. 再次 JPEG 压缩（模拟下载后再分享）

    这种串行降质能有效训练模型对频率伪影的鲁棒性。
    """
    def __init__(self, quality_range=(40, 85), scale_range=(0.5, 0.9)):
        self.quality_range = quality_range
        self.scale_range = scale_range

    def __call__(self, img):
        w, h = img.size

        # Step 1: 第一次 JPEG 压缩
        quality1 = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality1)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')

        # Step 2: 随机缩放
        scale = random.uniform(*self.scale_range)
        interp = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        new_w, new_h = max(int(w * scale), 16), max(int(h * scale), 16)
        img = img.resize((new_w, new_h), interp)
        img = img.resize((w, h), interp)

        # Step 3: 第二次 JPEG 压缩（质量略低）
        quality2 = random.randint(max(30, quality1 - 15), quality1)
        buffer2 = io.BytesIO()
        img.save(buffer2, format='JPEG', quality=quality2)
        buffer2.seek(0)
        img = Image.open(buffer2).convert('RGB')

        return img


class MixUp:
    """MixUp增强（需要在batch级别应用）"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, labels):
        """
        Args:
            images: [B, C, H, W] tensor
            labels: [B] tensor
        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix增强（需要在batch级别应用）"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, images, labels):
        """
        Args:
            images: [B, C, H, W] tensor
            labels: [B] tensor
        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # 计算裁剪区域
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 应用CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))

        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class RobustAugmentation:
    """
    综合鲁棒性增强
    组合多种增强方法，模拟真实世界的各种图像变化

    频率类增强（JPEG/Blur/Noise/FreqMask）互斥：最多应用一种，
    避免串行叠加破坏 FDAA 需要学习的频率伪影信号。
    """
    def __init__(
        self,
        jpeg_prob=0.3,
        blur_prob=0.3,
        noise_prob=0.3,
        resize_prob=0.3,
        color_prob=0.5,
        cutout_prob=0.2,
        freq_prob=0.2,
        multiscale_prob=0.3
    ):
        # 频率类增强（互斥，最多应用一种）
        self.freq_augmentations = []
        if jpeg_prob > 0:
            self.freq_augmentations.append((JPEGCompression(quality_range=(40, 95)), jpeg_prob))
        if blur_prob > 0:
            self.freq_augmentations.append((GaussianBlur(radius_range=(0.5, 1.5)), blur_prob))
        if noise_prob > 0:
            self.freq_augmentations.append((GaussianNoise(std_range=(0.01, 0.03)), noise_prob))
        if freq_prob > 0:
            self.freq_augmentations.append((FrequencyMask(mask_type='random'), freq_prob))

        # 非频率类增强（独立应用）
        self.other_augmentations = []
        if resize_prob > 0:
            self.other_augmentations.append((DownsampleUpsample(scale_range=(0.5, 0.8)), resize_prob))
        if multiscale_prob > 0:
            self.other_augmentations.append((MultiScaleResize(scale_range=(0.3, 1.5)), multiscale_prob))
        if color_prob > 0:
            self.other_augmentations.append((RandomBrightness(factor_range=(0.8, 1.2)), color_prob / 4))
            self.other_augmentations.append((RandomContrast(factor_range=(0.8, 1.2)), color_prob / 4))
            self.other_augmentations.append((RandomSaturation(factor_range=(0.8, 1.2)), color_prob / 4))
            self.other_augmentations.append((RandomSharpness(factor_range=(0.7, 1.5)), color_prob / 4))
        if cutout_prob > 0:
            self.other_augmentations.append((RandomCutout(num_holes=1, max_h_size=24, max_w_size=24), cutout_prob))

    def __call__(self, img):
        # 频率类增强：互斥，随机打乱后最多应用一种
        if self.freq_augmentations:
            shuffled = list(self.freq_augmentations)
            random.shuffle(shuffled)
            for aug, prob in shuffled:
                if random.random() < prob:
                    img = aug(img)
                    break  # 最多应用一种频率增强

        # 非频率类增强：独立应用
        for aug, prob in self.other_augmentations:
            if random.random() < prob:
                img = aug(img)
        return img


def get_robust_transforms(
    img_size: int = 224,
    split: str = 'train',
    augmentation_level: str = 'strong'
) -> T.Compose:
    """
    获取增强的数据变换

    Args:
        img_size: 图像大小
        split: 数据集划分 ('train', 'val', 'test')
        augmentation_level: 增强强度 ('weak', 'medium', 'strong')

    Returns:
        transforms.Compose
    """
    # CLIP ViT-L/14 标准化参数
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    if split == 'train':
        # 基础几何变换
        base_transforms = [
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
        ]

        # 根据增强强度添加不同的增强
        if augmentation_level == 'weak':
            robust_aug = RobustAugmentation(
                jpeg_prob=0.1,
                blur_prob=0.1,
                noise_prob=0.1,
                resize_prob=0.1,
                color_prob=0.3,
                cutout_prob=0.0,
                freq_prob=0.0
            )
        elif augmentation_level == 'medium':
            robust_aug = RobustAugmentation(
                jpeg_prob=0.2,
                blur_prob=0.2,
                noise_prob=0.2,
                resize_prob=0.2,
                color_prob=0.4,
                cutout_prob=0.1,
                freq_prob=0.1
            )
        else:  # strong
            robust_aug = RobustAugmentation(
                jpeg_prob=0.3,
                blur_prob=0.3,
                noise_prob=0.3,
                resize_prob=0.3,
                color_prob=0.5,
                cutout_prob=0.2,
                freq_prob=0.2
            )

        return T.Compose([
            *base_transforms,
            robust_aug,
            T.ToTensor(),
            normalize
        ])
    else:
        # 验证/测试集只做基础变换
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize
        ])


def get_v3_transforms(
    img_size: int = 224,
    split: str = 'train',
) -> T.Compose:
    """
    V3 增强变换（更强的频率扰动 + 社交媒体模拟链）

    与 get_robust_transforms 的区别：
    - 频率增强概率更高：jpeg=0.3, blur=0.25, noise=0.2, freq=0.1
    - 新增 JPEGResizeChain（模拟社交媒体处理链）
    """
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    if split == 'train':
        base_transforms = [
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
        ]

        robust_aug = RobustAugmentation(
            jpeg_prob=0.3,
            blur_prob=0.25,
            noise_prob=0.2,
            resize_prob=0.3,
            color_prob=0.5,
            cutout_prob=0.2,
            freq_prob=0.1,
        )

        # JPEGResizeChain 独立于 RobustAugmentation（低概率应用）
        jpeg_resize_chain = JPEGResizeChain(quality_range=(40, 85), scale_range=(0.5, 0.9))

        class V3AugPipeline:
            def __init__(self, robust, chain, chain_prob=0.1):
                self.robust = robust
                self.chain = chain
                self.chain_prob = chain_prob

            def __call__(self, img):
                # 10% 概率使用社交媒体处理链（替代 robust_aug）
                if random.random() < self.chain_prob:
                    return self.chain(img)
                return self.robust(img)

        return T.Compose([
            *base_transforms,
            V3AugPipeline(robust_aug, jpeg_resize_chain, chain_prob=0.1),
            T.ToTensor(),
            normalize
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize
        ])


def get_test_time_augmentation(img_size: int = 224) -> list:
    """
    获取测试时增强 (TTA)

    Returns:
        list of transforms
    """
    # CLIP ViT-L/14 标准化参数
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    return [
        # 原始
        T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize]),
        # 水平翻转
        T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), normalize]),
        # 轻微缩放
        T.Compose([T.Resize((int(img_size * 1.1), int(img_size * 1.1))), T.CenterCrop(img_size), T.ToTensor(), normalize]),
    ]


# 测试代码
if __name__ == "__main__":
    print("Testing augmentations...")

    # 创建测试图像
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    for x in range(256):
        for y in range(256):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))

    # 测试各个增强
    print("\nTesting individual augmentations:")

    augs = [
        ("JPEG Compression", JPEGCompression()),
        ("Gaussian Blur", GaussianBlur()),
        ("Gaussian Noise", GaussianNoise()),
        ("Downsample Upsample", DownsampleUpsample()),
        ("Random Brightness", RandomBrightness()),
        ("Random Contrast", RandomContrast()),
        ("Random Saturation", RandomSaturation()),
        ("Random Sharpness", RandomSharpness()),
        ("Random Cutout", RandomCutout()),
        ("Frequency Mask", FrequencyMask()),
    ]

    for name, aug in augs:
        out = aug(img.copy())
        print(f"  {name}: output size = {out.size}")

    # 测试综合增强
    print("\nTesting RobustAugmentation:")
    robust_aug = RobustAugmentation()
    out = robust_aug(img.copy())
    print(f"  Output size: {out.size}")

    # 测试完整变换流程
    print("\nTesting full transforms pipeline:")
    for level in ['weak', 'medium', 'strong']:
        transform = get_robust_transforms(224, 'train', level)
        tensor = transform(img.copy())
        print(f"  {level}: output shape = {tensor.shape}")

    # 测试MixUp和CutMix（需要batch）
    print("\nTesting batch-level augmentations:")
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])

    mixup = MixUp(alpha=0.2)
    mixed, la, lb, lam = mixup(images, labels)
    print(f"  MixUp: shape = {mixed.shape}, lambda = {lam:.3f}")

    cutmix = CutMix(alpha=1.0)
    mixed, la, lb, lam = cutmix(images, labels)
    print(f"  CutMix: shape = {mixed.shape}, lambda = {lam:.3f}")

    print("\nAll augmentation tests passed!")
