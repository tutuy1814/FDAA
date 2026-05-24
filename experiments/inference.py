"""
推理脚本

Usage:
    python experiments/inference.py --checkpoint model_best.pth --image test.jpg
    python experiments/inference.py --checkpoint model_best.pth --image_dir ./test_images --output results.json
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AIGCDetector, AIGCDetectorLite
from data import get_transforms
from utils.checkpoint import load_checkpoint


class AIGCDetectorInference:
    """
    AIGC检测推理类
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        img_size: int = 224,
        use_lite: bool = True
    ):
        """
        Args:
            checkpoint_path: 检查点路径
            device: 设备
            img_size: 图像大小
            use_lite: 是否使用轻量版模型
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size

        # 构建模型
        if use_lite:
            self.model = AIGCDetectorLite(
                num_classes=2,
                img_size=img_size,
                embed_dim=768
            )
        else:
            self.model = AIGCDetector(
                backbone='ViT-L/14',
                num_classes=2,
                img_size=img_size
            )

        # 加载权重
        load_checkpoint(checkpoint_path, self.model, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据变换
        self.transform = get_transforms(img_size, split='test')

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        return_attention: bool = False
    ) -> dict:
        """
        预测单张图像

        Args:
            image_path: 图像路径
            return_attention: 是否返回注意力图
        Returns:
            result: 预测结果
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 前向传播
        outputs = self.model(image_tensor, return_attention=return_attention)

        # 计算概率
        probs = torch.softmax(outputs['logits'], dim=1)[0]
        pred_prob = probs[1].item()  # fake的概率
        pred_label = 'fake' if pred_prob >= 0.5 else 'real'

        result = {
            'path': image_path,
            'prediction': pred_label,
            'fake_probability': pred_prob,
            'real_probability': probs[0].item(),
            'confidence': max(pred_prob, 1 - pred_prob)
        }

        if return_attention and 'forgery_map' in outputs:
            result['forgery_map'] = outputs['forgery_map'][0].cpu().numpy()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: list,
        batch_size: int = 16
    ) -> list:
        """
        批量预测

        Args:
            image_paths: 图像路径列表
            batch_size: 批大小
        Returns:
            results: 预测结果列表
        """
        results = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc='Predicting'):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs['logits'], dim=1)

            for j, path in enumerate(batch_paths[:len(batch_images)]):
                pred_prob = probs[j, 1].item()
                results.append({
                    'path': path,
                    'prediction': 'fake' if pred_prob >= 0.5 else 'real',
                    'fake_probability': pred_prob,
                    'confidence': max(pred_prob, 1 - pred_prob)
                })

        return results

    def predict_directory(
        self,
        image_dir: str,
        batch_size: int = 16,
        extensions: list = None
    ) -> list:
        """
        预测目录中的所有图像

        Args:
            image_dir: 图像目录
            batch_size: 批大小
            extensions: 支持的文件扩展名
        Returns:
            results: 预测结果列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        image_dir = Path(image_dir)
        image_paths = []

        for ext in extensions:
            image_paths.extend(image_dir.glob(f'**/*{ext}'))
            image_paths.extend(image_dir.glob(f'**/*{ext.upper()}'))

        image_paths = [str(p) for p in image_paths]
        print(f"Found {len(image_paths)} images in {image_dir}")

        return self.predict_batch(image_paths, batch_size)


def visualize_result(result: dict, output_path: str = None):
    """
    可视化预测结果

    Args:
        result: 预测结果
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    image = Image.open(result['path'])

    fig, axes = plt.subplots(1, 2 if 'forgery_map' in result else 1, figsize=(12, 5))

    if 'forgery_map' not in result:
        axes = [axes]

    # 原始图像
    axes[0].imshow(image)
    pred = result['prediction']
    prob = result['fake_probability']
    color = 'red' if pred == 'fake' else 'green'
    axes[0].set_title(f"Prediction: {pred.upper()}\nFake Probability: {prob:.2%}", color=color)
    axes[0].axis('off')

    # 伪造热力图
    if 'forgery_map' in result:
        forgery_map = result['forgery_map']
        side = int(np.sqrt(len(forgery_map)))
        forgery_2d = forgery_map.reshape(side, side)

        # 上采样
        from scipy.ndimage import zoom
        h, w = image.size[1], image.size[0]
        scale_h, scale_w = h / side, w / side
        forgery_resized = zoom(forgery_2d, (scale_h, scale_w), order=1)

        axes[1].imshow(image)
        im = axes[1].imshow(forgery_resized, cmap='hot', alpha=0.6)
        axes[1].set_title('Forgery Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main(args):
    """主函数"""
    # 创建推理器
    detector = AIGCDetectorInference(
        checkpoint_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu',
        img_size=args.img_size,
        use_lite=not args.full_model
    )

    if args.image:
        # 单张图像
        result = detector.predict(args.image, return_attention=args.visualize)
        print(f"\nPrediction for {args.image}:")
        print(f"  Result: {result['prediction'].upper()}")
        print(f"  Fake Probability: {result['fake_probability']:.2%}")
        print(f"  Confidence: {result['confidence']:.2%}")

        if args.visualize:
            output_path = args.output or args.image.replace('.', '_prediction.')
            visualize_result(result, output_path)

    elif args.image_dir:
        # 目录批量处理
        results = detector.predict_directory(
            args.image_dir,
            batch_size=args.batch_size
        )

        # 统计
        fake_count = sum(1 for r in results if r['prediction'] == 'fake')
        real_count = len(results) - fake_count

        print(f"\nResults Summary:")
        print(f"  Total images: {len(results)}")
        print(f"  Predicted fake: {fake_count} ({fake_count / len(results):.1%})")
        print(f"  Predicted real: {real_count} ({real_count / len(results):.1%})")

        # 保存结果
        output_path = args.output or 'predictions.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    else:
        print("Please provide --image or --image_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIGC Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to image directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for directory processing')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--full_model', action='store_true',
                        help='Use full model with CLIP backbone')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize prediction with attention map')

    args = parser.parse_args()
    main(args)
