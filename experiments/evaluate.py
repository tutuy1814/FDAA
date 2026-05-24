"""
评估脚本

Usage:
    python experiments/evaluate.py --config configs/default.yaml --checkpoint checkpoints/model_best.pth
    python experiments/evaluate.py --checkpoint checkpoints/model_best.pth --data_root ./datasets/test
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AIGCDetector, AIGCDetectorLite
from data import AIGCDataset, FFppDataset, get_transforms
from utils.metrics import MetricsAccumulator, compute_metrics, compute_optimal_threshold
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict, checkpoint_path: str) -> torch.nn.Module:
    """构建并加载模型"""
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
            backbone=model_config.get('backbone', 'ViT-L/14'),
            num_classes=model_config.get('num_classes', 2),
            img_size=model_config.get('img_size', 224),
            num_adapter_layers=model_config.get('num_adapter_layers', 3),
            num_prototypes=model_config.get('num_prototypes', 4),
            use_hierarchical=model_config.get('use_hierarchical', True),
            freeze_backbone=model_config.get('freeze_backbone', True),
            dropout=model_config.get('dropout', 0.1)
        )

    # 加载检查点
    load_checkpoint(checkpoint_path, model, strict=False)

    return model


def build_test_loader(
    data_root: str,
    split: str = 'test',
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8
) -> DataLoader:
    """构建测试数据加载器"""
    transform = get_transforms(img_size, split='test')

    dataset = AIGCDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        balance_classes=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    save_predictions: bool = True,
    output_path: str = None
) -> dict:
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        save_predictions: 是否保存预测结果
        output_path: 输出路径
    Returns:
        results: 评估结果
    """
    model.eval()
    model = model.to(device)

    accumulator = MetricsAccumulator()
    all_predictions = []

    for batch in tqdm(test_loader, desc='Evaluating'):
        images = batch['image'].to(device)
        labels = batch['label']
        paths = batch.get('path', [''] * len(labels))
        methods = batch.get('class_name', ['unknown'] * len(labels))

        # 前向传播
        outputs = model(images)
        probs = torch.softmax(outputs['logits'], dim=1)

        # 更新累积器
        accumulator.update(
            probs[:, 1].cpu(),
            labels,
            paths=paths,
            methods=methods
        )

        # 保存预测
        if save_predictions:
            for i in range(len(labels)):
                all_predictions.append({
                    'path': paths[i],
                    'label': int(labels[i]),
                    'pred_prob': float(probs[i, 1].cpu()),
                    'pred_label': int(probs[i, 1].cpu() >= 0.5),
                    'method': methods[i] if isinstance(methods, list) else methods
                })

    # 计算整体指标
    overall_metrics = accumulator.compute()

    # 按方法分组指标
    per_method_metrics = accumulator.compute_per_method()

    # 计算最优阈值
    labels_np = np.array(accumulator.labels)
    preds_np = np.array(accumulator.predictions)
    optimal_threshold, _ = compute_optimal_threshold(labels_np, preds_np, 'youden')

    # 使用最优阈值重新计算指标
    optimal_metrics = compute_metrics(preds_np, labels_np, threshold=optimal_threshold)

    results = {
        'overall': overall_metrics,
        'per_method': per_method_metrics,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics
    }

    # 保存预测结果
    if save_predictions and output_path:
        predictions_path = output_path.replace('.json', '_predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)

    return results


def evaluate_cross_dataset(
    model: torch.nn.Module,
    config: dict,
    device: str = 'cuda'
) -> dict:
    """
    跨数据集评估

    Args:
        model: 模型
        config: 配置
        device: 设备
    Returns:
        results: 各数据集的评估结果
    """
    data_config = config.get('data', {})
    test_datasets = data_config.get('test_datasets', [])

    results = {}

    for dataset_name in test_datasets:
        print(f"\nEvaluating on {dataset_name}...")

        # 根据数据集名称构建路径
        data_root = os.path.join(
            config.get('paths', {}).get('data_root', './datasets'),
            dataset_name
        )

        if not os.path.exists(data_root):
            print(f"  Dataset not found: {data_root}")
            continue

        test_loader = build_test_loader(
            data_root=data_root,
            split='test',
            batch_size=config.get('training', {}).get('batch_size', 32),
            img_size=config.get('augmentation', {}).get('resize', 224),
            num_workers=config.get('data', {}).get('num_workers', 8)
        )

        dataset_results = evaluate(model, test_loader, device, save_predictions=False)
        results[dataset_name] = dataset_results['overall']

        print(f"  Accuracy: {dataset_results['overall']['accuracy']:.4f}")
        print(f"  AUC: {dataset_results['overall']['auc']:.4f}")
        print(f"  AP: {dataset_results['overall']['ap']:.4f}")

    # 计算平均值
    if results:
        avg_metrics = {}
        for metric in ['accuracy', 'auc', 'ap', 'eer']:
            values = [r[metric] for r in results.values()]
            avg_metrics[metric] = np.mean(values)
        results['average'] = avg_metrics

    return results


def print_results(results: dict, title: str = "Evaluation Results"):
    """打印评估结果"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)

    if 'overall' in results:
        print("\nOverall Metrics:")
        for name, value in results['overall'].items():
            print(f"  {name}: {value:.4f}")

    if 'per_method' in results:
        print("\nPer-Method Metrics:")
        for method, metrics in results['per_method'].items():
            print(f"\n  {method}:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")

    if 'optimal_threshold' in results:
        print(f"\nOptimal Threshold: {results['optimal_threshold']:.4f}")

    if 'optimal_metrics' in results:
        print("\nMetrics at Optimal Threshold:")
        for name, value in results['optimal_metrics'].items():
            print(f"  {name}: {value:.4f}")


def main(args):
    """主函数"""
    # 加载配置
    config = load_config(args.config) if args.config else {}

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = args.output_dir or config.get('paths', {}).get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger('evaluate', log_dir=os.path.join(output_dir, 'logs'))

    # 构建模型
    logger.info("Building model...")
    model = build_model(config, args.checkpoint)
    model = model.to(device)
    logger.info("Model loaded successfully")

    if args.cross_dataset:
        # 跨数据集评估
        results = evaluate_cross_dataset(model, config, device)
        output_path = os.path.join(output_dir, 'cross_dataset_results.json')
    else:
        # 单数据集评估
        data_root = args.data_root or config.get('paths', {}).get('data_root', './datasets')
        test_loader = build_test_loader(
            data_root=data_root,
            split=args.split,
            batch_size=args.batch_size,
            img_size=config.get('augmentation', {}).get('resize', 224),
            num_workers=args.num_workers
        )

        logger.info(f"Test samples: {len(test_loader.dataset)}")

        output_path = os.path.join(output_dir, 'evaluation_results.json')
        results = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            save_predictions=args.save_predictions,
            output_path=output_path
        )

    # 打印结果
    print_results(results)

    # 保存结果
    with open(output_path, 'w') as f:
        # 转换numpy类型为Python类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AIGC Detection Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to test data')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--cross_dataset', action='store_true',
                        help='Evaluate on multiple datasets')

    args = parser.parse_args()
    main(args)
