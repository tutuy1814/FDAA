#!/usr/bin/env python3
"""
鲁棒性消融实验: 对比 baseline / +FDAA / +MGFP / full 在7种扰动下的表现
证明 FDAA 模块对鲁棒性的核心贡献
"""

import sys, os, json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt,
    evaluate_model, DegradedDataset,
    get_multi_source_transforms,
    _create_ablation_model,
)
from data.multi_source_dataset import MultiSourceGenImageDataset
from models.detector import AIGCDetectorV2


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    config = PAPER_CONFIG.copy()
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']

    # 模型变体和对应的 checkpoint
    variants = {
        'Baseline': ('baseline', 'abl_baseline_best.pth'),
        '+FDAA': ('baseline+fdaa', 'abl_baseline_fdaa_best.pth'),
        '+MGFP': ('baseline+mgfp', 'abl_baseline_mgfp_best.pth'),
        'Full': ('full', 'fdaa_net_v2_6src_best.pth'),
    }

    # Full 的备选名
    full_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_6src_best.pth')
    if not os.path.exists(full_ckpt):
        full_ckpt_alt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
        if os.path.exists(full_ckpt_alt):
            variants['Full'] = ('full', 'fdaa_net_v2_best.pth')

    # 加载模型
    print("=" * 60)
    print("Loading Ablation Models for Robustness Test")
    print("=" * 60)

    loaded_models = {}
    for display_name, (variant, ckpt_name) in variants.items():
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] {display_name}: {ckpt_name} not found")
            continue
        try:
            m = _create_ablation_model(config, variant)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[display_name] = m
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  [Loaded] {display_name} ({params/1e6:.2f}M trainable)")
        except Exception as e:
            print(f"  [Error] {display_name}: {e}")

    if not loaded_models:
        print("[Error] No models loaded!")
        return

    # 创建基础验证数据集
    base_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        transform=None,
        max_samples_per_source=2000,
        balance_classes=True,
    )

    # 鲁棒性测试
    print(f"\n{'='*60}")
    print("Robustness Ablation Evaluation")
    print(f"{'='*60}")

    robustness_tests = config['robustness_tests']
    results = {}

    for test_name, test_cfg in robustness_tests.items():
        deg_type = test_cfg['type']
        deg_param = test_cfg.get('quality') or test_cfg.get('radius') or test_cfg.get('std')

        print(f"\n--- {test_name} ({deg_type}={deg_param}) ---")

        degraded = DegradedDataset(base_dataset, deg_type, deg_param, config['img_size'])
        loader = DataLoader(degraded, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

        results[test_name] = {'type': deg_type, 'param': deg_param}

        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {test_name}')
                results[test_name][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[test_name][model_name] = {'error': str(e)}

        save_json(results, os.path.join(results_dir, 'robustness_ablation_results.json'))
        torch.cuda.empty_cache()

    # 生成表格
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    models = list(loaded_models.keys())
    tests = list(robustness_tests.keys())

    for model_name in models:
        aucs = []
        for t in tests:
            auc = results.get(t, {}).get(model_name, {}).get('auc', 0) * 100
            aucs.append(auc)
        avg = np.mean(aucs)
        print(f"  {model_name:<12}: avg={avg:.2f}%  " +
              "  ".join([f"{t}={a:.1f}" for t, a in zip(tests, aucs)]))

    del loaded_models
    torch.cuda.empty_cache()
    print("\n[Done]")


if __name__ == '__main__':
    main()
