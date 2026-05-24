#!/usr/bin/env python3
"""
消融鲁棒性 + 跨数据集补充实验

对4个模块消融变体 (baseline / +FDAA / +MGFP / Full) 和3个损失消融变体
分别在以下维度进行评估:

1. 鲁棒性: 7种扰动 (JPEG-70/50/30, Blur-1.0/2.0, Noise-0.02/0.05)
2. 跨数据集泛化: UniversalFakeDetect (Ojha), ForenSynths, Synthbuster

用于证明 FDAA 模块在扰动/泛化维度的不可替代性。

使用方法:
    python experiments/run_ablation_robustness.py --mode all
    python experiments/run_ablation_robustness.py --mode robustness
    python experiments/run_ablation_robustness.py --mode crossdataset
    python experiments/run_ablation_robustness.py --mode report
"""

import sys
import os
import json
import copy
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt, evaluate_model,
    get_multi_source_transforms, DegradedDataset,
)
from data.multi_source_dataset import MultiSourceGenImageDataset
from models.detector import AIGCDetectorV2


# =============================================================================
# 消融模型创建 (复用 run_paper_experiments 的逻辑)
# =============================================================================

def _create_ablation_model(config, variant):
    """创建消融变体模型，与主脚本中一致"""
    embed_dim = config['embed_dim']
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

    class ZeroFreq(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, x):
            return torch.zeros(x.shape[0], self.dim, device=x.device)

    if variant == 'baseline':
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

    return model


# =============================================================================
# 加载消融模型 checkpoint
# =============================================================================

def load_ablation_models(config, device):
    """加载所有消融变体的 checkpoint"""
    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')

    # 消融变体与 checkpoint 映射
    variant_ckpt_map = OrderedDict([
        ('baseline',       'abl_baseline_best.pth'),
        ('baseline+fdaa',  'abl_baseline_fdaa_best.pth'),
        ('baseline+mgfp',  'abl_baseline_mgfp_best.pth'),
        ('full',           'fdaa_net_v2_best.pth'),  # full 用主模型 checkpoint
    ])

    # 损失消融变体
    loss_variant_ckpt_map = OrderedDict([
        ('focal_only',        'abl_loss_focal_only_best.pth'),
        ('focal+contrastive', 'abl_loss_focal_contrastive_best.pth'),
        ('focal+aux',         'abl_loss_focal_aux_best.pth'),
        ('focal+contr+aux',   'fdaa_net_v2_best.pth'),  # 复用 full
    ])

    loaded = OrderedDict()

    # 加载模块消融变体
    for variant, ckpt_name in variant_ckpt_map.items():
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] {variant}: {ckpt_name} not found")
            continue

        try:
            model = _create_ablation_model(config, variant)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            loaded[f"abl_{variant}"] = model
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [Loaded] {variant} ({trainable/1e6:.2f}M params)")
        except Exception as e:
            print(f"  [Error] {variant}: {e}")
            import traceback; traceback.print_exc()

    # 加载损失消融变体
    for lv_name, ckpt_name in loss_variant_ckpt_map.items():
        key = f"loss_{lv_name}"
        if key == "loss_focal+contr+aux" and "abl_full" in loaded:
            # 复用 full 模型实例
            loaded[key] = loaded["abl_full"]
            print(f"  [Reuse] {lv_name} -> full model")
            continue

        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] loss_{lv_name}: {ckpt_name} not found")
            continue

        try:
            # 损失消融都是 full 架构，只是训练时损失不同
            use_hier = lv_name in ('focal+aux', 'focal+contr+aux')
            model = AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=use_hier,
                dropout=config['dropout'],
                freeze_backbone=True,
            )
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            loaded[key] = model
            print(f"  [Loaded] loss_{lv_name}")
        except Exception as e:
            print(f"  [Error] loss_{lv_name}: {e}")

    return loaded


# =============================================================================
# 实验1: 消融鲁棒性
# =============================================================================

def exp_ablation_robustness(config, device, loaded_models):
    """对消融变体进行7种鲁棒性扰动测试"""
    output_dir = config['output_dir']
    results = {}

    # 创建基础验证数据集 (不带 transform)
    base_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        transform=None,
        max_samples_per_source=2000,
        balance_classes=True,
    )
    print(f"\n  Base dataset: {len(base_dataset)} samples")

    print(f"\n{'='*70}")
    print("Ablation Robustness Evaluation")
    print(f"{'='*70}")

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
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%  Acc={metrics['accuracy']*100:.2f}%")
            except Exception as e:
                results[test_name][model_name] = {'error': str(e)}
                print(f"  [Error] {model_name}: {e}")

    save_json(results, os.path.join(output_dir, 'results', 'ablation_robustness_results.json'))
    return results


# =============================================================================
# 实验2: 消融跨数据集泛化
# =============================================================================

def exp_ablation_crossdataset(config, device, loaded_models):
    """对消融变体进行跨数据集泛化测试 (Ojha/ForenSynths/Synthbuster)"""
    output_dir = config['output_dir']
    results = {}

    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # 2.1 尝试加载 Ojha (UniversalFakeDetect)
    print(f"\n{'='*70}")
    print("Ablation Cross-Dataset: Ojha (UniversalFakeDetect)")
    print(f"{'='*70}")

    try:
        from run_crossdataset_eval import ArrowImageDataset, download_arrow_files

        ojha_generators = ['dalle', 'glide_100_27', 'guided', 'ldm_200', 'ldm_200_cfg']
        ojha_display = {
            'dalle': 'DALL-E', 'glide_100_27': 'GLIDE', 'guided': 'Guided Diffusion',
            'ldm_200': 'LDM-200', 'ldm_200_cfg': 'LDM-200-CFG',
        }

        print("  Downloading Ojha arrow files...")
        ojha_files = download_arrow_files("Ojha", 3)

        ojha_results = {}
        for gen in ojha_generators:
            gen_disp = ojha_display.get(gen, gen)
            print(f"\n--- {gen_disp} ---")
            dataset = ArrowImageDataset(ojha_files, generator_filter=gen, transform=transform)
            if len(dataset) < 10:
                print(f"  [Skip] Too few samples")
                continue

            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            ojha_results[gen_disp] = {}

            for model_name, m in loaded_models.items():
                try:
                    metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_disp}')
                    ojha_results[gen_disp][model_name] = metrics
                    print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
                except Exception as e:
                    ojha_results[gen_disp][model_name] = {'error': str(e)}

            torch.cuda.empty_cache()

        results['ojha'] = ojha_results
        save_json(results, os.path.join(output_dir, 'results', 'ablation_crossdataset_results.json'))

    except Exception as e:
        print(f"  [Error] Ojha evaluation failed: {e}")
        import traceback; traceback.print_exc()

    # 2.2 ForenSynths
    print(f"\n{'='*70}")
    print("Ablation Cross-Dataset: ForenSynths")
    print(f"{'='*70}")

    try:
        from run_crossdataset_eval import ArrowImageDataset, download_arrow_files
        import pyarrow.ipc as ipc
        from collections import Counter

        print("  Downloading ForenSynths arrow files...")
        forensynths_files = download_arrow_files("ForenSynths", 192)

        # 扫描生成器列表
        gen_counter = Counter()
        for f in forensynths_files[:5]:
            reader = ipc.open_stream(f)
            table = reader.read_all()
            for path in table['image_path'].to_pylist():
                gen = path.split('/')[0]
                gen_counter[gen] += 1
        forensynths_gens = sorted(gen_counter.keys())
        print(f"  Detected generators: {forensynths_gens}")

        forensynths_results = {}
        for gen in forensynths_gens:
            gen_disp = gen.replace('_', ' ').title()
            print(f"\n--- {gen_disp} ---")
            dataset = ArrowImageDataset(forensynths_files, generator_filter=gen, transform=transform)
            if len(dataset) < 30:
                print(f"  [Skip] Too few samples ({len(dataset)})")
                continue

            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            forensynths_results[gen_disp] = {}

            for model_name, m in loaded_models.items():
                try:
                    metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_disp}')
                    forensynths_results[gen_disp][model_name] = metrics
                    print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
                except Exception as e:
                    forensynths_results[gen_disp][model_name] = {'error': str(e)}

            torch.cuda.empty_cache()

        results['forensynths'] = forensynths_results
        save_json(results, os.path.join(output_dir, 'results', 'ablation_crossdataset_results.json'))

    except Exception as e:
        print(f"  [Error] ForenSynths evaluation failed: {e}")
        import traceback; traceback.print_exc()

    # 2.3 Synthbuster (如果有本地数据)
    print(f"\n{'='*70}")
    print("Ablation Cross-Dataset: Synthbuster")
    print(f"{'='*70}")

    try:
        from run_crossdataset_eval import ArrowImageDataset, download_arrow_files

        print("  Downloading Synthbuster arrow files...")
        synthbuster_files = download_arrow_files("Synthbuster", 9)

        if synthbuster_files:
            import pyarrow.ipc as ipc
            from collections import Counter

            gen_counter = Counter()
            for f in synthbuster_files[:3]:
                reader = ipc.open_stream(f)
                table = reader.read_all()
                for path in table['image_path'].to_pylist():
                    gen = path.split('/')[0]
                    gen_counter[gen] += 1

            synthbuster_gens = sorted(gen_counter.keys())
            print(f"  Detected generators: {synthbuster_gens}")

            synthbuster_results = {}
            for gen in synthbuster_gens:
                gen_disp = gen.replace('_', ' ').title()
                print(f"\n--- {gen_disp} ---")
                dataset = ArrowImageDataset(synthbuster_files, generator_filter=gen, transform=transform)
                if len(dataset) < 10:
                    print(f"  [Skip] Too few samples")
                    continue

                loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
                synthbuster_results[gen_disp] = {}

                for model_name, m in loaded_models.items():
                    try:
                        metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_disp}')
                        synthbuster_results[gen_disp][model_name] = metrics
                        print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
                    except Exception as e:
                        synthbuster_results[gen_disp][model_name] = {'error': str(e)}

                torch.cuda.empty_cache()

            results['synthbuster'] = synthbuster_results
        else:
            print("  [Skip] No Synthbuster files available")

    except Exception as e:
        print(f"  [Error] Synthbuster evaluation failed: {e}")
        import traceback; traceback.print_exc()

    # 2.4 域内额外数据集 (DiffusionForensics, CIFAKE)
    print(f"\n{'='*70}")
    print("Ablation Cross-Dataset: DiffusionForensics & CIFAKE")
    print(f"{'='*70}")

    from run_paper_experiments import create_eval_loader

    for ds_name, ds_root in config.get('eval_datasets', {}).items():
        if ds_name == 'ntire2026':
            continue  # NTIRE 可跳过

        print(f"\n--- {ds_name} ---")
        loader = create_eval_loader(ds_root, split='test', batch_size=64,
                                     img_size=config['img_size'], num_workers=4)
        if loader is None or len(loader.dataset) == 0:
            print(f"  [Skip] Empty dataset")
            continue

        results[ds_name] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {ds_name}')
                results[ds_name][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                results[ds_name][model_name] = {'error': str(e)}

    save_json(results, os.path.join(output_dir, 'results', 'ablation_crossdataset_results.json'))
    return results


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(config):
    """生成消融鲁棒性与跨数据集的论文表格"""
    output_dir = config['output_dir']
    results_dir = os.path.join(output_dir, 'results')
    report_path = os.path.join(output_dir, 'reports', 'ablation_robustness_report.md')

    lines = []
    lines.append("# 消融实验补充: 鲁棒性 & 跨数据集泛化\n\n")

    # ---- 鲁棒性表格 ----
    rob_path = os.path.join(results_dir, 'ablation_robustness_results.json')
    if os.path.exists(rob_path):
        with open(rob_path) as f:
            rob = json.load(f)

        lines.append("## Table A1: 模块消融 — 鲁棒性评估 (AUC %)\n\n")

        # 提取模块消融模型
        module_variants = ['abl_baseline', 'abl_baseline+fdaa', 'abl_baseline+mgfp', 'abl_full']
        variant_display = {
            'abl_baseline': 'Baseline',
            'abl_baseline+fdaa': '+ FDAA',
            'abl_baseline+mgfp': '+ MGFP',
            'abl_full': 'Full (Ours)',
        }

        tests = [k for k in rob.keys() if k.startswith(('jpeg', 'blur', 'noise'))]
        tests.sort(key=lambda x: (x.split('_')[0], float(x.split('_')[1])))

        # 表头
        header = "| Configuration |"
        sep = "|:---|"
        for t in tests:
            header += f" {t} |"
            sep += ":---:|"
        header += " **Avg** | **ΔAUC** |"
        sep += ":---:|:---:|"
        lines.append(header + "\n")
        lines.append(sep + "\n")

        # 表内容
        avgs = {}
        for var in module_variants:
            row = f"| {variant_display.get(var, var)} |"
            aucs = []
            for t in tests:
                auc = rob.get(t, {}).get(var, {}).get('auc', None)
                if auc is not None:
                    auc_pct = auc * 100
                    aucs.append(auc_pct)
                    row += f" {auc_pct:.1f} |"
                else:
                    row += " — |"
            avg = np.mean(aucs) if aucs else 0
            avgs[var] = avg
            row += f" **{avg:.1f}** |"

            # ΔAUC vs baseline
            if var == 'abl_baseline':
                row += " — |"
            else:
                delta = avg - avgs.get('abl_baseline', avg)
                row += f" +{delta:.1f} |" if delta >= 0 else f" {delta:.1f} |"

            lines.append(row + "\n")

        # 关键分析
        lines.append("\n**关键发现:**\n")
        if 'abl_baseline+fdaa' in avgs and 'abl_baseline' in avgs:
            fdaa_gain = avgs['abl_baseline+fdaa'] - avgs['abl_baseline']
            lines.append(f"- FDAA模块在鲁棒性维度贡献 **+{fdaa_gain:.1f}pp** 平均AUC提升\n")
        if 'abl_full' in avgs and 'abl_baseline+mgfp' in avgs:
            fdaa_in_full = avgs['abl_full'] - avgs['abl_baseline+mgfp']
            lines.append(f"- 在已有MGFP基础上，FDAA额外贡献 **+{fdaa_in_full:.1f}pp** 鲁棒性提升\n")
        if 'abl_full' in avgs and 'abl_baseline' in avgs:
            total_gain = avgs['abl_full'] - avgs['abl_baseline']
            lines.append(f"- 完整模型相比Baseline总提升 **+{total_gain:.1f}pp**\n")
        lines.append("\n")

        # 损失消融鲁棒性表格
        loss_variants = ['loss_focal_only', 'loss_focal+contrastive', 'loss_focal+aux', 'loss_focal+contr+aux']
        loss_display = {
            'loss_focal_only': 'Focal Only',
            'loss_focal+contrastive': 'Focal + Contr',
            'loss_focal+aux': 'Focal + Aux',
            'loss_focal+contr+aux': 'Focal + Contr + Aux',
        }

        # 检查是否有损失消融数据
        has_loss = any(
            rob.get(tests[0], {}).get(lv) for lv in loss_variants
        ) if tests else False

        if has_loss:
            lines.append("## Table A2: 损失函数消融 — 鲁棒性评估 (AUC %)\n\n")
            header = "| Loss Configuration |"
            sep = "|:---|"
            for t in tests:
                header += f" {t} |"
                sep += ":---:|"
            header += " **Avg** |"
            sep += ":---:|"
            lines.append(header + "\n")
            lines.append(sep + "\n")

            for lv in loss_variants:
                row = f"| {loss_display.get(lv, lv)} |"
                aucs = []
                for t in tests:
                    auc = rob.get(t, {}).get(lv, {}).get('auc', None)
                    if auc is not None:
                        auc_pct = auc * 100
                        aucs.append(auc_pct)
                        row += f" {auc_pct:.1f} |"
                    else:
                        row += " — |"
                avg = np.mean(aucs) if aucs else 0
                row += f" **{avg:.1f}** |"
                lines.append(row + "\n")
            lines.append("\n")

    # ---- 跨数据集表格 ----
    cross_path = os.path.join(results_dir, 'ablation_crossdataset_results.json')
    if os.path.exists(cross_path):
        with open(cross_path) as f:
            cross = json.load(f)

        for benchmark_name, bench_data in cross.items():
            if not isinstance(bench_data, dict):
                continue

            lines.append(f"## Table A3: 模块消融 — {benchmark_name} 跨数据集泛化 (AUC %)\n\n")

            generators = [g for g in bench_data.keys() if isinstance(bench_data[g], dict)]
            if not generators:
                continue

            header = "| Configuration |"
            sep = "|:---|"
            for g in generators:
                header += f" {g} |"
                sep += ":---:|"
            header += " **Avg** |"
            sep += ":---:|"
            lines.append(header + "\n")
            lines.append(sep + "\n")

            for var in module_variants:
                var_disp = variant_display.get(var, var)
                row = f"| {var_disp} |"
                aucs = []
                for g in generators:
                    auc = bench_data.get(g, {}).get(var, {}).get('auc', None)
                    if auc is not None:
                        auc_pct = auc * 100
                        aucs.append(auc_pct)
                        row += f" {auc_pct:.1f} |"
                    else:
                        row += " — |"
                avg = np.mean(aucs) if aucs else 0
                row += f" **{avg:.1f}** |"
                lines.append(row + "\n")
            lines.append("\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\n[Save] Report: {report_path}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融鲁棒性 + 跨数据集补充实验')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['robustness', 'crossdataset', 'report', 'all'])
    parser.add_argument('--force', action='store_true', help='强制重新运行')
    parser.add_argument('--module-only', action='store_true',
                        help='仅评估模块消融变体（不含损失消融）')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # 配置
    config = copy.deepcopy(PAPER_CONFIG)
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    config['output_dir'] = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    os.makedirs(os.path.join(config['output_dir'], 'results'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'reports'), exist_ok=True)

    if args.mode == 'report':
        generate_report(config)
        return

    # 加载消融模型
    print("\n" + "=" * 70)
    print("Loading Ablation Models")
    print("=" * 70)
    loaded_models = load_ablation_models(config, device)

    if not loaded_models:
        print("[Error] No ablation models loaded!")
        return

    if args.module_only:
        # 仅保留模块消融变体
        loaded_models = OrderedDict(
            (k, v) for k, v in loaded_models.items()
            if k.startswith('abl_')
        )

    print(f"\n  Total models to evaluate: {len(loaded_models)}")
    for name in loaded_models:
        print(f"    - {name}")

    # 设置种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 执行实验
    if args.mode in ('all', 'robustness'):
        result_file = os.path.join(config['output_dir'], 'results', 'ablation_robustness_results.json')
        if args.force or not os.path.exists(result_file):
            exp_ablation_robustness(config, device, loaded_models)
        else:
            print(f"\n[Skip] Robustness: {result_file} exists (use --force)")

    if args.mode in ('all', 'crossdataset'):
        result_file = os.path.join(config['output_dir'], 'results', 'ablation_crossdataset_results.json')
        if args.force or not os.path.exists(result_file):
            exp_ablation_crossdataset(config, device, loaded_models)
        else:
            print(f"\n[Skip] Cross-dataset: {result_file} exists (use --force)")

    # 释放模型
    del loaded_models
    torch.cuda.empty_cache()

    # 生成报告
    generate_report(config)

    print("\n" + "=" * 70)
    print("[Done] Ablation robustness & cross-dataset evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
