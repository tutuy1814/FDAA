#!/usr/bin/env python3
"""
仅 Ojha (UniversalFakeDetect, CVPR 2023) 跨数据集评估
Ojha 数据已缓存，无需再下载
包含: DALL-E, GLIDE, Guided Diffusion (ADM), LDM (Latent Diffusion Model)
"""

import sys, os, json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
import pyarrow.ipc as ipc
import glob

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt,
    evaluate_model, create_sota_model, get_multi_source_transforms,
)


class ArrowImageDataset(Dataset):
    def __init__(self, arrow_files, generator_filter=None, transform=None, max_per_class=1000):
        self.transform = transform
        self.samples = []

        for f in arrow_files:
            reader = ipc.open_stream(f)
            table = reader.read_all()
            for i in range(table.num_rows):
                path = table['image_path'][i].as_py()
                gen = path.split('/')[0]
                if generator_filter and gen != generator_filter:
                    continue
                label = 1 if '1_fake' in path else 0
                img_data = table['image'][i].as_py()
                self.samples.append((img_data, label))

        real = [(d, l) for d, l in self.samples if l == 0][:max_per_class]
        fake = [(d, l) for d, l in self.samples if l == 1][:max_per_class]
        self.samples = real + fake
        print(f"    {len(real)} real + {len(fake)} fake = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_data, label = self.samples[idx]
        try:
            img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        except:
            try:
                img = Image.open(BytesIO(img_data)).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label, 'path': ''}


def main():
    device = 'cuda:0'
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    config = PAPER_CONFIG.copy()
    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # 加载模型
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)

    from models.detector import AIGCDetectorV2
    loaded_models = {}

    # 我们的模型
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if not os.path.exists(ours_ckpt):
        ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_6src_best.pth')
    m = AIGCDetectorV2(backbone_name=config['backbone'], num_classes=2,
                       img_size=config['img_size'], embed_dim=config['embed_dim'],
                       use_hierarchical=config['use_hierarchical'], dropout=config['dropout'])
    ckpt = torch.load(ours_ckpt, map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    loaded_models['Ours_V2'] = m.to(device).eval()
    print("  [Loaded] Ours_V2")

    # SOTA
    for method in config['sota_methods']:
        ckpt_path = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt_path) and not os.path.islink(ckpt_path):
            try:
                sm = create_sota_model(method, num_classes=2)
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                sm.load_state_dict(ckpt['model_state_dict'])
                loaded_models[method] = sm.to(device).eval()
                print(f"  [Loaded] {method}")
            except Exception as e:
                print(f"  [Error] {method}: {e}")

    # 找 Ojha arrow 文件 (已缓存)
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/Ojha/*.arrow")
    ojha_files = sorted(glob.glob(cache_pattern))
    print(f"\n  Ojha arrow files: {len(ojha_files)}")

    if not ojha_files:
        print("[Error] No cached Ojha files found!")
        return

    # 评估
    print("\n" + "=" * 60)
    print("Ojha/UniversalFakeDetect Cross-Dataset Evaluation")
    print("(DALL-E, GLIDE, Guided Diffusion, LDM)")
    print("=" * 60)

    # 选择代表性生成器
    generators = {
        'dalle': 'DALL-E',
        'glide_100_27': 'GLIDE',
        'guided': 'Guided Diffusion',
        'ldm_200': 'LDM-200',
        'ldm_200_cfg': 'LDM-200-CFG',
    }

    results = {}
    for gen_key, gen_name in generators.items():
        print(f"\n--- {gen_name} ---")
        dataset = ArrowImageDataset(ojha_files, generator_filter=gen_key, transform=transform)
        if len(dataset) < 10:
            continue
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        results[gen_name] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_name}')
                results[gen_name][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[gen_name][model_name] = {'error': str(e)}

        # 及时保存
        save_json(results, os.path.join(results_dir, 'crossdataset_ojha_results.json'))

    # 生成表格
    print("\n" + "=" * 60)
    print("Results Table")
    print("=" * 60)

    all_models = set()
    for gen_results in results.values():
        for m, v in gen_results.items():
            if isinstance(v, dict) and 'auc' in v:
                all_models.add(m)
    models = sorted(all_models, key=lambda x: (0 if 'Ours' in x else 1, x))
    gens = list(results.keys())

    # 找每列最好
    best = {}
    for g in gens:
        best[g] = max(models, key=lambda m: results.get(g, {}).get(m, {}).get('auc', 0))

    header = f"| {'Method':<15} |" + "".join([f" {g:<18} |" for g in gens]) + " Avg |"
    print(header)
    print("|" + "-" * 17 + "|" + ("|".join(["-" * 20 for _ in gens])) + "|-----|")

    for model in models:
        aucs = []
        row = f"| {'**Ours**' if 'Ours' in model else model:<15} |"
        for g in gens:
            auc = results.get(g, {}).get(model, {}).get('auc', 0) * 100
            aucs.append(auc)
            mark = "**" if best[g] == model else ""
            row += f" {mark}{auc:.1f}{mark:<{18-len(mark)*2-len(f'{auc:.1f}')}} |"
        avg = np.mean(aucs)
        print(f"  {model:<15} |" + "".join([f" {a:.1f}{'*' if best[g]==model else '':<10}" for a, g in zip(aucs, gens)]) + f" | {avg:.1f}")

    # 保存 markdown
    lines = ["# UniversalFakeDetect (Ojha et al., CVPR 2023) Cross-Dataset Evaluation\n\n"]
    lines.append("All models trained on GenImage 6-source. Zero-shot on UniversalFakeDetect benchmark.\n\n")
    h = "| Method |" + "".join([f" {g} |" for g in gens]) + " **Avg** |\n"
    s = "|:---|" + "".join([":---:|" for _ in gens]) + ":---:|\n"
    lines.append(h)
    lines.append(s)

    for model in models:
        name = "**FDAA-Net (Ours)**" if 'Ours' in model else model
        row = f"| {name} |"
        aucs = []
        for g in gens:
            auc = results.get(g, {}).get(model, {}).get('auc', 0) * 100
            aucs.append(auc)
            b = "**" if best[g] == model else ""
            row += f" {b}{auc:.1f}{b} |"
        avg = np.mean(aucs)
        row += f" {'**' if 'Ours' in model else ''}{avg:.1f}{'**' if 'Ours' in model else ''} |"
        lines.append(row + "\n")

    table_path = os.path.join(results_dir, 'crossdataset_ojha_table.md')
    with open(table_path, 'w') as f:
        f.writelines(lines)
    print(f"\n[Saved] {table_path}")

    del loaded_models; torch.cuda.empty_cache()
    print("\n[Done]")


if __name__ == '__main__':
    main()
