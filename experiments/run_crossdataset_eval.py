#!/usr/bin/env python3
"""
跨数据集评估: UniversalFakeDetect (Ojha, CVPR 2023) + ForenSynths (CNNDetection, CVPR 2020)
从 HuggingFace 数据集 nebula/DF-arrow 加载
"""

import sys
import os
import json
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

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt,
    evaluate_model, create_sota_model, get_multi_source_transforms,
)


class ArrowImageDataset(Dataset):
    """从 HF arrow 文件加载图像数据"""

    def __init__(self, arrow_files, generator_filter=None, transform=None, max_per_class=1000):
        self.transform = transform
        self.samples = []  # (image_bytes, label)

        for f in arrow_files:
            reader = ipc.open_stream(f)
            table = reader.read_all()

            for i in range(table.num_rows):
                path = table['image_path'][i].as_py()
                gen = path.split('/')[0]

                if generator_filter and gen != generator_filter:
                    continue

                label = 1 if '1_fake' in path else 0
                img_data = table['image'][i].as_py()  # bytes
                self.samples.append((img_data, label))

        # 平衡采样
        real = [(d, l) for d, l in self.samples if l == 0][:max_per_class]
        fake = [(d, l) for d, l in self.samples if l == 1][:max_per_class]
        self.samples = real + fake
        print(f"    Loaded: {len(real)} real + {len(fake)} fake = {len(self.samples)}")

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


def download_arrow_files(dataset_name, num_files):
    """从 HF 镜像下载 arrow 文件"""
    from huggingface_hub import hf_hub_download

    files = []
    for i in range(num_files):
        fname = f"{dataset_name}/data-{i:05d}-of-{num_files:05d}.arrow"
        try:
            f = hf_hub_download("nebula/DF-arrow", fname, repo_type="dataset")
            files.append(f)
        except Exception as e:
            print(f"  [Warning] Failed to download {fname}: {e}")
    return files


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    config = PAPER_CONFIG.copy()
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # ====== Step 1: 加载模型 ======
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)

    from models.detector import AIGCDetectorV2

    model_configs = {}
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if not os.path.exists(ours_ckpt):
        ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_6src_best.pth')

    if os.path.exists(ours_ckpt):
        model_configs['Ours_V2'] = {
            'checkpoint': ours_ckpt,
            'create_fn': lambda: AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    for method in config['sota_methods']:
        ckpt = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt) and not os.path.islink(ckpt):
            m = method
            model_configs[method] = {
                'checkpoint': ckpt,
                'create_fn': lambda m=m: create_sota_model(m, num_classes=2)
            }

    loaded_models = {}
    for model_name, mcfg in model_configs.items():
        try:
            m = mcfg['create_fn']()
            ckpt = torch.load(mcfg['checkpoint'], map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[model_name] = m
            print(f"  [Loaded] {model_name}")
        except Exception as e:
            print(f"  [Error] {model_name}: {e}")

    print(f"\n  Total: {len(loaded_models)} models")

    # ====== Step 2: 下载数据 ======
    print("\n" + "=" * 60)
    print("Downloading Arrow Files from HF Mirror")
    print("=" * 60)

    print("\n--- Ojha (UniversalFakeDetect, CVPR 2023) ---")
    ojha_files = download_arrow_files("Ojha", 3)
    print(f"  Downloaded {len(ojha_files)} files")

    print("\n--- ForenSynths (CNNDetection, CVPR 2020) ---")
    forensynths_files = download_arrow_files("ForenSynths", 192)
    print(f"  Downloaded {len(forensynths_files)} files")

    # ====== Step 3: Ojha 逐生成器评估 ======
    print("\n" + "=" * 60)
    print("Ojha Per-Generator Evaluation")
    print("=" * 60)

    ojha_generators = ['dalle', 'glide_100_27', 'guided', 'ldm_200', 'ldm_200_cfg']
    # 选择代表性的生成器（去重复配置）

    results = {}

    for gen in ojha_generators:
        gen_display = {
            'dalle': 'DALL-E',
            'glide_100_27': 'GLIDE',
            'guided': 'Guided Diffusion',
            'ldm_200': 'LDM-200',
            'ldm_200_cfg': 'LDM-200-CFG',
        }.get(gen, gen)

        print(f"\n--- {gen_display} ({gen}) ---")
        dataset = ArrowImageDataset(ojha_files, generator_filter=gen, transform=transform)

        if len(dataset) < 10:
            print(f"  [Skip] Too few samples")
            continue

        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        results[gen_display] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_display}')
                results[gen_display][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[gen_display][model_name] = {'error': str(e)}

        save_json(results, os.path.join(results_dir, 'crossdataset_ojha_results.json'))
        torch.cuda.empty_cache()

    # ====== Step 4: ForenSynths 评估 ======
    print("\n" + "=" * 60)
    print("ForenSynths Per-Generator Evaluation")
    print("=" * 60)

    # 先扫描有哪些生成器
    gen_counter = Counter()
    for f in forensynths_files[:5]:  # 只扫前5个文件确定结构
        reader = ipc.open_stream(f)
        table = reader.read_all()
        for path in table['image_path'].to_pylist():
            gen = path.split('/')[0]
            gen_counter[gen] += 1

    print(f"  Detected generators: {list(gen_counter.keys())}")

    forensynths_gens = sorted(gen_counter.keys())
    forensynths_results = {}

    for gen in forensynths_gens:
        gen_display = gen.replace('_', ' ').title()
        print(f"\n--- {gen_display} ---")
        dataset = ArrowImageDataset(forensynths_files, generator_filter=gen, transform=transform)

        if len(dataset) < 10:
            print(f"  [Skip] Too few samples")
            continue

        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        forensynths_results[gen_display] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_display}')
                forensynths_results[gen_display][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                forensynths_results[gen_display][model_name] = {'error': str(e)}

        save_json(forensynths_results, os.path.join(results_dir, 'crossdataset_forensynths_results.json'))
        torch.cuda.empty_cache()

    # ====== Step 5: 生成论文表格 ======
    print("\n" + "=" * 60)
    print("Generating Paper Tables")
    print("=" * 60)

    all_results = {**results, **forensynths_results}
    save_json(all_results, os.path.join(results_dir, 'crossdataset_all_results.json'))

    # 生成 Markdown 表格
    generate_table(all_results, os.path.join(results_dir, 'crossdataset_table.md'))

    # 释放模型
    del loaded_models
    torch.cuda.empty_cache()

    print("\n[Done] Cross-dataset evaluation complete!")


def generate_table(results, output_path):
    """生成 Markdown 格式的论文表格"""
    all_models = set()
    for gen_results in results.values():
        for m, v in gen_results.items():
            if isinstance(v, dict) and 'auc' in v:
                all_models.add(m)

    models = sorted(all_models, key=lambda x: (0 if 'Ours' in x else 1, x))
    generators = list(results.keys())

    lines = []
    lines.append("# Cross-Dataset Evaluation (AUC %)\n\n")
    lines.append("Models trained on GenImage 6-source. Zero-shot evaluation on external benchmarks.\n\n")

    # 表头
    header = "| Method |"
    sep = "|:---|"
    for gen in generators:
        header += f" {gen} |"
        sep += ":---:|"
    header += " **Avg** |"
    sep += ":---:|"
    lines.append(header + "\n")
    lines.append(sep + "\n")

    # 找每列最高
    best_per_gen = {}
    for gen in generators:
        best_auc = 0
        for model in models:
            auc = results.get(gen, {}).get(model, {}).get('auc', 0)
            if auc > best_auc:
                best_auc = auc
                best_per_gen[gen] = model

    for model in models:
        display_name = "**FDAA-Net (Ours)**" if 'Ours' in model else model
        row = f"| {display_name} |"
        aucs = []
        for gen in generators:
            auc = results.get(gen, {}).get(model, {}).get('auc', 0) * 100
            is_best = best_per_gen.get(gen) == model
            if is_best:
                row += f" **{auc:.1f}** |"
            else:
                row += f" {auc:.1f} |"
            aucs.append(auc)

        avg = np.mean(aucs) if aucs else 0
        row += f" **{avg:.1f}** |" if 'Ours' in model else f" {avg:.1f} |"
        lines.append(row + "\n")

    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"[Table] {output_path}")
    print("\n" + "".join(lines))


if __name__ == '__main__':
    main()
