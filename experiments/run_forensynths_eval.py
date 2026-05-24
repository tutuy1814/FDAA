#!/usr/bin/env python3
"""
ForenSynths (CNNDetection, CVPR 2020) 跨数据集评估
从 HF 缓存加载 + 下载更多文件
"""

import sys, os, json, glob
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


class ForenSynthsDataset(Dataset):
    """从 HF arrow 文件加载 ForenSynths 测试集"""

    def __init__(self, arrow_files, generator_filter=None, transform=None, max_per_class=500):
        self.transform = transform
        self.samples = []

        for f in arrow_files:
            reader = ipc.open_stream(f)
            table = reader.read_all()

            for i in range(table.num_rows):
                path = table['image_path'][i].as_py()
                # ForenSynths structure: test/generator/0_real or 1_fake/xxx.png
                if not path.startswith('test/'):
                    continue

                parts = path.split('/')
                if len(parts) < 4:
                    continue

                gen = parts[1]
                if generator_filter and gen != generator_filter:
                    continue

                label = 1 if '1_fake' in path else 0
                img_data = table['image'][i].as_py()
                self.samples.append((img_data, label))

        # Balance
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


def download_more_files(current_count, target_count=20):
    """下载更多 ForenSynths arrow 文件"""
    from huggingface_hub import hf_hub_download

    files = []
    for i in range(current_count, target_count):
        fname = f"ForenSynths/data-{i:05d}-of-00192.arrow"
        try:
            f = hf_hub_download("nebula/DF-arrow", fname, repo_type="dataset")
            files.append(f)
            print(f"  Downloaded {fname}")
        except Exception as e:
            print(f"  [Warning] {fname}: {e}")
            break
    return files


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    config = PAPER_CONFIG.copy()
    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # Load models
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)

    from models.detector import AIGCDetectorV2
    loaded_models = {}

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

    # Get arrow files
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/ForenSynths/*.arrow")
    existing_files = sorted(glob.glob(cache_pattern))
    print(f"\n  Cached ForenSynths files: {len(existing_files)}")

    # Download more files if needed (up to 30 for better coverage)
    if len(existing_files) < 30:
        print("\n  Downloading more ForenSynths files...")
        new_files = download_more_files(len(existing_files), target_count=30)
        existing_files = sorted(glob.glob(cache_pattern))
        print(f"  Total ForenSynths files: {len(existing_files)}")

    # Scan generators
    gen_counter = Counter()
    for f in existing_files:
        reader = ipc.open_stream(f)
        table = reader.read_all()
        for path in table['image_path'].to_pylist():
            if not path.startswith('test/'):
                continue
            parts = path.split('/')
            if len(parts) >= 4:
                gen = parts[1]
                label = 'real' if '0_real' in path else 'fake'
                gen_counter[(gen, label)] += 1

    gens_available = sorted(set(g for g, l in gen_counter))
    print(f"\n  Available test generators: {gens_available}")
    for g in gens_available:
        real = gen_counter.get((g, 'real'), 0)
        fake = gen_counter.get((g, 'fake'), 0)
        print(f"    {g:<20} real={real:>4}  fake={fake:>4}")

    # Select generators with enough samples (at least 30 real + 30 fake)
    valid_gens = []
    for g in gens_available:
        real = gen_counter.get((g, 'real'), 0)
        fake = gen_counter.get((g, 'fake'), 0)
        if real >= 30 and fake >= 30:
            valid_gens.append(g)
    print(f"\n  Valid generators (≥30 per class): {valid_gens}")

    # Evaluate
    print("\n" + "=" * 60)
    print("ForenSynths (CNNDetection, CVPR 2020) Cross-Dataset Evaluation")
    print("=" * 60)

    gen_display = {
        'progan': 'ProGAN',
        'stylegan': 'StyleGAN',
        'stylegan2': 'StyleGAN2',
        'biggan': 'BigGAN',
        'cyclegan': 'CycleGAN',
        'stargan': 'StarGAN',
        'gaugan': 'GauGAN',
        'deepfake': 'Deepfake',
        'crn': 'CRN',
        'imle': 'IMLE',
        'san': 'SAN',
        'seeingdark': 'SeeingDark',
        'whichfaceisreal': 'WhichFace',
    }

    results = {}
    for gen in valid_gens:
        display = gen_display.get(gen, gen)
        print(f"\n--- {display} ({gen}) ---")
        dataset = ForenSynthsDataset(existing_files, generator_filter=gen, transform=transform)
        if len(dataset) < 10:
            continue
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        results[display] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {display}')
                results[display][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[display][model_name] = {'error': str(e)}

        save_json(results, os.path.join(results_dir, 'crossdataset_forensynths_results.json'))
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    all_models = set()
    for gen_results in results.values():
        for m, v in gen_results.items():
            if isinstance(v, dict) and 'auc' in v:
                all_models.add(m)
    models = sorted(all_models, key=lambda x: (0 if 'Ours' in x else 1, x))
    gens = list(results.keys())

    for model in models:
        aucs = []
        for g in gens:
            auc = results.get(g, {}).get(model, {}).get('auc', 0) * 100
            aucs.append(auc)
        avg = np.mean(aucs)
        print(f"  {model:<15}: avg={avg:.2f}%  " + "  ".join([f"{g}={a:.1f}" for g, a in zip(gens, aucs)]))

    del loaded_models
    torch.cuda.empty_cache()
    print("\n[Done]")


if __name__ == '__main__':
    main()
