# Dataset Preparation Guide

## 1. Training Data: GenImage (6 Sources)

Download GenImage dataset from: https://github.com/GenImage-Dataset/GenImage

Required 6 subsets (each contains `train/` and `val/` with `real/` and `fake/` subdirectories):

```
datasets/authoritative/genimage/
├── biggan/
│   ├── train/
│   │   ├── real/     # 10,000 real images
│   │   └── fake/     # 10,000 generated images
│   └── val/
│       ├── real/
│       └── fake/
├── adm/
├── glide/
├── vqdm/
├── midjourney/
└── sdv4/              # Stable Diffusion v1.4
    ├── train/
    │   ├── real -> ../../../biggan/train/real   # symlink to BigGAN real images
    │   └── fake/
    └── val/
```

**Notes:**
- Total training data: 6 sources x 10K real + 10K fake = 120K images
- SDv4 real images are symlinked to BigGAN real images (same source)
- Images are resized to 224x224 during data loading

## 2. Cross-Dataset Evaluation Benchmarks

### 2.1 UniversalFakeDetect (Ojha, CVPR 2023)
- Source: https://github.com/Yuheng-Li/UniversalFakeDetect
- HuggingFace: `datasets load` in `run_crossdataset_eval.py`
- Contains: DALL-E, GLIDE, Guided Diffusion, LDM-200, LDM-200-CFG

### 2.2 ForenSynths (Wang, CVPR 2020)
- Source: https://github.com/peterwang512/CNNDetection
- Contains: ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake, CRN, IMLE

### 2.3 Synthbuster (Bammey, WIFS 2023)
- HuggingFace: `bammey/synthbuster`
- Contains: DALL-E 2/3, Firefly, GLIDE, MJ v5, SD 1.3/1.4/2.0/XL

### 2.4 DiffusionForensics
- 200 test images per class (real/fake)

## 3. Directory Configuration

Update dataset paths in:
- `configs/default.yaml` — `data_root` field
- `experiments/run_paper_experiments.py` — `DATA_ROOT` variable

Default expected path:
```
/path/to/datasets/authoritative/genimage/
```

## 4. Hardware Requirements

- GPU: 2x NVIDIA RTX 4090 (24GB each), using DataParallel
- CLIP ViT-L/14 backbone requires ~1.2GB GPU memory (frozen)
- Batch size: 32 per GPU (default)
- Training time: ~2 hours for 20 epochs on 6-source data
