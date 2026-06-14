# Toward E cient Test Time Adaptation With Hierarchical Distribution Alignment


## Overview

HDA performs test-time adaptation by aligning the feature distribution of corrupted test samples to that of clean training data. The alignment operates at **three hierarchical levels**:

1. **Global level** — aligns the overall feature distribution between source and target domains via bidirectional KL divergence.
2. **Category level** — aligns per-class Gaussian components using online EMA updates, with pseudo-label filtering based on prediction consistency.
3. **Cluster level** — discovers latent subcategories within each class via online clustering and aligns fine-grained cluster distributions.

The method also incorporates **feature matching losses** that measure semantic correspondence between source and target features at both category and cluster levels, using a learnable affinity-based matching module.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- FAISS (GPU) >= 1.7.0
- scikit-learn >= 0.24.0
- NumPy, Pillow, colorama

---

## Dataset Setup

Download the CIFAR-10-C corruption benchmark and create the directory structure:

```bash
export DATADIR=/path/to/data
mkdir -p ${DATADIR} && cd ${DATADIR}

# Download CIFAR-10-C
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar

# The CIFAR-10 dataset itself will be downloaded automatically by torchvision
```

Expected structure:
```
${DATADIR}/
├── cifar-10-batches-py/          # Downloaded by torchvision
└── CIFAR-10-C/
    ├── gaussian_noise.npy
    ├── shot_noise.npy
    ├── impulse_noise.npy
    ├── defocus_blur.npy
    ├── glass_blur.npy
    ├── motion_blur.npy
    ├── zoom_blur.npy
    ├── snow.npy
    ├── frost.npy
    ├── fog.npy
    ├── brightness.npy
    ├── contrast.npy
    ├── elastic_transform.npy
    ├── pixelate.npy
    └── jpeg_compression.npy
```

---

## Pre-trained Model

Download the jointly trained ResNet-50 checkpoint (classification + SimCLR):

```bash
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
```

This checkpoint was obtained by training on clean CIFAR-10 training images with a semi-supervised SimCLR objective.

---

## Running

### Single Corruption

Run HTTAC on a single corruption type:

```bash
bash scripts/run_ttac_cifar10.sh snow 100000
```

Where the first argument is the corruption type and the second is the maximum number of samples.

### All Corruptions

The script iterates through all 15 common corruptions automatically:

```bash
bash scripts/run_ttac_cifar10.sh
```

### Configuration

Key hyperparameters:
- `--iters 4`: Number of adaptation iterations per batch.
- `--batch_size 256`: Batch size for self-supervised loss.
- `--batch_size_align 256`: Batch size for distribution alignment.
- `--lr 0.01`: Learning rate.
- `--num_samples_per_cluster 20`: Number of samples drawn per cluster/class for matching loss.
- `--weight_cluster_kl_loss 0.0001`: Weight for cluster-level KL divergence.

---

## Code Structure

```
cifar10/
├── main.py                  # Main entry point: parsing, setup, adaptation loop
├── offline.py               # Offline source feature extraction and clustering
├── models/
│   ├── resnet.py            # ResNet-50 backbone, contrastive head, classifiers
│   └── heads.py             # ExtractorHead combination module
├── utils/
│   ├── datasets.py          # CIFAR-10(-C) dataset loading and transforms
│   ├── augmentations.py     # RandAugment for self-supervised training
│   ├── model_utils.py       # Model building, checkpoint loading, entropy
│   ├── losses.py            # Supervised contrastive loss
│   ├── matching.py          # Feature matching (Affinity + FeatureMatcher)
│   ├── clustering.py        # Cluster classifier training
│   ├── stats.py             # Covariance, CORAL, MMD utilities
│   └── helpers.py           # Logging, directory, average meter
└── scripts/
    └── run_ttac_cifar10.sh  # Example run script
```

---

## Protocol

The single-pass (N-O / sTTT) protocol processes the corrupted test set in a single online pass:

1. **Offline phase**: Extract source features from clean training data, compute per-class Gaussian parameters, and optionally perform clustering.
2. **Online adaptation**: For each test batch:
   - Generate pseudo-labels via EMA of prediction logits.
   - Filter reliable samples using prediction consistency.
   - Update target distribution parameters (global, category-level, cluster-level).
   - Optimize alignment losses (KL divergence + feature matching).
   - Evaluate on the current batch.

---

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{liu2025towards,
  title={Towards Efficient Test time Adaptation with Hierarchical Distribution Alignment},
  author={Liu, Yabo and Huang, Chao and Xu, Yong and Cao, Xiaochun and Wang, Jinghua},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```

---

## Acknowledgements

This implementation builds upon the public codebase of [TTT++](https://github.com/vita-epfl/ttt-plus-plus) and [TTAC](https://github.com/Gorilla-Lab-SCUT/TTAC), and is substantially refactored for clarity and modularity.