# Unsupervised Semantic Segmentation Facilitates Model Understanding

This is the official implementation of the paper [**Unsupervised Semantic Segmentation Facilitates Model Understanding**](https://arxiv.org/pdf/2605.29691).

---

## 🛠️ Installation

### 1. Sync Submodules

Before setting up the environment, sync the submodules to ensure the `dinov3` folder contains the required code:

```bash
git submodule sync
git submodule update --init --recursive
```

> **Note:** Make sure the `DINOv3` folder is in the same directory as `pretrained/`, so the DINOv3 backbone can be initialized correctly.

### 2. Environment Setup

Create and activate an isolated virtual environment using `mamba` (or `conda`):

```bash
mamba create -n ssl_rep_seg python=3.11 -y
mamba activate ssl_rep_seg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ssl_rep_seg
```

### 3. Pretrained Weights

We collect several SSL pretrained weights and rename them for convenience. Download them by running:

```bash
hf download xiaoyan-xyy/ssl_pretrained_backbones --local-dir pretrained
```

---

## 📦 Dataset

Dataset preparation is left to the user. We demonstrate the protocol on three example datasets.

- **COCO-Stuff & Cityscapes** — Download from their original sources, or follow the [STEGO](https://github.com/mhamilton723/STEGO) dataset preparation guide.
- **PascalPart** — Please refer to this [website](https://roozbehm.info/pascal-parts/pascal-parts.html).

---

## 🚀 Usage

### 1. Train an Unsupervised Segmentation Model

All example code is documented under the `examples/` folder. Before running any protocol, ensure you have a trained unsupervised segmenter. You can train one by running:

```bash
bash train.sh
```

Alternatively, you may use your own segmentation model.

### 2. Run Protocol Examples

```bash
python -m examples.protocol_1_layerwise.run        # Protocol 1
python -m examples.protocol_2_compare_ssl.run      # Protocol 2
python -m examples.protocol_3_compare_modelsize.run # Protocol 3
```

### 3. Run Locality Bias Examples

```bash
python -m examples.locality_bias.run1  # Attention matrix for representative images
python -m examples.locality_bias.run2  # Locality bias MI calculation
```

---

## 📖 Citation

If you find this repository useful, please consider giving a star ⭐ and citing our work:

```bibtex
@article{yu2026unsupervised,
  title={Unsupervised Semantic Segmentation Facilitates Model Understanding},
  author={Yu, Xiaoyan and Mais, Lisa and Franzen, Jannik and Hirsch, Peter and Lechtenb{\"o}rger, Nick and Mardt, Andreas and Kainm{\"u}ller, Dagmar},
  journal={arXiv preprint arXiv:2605.29691},
  year={2026}
}
```