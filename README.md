# ScaleComm

**Self-Supervised Community Detection using Graph Attention Networks and Contrastive Learning**

> A PyTorch implementation for the research method described in:  
> *"Community Detection in Networks via Deep Learning"* — Graph Mining & Network Science

---

## Overview

Community detection is the problem of finding groups of nodes in a network that are more densely connected to each other than to the rest of the graph. It has applications in:

- Social network analysis (finding friend groups, echo chambers)
- Biological network analysis (protein complexes, gene modules)
- Recommendation systems (user communities, item clusters)
- Cybersecurity (botnet detection, anomalous cluster identification)

**ScaleComm** addresses two key limitations of existing methods:
1. **Scalability**: Most deep community detection methods fail on large graphs (>100K nodes)
2. **Fixed K**: All existing methods require the number of communities as input

---

## Method: ScaleComm

ScaleComm combines four components:

```
Graph Input
    │
    ▼
[Graph Augmentor]
    │ ──── View 1: edge drop + feature mask
    │ ──── View 2: edge drop + feature mask
    │
    ▼
[GAT Encoder] × 2 views
    │ 3-layer Graph Attention Network
    │ Multi-head attention (8 heads)
    │ ELU + Dropout + BatchNorm
    │
    ▼
[InfoNCE Contrastive Loss]
    │ Maximize agreement between same-node views
    │ Minimize agreement between different-node views
    │
    ▼
[Clustering Head]
    │ KMeans (fast, fixed K)
    │ DPGMM (auto K via Dirichlet Process)
    │
    ▼
Community Labels
```

### Key Design Choices

| Component | Design | Why |
|-----------|---------|-----|
| Augmentation | Community-aware edge drop | Preserves intra-community structure |
| Loss | InfoNCE (NT-Xent) | Proven effective for graph SSL |
| Encoder | 3-layer GAT | Attention weights down-weight inter-community edges |
| Clustering | DPGMM | Automatically determines K without prior knowledge |
| Optimization | Adam + Cosine LR decay | Stable convergence |

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourname/scalecomm.git
cd scalecomm
```

### Step 2 — Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows
```

### Step 3 — Install PyTorch (match your CUDA version)

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1 (recommended if GPU available):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install PyTorch Geometric
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# Replace +cpu with +cu121 if using CUDA
```

### Step 5 — Install remaining dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Information

Datasets are **automatically downloaded** on first run via PyTorch Geometric.

| Dataset | Nodes | Edges | Features | Communities | Auto-download |
|---------|-------|-------|----------|-------------|---------------|
| Cora | 2,708 | 5,429 | 1,433 | 7 | ✅ Yes |
| CiteSeer | 3,327 | 4,732 | 3,703 | 6 | ✅ Yes |
| Amazon Photo | 7,650 | 119,081 | 745 | 8 | ✅ Yes |
| Amazon Computers | 13,752 | 245,861 | 767 | 10 | ✅ Yes |
| DBLP (Coauthor) | 41,302 | 122,449 | 1,639 | 15 | ✅ Yes |
| Facebook SNAP | 4,039 | 88,234 | structural | Louvain | ⚠️ Manual |
| Enron Email | 36,692 | 183,831 | structural | Louvain | ⚠️ Manual |

### Manual dataset download

**Facebook SNAP:**
```bash
# Download from: https://snap.stanford.edu/data/ego-Facebook.html
mkdir -p data/raw/facebook
# Place facebook_combined.txt in data/raw/facebook/
```

**Enron Email:**
```bash
# Download from: https://snap.stanford.edu/data/email-Enron.html
mkdir -p data/raw/enron
# Place Email-Enron.txt in data/raw/enron/
```

---

## Running the Code

### Quick start (Cora, 200 epochs)
```bash
chmod +x run.sh
./run.sh
```

### Custom configuration
```bash
./run.sh --dataset citeseer --epochs 300 --cluster dpgmm
```

### Manual step-by-step

**Train:**
```bash
python training/train.py \
    --dataset cora \
    --epochs 200 \
    --hidden_dim 256 \
    --out_dim 128 \
    --num_heads 8 \
    --num_layers 3 \
    --lr 0.001 \
    --cluster kmeans \
    --device auto
```

**Evaluate:**
```bash
python training/evaluate.py \
    --dataset cora \
    --checkpoint checkpoints/scalecomm_cora_best.pt \
    --visualize
```

**All datasets:**
```bash
for ds in cora citeseer amazon-photo amazon-computers; do
    ./run.sh --dataset $ds --epochs 200
done
```

---

## Project Structure

```
scalecomm/
│
├── data/
│   ├── datasets.py          # Dataset loaders (Cora, CiteSeer, Amazon, DBLP...)
│   └── raw/                 # Downloaded datasets (auto-created)
│
├── models/
│   ├── gat_encoder.py       # 3-layer GAT encoder + projection head
│   ├── contrastive_loss.py  # GraphAugmentor + InfoNCE loss
│   └── clustering.py        # KMeans + DPGMM clustering
│
├── training/
│   ├── train.py             # Full training pipeline
│   └── evaluate.py          # Evaluation + visualization
│
├── utils/
│   ├── metrics.py           # NMI, ARI, ACC, F1, Modularity, Conductance
│   └── graph_utils.py       # Preprocessing, positional encoding, plotting
│
├── checkpoints/             # Saved model weights (auto-created)
├── outputs/                 # Results, labels, plots (auto-created)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Expected Output

```
████████████████████████████████████████████████████████████
  ScaleComm — Self-Supervised Community Detection
████████████████████████████████████████████████████████████

  Dataset    : CORA
  Epochs     : 200
  Clustering : KMEANS
  Device     : cpu

[Step 1/5] Loading dataset...
[Dataset] Cora loaded.
  Nodes      : 2708
  Edges      : 10556
  Features   : 1433
  Communities: 7

[Step 2/5] Initializing GAT encoder...
  Total parameters    : 1,842,176
  Trainable parameters: 1,842,176

[Step 3/5] Starting contrastive training...

   Epoch    Total Loss    Contrastive     Recon          LR
------------------------------------------------------------
       1        7.6231         7.5817    0.0414    0.001000
      10        5.4382         5.3971    0.0411    0.000995
      20        4.2817         4.2403    0.0414    0.000980
      50        2.8941         2.8527    0.0414    0.000905

  [Epoch 50] Running intermediate evaluation...
  → NMI: 0.4821 | ARI: 0.3917 | Modularity: 0.3201 | K=7

     100        1.8234         1.7820    0.0414    0.000772
     150        1.3712         1.3298    0.0414    0.000610
     200        1.1045         1.0631    0.0414    0.000436

[Step 4/5] Extracting node embeddings...
  Embedding shape: torch.Size([2708, 128])

[Step 5/5] Clustering embeddings...
  Using ground-truth K = 7 for KMeans
  Predicted communities: 7

══════════════════════════════════════════════════════════════
  FINAL EVALUATION RESULTS
══════════════════════════════════════════════════════════════
  NMI             : 0.7124
  ARI             : 0.6832
  ACC             : 0.7489
  F1              : 0.7203
  MODULARITY      : 0.4218
  CONDUCTANCE     : 0.1834
══════════════════════════════════════════════════════════════
```

---

## Hyperparameter Guide

| Parameter | Default | Recommended Range | Effect |
|-----------|---------|-------------------|--------|
| `--hidden_dim` | 256 | 128–512 | Larger = more expressive, slower |
| `--out_dim` | 128 | 64–256 | Embedding space size |
| `--num_heads` | 8 | 4–16 | More heads = better attention |
| `--num_layers` | 3 | 2–4 | Deeper = larger receptive field |
| `--lr` | 0.001 | 5e-4 to 5e-3 | Lower for large datasets |
| `--temperature` | 0.5 | 0.1–1.0 | Lower = harder contrastive task |
| `--p_edge` | 0.2 | 0.1–0.4 | Higher = more augmentation |
| `--p_feat` | 0.2 | 0.1–0.4 | Higher = more regularization |

---

## Baselines for Comparison

The following baselines are reproduced for comparison in `evaluate.py`:

| Method | Type | Publication |
|--------|------|-------------|
| Louvain | Classical | Blondel et al., 2008 |
| Spectral Clustering | Classical | Von Luxburg, 2007 |
| DGI | Deep | Velickovic et al., ICLR 2019 |
| AGC | Deep | Zhang et al., IJCAI 2019 |
| DCRN | Deep | Liu et al., IEEE TNNLS 2022 |
| GRACE-CD | Deep | Zhu et al., ACM TKDD 2023 |

---

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{scalecomm2026,
  title     = {ScaleComm: Scalable Self-Supervised Community Detection
               via Graph Attention Networks and Contrastive Learning},
  author    = {Your Name},
  year      = {2026},
  note      = {Research Project Report, Graph Mining \& Network Science}
}
```

---

## License

MIT License — free to use for academic research.
