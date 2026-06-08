```markdown
# CL3E-GNN: Curriculum Learning with Engage, Enact, and Embed Phases for Imbalanced Node Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.0+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**CL3E-GNN** is a novel curriculum learning framework for imbalanced node classification on graphs. Standard GNNs struggle under class imbalance because majority nodes dominate message passing, causing minority class signals to become statistically diluted. CL3E-GNN addresses this through three synergistic phases — **Engage**, **Enact**, and **Embed** — that progressively refine node representations while preserving graph topology.

### Key Features

- Three-Phase Curriculum Learning**:
  - Engage — multi-head attention identifies critical node/edge connections; dual GCN+GAT aggregation generates enhanced feature representations
  - Enact — refines minority class representations via importance sampling, metapath propagation, and anomaly detection
  - Embed — memory-augmented reinforcement learning progressively refines embeddings via reward-modulated updates
- Dual-Loss Optimisation: Cross-entropy combined with adaptive curriculum loss for principled difficulty progression
- Four Loss Strategies: Standard, curriculum, entropy-regularised, and combined — selectable or dynamically scheduled
- Progressive Training: Phase-locked training (Engage → Enact → Embed → Finetune) with independent optimisers per phase
- Comprehensive Analysis: Gradient stability, t-SNE visualisation (per phase), ROC curves, confusion matrices, theory validation plots

---

Here's the model structure section to add to your README:

```markdown
## Model Architecture Details

### CL3EGNN — Full Three-Phase Architecture

```
Input Node Features (in_dim)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│               Phase 1: ENGAGE                       │
│                                                     │
│  node_proj (MLP) → FeatureResourceConnector         │
│       ↓                    ↓                        │
│   GCNLayer            MultiHeadGATLayer             │
│       └──────── Gated Fusion ───────────┘           │
│                      ↓                              │
│         TopologicalPositionEncoder (TPE)            │
│    (degree, 2-hop degree, clustering,               │
│     reachability, random-walk diagonal)             │
└──────────────────────┬──────────────────────────────┘
                       │ + TPE bias
                       ▼
          [Optional Backbone: ResidualGNNBlocks]
          (GCNConv + GATConv + JumpingKnowledge)
                       │
┌──────────────────────┼──────────────────────────────┐
│               Phase 2: ENACT                        │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ NodeImportanceWeighting                     │    │
│  │  (prototype-based similarity scoring)       │    │
│  ├─────────────────────────────────────────────┤    │
│  │ MetapathExtractor                           │    │
│  │  (1-hop + 2-hop aggregation, weighted)      │    │
│  ├─────────────────────────────────────────────┤    │
│  │ AdaptiveSampling                            │    │
│  │  (class-prototype distance + degree bias)   │    │
│  ├─────────────────────────────────────────────┤    │
│  │ EdgeIntensityModelling                      │    │
│  │  (edge weight prediction + aggregation)     │    │
│  ├─────────────────────────────────────────────┤    │
│  │ AnomalyDetection                            │    │
│  │  (feature + structural anomaly scoring)     │    │
│  └──────────────┬──────────────────────────────┘    │
│                 │                                   │
│    Component Attention (softmax over 5 components)  │
│                 ↓                                   │
│         EnactFeedback (gated update)                │
└──────────────────────┬──────────────────────────────┘
                       │ + residual from Engage
                       ▼
          Gated Phase-to-Phase Skip Connection
                       │
┌──────────────────────┼──────────────────────────────┐
│               Phase 3: EMBED                        │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ MemoryAugmentedUpdate                       │    │
│  │  (16 memory slots, online updates)          │    │
│  ├─────────────────────────────────────────────┤    │
│  │ KnowledgeGraphIntegration                   │    │
│  │  (1-hop + 2-hop knowledge propagation)      │    │
│  ├─────────────────────────────────────────────┤    │
│  │ GradientHistoricalUpdate                    │    │
│  │  (reward-modulated, 3-strategy selection)   │    │
│  │   Strategy 1: MLP expansion                 │    │
│  │   Strategy 2: Bottleneck compression        │    │
│  │   Strategy 3: Wide expansion                │    │
│  └──────────────┬──────────────────────────────┘    │
│                 │                                   │
│    Component Attention (softmax over 4 components)  │
│                 ↓                                   │
│         ReflectiveLearning (gated reflection)       │
│                 ↓                                   │
│         Node Embeddings + Classification Head       │
└──────────────────────┬──────────────────────────────┘
                       │
          Learned Phase Ensemble (softmax weights)
          pw[0]×logits_engage + pw[1]×logits_enact
                     + pw[2]×logits_embed
                       │
                       ▼
              Final Node Classification
```

---

### Component Summary

| Component | Module | Role |
|---|---|---|
| TPE | `TopologicalPositionEncoder` | Structural bias: degree, clustering, reachability, RW-diagonal |
| Engage | `EngageModule` | Dual GCN+GAT aggregation with gated fusion |
| Backbone | `ResidualGNNBlock` × L | Optional stochastic-depth GCN+GAT residual layers |
| Enact | `EnactModule` | 5-component minority refinement with feedback |
| Embed | `EmbedModule` | Memory-augmented RL with knowledge graph integration |
| Aux heads | `aux_engage`, `aux_enact` | Intermediate supervision (label smoothing) |
| Phase weights | `phase_weights` | Learned softmax ensemble over 3 phases |
| Curriculum | `curriculum_factor` | Per-class difficulty scaling |

---

### Model Variants

| Variant | Class | Description |
|---|---|---|
| `cl3e` | `CL3EGNN(backbone_layers=0)` | Pure three-phase: Engage → Enact → Embed |
| `v1` | `CL3EGNN(backbone_layers=N)` | Three phases + N residual backbone layers |
| `v2orig` | `CL3E_ModelV2` | Original standalone V2: Engage → Enact → GCN backbone |

---

### Enact Phase — 5 Components

```
x ──┬── NodeImportanceWeighting ──┐
    ├── AdaptiveSampling          ├── Component Attention ──► EnactFeedback
    ├── EdgeIntensityModelling    │   (softmax over 5)
    ├── AnomalyDetection         ─┤
    └── KnowledgeTransfer ────────┘
```

- NodeImportanceWeighting: Prototype-based similarity; EMA prototype updates per class
- AdaptiveSampling: Class-prototype distance + neighbour degree for minority upweighting
- EdgeIntensityModelling: Learns edge intensities to weight neighbourhood aggregation
- AnomalyDetection: Feature + structural anomaly scores; class-conditional deviation
- KnowledgeTransfer: Prototype similarity projection for cross-class knowledge sharing

---

### Embed Phase — Reward Computation

```
rewards_i = 1.0  if model correctly classifies node i (labelled)
          = propagated via 2-hop max-pooling for unlabelled nodes
          = degree-normalised fallback if no labels available
```

Rewards modulate the `GradientHistoricalUpdate` to emphasise harder, 
misclassified nodes during embedding refinement.

---

### Loss Functions

```
Total Loss = Primary Loss + Auxiliary Loss + Embedding Regularisation

Primary Loss (one of):
  standard    → CrossEntropy(logits, targets)
  curriculum  → CrossEntropy weighted by class difficulty
  entropy     → CrossEntropy + confidence-based entropy regularisation
  combined    → all three combined

Auxiliary Loss (after epoch 10):
  aux_weight × 0.5 × (LabelSmoothCE(logits_engage) + LabelSmoothCE(logits_enact))

Embedding Regularisation:
  emb_reg × ||embeddings||²_F
```
```Classification Output
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- scikit-learn
- CUDA (optional, recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/afofanah/CL3E-GNN.git
cd CL3E-GNN

# Create conda environment
conda create -n cl3e python=3.8
conda activate cl3e

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install numpy scikit-learn matplotlib seaborn tqdm
```

---

## Datasets

CL3E-GNN is evaluated on **8 benchmark datasets** spanning citation networks, co-purchase graphs, co-authorship networks, and heterophilic graphs:

| Dataset      | Type                  | Nodes   | Classes | Source          |
|--------------|-----------------------|---------|---------|-----------------|
| Cora         | Citation network      | 2,708   | 7       | Auto-downloaded |
| CiteSeer     | Citation network      | 3,327   | 6       | Auto-downloaded |
| PubMed       | Citation network      | 19,717  | 3       | Auto-downloaded |
| Photo        | Co-purchase graph     | 7,650   | 8       | Auto-downloaded |
| Computers    | Co-purchase graph     | 13,752  | 10      | Auto-downloaded |
| CS           | Co-authorship network | 18,333  | 15      | Auto-downloaded |
| Chameleon    | Wikipedia (heteroph.) | 2,277   | 5       | Auto-downloaded |
| ogbn-arxiv   | arXiv citation graph  | 169,343 | 40      | Auto-downloaded |

All datasets are automatically downloaded to `./data/` on first run via PyTorch Geometric.

---

## Usage

### Quick Start

```bash
# Train on single dataset (default: computers)
python main.py

# Train on a specific dataset
python main.py --dataset cora

# Train on all 8 datasets
python main.py --all_datasets

# Train on a custom subset
python main.py --datasets cora citeseer pubmed
```

### Progressive Three-Phase Training (Recommended)

```bash
python main.py --dataset cora --progressive --model cl3e
```

### Dynamic Loss Schedule

```bash
# Cycles: standard → curriculum → entropy → combined
python main.py --dataset computers --dynamic_schedule
```

### Full CLI Options

```bash
python main.py \
  --dataset cora \                  # Target dataset
  --model cl3e \                    # cl3e | v1 | v2 | v2orig
  --runs 5 \                        # Number of independent runs
  --epochs 500 \                    # Training epochs
  --hidden_dim 128 \                # Hidden dimension
  --num_gcn_layers 3 \              # GCN layers
  --attention_type sigmoid \        # sigmoid | softmax
  --dropout 0.3 \                   # Dropout rate
  --lr 0.005 \                      # Learning rate
  --weight_decay 5e-4 \             # Weight decay
  --grad_clip 1.0 \                 # Gradient clipping norm
  --dropedge 0.3 \                  # DropEdge rate
  --emb_reg 1e-4 \                  # Embedding L2 regularisation
  --patience 100 \                  # Early stopping patience
  --loss_type combined \            # standard | curriculum | entropy | combined
  --progressive \                   # Phase-locked progressive training
  --dynamic_schedule \              # Dynamic loss schedule
  --seed 42 \                       # Random seed
  --data_root ./data \              # Data directory
  --out_dir ./results               # Output directory
```

---

## Training Strategies

### Standard Training
Single loss function applied throughout training:
```bash
python main.py --dataset cora --loss_type combined
```

### Progressive Training (Phase-Locked)
Each phase trains independently with frozen parameters elsewhere:
```
Phase 1 (Engage)   → Phase 2 (Enact) → Phase 3 (Embed) → Phase 4 (Finetune)
epochs/4 each phase, lr/5 in finetune phase
```
```bash
python main.py --dataset cora --progressive
```

### Dynamic Schedule
Cycles through all four loss functions sequentially:
```
Standard → Curriculum → Entropy → Combined
epochs/4 each segment
```
```bash
python main.py --dataset cora --dynamic_schedule
```

### Multi-Run Evaluation
Average results over multiple seeds for statistical reliability:
```bash
python main.py --dataset cora --runs 5
```

---

## Loss Functions

| Loss Type    | Description |
|--------------|-------------|
| `standard`   | Cross-entropy loss |
| `curriculum` | Adaptive curriculum loss with class-performance weighting |
| `entropy`    | Entropy-regularised loss with confidence-based dynamic weighting |
| `combined`   | Cross-entropy + curriculum + entropy (recommended) |

---

## Output Structure

```
results/
├── theory/                          # Static theory validation plots
│   ├── thm3_curriculum_schedule.png
│   ├── cor4_memory_forgetting.png
│   ├── thm1_thm2_imbalance_penalty.png
│   └── thm3_weight_evolution.png
├── {dataset}/                       # Per-dataset results
│   ├── summary.json                 # Mean ± std metrics
│   ├── all_runs.json                # Per-run metrics
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_f1.png
│   ├── roc_curves.png
│   ├── tsne.png
│   ├── tsne_three_phases.png        # Engage → Enact → Embed progression
│   ├── curriculum_factors.png
│   ├── gradient_analysis/           # Gradient stability per phase
│   └── theory/                      # Per-dataset theory validation
├── checkpoints/
│   └── {dataset}/seed{N}/
│       └── {dataset}_best.pt
├── all_summaries.json
├── multi_run_summary.png
└── model_comparison.png
```

---

## Results

Results averaged over 5 independent runs:

| Dataset    | Accuracy | AUC    | F1-Macro |
|------------|----------|--------|----------|
| Cora       | —        | —      | —        |
| CiteSeer   | —        | —      | —        |
| PubMed     | —        | —      | —        |
| Photo      | —        | —      | —        |
| Computers  | —        | —      | —        |
| CS         | —        | —      | —        |
| Chameleon  | —        | —      | —        |
| ogbn-arxiv | —        | —      | —        |

*Full results available in the paper.*

---

## Project Structure

```
CL3E-GNN/
├── data/                        # Auto-downloaded datasets
├── checkpoints/                 # Pre-trained model checkpoints
├── models/
│   ├── model.py                 # CL3EGNN + CL3E_ModelV2 architecture
│   └── loss.py                  # CurriculumLoss, EntropyLoss, CombinedLoss
├── architecture/                # Architecture diagrams (PDF)
├── results/                     # Experiment outputs and plots
├── plots/                       # Global training plots
├── main.py                      # Entry point + CLI
├── train.py                     # Trainer (standard + progressive + dynamic)
├── gradient_analysis.py         # Gradient stability analysis
├── preprocess_datasets.py       # Dataset loading and preprocessing
├── visualise.py                 # Visualisation utilities
├── utils.py                     # Metrics, plotting, saving
└── test_train.py                # Unit tests
```

---

## Theoretical Contributions

CL3E-GNN provides formal theoretical analysis including:

- **Theorem 1–2**: Imbalance penalty bounds and Rademacher complexity analysis
- **Theorem 3**: Curriculum schedule convergence guarantees
- **Theorem 4**: Weight evolution analysis
- **Corollary 4**: Memory forgetting factor bounds

Theory validation plots are automatically generated in `results/theory/`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fofanah2025cl3e,
  title={CL3E-GNN: Curriculum Learning with Engage, Enact, and Embed Phases 
         for Imbalanced Node Classification},
  author={Fofanah, Abdul Joseph and others},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

