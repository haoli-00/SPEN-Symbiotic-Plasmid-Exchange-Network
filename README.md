# SPEN: Symbiotic Plasmid Exchange Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

**A novel bio-inspired deep learning paradigm.** SPEN treats neural network modules as bacterial *plasmids*—self-contained, independently evolving sub-networks that flow horizontally across a population of host models through quorum-gated conjugation bridges, guided by a mycorrhizal latent communication space, with CRISPR-like immune memory and endosymbiotic integration.

> If you've ever wondered what happens when you merge six biological mechanisms into one learning algorithm, this is it.

---

## Biological Inspirations → Computational Mechanisms

| Biology | SPEN Mechanism |
|---------|---------------|
| Bacterial plasmid conjugation | Horizontal transfer of sub-network modules between host models |
| Quorum sensing | Population-density-gated communication bursts |
| CRISPR adaptive immunity | Persistent memory bank of successful plasmid-task pairings |
| Mycorrhizal fungal networks | Always-on latent space for diffusion-based distress signaling |
| Endosymbiosis (mitochondria origin) | Foreign plasmids permanently integrated into host architecture |
| Apoptosis (programmed cell death) | Automatic pruning of low-vitality modules via knowledge distillation |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              CRISPR Memory Bank                  │
│    task_signature → plasmid_fingerprint → gain   │
└────────────────────┬─────────────────────────────┘
                     │ recall / store
┌────────────────────┼─────────────────────────────┐
│           Mycorrhizal Latent Space M             │
│   v₁    v₂    v₃    v₄    v₅   (host embeddings) │
│   ↕     ↕     ↕     ↕     ↕                      │
│  HOST₁ HOST₂ HOST₃ HOST₄ HOST₅                   │
│  [p₁]  [p₂]  [p₃]  [p₄]  [p₅]  ← plasmids       │
│  [p₆]  [p₇]  [p₈]  [p₉]  [p₁₀]                  │
│    ↕ Conjugation Bridge (quorum-gated)            │
│    ≈ Diffusion Signal (always-on, distance-decay) │
└──────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Minimal Example

```python
import torch
from spen import Ecosystem, HostConfig, PlasmidConfig

# 5 hosts, each with 3 plasmids
ecosystem = Ecosystem(
    num_hosts=5,
    host_config=HostConfig(base_dim=128, num_plasmids=3),
    plasmid_config=PlasmidConfig(dim=128, rank=8),
)

# Train on 5-way split of MNIST
ecosystem.fit(train_loaders, epochs=20)

# Inference on a new task: CRISPR Bank auto-recalls relevant plasmids
output = ecosystem.forward(x_new, task_signature=task_sig)
```

---

## Key Concepts

### Plasmid
A low-rank adapter module (LoRA-style) that can be inserted into any host's forward pass. Each plasmid tracks its own fitness score, mutation counter, and niche profile. Plasmids mutate independently and undergo apoptosis when vitality drops.

### Host Cell
A neural network instance that carries a population of plasmids. Hosts train individually on local data shards, but exchange high-fitness plasmids with peers during conjugation bursts.

### Quorum Sensing
Each host emits an autoinducer signal proportional to its training progress. When the global average crosses a threshold, all hosts enter a synchronized **Conjugation Burst**—efficiently exchanging top-performing plasmids in batch.

### CRISPR Memory Bank
A persistent store mapping `(task_signature, plasmid_fingerprint) → gain`. When a new task arrives, the bank is queried for historically successful plasmid configurations, enabling one-shot transfer learning without retraining.

### Mycorrhizal Latent Space
A shared high-dimensional embedding space where hosts project their current input distribution. Hosts emit distress signals (high gradient norm) that diffuse with distance decay. Nearby hosts respond by pushing niche-matched plasmids.

### Endosymbiosis
Foreign plasmids that sustain high gain for E consecutive epochs are permanently absorbed into the host's base network, freeing plasmid slots for new exploration.

### Apoptosis
Plasmids with continuously low vitality scores trigger self-destruction. Before removal, essential knowledge is distilled into the base network via directional weight extraction.

### Ecological Niche Specialization (V0.4)
Plasmids declare niche profiles (task type, difficulty range, modality). Hosts cluster by niche in the mycorrhizal space. Communication prioritizes same-niche neighbors for efficiency.

### Coevolutionary Arms Race (V0.4)
Adversarial host pairs (Generator/Discriminator) engage in minimax games. Winning plasmids spread through the population, making SPEN naturally suited for adversarial training and anomaly detection.

---

## Convergence

Under standard conditions (finite plasmid pool, monotonic gating, connected communication graph, apoptosis as L₀ regularization), SPEN's population-averaged loss converges almost surely to a local optimum.

Define Lyapunov function: `V(t) = (1/N) Σ L(f_θᵢ, Dᵢ)`

- Individual phase: gradient descent → ΔV ≤ 0
- Conjugation phase: gated insertion → ΔV ≤ 0
- Apoptosis: removal of negative-contribution modules → ΔV ≤ 0
- V(t) is monotonic non-increasing and bounded below → converges.

---

## Project Structure

```
spen/
├── spen/                   # Core library
│   ├── __init__.py
│   ├── plasmid.py          # Plasmid: low-rank adapter module
│   ├── host.py             # HostCell: neural network + plasmid population
│   ├── conjugation.py      # ConjugationBridge: plasmid transfer
│   ├── quorum.py           # QuorumSensor: population density gating
│   ├── crispr.py           # CRISPRBank: immune memory
│   ├── mycorrhiza.py       # MycorrhizalSpace: latent communication
│   └── ecosystem.py        # Ecosystem: full orchestrator
├── examples/
│   └── basic_usage.py      # Minimal working example
├── docs/
│   └── algorithm.md        # Full algorithm design (CN/EN)
├── tests/
│   └── test_core.py
├── README.md
├── LICENSE
├── .gitignore
└── pyproject.toml
```

---

## Comparison with Existing Paradigms

| Feature | Transfer Learning | Federated Learning | Mixture of Experts | **SPEN** |
|---------|-------------------|-------------------|---------------------|----------|
| Module granularity | Full model | Full model | Expert-level | **Plasmid-level (finer)** |
| Communication | None | Centralized | None | **Decentralized + quorum-gated** |
| Historical memory | None | None | None | **CRISPR Bank** |
| Auto-pruning | None | None | Manual | **Apoptosis-driven** |
| Synergy detection | None | None | None | **Plasmid co-occurrence mining** |
| Adversarial evolution | No | No | No | **Coevolution mode** |
| Bio-inspirations fused | 0 | 0 | 0 | **6 mechanisms** |

---

## Citation

If you use SPEN in your research, please cite:

```bibtex
@software{spen2026,
  title     = {SPEN: Symbiotic Plasmid Exchange Network},
  year      = {2026},
  url       = {https://github.com/haoli-00/SPEN-Symbiotic-Plasmid-Exchange-Network},
  note      = {A bio-inspired deep learning paradigm fusing plasmid conjugation,
               quorum sensing, CRISPR memory, mycorrhizal communication,
               endosymbiosis, and apoptosis}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"Knowledge should not always crawl down the gradient. Sometimes it should jump sideways."*
