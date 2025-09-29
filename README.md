# Boundary-to-Region Supervision for Offline Safe Reinforcement Learning (NeurIPS 2025)

This is the official implementation of **B2R**, a new method for offline safe RL that fixes a core "symmetry fallacy" in Decision Transformer-style models. B2R is evaluated on the [DSRL benchmark](https://github.com/liuzuxin/dsrl).

**The Idea:** Stop treating safety and reward symmetrically. Safety is a hard **boundary**, while reward is a flexible **target**. B2R implements this insight via **Boundary-to-Region** supervision, which realigns all cost-to-go signals to the true safety budget. This provides **dense, region-wide supervision** without changing the model's architecture.

![Figure 0](./figure_1_score0.97.jpg)

### Highlights

-   **Fixes a Core Flaw**: Solves the brittle deployment and sparse signal problems caused by the "symmetry fallacy."
-   **Zero Architectural Change**: Drastically improves safety without modifying the Transformer model or its objective function.
-   **SOTA Safety & Performance**:
    -   Satisfies safety in **35/38** tasks.
    -   Achieves the highest reward in **20/38** tasks.
    -   Consistently safer than CDT while delivering competitive rewards.
-   **Robust**: Degrades gracefully under safe data scarcity (down to 5-20%).
-   **Flexible**: A single model can be trained on multiple safety budgets simultaneously, enabling it to satisfy different constraints at deployment.

**In short:** B2R adds robust safety guarantees to the simple and powerful Decision Transformer framework by fundamentally changing how cost signals are supervised.
## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/B2R-ULTRA.git
cd B2R-ULTRA

# Install the package
pip install -e .
```

### Training

```bash
python main/B2R_main.py --env CarButton1 --cost_limit 20 --seed 0
```

## 🙏 Acknowledgements

- [DSRL](https://github.com/decisionintelligence/DSRL) – Benchmark for offline safe RL
- [Decision Transformer](https://github.com/kzl/decision-transformer) – Transformer-based RL modeling

## 📄 Citation

```bibtex
@inproceedings{b2r2025,
  title={Boundary-to-Region Supervision for Offline Safe Reinforcement Learning},
  author={Huikang Su, Dengyun Peng, Zifeng Zhuang, YuHan Liu, Qiguang Chen, Donglin Wang, Qinghe Liu},
  booktitle={NeurIPS},
  year={2025}
}
```

## 🛠 License
This project is licensed under the MIT License.
