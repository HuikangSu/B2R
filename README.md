# 🌟B2R: Boundary-to-Region Supervision for Offline Safe Reinforcement Learning (NIPS2025 Poster)

This repository provides the official implementation of **B2R**. It is evaluated on the [DSRL benchmark](https://github.com/decisionintelligence/DSRL) across SafetyGymnasium, BulletSafetyGym, and MetaDrive environments.

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
