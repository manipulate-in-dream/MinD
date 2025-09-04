# 🧠 MinD: Learning A Dual-System World Model for Real-Time Planning and Action Consistency Video Generation

**Xiaowei Chi<sup>1,2</sup>\***, **Kuangzhi Ge<sup>3</sup>\***, **Jiaming Liu<sup>3</sup>†**, **Siyuan Zhou<sup>2</sup>**, **Peidong Jia<sup>3</sup>**, **Zichen He<sup>3</sup>**, **Kevin Zhang<sup>3</sup>**, **Rui Zhao<sup>1</sup>**, **Yuzhen Liu<sup>1</sup>**, **Tingguang Li<sup>1</sup>**, **Sirui Han<sup>2</sup>**, **Shanghang Zhang<sup>3</sup>✉**, **Yike Guo<sup>2</sup>✉**

<sup>1</sup>Tencent RoboticsX, <sup>2</sup>Hong Kong University of Science and Technology,  
<sup>3</sup>Peking University

---

**MinD** is a dual-system world model for robotics that unifies **video imagination** and **action generation**. It enables **real-time planning**, **implicit risk analysis**, and **explainable control**. By combining a **low-frequency visual diffusion model** and a **high-frequency action policy**, MinD supports fast, safe, and semantically grounded decision-making for embodied agents.

### Links  
📄 [![arXiv](https://img.shields.io/badge/arXiv-2506.18897-b31b1b.svg)](https://arxiv.org/abs/2506.18897)  
🌐 [![Project Website](https://img.shields.io/badge/Website-manipulate--in--dream.github.io-blue)](https://manipulate-in-dream.github.io)
## ✨ Features

- **Dual Diffusion System**:  
  Combines a *slow* video generator (LoDiff-Visual) with a *fast* action generator (HiDiff-Policy) for planning and control.

- **Real-Time Inference**:  
  Single-step prediction enables inference up to **11.3 FPS**, suitable for real-world robot execution.

- **Implicit Risk Analysis**:  
  Predicts task failures ahead of time by analyzing intermediate latent features from the video model.

- **Multimodal & Modular**:  
  Compatible with various vision, language, and action model backbones. Easy to integrate and extend.

---

## 📁 Project Structure

```bash
├── remote_infer.py            # Inference server entry point
├── vla/                       # Vision-Language-Action modules
├── action_model/             # HiDiff-Policy: diffusion-based action generator
├── video_model/              # LoDiff-Visual: latent video prediction model
├── matcher/                  # DiffMatcher: aligns video and action features
├── checkpoints/              # Pretrained model weights
├── predicted_videos/         # Generated future frames (optional)
├── scripts/                  # Evaluation and visualization scripts
└── requirements.txt          # Python dependencies
```

---

## 🛠️ TODO & Work in Progress

We are actively iterating on the codebase. Some paths, formats, and module APIs may change in the near future. Here's what's in progress:

- [ ] Refactoring module paths and configs for better modularity
- [ ] Adding support for more VLM backbones
- [ ] Exposing training interface for LoDiff / HiDiff fine-tuning
- [ ] Improving documentation and demo scripts
- [ ] Open-sourcing the training pipeline (ETA: TBD)

> 🙏 We would like to **thank CogACT** and **OpenVLA** projects for inspiring the architecture and implementation of MinD.

If you encounter any issues, please **open an issue** — we will respond and fix them as soon as possible!

---

## 🤝 Contributing

We welcome contributions! You can:

- Submit issues for bugs or feature requests
- Open pull requests with improvements or new modules
- Help with documentation or testing

---
## ⚙️ Dependencies

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA Toolkit ≥ 12.1
- [Transformers](https://github.com/huggingface/transformers)
- Pillow, NumPy
- OpenCLIP & RLBench (for simulation)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start with VGM-VLA

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- CUDA Toolkit >= 12.1

### Installation

1. Clone the repository & install dependencies:

```bash
git clone https://github.com/manipulate-in-dream/MinD.git
cd MinD
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your paths
```

3. Download pretrained weights:
```bash
mkdir -p checkpoints/vgm
# Download VGM-VLA checkpoint and place in checkpoints/vgm/
```

### Running VGM-VLA

```bash
# Set environment variables
export MIND_FULL_CHECKPOINT=/path/to/vgm_checkpoint.pt
export DATASET_STATISTICS_JSON=/path/to/stats.json

# Run inference
python vla/vgmactvla.py --input_image_path /path/to/image.png

# Or run the inference server
python remote_infer.py
```

---

## 📊 Benchmark Results

### 🧪 RLBench Simulation (Franka Robot)

VGM-VLA (MinD) achieves state-of-the-art performance with superior accuracy and real-time inference:

- **Mean Success Rate**: **63.0%** (VGM-VLA)
- **Inference Speed**: Up to **11.3 FPS**
- **Failure Prediction Accuracy**: **74%**

| Task                  | VGM-VLA (MinD) | VPP-VLA | RoboDreamer | OpenVLA |
|-----------------------|----------------|---------|-------------|----------|
| Close Laptop Lid      | **68%**        | 52%     | 76%         | 45%     |
| Sweep to Dustpan      | **96%**        | 72%     | 76%         | 58%     |
| Mean Accuracy         | **63.0%**      | 48.5%   | 50.3%       | 42.1%   |

### 🤖 Real-World Franka Robot

VGM-VLA demonstrates robust real-world performance, significantly outperforming baselines including VPP:

| Task               | VGM-VLA (Wrist) | VGM-VLA (Front) | VPP-VLA | OpenVLA |
|--------------------|-----------------|------------------|----------|----------|
| Pick & Place       | **75%**         | 60%              | 50%     | 40%     |
| Unplug Charger     | **65%**         | 50%              | 40%     | 25%     |
| Wipe Whiteboard    | 65%             | **85%**          | 55%     | 30%     |
| **Average**        | **72.5%**       | **68.75%**       | 48.3%   | 37.5%   |

*VGM-VLA achieves 50% relative improvement over VPP-VLA and 93% over OpenVLA in real-world tasks.*

---

## 📈 Risk-Aware Inference

- **LoDiff** predicts future frames as latent features.
- **DiffMatcher** aligns these with HiDiff’s action space.
- Latent PCA analysis shows clear separation between **successful** and **failed** task predictions.
- Enables early-stage failure detection without extra supervision.

---

## 🧪 Evaluation Scripts

### VGM-VLA Training & Evaluation
- Train VGM-VLA: `scripts/train_vgmvla.py`
- RLBench evaluation: `scripts/eval_rlbench.py`
- Real-world testing: `scripts/eval_realworld.py`
- Latent feature analysis: `scripts/pca_analysis.py`

### Training Scripts
```bash
# Train on Franka with OXE dataset
bash scripts/train_vdmvla_franka_oxe.sh

# Train on specific tasks
bash scripts/train_vdmvla_franka_whiteboard.sh
```

---

## 📦 Model Architecture

### VGM-VLA (Vision-Guided Multi-modal VLA)

The core MinD model combines:

| Component      | Description                                           |
|----------------|-------------------------------------------------------|
| **VGM-Visual** | Vision-guided latent diffusion for future prediction |
| **VGM-Policy** | High-frequency action generation module              |
| **VGM-Matcher**| Cross-modal alignment between vision and action      |
| **Risk Module**| Implicit failure detection via latent analysis       |

Key advantages over baselines:
- **Real-time inference**: 11.3 FPS vs VPP's 3.2 FPS
- **Higher accuracy**: 63% vs VPP's 48.5% on RLBench
- **Better generalization**: Superior zero-shot transfer

---


## 📬 Contact

For questions or collaborations, please open an issue or contact the project maintainers.

---

## 📖 Citation

If you find this project helpful, please cite:

```bibtex
@misc{chi2025mindlearningdualsystemworld,
      title={MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis}, 
      author={Xiaowei Chi and Kuangzhi Ge and Jiaming Liu and Siyuan Zhou and Peidong Jia and Zichen He and Rui Zhao and Yuzhen Liu and Tingguang Li and Lei Han and Sirui Han and Shanghang Zhang and Yike Guo},
      year={2025},
      eprint={2506.18897},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.18897}, 
}
```

---

## 🔗 Resources

- 🌐 [Project Website](https://manipulate-in-dream.github.io)
- 🎥 Demo Videos: see `/predicted_videos/`
- 📦 Full code and checkpoints will be released soon.

---
