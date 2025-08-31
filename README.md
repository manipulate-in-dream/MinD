# 🧠 MinD: Learning A Dual-System World Model for Real-Time Planning and Action Consistency Video Generation
*Xiaowei Chi1,2*, Kuangzhi Ge3*, Jiaming Liu3†, Siyuan Zhou2, Peidong Jia3, Zichen He3,
Rui Zhao1, Yuzhen Liu1, Tingguang Li1, Sirui Han2, Shanghang Zhang3✉, Yike Guo2✉

1Tencent RoboticsX, 2Hong Kong University of Science and Technology 3Peking University*

**MinD** is a dual-system world model for robotics that unifies video imagination and action generation. It enables real-time planning, implicit risk analysis, and explainable control. By combining a low-frequency visual diffusion model and a high-frequency action policy, MinD supports fast, safe, and semantically grounded decision-making for embodied agents.

> 📄 [arXiv Paper](https://arxiv.org/abs/2506.18897)
> 🌐 [Project Website](https://manipulate-in-dream.github.io)
---

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

## 🚀 Quick Start

1. Clone the repository & install dependencies:

```bash
git clone https://github.com/your-org/mind-world-model.git
cd mind-world-model
pip install -r requirements.txt
```

2. Download pretrained weights and place them in:

```
checkpoints/
  ├── lodiff_visual.pth
  ├── hidiff_policy.pth
  └── matcher.pth
```

3. Run the remote inference server:

```bash
python remote_infer.py
```

4. Send images and instructions from the client; the server will return predicted actions.

---

## 📊 Benchmark Results

### 🧪 RLBench Simulation (Franka Robot)

- **Mean Success Rate**: Up to **63.0%**
- **Inference Speed**: Up to **11.3 FPS**
- **Failure Prediction Accuracy**: **74%**

| Task                  | MinD-B | VPP | RoboDreamer |
|-----------------------|--------|-----|-------------|
| Close Laptop Lid      | 68%    | 84% | 76%         |
| Sweep to Dustpan      | 96%    | 80% | 76%         |
| Mean Accuracy         | 63.0%  | 53% | 50.3%       |

### 🤖 Real-World Franka Robot

| Task               | MinD (Wrist) | MinD (Front) | VPP  | OpenVLA |
|--------------------|--------------|--------------|------|---------|
| Pick & Place       | 75%          | 60%          | 55%  | 40%     |
| Unplug Charger     | 65%          | 50%          | 40%  | 25%     |
| Wipe Whiteboard    | 65%          | 85%          | 60%  | 30%     |
| **Average**        | **72.5%**    | **68.75%**   | 52.5%| 37.5%   |

---

## 📈 Risk-Aware Inference

- **LoDiff** predicts future frames as latent features.
- **DiffMatcher** aligns these with HiDiff’s action space.
- Latent PCA analysis shows clear separation between **successful** and **failed** task predictions.
- Enables early-stage failure detection without extra supervision.

---

## 🧪 Evaluation Scripts

- RLBench evaluation: `scripts/eval_rlbench.py`
- Real-world testing: `scripts/eval_realworld.py`
- Latent feature PCA: `scripts/pca_analysis.py`
- Video visualization: `scripts/visualize_videos.py`

---

## 📦 Pretrained Model Info

| Module         | Description                                  |
|----------------|----------------------------------------------|
| LoDiff-Visual  | Latent video diffusion model (slow, 1000 steps) |
| HiDiff-Policy  | Diffusion Transformer for action generation  |
| DiffMatcher    | Visual-to-action latent adapter              |

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
