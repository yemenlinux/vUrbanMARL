# vUrbanMARL: Vectorized Urban Multi-Agent Reinforcement Learning

<a href="https://su.edu.ye/fcit/"><img src="docs/resources/fcit_logo.png" height="40" alt="Faculty of Computer and Information Technology"></a>
<a href="http://su.edu.ye/"><img src="docs/resources/su_logo.png" height="40" alt="Sana'a University"></a>
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.12+](https://img.shields.io/badge/pytorch-2.12%2B-orange.svg)](https://pytorch.org/)
[![TorchRL 0.13.3](https://img.shields.io/badge/TorchRL-0.13.3-red.svg)](https://github.com/pytorch/rl)
[![BenchMARL 1.5.2+](https://img.shields.io/badge/BenchMARL-1.5.2%2B-green.svg)](https://github.com/facebookresearch/BenchMARL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**vUrbanMARL** is a high-performance, GPU-vectorized, data-driven multi-agent reinforcement learning (MARL) framework and digital twin simulator tailored for **6G geospatial radio environments, Unmanned Aerial Vehicles (UAVs), Mobile Edge Computing (MEC) networks, and other emerging applications**.

**vUrbanMARL** is built natively on **TorchRL**, **PyTorch**, **TensorDict**, and **BenchMARL**, vectorizes 3D spatial ray-casting of Line-of-Sight (LoS) paths, ITU-R P.1410 urban map procedural generation, mmWave radio propagation, and M/M/c queuing dynamics across hundreds of concurrent environments.

**vUrbanMARL** is developed to extend the PhD research project titled **"Three-Dimensional Mobility for Multi-UAV Assisted Multi-Access Edge Computing"** conducted by **Dr. Basheer Raddwan** under the supervision of **Prof. Ibrahim Al-Baltah**, at the
**Department of Information Technology, Faculty of Computer and Information Technology, Sana’a University, Yemen**,
by refactoring the 
[**Panda5gSim**](https://github.com/yemenlinux/panda5gsim) project into a fully vectorized, data-driven, GPU-accelerated, and TorchRL-compatible MARL framework.

---

## 🎥 Video Overview
Rendering example of 3D and 2D geospatial and building heatmap during training of 6 MARL algorithms with 3 task scenarios; using 3 UAV-BS and 10 UE. Orange lines show UAV trajectories, green and red lines show LoS/NLoS conditions.

[![Watch the video](https://img.youtube.com/vi/NLE_W9oxWu0/maxresdefault.jpg)](https://www.youtube.com/watch?v=NLE_W9oxWu0)

Example of analysis and result evaluation images rendered in a single video.

[![Watch the video](https://img.youtube.com/vi/lzPbTOS-RSc/maxresdefault.jpg)](https://www.youtube.com/watch?v=lzPbTOS-RSc)

---

## 💡 Key Features

- ⚡ **100% Vectorized PyTorch Core**: Batched 3D spatial movement, line-of-sight (LoS) calculations, building collision detection, radio propagation, and queuing systems run natively on GPU/CPU without python-loop bottlenecks.
- 🏢 **ITU-R P.1410 3D Procedural Urban Maps**: Realistic 3D urban environments parameterized by building coverage ($\alpha$), building density ($\beta$), height distribution ($\gamma$), street width, and building dimensions using Poisson Point Processes.
- 📡 **Vectorized mmWave Radio Channel Model**: Computes Friis path loss, LoS/NLoS attenuation exponent shifts, Signal-to-Interference-plus-Noise Ratio (SINR), and Shannon channel capacity for 29 GHz / 6G networks.
- 🖥️ **Vectorized M/M/c MEC Queue Model**: Pure PyTorch tensor implementation of M/M/c queuing theory calculating server utilization, average queue lengths, waiting times, and total response delay.
- 🤖 **BenchMARL & TorchRL Native**: Fully compatible with BenchMARL task APIs (`UrbanEnvTask`). Supports SOTA MARL algorithms (MAPPO, MADDPG, MASAC, IPPO, IDDPG, ISAC, QMIX).
- 🎬 **3D Rendering & Analytics Suite**: Includes interactive Matplotlib/MP4 3D trajectory rendering, CSV/TensorBoard logging, and evaluation plotters.

---

## 🎮 Scenarios Overview

| Scenario Name | Description | Agent Action Space | Observation Space |
| :--- | :--- | :--- | :--- |
| `uav_navigation` | Multi-UAV assest 5G+/6G wireless network independently/cooperatively learn to improve the coverage by navigating the simulation volume to find the best positions. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |
| `uav_ue_los` | Multi-UAV assest 5G+/6G wireless network independently/cooperatively learn to maximize the Line-of-sight status with ground IoTs/users for high-altitude platform stations and UAV relays. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |
| `coverage` | Multi-UAV assest 5G+/6G wireless network independently/cooperatively learn to maximize the spatial coverage. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |


---

## ⚙️ Installation

### Prerequisites
- **Python**: $\ge 3.10$ tested on 3.12
- **PyTorch**: $\ge 2.2.0$ (CUDA recommended for large batch sizes)

### Setup Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yemenlinux/vUrbanMARL.git
   cd vUrbanMARL
   ```

2. **Create a virtual environment** (optional but recommended):

    Using conda:

    ```bash
        conda create -n vurbanmarl python=3.12
        conda activate vurbanmarl
    ```

    Using standard Python venv:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies and the package**:

    For CPU-only installation:

    ```bash
    pip install -r requirements/cpu.txt
    ```

    For GPU installation (CUDA 12.6 - recommended for old GPUs):

    ```bash
    pip install -r requirements/cuda126.txt
    ```

    For GPU installation (CUDA 13.0 - recommended for new GPUs):

    ```bash
    pip install -r requirements/cuda130.txt
    ```

---

## 🚀 Quick Start

### 1. Direct PyTorch / TorchRL Usage

You can instantiate and step through `UrbanEnv` directly using TorchRL semantics:

```python
import torch
from urbanmarl.envs.base_env import UrbanEnv

# Initialize environment with 64 parallel batched environments on GPU/CPU
env = UrbanEnv(
    num_envs=64,
    scenario="uav_navigation",
    num_uavs=3,
    num_ues=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Reset environment
tensordict = env.reset()
print("Initial Observation:", tensordict["agents", "observation"].shape)

# Sample action and step
actions = env.action_spec.sample()
tensordict.update(actions)
next_tensordict = env.step(tensordict)

print("Reward:", next_tensordict["next", "agents", "reward"].shape)
print("Done:", next_tensordict["next", "done"].shape)
```

### 2. Running MARL Benchmarks via BenchMARL

Run comparative benchmark training across multiple MARL algorithms (e.g., MAPPO, MADDPG):

```bash
python scripts/run_benchmark.py
```

Or execute full benchmark experiments with automated result collection and plotting:

```bash
python scripts/full_experiment.py
```

---

## 📊 Evaluation & Visualization

- **Interactive Notebooks**: Explore [evaluate.ipynb](notebooks/evaluate.ipynb) and [test_renderer.ipynb](notebooks/test_renderer.ipynb) for plotting scalar metrics and rendering 3D UAV trajectories over procedurally generated building maps.
- **Result Processing**: Utilities in [urbanmarl/eval_results.py](urbanmarl/eval_results.py) format CSV/TensorBoard outputs into publication-ready figures.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## Affiliation and Sponsors

Department of Information Technology, Faculty of Computer and Information Technology, Sana’a University, Yemen.

---

## ✍️ Authors & Citation

Created by: Dr. Basheer Raddwan

Supervised by: Prof. Ibrahim Al-Baltah

If you use **vUrbanMARL** in your research, please cite:
```bibtex
@inproceedings{raddwan_urbanmarl_2026,
	title = {{UrbanMARL}: {A} {Vectorized} {Urban} {Simulator} for {Multi}-{Agent} {Reinforcement} {Learning}},
	url = {https://www.researchgate.net/doi/10.13140/RG.2.2.17864.69122},
	doi = {10.13140/RG.2.2.17864.69122},
	urldate = {2026-07-31},
	author = {{Basheer A. Raddwan} and {Ibrahim A. Al-Baltah} and {Ammar Thabit Zahary} and {Anwar Alshamery}},
	year = {2026},
}
```

## Contributors:
  * Dr. Basheer Raddwan
  * Prof. Ibrahim Al-Baltah