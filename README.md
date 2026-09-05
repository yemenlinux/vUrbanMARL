# UrbanMARL: Vectorized Urban Multi-Agent Reinforcement Learning

<a href="https://su.edu.ye/fcit/"><img src="docs/source/resources/fcit_logo.png" height="40" alt="Faculty of Computer and Information Technology"></a>
<a href="http://su.edu.ye/"><img src="docs/source/resources/su_logo.png" height="40" alt="Sana'a University"></a>
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.12+](https://img.shields.io/badge/pytorch-2.12%2B-orange.svg)](https://pytorch.org/)
[![TorchRL 0.13.3](https://img.shields.io/badge/TorchRL-0.13.3-red.svg)](https://github.com/pytorch/rl)
[![BenchMARL 1.5.2+](https://img.shields.io/badge/BenchMARL-1.5.2%2B-green.svg)](https://github.com/facebookresearch/BenchMARL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**UrbanMARL** is a high-performance, GPU-vectorized, data-driven multi-agent reinforcement learning (MARL) framework and digital twin simulator tailored for **6G geospatial radio environments, Unmanned Aerial Vehicles (UAVs), Mobile Edge Computing (MEC) networks, and other emerging applications**.

**UrbanMARL** is built natively on **TorchRL**, **PyTorch**, **TensorDict**, and **BenchMARL**, vectorizes 3D spatial ray-casting of Line-of-Sight (LoS) paths, ITU-R P.1410 urban map procedural generation, mmWave radio propagation, and M/M/c queuing dynamics across hundreds of concurrent environments.

**UrbanMARL** is developed to extend the PhD research project titled **"Three-Dimensional Mobility for Multi-UAV Assisted Multi-Access Edge Computing"** conducted by **Dr. Basheer Raddwan** under the supervision of **Prof. Ibrahim Al-Baltah**, at the
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

UrbanMARL features a comprehensive suite of multi-agent urban scenarios covering navigation, wireless communications, edge computing, user mobility, and sensor perception:

| Scenario Key | Scenario Class | Primary Objective & Description | Action Space | Observation Space |
| :--- | :--- | :--- | :--- | :--- |
| `uav_navigation` | `NavigationScenario` | Multi-UAV swarm 3D waypoint navigation and obstacle avoidance across procedural urban canyons. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |
| `uav_ue_los` | `UavUeLosScenario` | Maintain dynamic unblocked Line-of-Sight (LoS) links to ground IoT users for aerial relay networks. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |
| `coverage` | `CoverageScenario` | Maximize 3D geospatial mmWave coverage over ground users with inter-UAV separation safety constraints. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |
| `uavmec_offloading` | `UAVMECScenario` | Joint UAV trajectory planning and M/M/c computation offloading under latency deadlines and sojourn time. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position, Battery, MEC Queue & LoS |
| `uav_mobile_ue` | `UavMobileUeScenario` | Persistent aerial tracking of dynamic ground users moving under Gauss-Markov, Manhattan Grid, and Hotspot flocking models. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position, Battery, UE Pos & LoS |
| `uav_lidar_navigation` | `UavLidarNavigationScenario` | POMDP safe urban navigation using batched 360° LiDAR ray-casting rangefinder beams and proximity margin penalties. | Continuous velocity $(v_h, \phi, v_z)$ | Target Vector, Velocity & 360° LiDAR Beams |
| `uavmec_advanced_physics` | `UavMecAdvancedPhysicsScenario` | High-fidelity rotary-wing aerodynamic propulsion power dissipation (Zeng et al.), 3GPP 38.901 3D directional antennas, and multi-core MEC digital twin. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Kinematics, Battery, Directional Rates & Queues |
| `default` | `DefaultScenario` | Baseline urban MARL template for rapid prototyping and custom scenario development. | Continuous velocity $(v_h, \phi, v_z)$ | UAV Position $(x,y,z)$ & Battery level |

---

## 🔬 Core Concepts & Theoretical Formulations

### 1. 3D Procedural Urban Maps (ITU-R P.1410)
Urban topographies are procedurally generated in parallel using statistical parameters defined by the International Telecommunication Union (ITU-R P.1410):
- **Building Coverage Ratio ($\alpha$)**: Fraction of land area covered by buildings.
- **Building Density ($\beta$)**: Average number of buildings per square kilometer.
- **Rayleigh Height Scale ($\gamma$)**: Scale parameter governing building height distributions:
  $$f(h) = \frac{h}{\gamma^2} \exp\left(-\frac{h^2}{2\gamma^2}\right), \quad h \ge 0$$

### 2. Sojourn Time & Dynamic LoS Channels
Quantifies consecutive Line-of-Sight connection durations between aerial UAVs and moving ground users:
- **LoS Ray-Casting**: Efficient segment-to-heightmap checking in pure PyTorch tensors without ray-triangle intersections.
- **Sojourn Bonus**: Formulated by Dr. Raddwan (*Ad Hoc Networks 2025, IEEE Access 2025*) to encourage UAVs to maintain stable, persistent coverage over mobile ground clusters:
  $$R_{\text{sojourn}} = w_{\text{sojourn}} \cdot \tanh(0.1 \cdot \text{Sojourn Time})$$

### 3. Multi-Core MEC Queuing Theory (M/M/c)
Simulates edge computation queues on GPU:
- Each UAV hosts $c$ parallel CPU processing cores operating at clock frequency $f_{\text{uav}}$.
- Tasks arrive stochastically with payload size $D_k$ (bits) and computational density $X_k$ (CPU cycles/bit).
- Computes server utilization $\rho = \frac{\lambda}{c \mu}$, waiting time in queue $W_q$, and system sojourn delay $W_s = W_q + \frac{1}{\mu}$.

### 4. High-Fidelity Rotary-Wing Aerodynamic Power Model
Accurately models flight energy dissipation as a function of 3D velocity $\mathbf{V}$ (Zeng et al., IEEE TWC):
$$P(V) = P_0 \left(1 + \frac{3 V^2}{U_{\text{tip}}^2}\right) + P_i \left(\sqrt{1 + \frac{V^4}{4 v_0^4}} - \frac{V^2}{2 v_0^2}\right)^{1/2} + \frac{1}{2} d_0 \rho s A V^3$$
where $P_0$ is blade profile power in hover, $P_i$ is induced power in hover, $U_{\text{tip}}$ is rotor tip speed, $v_0$ is mean rotor induced velocity, $d_0$ is fuselage drag ratio, and $s$ is rotor solidity.

### 5. 360° LiDAR Ray-Casting Rangefinder
Provides depth perception for urban canyon navigation under Partial Observability (POMDP):
- Casts $K$ radial rangefinder beams covering $360^\circ$ around each UAV.
- Detects obstacle proximity distances $d_k$ across building heightmaps.
- Applies proximity margin safety penalties $c_{\text{prox}} \cdot \max(0, d_{\text{safe}} - \min_k d_k)$ to prevent collisions before they happen.

### 6. Dynamic Scenario Auto-Registration
Scenarios located in `urbanmarl/scenarios/` are **automatically discovered and registered at runtime**:
- No need to manually edit `__init__.py` when creating new scenarios.
- Any subclass of `UrbanScenario` defining a `Scenario` class is immediately available via `load_scenario("my_scenario")` or `UrbanEnv(scenario="my_scenario")`.


---

## ⚙️ Installation

### Prerequisites
- **Python**: $\ge 3.10$ tested on 3.12
- **PyTorch**: $\ge 2.12.0$ (CUDA recommended for large batch sizes)

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

### 2. Multi-UAV MEC Advanced Physics with 6G Digital Twin

Run the high-fidelity aerodynamic flight and multi-core MEC digital twin:

```python
import torch
from urbanmarl.envs.base_env import UrbanEnv

# 32 parallel environments with high-fidelity aerodynamics and MEC queuing
env = UrbanEnv(
    num_envs=32,
    scenario="uavmec_advanced_physics",
    num_uavs=3,
    num_ues=10,
    frequency_ghz=29.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

td = env.reset()
action = env.full_action_spec.rand()
td = env.step(td.update(action))

# Access real-time Network Digital Twin telemetry
print("Telemetry info:", td.get("info", {}))
# Render 3D environment frame with HUD overlay
frame = env.render(mode="rgb_array")
print("Rendered frame size:", frame.shape)
```

### 3. Dynamic Ground User Mobility Tracking

Simulate aerial UAV relays tracking moving ground users with Gauss-Markov or Manhattan Grid dynamics:

```python
from urbanmarl.envs.base_env import UrbanEnv

env = UrbanEnv(
    num_envs=16,
    scenario="uav_mobile_ue",
    num_uavs=3,
    num_ues=20,
    mobility_model="gauss_markov",  # or "manhattan", "hotspot", "rwp"
)

td = env.reset()
for step in range(50):
    action = env.full_action_spec.rand()
    td = env.step(td.update(action))
```

### 4. Running MARL Benchmarks via BenchMARL

Train SOTA MARL algorithms (MAPPO, MADDPG, MASAC) on any UrbanMARL task:

```python
from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import UrbanEnvTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

task = UrbanEnvTask.UAVMEC_ADVANCED_PHYSICS.get_from_yaml()
experiment = Experiment(
    task=task,
    algorithm_config=MaddpgConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=ExperimentConfig.get_from_yaml(),
)
experiment.run()
```

### 5. Defining Custom Scenarios with Auto-Registration

Creating a new scenario requires **zero configuration**. Simply define a new file in `urbanmarl/scenarios/` inheriting from `UrbanScenario`:

```python
# urbanmarl/scenarios/my_custom_task.py
from urbanmarl.scenarios.base import UrbanScenario
import torch

class Scenario(UrbanScenario):
    """Custom research scenario."""
    def reset(self, env, tensordict=None, **kwargs):
        env.uav_agents_pos = env._env.gen_pos(num_pos=env.n_uavs, min_z=30.0, max_z=100.0, outdoor=True)
        # Initialize custom variables...

    def process_actions(self, env, tensordict):
        # Update flight dynamics...
        pass

    def reward(self, env, group):
        # Calculate custom research reward...
        return torch.zeros((env.batch_size[0], env.n_uavs, 1), device=env.device)
```

The new scenario is immediately recognized:
```python
from urbanmarl.envs.base_env import UrbanEnv
env = UrbanEnv(scenario="my_custom_task", num_envs=4)
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

If you use **UrbanMARL** in your research, please cite:

```bibtex
@inproceedings{raddwan_urbanmarl_2026,
location = {Sana'a, Yemen},
title = {{UrbanMARL}: A Vectorized Urban Simulator for Multi-Agent Reinforcement Learning},
url = {https://www.researchgate.net/doi/10.13140/RG.2.2.17864.69122},
booktitle = {1st International Conference on Intelligent, Dependable, Emerging, Autonomous, and Sustainable Engineering Technology ({IDEASET}2026)},
author = {{Raddwan, Basheer and Al-Baltah, Ibrahim and Zahary, Ammar and Alshamery, Anwar}},
date = {2026-12-01},
}
```

Related work

```bibtex
@article{RADDWAN2025104019,
title = {Quantify the joint effect of mobility and urban environment on computation offloading to multi-UAV MEC network: Sojourn time},
journal = {Ad Hoc Networks},
volume = {179},
pages = {104019},
year = {2025},
issn = {1570-8705},
doi = {https://doi.org/10.1016/j.adhoc.2025.104019},
url = {https://www.sciencedirect.com/science/article/pii/S1570870525002677},
author = {Raddwan, Basheer and Al-Baltah, Ibrahim}
}

@ARTICLE{11050367,
  author={Raddwan, Basheer and Al-Baltah, Ibrahim},
  journal={IEEE Access}, 
  title={Mobility-Aware Bivariate Line-of-Sight Probability for Air-to-Ground Communications Using Millimeter and Terahertz Waves}, 
  year={2025},
  volume={13},
  number={},
  pages={123913-123930},
  keywords={Air to ground communication;Atmospheric modeling;Layout;Geometry;Line-of-sight propagation;Directional antennas;Ray tracing;Communication channels;Buildings;ITU;Line-of-sight probability;mobility;air-to-ground communication;multi-access edge computing;ray-tracing;unmanned aerial vehicles;service time;sojourn time;urban;simulation},
  doi={10.1109/ACCESS.2025.3582890}}

@INPROCEEDINGS{10777167,
  author={Raddwan, Basheer and Al-Baltah, Ibrahim and Ghaleb, Mukhtar},
  booktitle={2024 1st International Conference on Emerging Technologies for Dependable Internet of Things (ICETI)}, 
  title={Environment-Aware 3D Mobility Simulation for the 5G and 6G Wireless Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Three-dimensional displays;Mobility models;Biological system modeling;Wireless networks;Urban areas;Interference;Ray tracing;Throughput;3GPP;Signal to noise ratio;urban;5G;6G;mobility;ray-tracing;simulation;open-source;handover rate;framework;multi-access edge computing;unmanned aerial vehicle;3D mobility;3D environment},
  doi={10.1109/ICETI63946.2024.10777167}}
```

## Contributors:
  * Dr. Basheer Raddwan
  * Prof. Ibrahim Al-Baltah
