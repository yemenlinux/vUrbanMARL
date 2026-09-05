Urban Scenario Benchmarks
=========================

**UrbanMARL** provides a comprehensive suite of multi-agent urban scenarios inheriting from ``UrbanScenario``.
All scenarios run natively as vectorized PyTorch tensors on CPU and CUDA devices, supporting both direct TorchRL execution and BenchMARL benchmark experimentation.

Scenario Overview
-----------------

============================ ================================= ===================================================
Scenario Key                 Scenario Class                    Primary Objective & Scope
============================ ================================= ===================================================
``uav_navigation``           ``NavigationScenario``            3D UAV path planning & building collision avoidance
``uav_ue_los``               ``UavUeLosScenario``              Maintain dynamic LoS wireless links to ground UEs
``coverage``                 ``CoverageScenario``              Maximize 3D geospatial mmWave area coverage
``uavmec_offloading``        ``UAVMECScenario``                End-to-end task offloading, M/M/c queuing & sojourn
``uav_mobile_ue``            ``UavMobileUeScenario``           Dynamic user mobility (Manhattan / Hotspot tracking)
``uav_lidar_navigation``     ``UavLidarNavigationScenario``     360-degree LiDAR POMDP safe canyon navigation
``uavmec_advanced_physics``  ``UavMecAdvancedPhysicsScenario`` High-fidelity aerodynamics & 3D directional beamforming
``default``                  ``DefaultScenario``               Baseline template for prototyping custom scenarios
============================ ================================= ===================================================


1. UAV Navigation (``uav_navigation``)
--------------------------------------

**Description & Objective**:
UAV swarms navigate through complex 3D urban terrain from randomized initial coordinates to target destinations while avoiding building collisions.

- **Action Space**: Continuous 3D velocity vector :math:`\mathbf{a}_i = [v_h, \phi, v_z]`, where :math:`v_h \in [-v_{h,\max}, v_{h,\max}]` is horizontal speed, :math:`\phi \in [-\pi, \pi]` is heading angle, and :math:`v_z \in [-v_{z,\max}, v_{z,\max}]` is vertical climb rate.
- **Observation Space**: 4-dimensional vector per UAV :math:`\mathbf{o}_i = [x_i, y_i, z_i, \text{Battery}_i]`.
- **Reward Function**:

.. math::

   R_i = -\Delta d_{\text{target}} - c_{\text{collision}} \cdot \mathbb{I}_{\text{collision}} + r_{\text{goal}} \cdot \mathbb{I}_{\text{reached}}

where :math:`\Delta d_{\text{target}}` is the reduction in Euclidean distance to the waypoint, :math:`c_{\text{collision}}` penalizes building collisions, and :math:`r_{\text{goal}}` rewards reaching the destination.

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv

   env = UrbanEnv(num_envs=16, scenario="uav_navigation", num_uavs=3)
   td = env.reset()
   action = env.action_spec.sample()
   next_td = env.step(td.update(action))


2. UAV-UE LoS Link Maintenance (``uav_ue_los``)
------------------------------------------------

**Description & Objective**:
UAVs coordinate 3D positions to maximize the Line-of-Sight (LoS) coverage ratio over ground IoT users, acting as airborne relays while avoiding building blockages.

- **Action Space**: Continuous 3D velocity vector :math:`\mathbf{a}_i = [v_h, \phi, v_z]`.
- **Observation Space**: 4-dimensional vector per UAV :math:`\mathbf{o}_i = [x_i, y_i, z_i, \text{Battery}_i]`.
- **Reward Function**:

.. math::

   R_i = \sum_{j \in \text{UEs}} \text{LoS}(i, j) - c_{\text{collision}} \cdot \mathbb{I}_{\text{collision}}

where :math:`\text{LoS}(i, j) \in \{0, 1\}` is the binary Line-of-Sight indicator between UAV :math:`i` and ground user :math:`j` evaluated via 3D ray-casting over the urban heightmap.


3. 3D Spatial Area Coverage (``coverage``)
------------------------------------------

**Description & Objective**:
UAVs learn cooperative spatial dispersion to maximize aggregate mmWave radio coverage over urban zones while penalizing UAVs that fly too close to one another.

- **Action Space**: Continuous 3D velocity vector :math:`\mathbf{a}_i = [v_h, \phi, v_z]`.
- **Observation Space**: 4-dimensional vector per UAV :math:`\mathbf{o}_i = [x_i, y_i, z_i, \text{Battery}_i]`.
- **Reward Function**:

.. math::

   R_i = \text{LoSRatio}_i - c_{\text{collision}} \cdot \mathbb{I}_{\text{collision}} - 5.0 \cdot \sum_{j \ne i} \mathbb{I}\left(\|\mathbf{p}_i - \mathbf{p}_j\| < d_{\text{thresh}}\right)

where :math:`d_{\text{thresh}}` (default: 100 m) is the minimum inter-UAV safety distance threshold.


4. UAV-MEC Task Offloading (``uavmec_offloading``)
---------------------------------------------------

**Description & Objective**:
Jointly optimizes 3D flight trajectories, dynamic DTLCM task offloading assignments, and multi-core MEC queuing resources to minimize system latency and energy dissipation while maximizing persistent sojourn coverage.

- **Action Space**: Continuous velocity :math:`[v_h, \phi, v_z]`.
- **Observation Space**: 8-dimensional vector per UAV: position :math:`(x, y, z)`, battery level, MEC server utilization, queue length, system latency, and completed task ratio.
- **Queuing & Wireless Channels**:
  - Tasks arrive according to a Poisson process with data size :math:`D_k \in [0.5, 2.0]\text{ MB}` and processing workload :math:`X_k = 1000\text{ cycles/bit}`.
  - Channel capacity follows Shannon's theorem over 29 GHz mmWave carrier with ITU-R P.1410 path loss.
  - MEC computation follows an M/M/c queue with server utilization :math:`\rho_i` and response delay :math:`W_{s, i}`.
- **Reward Function**:

.. math::

   R_i = w_1 \left(\frac{N_{\text{completed}, i}}{M}\right) - w_2 \left(\frac{\min(W_{s, i}, T_{\text{max}})}{T_{\text{max}}}\right) - w_3 \left(\frac{E_i}{E_{\text{max}}}\right) - w_4 \mathbb{I}_{\text{collision}} + w_5 \tanh(0.1 \cdot \text{Sojourn}_i)

where :math:`\text{Sojourn}_i` is Dr. Raddwan's formulation quantifying persistent consecutive LoS coverage steps.

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv

   env = UrbanEnv(
       num_envs=32,
       scenario="uavmec_offloading",
       num_uavs=3,
       num_ues=50,
       frequency_ghz=29.0,
   )
   td = env.reset()


5. Dynamic Ground User Mobility (``uav_mobile_ue``)
----------------------------------------------------

**Description & Objective**:
Ground users move continuously along urban street canyons under realistic mobility models. UAV swarms learn to track dynamic user clusters and maintain persistent communication links.

- **Mobility Engines**:
  1. **Gauss-Markov Mobility**: Correlated velocity updates with temporal memory factor :math:`\alpha_m \in [0, 1]`:
     
     .. math::
        \mathbf{v}_j(t+1) = \alpha_m \mathbf{v}_j(t) + (1-\alpha_m) \bar{\mathbf{v}} + \sqrt{1-\alpha_m^2} \mathbf{n}(t)

  2. **Manhattan Grid Mobility**: Constrains users to street canyon intersections with turn probability matrices.
  3. **Hotspot Flocking**: Models crowd gatherings drifting between urban points of interest.
  4. **Random Waypoint (RWP)**: Standard baseline random destination model with pause times.

- **Action Space**: Continuous velocity :math:`[v_h, \phi, v_z]`.
- **Observation Space**: UAV coordinates, battery, ground user centroid positions, and active LoS matrix.
- **Reward Function**:

.. math::

   R_i = w_{\text{los}} \cdot \text{LoSRatio}_i - w_{\text{coll}} \cdot \mathbb{I}_{\text{collision}} + w_{\text{sojourn}} \cdot \tanh(0.1 \cdot \text{Sojourn}_i)

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv

   env = UrbanEnv(
       num_envs=16,
       scenario="uav_mobile_ue",
       num_uavs=3,
       num_ues=20,
       mobility_model="gauss_markov",
   )


6. Autonomous LiDAR Navigation (``uav_lidar_navigation``)
----------------------------------------------------------

**Description & Objective**:
Addresses Partial Observability (POMDP) in GPS-denied urban canyons by equipping each UAV with a 360-degree radial LiDAR rangefinder sensor. Rangefinder beams detect obstacle boundaries before collisions occur, providing a continuous proximity safety buffer.

- **LiDAR Sensor Engine**:
  - Casts :math:`K` radial laser beams (default: :math:`K = 16` or :math:`32`) spanning :math:`360^\circ` azimuth.
  - Queries building heightmaps in parallel tensor operations to determine obstacle proximity distances :math:`d_k \in [0, d_{\max}]`.
- **Action Space**: Continuous velocity :math:`[v_h, \phi, v_z]`.
- **Observation Space**: Normalized target displacement vector, current velocity vector, battery level, and :math:`K` normalized LiDAR depth readings.
- **Reward Function**:

.. math::

   R_i = -w_{\text{dist}} \cdot \frac{d_{\text{target}}}{D_{\max}} + r_{\text{goal}} \cdot \mathbb{I}_{\text{reached}} - c_{\text{coll}} \cdot \mathbb{I}_{\text{collision}} - c_{\text{prox}} \cdot \max\left(0, d_{\text{safe}} - \min_k d_k\right)

where :math:`d_{\text{safe}}` is the safety margin distance (default: 15 m) and :math:`c_{\text{prox}}` penalizes proximity violations.

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv

   env = UrbanEnv(
       num_envs=16,
       scenario="uav_lidar_navigation",
       num_uavs=3,
       num_beams=16,
       max_range=50.0,
   )


7. High-Fidelity Physics & Propagation (``uavmec_advanced_physics``)
--------------------------------------------------------------------

**Description & Objective**:
Combines rigorous rotary-wing aerodynamic propulsion power dissipation with 3GPP TR 38.901 3D directional beamforming antennas and multi-core MEC queuing.

- **Aerodynamic Propulsion Power Model (Zeng et al., IEEE TWC)**:
  Models flight power dissipation as a function of 3D speed :math:`V = \|\mathbf{v}_i\|`:

  .. math::

     P(V) = \underbrace{P_0 \left(1 + \frac{3 V^2}{U_{\text{tip}}^2}\right)}_{\text{Blade Profile Power}} + \underbrace{P_i \left(\sqrt{1 + \frac{V^4}{4 v_0^4}} - \frac{V^2}{2 v_0^2}\right)^{1/2}}_{\text{Induced Flight Power}} + \underbrace{\frac{1}{2} d_0 \rho s A V^3}_{\text{Parasite Drag Power}}

  where :math:`P_0` is blade profile power in hover, :math:`P_i` is induced hover power, :math:`U_{\text{tip}}` is rotor tip speed, :math:`v_0` is mean rotor induced velocity, :math:`d_0` is fuselage drag ratio, :math:`\rho` is air density (:math:`1.225\text{ kg/m}^3`), and :math:`s` is rotor solidity.

- **3GPP TR 38.901 3D Directional Antennas**:
  Computes azimuth and elevation angle attenuation:

  .. math::

     A(\theta, \phi) = -\min\left(-(A_H(\phi) + A_V(\theta)), A_{\max}\right)

  yielding realistic directional beamforming gains for mmWave access links.

- **Action Space**: Continuous velocity :math:`[v_h, \phi, v_z]`.
- **Observation Space**: UAV kinematics, aerodynamic battery consumption rate, 3D directional achievable data rates, and multi-core MEC queue telemetry.
- **Reward Function**:

.. math::

   R_i = w_{\text{tasks}} \cdot \left(\frac{N_{\text{completed}, i}}{M}\right) - w_{\text{delay}} \cdot W_{s, i} - w_{\text{coll}} \cdot \mathbb{I}_{\text{collision}} + w_{\text{sojourn}} \cdot \tanh(0.1 \cdot \text{Sojourn}_i)

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv

   env = UrbanEnv(
       num_envs=32,
       scenario="uavmec_advanced_physics",
       num_uavs=3,
       num_ues=10,
       frequency_ghz=29.0,
   )


8. Default Scenario (``default``)
---------------------------------

**Description**:
Standard baseline scenario template inheriting directly from ``NavigationScenario``. Serves as a reference implementation for new researchers creating novel custom tasks.


Dynamic Scenario Auto-Registration
----------------------------------

UrbanMARL features an automated discovery system in ``urbanmarl.scenarios``.
Any new scenario placed in the ``urbanmarl/scenarios/`` directory that inherits from ``UrbanScenario`` is automatically discovered, registered, and available across both TorchRL and BenchMARL without modifying any registry code.

Creating a Custom Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new research scenario:

1. Create a new Python file in ``urbanmarl/scenarios/`` (e.g. ``urbanmarl/scenarios/my_scenario.py``).
2. Subclass ``UrbanScenario`` and define the required interface methods:

.. code-block:: python

   # urbanmarl/scenarios/my_scenario.py
   from typing import Tuple, Optional
   import torch
   from torchrl.data import Composite, Unbounded
   from urbanmarl.scenarios.base import UrbanScenario

   class Scenario(UrbanScenario):
       """Custom urban research scenario."""

       def _reset_all(self, env, tensordict=None, **kwargs):
           env.uav_agents_pos = env._env.gen_pos(
               num_pos=env.n_uavs, min_z=25.0, max_z=100.0, outdoor=True
           )
           env.uav_battery = torch.full((env.batch_size[0], env.n_uavs, 1), 100.0, device=env.device)
           env.uav_collisions = torch.zeros((env.batch_size[0], env.n_uavs, 1), dtype=torch.bool, device=env.device)

       def _reset_at(self, env, env_index: int, tensordict=None, **kwargs):
           # Reset single environment instance in the batch
           pass

       def process_actions(self, env, tensordict):
           # Update flight kinematics
           pass

       def observation(self, env) -> torch.Tensor:
           return torch.cat([env.uav_agents_pos, env.uav_battery], dim=-1)

       def observation_spec(self, env, group: str) -> Composite:
           return Unbounded(shape=torch.Size([4]), dtype=torch.float32, device=env.device)

       def action_spec(self, env, group: str) -> Composite:
           from torchrl.data.tensor_specs import Bounded
           return Bounded(low=-1.0, high=1.0, shape=torch.Size([3]), dtype=torch.float32, device=env.device)

       def reward(self, env, group: str) -> torch.Tensor:
           return -env.uav_collisions.float()

       def done(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           terminated = env.uav_collisions.any(dim=1)
           truncated = env.current_step >= env.max_steps
           return terminated | truncated, terminated, truncated

       def _render(self, env, mode="rgb_array") -> dict:
           return {"uav_positions": env.uav_agents_pos[0].detach().cpu()}

3. The scenario is now automatically available:

.. code-block:: python

   from urbanmarl.envs.base_env import UrbanEnv
   from urbanmarl.scenarios import list_scenarios

   print(list_scenarios())  # 'my_scenario' appears automatically!
   env = UrbanEnv(scenario="my_scenario", num_envs=8)
