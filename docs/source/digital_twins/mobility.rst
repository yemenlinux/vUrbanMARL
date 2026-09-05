Kinematic & Mobility Digital Twin
===================================

The **Kinematic & Mobility Digital Twin** (`VectorizedUserMobility`) models 3D UAV kinematics alongside GPU-accelerated, pure PyTorch mobility engines for ground User Equipments (UEs).

UAV 3D Kinematics
-----------------

Each UAV :math:`i \in \{1, \dots, N\}` is modeled by its continuous 3D spatial coordinate :math:`\mathbf{p}_i(t) = [x_i(t), y_i(t), z_i(t)]^T` and 3D velocity vector :math:`\mathbf{v}_i(t) = [v_{i,x}(t), v_{i,y}(t), v_{i,z}(t)]^T`.

Discrete-time updates over interval :math:`\Delta t`:

.. math::

   \mathbf{p}_i(t+1) = \mathbf{p}_i(t) + \mathbf{v}_i(t) \cdot \Delta t

where continuous action :math:`\mathbf{a}_i(t) \in [-1, 1]^3` controls velocity directly or applies acceleration:

.. math::

   \mathbf{v}_i(t+1) = \text{clip}\left(\mathbf{v}_i(t) + \mathbf{a}_i(t) \cdot a_{\max} \cdot \Delta t, \; -\mathbf{v}_{\max}, \; \mathbf{v}_{\max}\right)

Boundary and Altitude Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UAV flight envelopes are strictly bound within urban air corridors:

- **Horizontal Boundaries**: :math:`-L_x/2 \le x_i \le L_x/2`, :math:`-L_y/2 \le y_i \le L_y/2`.
- **Altitude Floor & Ceiling**: :math:`z_{\min} \le z_i(t) \le z_{\max}` (e.g., :math:`20\text{ m} \le z_i \le 120\text{ m}`).

Ground User Mobility Engine
---------------------------

The `VectorizedUserMobility` class simulates dynamic pedestrian and vehicular ground traffic across thousands of parallel environments. It supports four distinct mobility algorithms:

1. Street-Constrained Manhattan Mobility (`model_type="manhattan"`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates street grid navigation between building blocks. UEs travel along orthogonal cardinal axes (East, North, West, South) and make probabilistic :math:`\pm 90^\circ` turns at street intersections:

.. math::

   \theta(t+1) = \begin{cases} 
   \theta(t) \pm \frac{\pi}{2}, & \text{with probability } p_{\text{turn}} = 0.15 \\
   \theta(t), & \text{with probability } 1 - p_{\text{turn}}
   \end{cases}

When a UE reaches a building footprint (:math:`H_{\text{map}}(x, y) > 0`), the digital twin reflects the velocity vector away from the facade, constraining users strictly to streets and sidewalks.

2. Gauss-Markov Mobility (`model_type="gauss_markov"`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Introduces temporal memory into pedestrian speed and heading to prevent unrealistic abrupt direction changes:

.. math::

   \mathbf{v}_{xy}(t+1) = \alpha_{\text{mem}} \mathbf{v}_{xy}(t) + (1 - \alpha_{\text{mem}}) \mathbf{n}(t)

where :math:`\alpha_{\text{mem}} \in [0, 1]` is the memory tuning coefficient (default 0.75) and :math:`\mathbf{n}(t) \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})` is Gaussian perturbation noise.

3. Dynamic Hotspot & Crowd Migration (`model_type="hotspot"`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models dynamic spatial clustering around mobile points of interest (e.g., public squares, transit hubs, concerts):

- Hotspot centers :math:`\mathbf{c}_k(t)` drift across the urban space with velocity :math:`\mathbf{v}_k \in [0.2, 1.0]\text{ m/s}`.
- Ground UEs experience an attractive vector pull towards the nearest active hotspot center:

.. math::

   \theta(t+1) = 0.4 \cdot \theta(t) + 0.6 \cdot \text{atan2}\left(\mathbf{c}_{k^*}(t) - \mathbf{p}_{xy}(t)\right)

4. Random Waypoint Mobility (`model_type="rwp"`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous random walk where users select random bearings in :math:`[0, 2\pi)` with speeds bounded in :math:`[v_{\min}, v_{\max}]`. Boundary collisions trigger specular reflection.

Python Usage Example
--------------------

.. code-block:: python

   import torch
   from urbanmarl.models.urban_map import VectorizedUrbanMap
   from urbanmarl.models.mobility import VectorizedUserMobility

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   batch_size = 64
   n_ues = 20

   # 1. Initialize urban map for street obstacle reflection
   urban_map = VectorizedUrbanMap(batch_size=batch_size, device=device)

   # 2. Instantiate mobility engine with Manhattan street grid logic
   mobility = VectorizedUserMobility(
       volume_size=(500.0, 500.0, 200.0),
       model_type="manhattan",
       speed_min=0.5,
       speed_max=2.5,
       device=device,
   )

   # 3. Initial positions on ground (z = 1.5m) and velocities
   ue_pos = torch.zeros((batch_size, n_ues, 3), device=device)
   ue_pos[..., :2] = (torch.rand((batch_size, n_ues, 2), device=device) - 0.5) * 400.0
   ue_pos[..., 2] = 1.5

   ue_vel = mobility.initialize_velocities(batch_size, n_ues)

   # 4. Advance simulation step by step
   dt = 1.0  # 1 second step
   for step in range(10):
       ue_pos, ue_vel = mobility.step(
           ue_pos=ue_pos,
           ue_vel=ue_vel,
           dt=dt,
           urban_map=urban_map,
       )

   print(f"Updated UE coordinates shape: {ue_pos.shape}")
