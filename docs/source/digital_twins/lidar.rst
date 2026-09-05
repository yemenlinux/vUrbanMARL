360° LiDAR Rangefinder Digital Twin
===================================

The **360° LiDAR Rangefinder Digital Twin** (`VectorizedLiDAR`) provides high-speed, GPU-accelerated ray-marching simulations for multi-rotor UAV obstacle detection and collision avoidance in 3D urban topographies.

Overview & Sensor Geometry
--------------------------

Autonomous UAV navigation in complex urban canyons requires real-time sensing of building facades and vertical structures. The `VectorizedLiDAR` digital twin equips each UAV with a horizontal radial array of rangefinding beams spanning a full :math:`360^\circ` azimuth.

.. math::

   \theta_k = \frac{2\pi k}{K}, \quad k \in \{0, 1, \dots, K-1\}

where :math:`K` is the number of azimuth beams (typically 8, 16, or 32).

Ray-Marching Over ITU-R P.1410 3D Heightmaps
--------------------------------------------

Unlike mesh-based ray-tracing which is computationally intensive on CPU, `VectorizedLiDAR` implements a tensorized **ray-marching algorithm** directly against `VectorizedUrbanMap` 2.5D/3D building elevation matrices:

1. **Radial Sample Points**: For each beam :math:`k` extending from UAV position :math:`(x_i, y_i, z_i)`, sample coordinates are evaluated at :math:`S` discrete radial step fractions:

.. math::

   x_s = x_i + s \cdot d_{\max} \cos(\theta_k), \quad y_s = y_i + s \cdot d_{\max} \sin(\theta_k), \quad s \in \left\{\frac{1}{S}, \frac{2}{S}, \dots, 1.0\right\}

2. **Heightmap Collision Condition**: Obstacle intersection occurs whenever the UAV altitude :math:`z_i` falls at or below the local building height:

.. math::

   \text{Hit}(s) = \mathbb{I}\left( z_i \le H_{\text{map}}\left(\lfloor x_s \rfloor, \lfloor y_s \rfloor\right) \right)

3. **Range Normalization**: The normalized distance reading :math:`d_k \in [0.0, 1.0]` is determined by the nearest hit distance:

.. math::

   d_k = \min \left\{ s \mid \text{Hit}(s) = 1 \right\}

If no building is intersected within maximum range :math:`d_{\max}`, :math:`d_k = 1.0` (unobstructed clearance).

Sensor Configuration Parameters
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Default
     - Unit
   * - `num_beams`
     - Number of radial azimuth beams
     - 8
     - integer
   * - `max_range`
     - Maximum sensor detection range
     - 60.0
     - meters
   * - `n_steps`
     - Ray-marching discretization intervals
     - 15
     - integer
   * - `device`
     - PyTorch execution device (CPU / CUDA)
     - `cpu`
     - device

Observation Space Integration
-----------------------------

In the `uav_lidar_navigation` scenario and POMDP control architectures, the LiDAR vector :math:`\mathbf{d}_i \in \mathbb{R}^K` is directly concatenated into the decentralized observation vector:

.. math::

   \mathbf{o}_i = \big[ \mathbf{p}_i, \; \mathbf{v}_i, \; \mathbf{p}_{\text{target}} - \mathbf{p}_i, \; \mathbf{d}_i \big]^T \in \mathbb{R}^{9 + K}

This provides reinforcement learning agents with local geometric awareness without requiring centralized global heightmap knowledge.

Python Usage Example
--------------------

.. code-block:: python

   import torch
   from urbanmarl.models.urban_map import VectorizedUrbanMap
   from urbanmarl.models.lidar import VectorizedLiDAR

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # 1. Create procedural 3D urban terrain
   urban_map = VectorizedUrbanMap(
       batch_size=32,
       volume_size=(500, 500, 50),
       device=device,
   )

   # 2. Initialize 8-beam LiDAR sensor with 60m maximum range
   lidar = VectorizedLiDAR(
       num_beams=8,
       max_range=60.0,
       n_steps=15,
       device=device,
   )

   # 3. UAV coordinates: shape (batch_size=32, num_uavs=5, 3)
   uav_pos = torch.zeros(32, 5, 3, device=device)
   uav_pos[..., 2] = 25.0  # flying at altitude 25m

   # 4. Perform parallel ray-marching scans
   # Output shape: (32, 5, 8) with values in [0.0, 1.0]
   lidar_readings = lidar.scan(uav_pos, urban_map)

   print(f"LiDAR scan tensor shape: {lidar_readings.shape}")
   print(f"UAV 0 Beam Clearances: {lidar_readings[0, 0].cpu().numpy()}")
