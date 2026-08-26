Geospatial 3D Urban Map Digital Twin
=====================================

The **Geospatial Digital Twin** (`VectorizedUrbanMap`) simulates realistic 3D urban topographies based on the **ITU-R P.1410** propagation standard using Poisson point processes.

Procedural Map Parameters
-------------------------

Urban environments are parameterized by three fundamental ITU-R P.1410 
statistics:

- **Building Coverage Ratio** (:math:`\alpha`): Fraction of land area covered by buildings relative to total area.
- **Building Density Ratio** (:math:`\beta`): Mean number of buildings per unit area (buildings per :math:`\text{km}^2`).
- **Height Distribution Parameter** (:math:`\gamma`): Scale parameter of the Rayleigh distribution governing building height variation:

.. math::

   f(h) = \frac{h}{\gamma^2} \exp\left(-\frac{h^2}{2\gamma^2}\right), \quad h \ge 0

Batched Map Generation
----------------------

`VectorizedUrbanMap` generates tens to hundreds of 3D urban maps in parallel on CPU or GPU:

.. code-block:: python

   from urbanmarl.models.urban_map import VectorizedUrbanMap

   # Generate 16 parallel urban maps with ITU-R parameters
   maps = VectorizedUrbanMap(
       batch_size=16,          # num_envs
       volume_size=(500, 500, 50),  # (X, Y, Z) in meters
       device="cpu",           # or "cuda"
       map_margin=5,            # default: 5 meters
   )

GPU-Accelerated 3D Ray-Casting & Line-of-Sight (LoS)
----------------------------------------------------

The digital twin determines Line-of-Sight (LoS) state between any 3D transmitter (e.g. UAV at :math:`\mathbf{p}_A`) and 3D receiver (e.g. ground UE at :math:`\mathbf{p}_B`) by sampling line segment points and querying building height grids:

.. math::

   \text{LoS}(\mathbf{p}_A, \mathbf{p}_B) = \mathbb{I}\left( z(t) > H_{\text{map}}(x(t), y(t)) \;\; \forall t \in [0, 1] \right)

where :math:`\mathbf{p}(t) = (1-t)\mathbf{p}_A + t\mathbf{p}_B`.

Building Collision Detection
----------------------------

UAV safety is enforced via vectorized 3D boundary checking against grid building heights:

.. code-block:: python

   # Check collisions for N UAVs across B environments
   uav_pos = torch.randn(16, 5, 3, device="cpu")  # (batch, uavs, 3)
   collisions = maps.check_collision_batch(uav_pos)
