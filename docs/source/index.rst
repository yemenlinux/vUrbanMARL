UrbanMARL Documentation
=======================

.. image:: resources/fcit_logo.png
   :height: 40px
   :alt: Faculty of Computer and Information Technology

.. image:: resources/su_logo.png
   :height: 40px
   :alt: Sana'a University

**UrbanMARL** is a scalable, GPU-vectorized multi-agent reinforcement learning (MARL) framework and digital twin platform for **6G geospatial radio environments, Unmanned Aerial Vehicle (UAV) swarms, Mobile Edge Computing (MEC) networks, and physical layer security**.

Built natively on **TorchRL**, **PyTorch**, **TensorDict**, and **BenchMARL**, UrbanMARL vectorizes 3D spatial ray-casting, ITU-R P.1410 urban map procedural generation, mmWave radio propagation, kinematic mobility models, and M/M/c queuing dynamics across hundreds of concurrent environments.


.. toctree::
   :maxdepth: 2
   :caption: Documentation Navigation:

   getting_started/index
   digital_twins/index
   agents_scenarios/index
   security/index
   api/index

Project Modules & Features
---------------------------

- **Geospatial 3D Digital Twin**: Procedural ITU-R P.1410 urban terrain parameterized by building coverage (:math:`\alpha`), density (:math:`\beta`), and height distribution (:math:`\gamma`). Includes GPU-accelerated ray-casting for Line-of-Sight (LoS) and building collision detection.
- **Radio Network Digital Twin**: Vectorized 29 GHz mmWave channel modeling Friis path loss, LoS/NLoS attenuation shifts, interference, SINR, and Shannon channel capacity.
- **MEC Queue Digital Twin**: Pure PyTorch M/M/c queuing system computing server utilization, queue lengths, task waiting times, and offloading execution delays.
- **Kinematic & Mobility Twin**: 3D motion models for UAVs and dynamic ground User Equipments (UEs).
- **Physical Layer & Network Security**: Frameworks for physical layer security (PLS), anti-jamming, adversarial agent detection, and privacy-preserving task offloading.
- **BenchMARL & TorchRL Integration**: Native support for BenchMARL task APIs (`UrbanEnvTask`). Compatible with SOTA MARL algorithms (MAPPO, MADDPG, MASAC, IPPO, IDDPG, ISAC, QMIX).

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
