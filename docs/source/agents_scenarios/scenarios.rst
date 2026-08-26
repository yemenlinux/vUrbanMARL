Urban Scenario Benchmarks
=========================

**UrbanMARL** includes five standard urban scenario benchmarks inheriting from `UrbanScenario`:


Scenario Overview
-----------------

======================= =========================== ===================================================
Scenario Key            Scenario Class              Primary Objective
======================= =========================== ===================================================
`uav_navigation`        `NavigationScenario`        3D UAV path planning & building collision avoidance
`uav_ue_los`            `UavUeLosScenario`          Maintain dynamic LoS wireless links to mobile UEs
`uavmec_offloading`     `UAVMECScenario`            Joint trajectory, offloading & MEC delay reduction
`coverage`              `CoverageScenario`          Maximize 3D geospatial mmWave area coverage
`mec_offloading`        `MecOffloadingScenario`     MEC task offloading in stationary urban layouts
======================= =========================== ===================================================

1. UAV Navigation (`uav_navigation`)
------------------------------------

- **Goal**: UAV swarms navigate through complex 3D urban building terrain from starting coordinates to target destination coordinates while avoiding building collisions.
- **Reward Function**:

.. math::

   R_i = -\Delta d_{\text{target}} - c_{\text{collision}} \cdot \mathbb{I}_{\text{collision}} + r_{\text{goal}} \cdot \mathbb{I}_{\text{reached}}

2. UAV-UE LoS Link Maintenance (`uav_ue_los`)
----------------------------------------------

- **Goal**: UAV swarms adjust 3D positions dynamically to maintain unblocked Line-of-Sight (LoS) links to moving ground UEs.
- **Reward Function**:

.. math::

   R_i = \sum_{j \in \text{UEs}} \text{LoS}(i, j) - c_{\text{interf}} \cdot \text{Interference}_i

3. UAV-MEC Task Offloading (`uavmec_offloading`)
-------------------------------------------------

- **Goal**: Jointly optimize UAV 3D flight trajectories, task offloading assignments, and MEC server queuing allocations to minimize total execution latency and energy consumption.
- **Reward Function**:

.. math::

   R_i = -\left( w_{\text{delay}} \cdot T_{\text{total}} + w_{\text{energy}} \cdot E_{\text{uav}} \right)
