Agent Taxonomy & State Spaces
=============================

**UrbanMARL** supports heterogeneous multi-agent topologies comprising aerial autonomous agents, stationary network infrastructure, and dynamic ground entities.


Agent Taxonomy
--------------

1. **UAV Swarm Agents (`uav`)**:
   Autonomous 3D aerial agents navigating urban airspaces.
   - *Role*: Relay communication, area coverage, MEC task execution, trajectory optimization.
   - *Action Space*: Continuous 3D displacement/velocity vectors :math:`[v_x, v_y, v_z] \in [-1, 1]^3` or discrete directional movements.
   - *Observation Space*: 3D position :math:`[x, y, z]`, velocity, relative vectors to target/obstacles, LoS states, SINR readings, and local MEC queue status.

2. **Ground User Equipments (`ue`)**:
   Dynamic or static mobile subscribers requesting wireless data or offloading computational workloads.
   - *Role*: Target tracking endpoints, task generators, signal receivers.
   - *Observation Space*: 2D/3D ground position, requested data rate, task queue length.

3. **Base Station Nodes (`bs`)**:
   Static ground macro/micro base stations providing cellular connectivity and centralized MEC compute resources.

4. **MEC Server Nodes (`mec`)**:
   Multi-server edge computing clusters attached to UAVs or Base Stations.

Multi-Agent Group Mapping (`group_map`)
---------------------------------------

`UrbanMARL` environments organize agents into TorchRL agent groups via `group_map`:


.. code-block:: python

   # Example group_map for 3 UAVs and 5 UEs
   group_map = {
       "uav": ["uav_0", "uav_1", "uav_2"],
       "ue": ["ue_0", "ue_1", "ue_2", "ue_3", "ue_4"],
   }
