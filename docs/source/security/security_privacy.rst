Physical Layer Security & Privacy
=================================

The **Security & Privacy Module** provides threat models and physical layer security (PLS) formulations for wireless UAV-MEC networks operating in hostile or non-cooperative 6G urban environments.

Threat Model & Eavesdropping
----------------------------

In the presence of passive eavesdroppers (e.g. unauthorized ground nodes or malicious aerial drones :math:`E`), the legitimate link secrecy capacity :math:`C_s` over the mmWave channel is defined as:

.. math::

   C_s = \left[ C_{\text{legitimate}} - C_{\text{eavesdropper}} \right]^+ = \left[ B \log_2(1 + \text{SINR}_{\text{UAV}\to\text{BS}}) - B \log_2(1 + \text{SINR}_{\text{UAV}\to E}) \right]^+

UAV swarms optimize 3D trajectories to maximize secrecy capacity by utilizing 3D urban building blockages to shield legitimate wireless signals from eavesdroppers.

Anti-Jamming Trajectory Optimization
------------------------------------

When active RF jammers emit high-power interference signals :math:`P_J`, UAV agents learn cooperative spatial beamforming and dynamic 3D positioning to move into jammer shadow zones created by tall urban buildings.

Privacy-Preserving Task Offloading
----------------------------------

To protect sensitive subscriber workload metadata (e.g., location traces, computation requirements), `UrbanMARL` supports:


1. **Differential Privacy Noise Injection**: Perturbing offloading request vectors before transmission.
2. **Federated MARL Training**: Keeping raw observations local to individual UAV agents while updating global policy weights securely.
