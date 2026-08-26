Kinematic & Mobility Digital Twin
===================================

The **Kinematic & Mobility Digital Twin** models motion dynamics, velocity limits, and trajectory constraints for 3D aerial UAV swarms and ground User Equipments (UEs).

UAV 3D Kinematics
-----------------

UAV state at discrete time step :math:`t` is defined by position :math:`\mathbf{p}_i(t) = [x_i, y_i, z_i]^T` and velocity :math:`\mathbf{v}_i(t) = [v_x, v_y, v_z]^T`.

Movement updates follow discrete-time kinematics:

.. math::

   \mathbf{p}_i(t+1) = \mathbf{p}_i(t) + \mathbf{v}_i(t) \cdot \Delta t

where action :math:`\mathbf{a}_i(t)` controls displacement or acceleration:

.. math::

   \mathbf{v}_i(t+1) = \text{clip}\left(\mathbf{v}_i(t) + \mathbf{a}_i(t) \cdot \Delta t, -v_{\max}, v_{\max}\right)

Operational Altitude & Boundary Constraints
-------------------------------------------

UAVs are strictly constrained within operational flight boundaries:

- **Altitude Box**: :math:`z_{\min} \le z_i(t) \le z_{\max}` (e.g. :math:`30\text{m} \le z_i \le 120\text{m}`).
- **Horizontal Bounds**: :math:`0 \le x_i(t) \le L_x`, :math:`0 \le y_i(t) \le L_y`.

Ground UE Mobility Models
-------------------------

Ground UEs move dynamically across 2D street grids using:

1. **Random Walk Mobility**: UEs change direction randomly at each step with speed :math:`v_{\text{ue}} \in [0.5, 2.0]\text{ m/s}`.
2. **Gauss-Markov Mobility**: Correlated velocity updates introducing temporal memory:

.. math::

   \mathbf{v}_{\text{ue}}(t+1) = \alpha_{\text{m}} \mathbf{v}_{\text{ue}}(t) + (1-\alpha_{\text{m}}) \bar{\mathbf{v}} + \sqrt{1-\alpha_{\text{m}}^2} \mathbf{n}(t)

3. **Street-Constrained Waypoint Navigation**: UEs navigate along street corridors between building blocks.
