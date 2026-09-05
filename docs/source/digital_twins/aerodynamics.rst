Rotary-Wing Aerodynamics Digital Twin
=======================================

The **Rotary-Wing Aerodynamics Digital Twin** (`VectorizedAerodynamics`) models high-fidelity flight dynamics and propulsion power consumption for multi-rotor unmanned aerial vehicles (UAVs) in 3D urban airspaces.

Theoretical Foundation
----------------------

UrbanMARL implements the analytical rotary-wing UAV power consumption model established by **Zeng, Zhang, and Lim** (IEEE Wireless Communications, 2016) and **Zeng and Zhang** (IEEE Transactions on Wireless Communications, 2017).

Unlike simplified linear models that assume power is strictly proportional to velocity (:math:`P \propto \|\mathbf{v}\|`), realistic rotary-wing aerodynamics exhibit a non-monotonic U-shaped power curve:
hovering requires substantial induced power, moderate forward speeds benefit from translational lift (reducing induced power), and high forward speeds suffer from steep parasite drag.

Propulsion Power Formulations
-----------------------------

Instantaneous aerodynamic propulsion power :math:`P_{\text{prop}}(\mathbf{v})` is calculated as the sum of four physical components:

.. math::

   P_{\text{prop}}(\mathbf{v}) = P_{\text{blade}}(v_h) + P_{\text{induced}}(v_h) + P_{\text{parasite}}(v_h) + P_{\text{vert}}(v_z)

where :math:`v_h = \sqrt{v_x^2 + v_y^2}` is the horizontal forward airspeed, and :math:`v_z` is the vertical climb/descent velocity.

1. Blade Profile Power
~~~~~~~~~~~~~~~~~~~~~~

Power required to overcome profile drag of the spinning rotor blades:

.. math::

   P_{\text{blade}}(v_h) = P_0 \left(1 + \frac{3 v_h^2}{U_{\text{tip}}^2}\right)

where :math:`P_0` is the blade profile power in hover, and :math:`U_{\text{tip}} = \Omega R` is the rotor blade tip speed.

2. Induced Power
~~~~~~~~~~~~~~~~

Power required to produce aerodynamic thrust counteracting gravity:

.. math::

   P_{\text{induced}}(v_h) = P_i \left(\sqrt{1 + \frac{v_h^4}{4 v_0^4}} - \frac{v_h^2}{2 v_0^2}\right)^{1/2}

where :math:`P_i` is the induced power in hover, and :math:`v_0` is the mean rotor induced velocity in hovering flight.

3. Parasite Drag Power
~~~~~~~~~~~~~~~~~~~~~~

Power required to overcome fuselage aerodynamic form drag through the air:

.. math::

   P_{\text{parasite}}(v_h) = \frac{1}{2} d_0 \rho s A v_h^3

where :math:`d_0` is fuselage drag ratio, :math:`\rho` is ambient air density (:math:`1.225\text{ kg/m}^3`), :math:`s` is rotor solidity, and :math:`A` is total rotor disc area.

4. Vertical Flight Power
~~~~~~~~~~~~~~~~~~~~~~~~

Gravitational work performed during climb or saved during descent:

.. math::

   P_{\text{vert}}(v_z) = m \cdot g \cdot v_z

where :math:`m` is UAV gross mass (kg) and :math:`g = 9.81\text{ m/s}^2`. Note that descent power is lower-bounded to account for motor idling limits during autorotation/descent.

Standard Aerodynamic Parameters
-------------------------------

Default physical parameters calibrated for standard commercial quadrotors (e.g., DJI Matrice 300 / Phantom 4):

.. list-table::
   :header-rows: 1
   :widths: 20 45 20 15

   * - Parameter
     - Physical Meaning
     - Default Value
     - Unit
   * - :math:`P_0`
     - Blade profile power in hover
     - 79.86
     - W
   * - :math:`P_i`
     - Induced power in hover
     - 88.63
     - W
   * - :math:`U_{\text{tip}}`
     - Rotor blade tip speed
     - 120.0
     - m/s
   * - :math:`v_0`
     - Mean induced velocity in hover
     - 4.03
     - m/s
   * - :math:`d_0`
     - Fuselage aerodynamic drag ratio
     - 0.60
     - dimensionless
   * - :math:`\rho`
     - Ambient air density at sea level
     - 1.225
     - :math:`\text{kg/m}^3`
   * - :math:`s`
     - Rotor disc solidity
     - 0.05
     - dimensionless
   * - :math:`A`
     - Total rotor disc area
     - 0.503
     - :math:`\text{m}^2`
   * - :math:`m`
     - Gross UAV mass
     - 2.0
     - kg

Battery Depletion Dynamics
--------------------------

The UAV onboard battery state :math:`E(t)` evolves over discrete time steps :math:`\Delta t`:

.. math::

   E(t+1) = \max\left(0, \; E(t) - \left[ P_{\text{prop}}(\mathbf{v}(t)) + P_{\text{MEC}}(t) + P_{\text{comm}}(t) \right] \cdot \Delta t \right)

where :math:`P_{\text{MEC}}` is computational power expended by the onboard mobile edge computing server, and :math:`P_{\text{comm}}` is radio transmission power.

Python Usage Example
--------------------

`VectorizedAerodynamics` operates directly on batched PyTorch tensors across CPU and GPU:

.. code-block:: python

   import torch
   from urbanmarl.models.aerodynamics import VectorizedAerodynamics

   # Initialize aerodynamic engine
   aero = VectorizedAerodynamics(
       p0=79.86,
       pi=88.63,
       u_tip=120.0,
       v0=4.03,
       mass_kg=2.0,
       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
   )

   # Batched 3D velocities: shape (batch_size=64, num_uavs=4, 3)
   velocities = torch.tensor([
       [0.0, 0.0, 0.0],    # Hover: ~168.5 W
       [10.0, 0.0, 0.0],   # Cruise (10 m/s): ~155.2 W (minimum power speed)
       [25.0, 0.0, 0.0],   # High-speed (25 m/s): ~310.8 W (parasite drag dominated)
       [5.0, 0.0, 2.0],    # Climbing at 2 m/s: hover + kinetic + gravitational work
   ], device=aero.device).unsqueeze(0).repeat(64, 1, 1)

   # Calculate instantaneous power in Watts: shape (64, 4, 1)
   power_watts = aero.compute_propulsion_power(velocities)

   # Integrate energy consumed over time step dt = 0.5s: shape (64, 4, 1)
   energy_joules = aero.compute_energy(velocities, dt=0.5)

   print(f"Hover Power:   {power_watts[0, 0].item():.2f} W")
   print(f"Cruise Power:  {power_watts[0, 1].item():.2f} W")
   print(f"High-Speed:    {power_watts[0, 2].item():.2f} W")
