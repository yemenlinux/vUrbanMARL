"""UrbanMARL High-Fidelity UAV Aerodynamic Propulsion Power Model.

Implements the rotary-wing UAV flight power consumption model from:
Zeng, Zhang, and Lim, "Wireless communications with unmanned aerial vehicles:
driving forces, key challenges, and options," IEEE Wireless Communications, 2016.
And:
Zeng and Zhang, "Energy-efficient UAV communication with trajectory optimization,"
IEEE Transactions on Wireless Communications, 2017.

Vectorized PyTorch implementation calculating instantaneous blade profile power,
induced power, parasite drag power, and vertical climb/descend power.
"""

from typing import Union

import torch


class VectorizedAerodynamics:
    """Calculates instantaneous aerodynamic power consumption for rotary-wing UAVs.

    Attributes:
        p0 (float): Blade profile power in hover (Watts).
        pi (float): Induced power in hover (Watts).
        u_tip (float): Tip speed of the rotor blade (m/s).
        v0 (float): Mean rotor induced velocity in hover (m/s).
        d0 (float): Fuselage drag ratio.
        rho (float): Air density (kg/m^3).
        s (float): Rotor solidity.
        rotor_area (float): Rotor disc area (m^2).
        mass_kg (float): UAV gross mass (kg).
    """

    def __init__(
        self,
        p0: float = 79.86,
        pi: float = 88.63,
        u_tip: float = 120.0,
        v0: float = 4.03,
        d0: float = 0.6,
        rho: float = 1.225,
        solidity: float = 0.05,
        rotor_area: float = 0.503,
        mass_kg: float = 2.0,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes aerodynamic parameters.

        Args:
            p0 (float): Blade profile power. Defaults to 79.86 W.
            pi (float): Induced power. Defaults to 88.63 W.
            u_tip (float): Rotor tip speed in m/s. Defaults to 120.0.
            v0 (float): Mean induced velocity in m/s. Defaults to 4.03.
            d0 (float): Fuselage drag ratio. Defaults to 0.6.
            rho (float): Air density in kg/m^3. Defaults to 1.225.
            solidity (float): Rotor solidity. Defaults to 0.05.
            rotor_area (float): Rotor disc area in m^2. Defaults to 0.503.
            mass_kg (float): UAV gross mass in kg. Defaults to 2.0.
            device (torch.device): PyTorch compute device.
        """
        self.p0 = p0
        self.pi = pi
        self.u_tip = u_tip
        self.v0 = v0
        self.d0 = d0
        self.rho = rho
        self.s = solidity
        self.rotor_area = rotor_area
        self.mass = mass_kg
        self.g = 9.81
        self.weight = mass_kg * self.g
        self.device = torch.device(device)

    def compute_propulsion_power(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Computes instantaneous aerodynamic propulsion power in Watts.

        Args:
            velocity (torch.Tensor): UAV 3D velocity vectors (..., 3) in m/s.

        Returns:
            torch.Tensor: Power dissipation tensor in Watts matching batch shapes (..., 1).
        """
        # Horizontal speed: v_h = sqrt(vx^2 + vy^2)
        v_h = torch.norm(velocity[..., :2], dim=-1, keepdim=True)
        v_z = velocity[..., 2:]

        # 1. Blade profile power: P_0 * (1 + 3 * v_h^2 / U_tip^2)
        p_blade = self.p0 * (1.0 + (3.0 * (v_h**2)) / (self.u_tip**2))

        # 2. Induced power: P_i * (sqrt(1 + v_h^4 / (4 * v0^4)) - v_h^2 / (2 * v0^2))^(1/2)
        term1 = torch.sqrt(1.0 + (v_h**4) / (4.0 * (self.v0**4)))
        term2 = (v_h**2) / (2.0 * (self.v0**2))
        p_induced = self.pi * torch.sqrt(torch.clamp(term1 - term2, min=1e-6))

        # 3. Parasite drag power: 0.5 * d0 * rho * s * A * v_h^3
        p_parasite = 0.5 * self.d0 * self.rho * self.s * self.rotor_area * (v_h**3)

        # 4. Vertical climb/descend power: m * g * v_z
        p_climb = self.weight * v_z

        total_power = p_blade + p_induced + p_parasite + p_climb
        # Aerodynamic power cannot drop below zero
        total_power = torch.clamp(total_power, min=0.0)

        return total_power
