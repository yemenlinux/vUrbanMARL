"""UrbanMARL Advanced 3D mmWave Channel Model.

Implements 3GPP TR 38.901 compliant wireless propagation modeling:

1. 3D Directional Antenna Radiation Patterns:
   - Elevation beam attenuation: A_V(theta) = -min(12 * ((theta - theta_tilt) / theta_3dB)^2, SLA_V)
   - Azimuth beam attenuation: A_H(phi) = -min(12 * (phi / phi_3dB)^2, A_max)
   - Combined 3D antenna gain: G(theta, phi) = G_max - min(-(A_V + A_H), A_max)
2. Elevation-Dependent Rician / Rayleigh Small-Scale Fading:
   - Elevation-dependent Rician K-factor for LoS paths.
   - Rayleigh fading for NLoS paths.
"""

from typing import Optional, Union

import torch


class AdvancedChannelModel:
    """3GPP compliant 3D directional wireless channel propagation model.

    Attributes:
        frequency_ghz (float): Carrier frequency in GHz.
        bandwidth (float): Transmission bandwidth in Hz.
        g_max_dbi (float): Maximum boresight antenna gain in dBi.
        theta_3db (float): Vertical half-power 3dB beamwidth in degrees.
        phi_3db (float): Horizontal half-power 3dB beamwidth in degrees.
        noise_figure_db (float): Receiver noise figure in dB.
    """

    def __init__(
        self,
        frequency_ghz: float = 29.0,
        bandwidth_hz: float = 10e6,
        g_max_dbi: float = 15.0,
        theta_3db_deg: float = 65.0,
        phi_3db_deg: float = 65.0,
        noise_figure_db: float = 7.0,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initializes the advanced channel model.

        Args:
            frequency_ghz (float): Carrier frequency. Defaults to 29.0 GHz.
            bandwidth_hz (float): Bandwidth. Defaults to 10 MHz.
            g_max_dbi (float): Antenna gain in dBi. Defaults to 15.0.
            theta_3db_deg (float): Vertical 3dB beamwidth. Defaults to 65 deg.
            phi_3db_deg (float): Horizontal 3dB beamwidth. Defaults to 65 deg.
            noise_figure_db (float): Receiver noise figure. Defaults to 7.0 dB.
            device (Union[torch.device, str]): PyTorch compute device.
        """
        self.frequency_ghz = frequency_ghz
        self.bandwidth = bandwidth_hz
        self.g_max_dbi = g_max_dbi
        self.theta_3db = theta_3db_deg
        self.phi_3db = phi_3db_deg
        self.noise_figure_db = noise_figure_db
        self.device = torch.device(device)

        k_b = 1.380649e-23
        t_k = 290.0
        thermal_noise = k_b * t_k * self.bandwidth
        self.noise_power = thermal_noise * (10 ** (self.noise_figure_db / 10.0))

        c = 299792458.0
        self.wavelength = c / (self.frequency_ghz * 1e9)

    def compute_3d_antenna_gain(
        self, tx_pos: torch.Tensor, rx_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes 3D directional antenna radiation pattern gain G(theta, phi) in linear scale.

        Args:
            tx_pos (torch.Tensor): UAV transmitter coordinates of shape (..., N, 3).
            rx_pos (torch.Tensor): UE receiver coordinates of shape (..., M, 3).

        Returns:
            torch.Tensor: Linear antenna gain array of shape (..., N, M).
        """
        # Relative vectors: shape (..., N, M, 3)
        diff = rx_pos.unsqueeze(-3) - tx_pos.unsqueeze(-2)
        d_xy = torch.clamp(torch.norm(diff[..., :2], dim=-1), min=0.1)

        # Elevation angle theta off nadir (downward pointing): theta = atan2(d_xy, -dz)
        dz = diff[..., 2]  # typically negative from UAV to ground
        theta_rad = torch.atan2(d_xy, -dz)
        theta_deg = theta_rad * (180.0 / torch.pi)

        # Horizontal azimuth angle phi
        phi_rad = torch.atan2(diff[..., 1], diff[..., 0])
        phi_deg = phi_rad * (180.0 / torch.pi)

        # 3GPP TR 38.901 antenna pattern
        a_v = -torch.clamp(12.0 * ((theta_deg / self.theta_3db) ** 2), max=30.0)
        a_h = -torch.clamp(12.0 * ((phi_deg / self.phi_3db) ** 2), max=30.0)

        # Combined attenuation: -min(-(A_V + A_H), A_max)
        combined_attenuation = torch.clamp(a_v + a_h, min=-30.0)
        gain_dbi = self.g_max_dbi + combined_attenuation
        gain_linear = 10.0 ** (gain_dbi / 10.0)

        return gain_linear

    def compute_data_rates(
        self,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_power: Union[float, torch.Tensor] = 2.0,
        los_mask: Optional[torch.Tensor] = None,
        include_fading: bool = False,
    ) -> torch.Tensor:
        """Computes Shannon data rates with 3D directional antenna patterns and fading.

        Args:
            tx_pos (torch.Tensor): UAV 3D coordinates (..., N, 3).
            rx_pos (torch.Tensor): UE 3D coordinates (..., M, 3).
            tx_power (Union[float, torch.Tensor]): Transmit power in Watts.
            los_mask (Optional[torch.Tensor]): Boolean LoS tensor of shape (..., N, M).
            include_fading (bool): Whether to sample stochastic small-scale fading.

        Returns:
            torch.Tensor: Achievable transmission data rate in bits per second (..., N, M).
        """
        diff = rx_pos.unsqueeze(-3) - tx_pos.unsqueeze(-2)
        distances = torch.clamp(torch.norm(diff, dim=-1), min=1.0)

        # Free space path loss: (4 * pi * d / lambda)^2
        fspl = (4.0 * torch.pi * distances / self.wavelength) ** 2

        if los_mask is None:
            los_mask = torch.ones_like(distances, dtype=torch.bool)

        pl_exponent = torch.where(los_mask, 2.0, 3.5)
        pathloss = fspl * (distances ** (pl_exponent - 2.0))

        # 3D Directional antenna gain
        antenna_gain = self.compute_3d_antenna_gain(tx_pos, rx_pos)

        # Small scale fading power gain
        if include_fading:
            # Elevation angle in [0, pi/2]
            dz = torch.abs(diff[..., 2])
            d_xy = torch.norm(diff[..., :2], dim=-1)
            elev_rad = torch.atan2(dz, torch.clamp(d_xy, min=0.1))
            k_factor = 1.0 + 15.0 * (elev_rad / (torch.pi / 2.0))

            # Rician fading power
            s = torch.sqrt(k_factor / (k_factor + 1.0))
            sigma = torch.sqrt(0.5 / (k_factor + 1.0))
            h_real = s + sigma * torch.randn_like(distances)
            h_imag = sigma * torch.randn_like(distances)
            rician_gain = h_real**2 + h_imag**2

            # Rayleigh fading power
            rayleigh_gain = -torch.log(
                torch.clamp(torch.rand_like(distances), min=1e-6)
            )

            fading_gain = torch.where(los_mask, rician_gain, rayleigh_gain)
        else:
            fading_gain = 1.0

        if isinstance(tx_power, torch.Tensor):
            p_tx = (
                tx_power.unsqueeze(-1) if tx_power.ndim < distances.ndim else tx_power
            )
        else:
            p_tx = tx_power

        rx_power = (p_tx * antenna_gain * fading_gain) / pathloss
        sinr = rx_power / self.noise_power
        data_rates = self.bandwidth * torch.log2(1.0 + sinr)

        return data_rates
