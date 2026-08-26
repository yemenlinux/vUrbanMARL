"""UrbanMARL Radio Channel Model.

Vectorized mmWave radio channel propagation model calculating Friis path loss,
LoS/NLoS attenuation exponent shifts, interference, and Shannon channel capacity.
"""

import torch


class VectorizedChannelModel:
    """Computes batched path loss, received power, interference, and data rates.

    Attributes:
        device (torch.device): Compute device for PyTorch tensor calculations.
        freq_ghz (float): Carrier frequency in Gigahertz (e.g. 29.0 GHz mmWave).
        bandwidth (float): Transmit channel bandwidth in Hertz.
        noise_figure_db (float): Receiver noise figure in decibels.
        noise_power (float): Receiver thermal noise power in Watts.
    """

    def __init__(self, config: dict, device: torch.device) -> None:
        """Initializes the VectorizedChannelModel.

        Args:
            config (dict): Channel configuration dictionary containing optional keys:
                - 'frequency_ghz': carrier frequency in GHz (default: 29.0).
                - 'g2a_bandwidth': channel bandwidth in Hz (default: 10e6).
                - 'noise_figure_db': noise figure in dB (default: 7.0).
            device (torch.device): Compute device.
        """
        self.device = device
        self.freq_ghz = config.get("frequency_ghz", 29.0)
        self.bandwidth = config.get("g2a_bandwidth", 10e6)
        self.noise_figure_db = config.get("noise_figure_db", 7.0)

        k_b = 1.380649e-23
        t_k = 290.0
        thermal_noise = k_b * t_k * self.bandwidth
        self.noise_power = thermal_noise * (10 ** (self.noise_figure_db / 10.0))

    def compute_data_rates(
        self,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_power: torch.Tensor,
        los_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates achievable transmission data rates (bps) using Shannon capacity.

        Args:
            tx_pos (torch.Tensor): Transmitter positions (UEs) of shape (B, M, 3).
            rx_pos (torch.Tensor): Receiver positions (UAVs) of shape (B, N, 3).
            tx_power (torch.Tensor): Transmission power (Watts) of shape (B, M).
            los_mask (torch.Tensor): Line-of-sight boolean tensor of shape (B, N, M).

        Returns:
            torch.Tensor: Achievable channel capacity data rates in bps of shape (B, N, M).
        """
        B, M, _ = tx_pos.shape
        _, N, _ = rx_pos.shape

        diff = rx_pos.unsqueeze(2) - tx_pos.unsqueeze(1)
        distances = torch.clamp(torch.norm(diff, dim=-1), min=1.0)

        c = 299792458.0
        freq_hz = self.freq_ghz * 1e9
        wavelength = c / freq_hz

        fspl = (4.0 * torch.pi * distances / wavelength) ** 2

        pl_exponent = torch.where(los_mask, 2.0, 3.5)
        pathloss = fspl * (distances ** (pl_exponent - 2.0))

        rx_power = tx_power.unsqueeze(1) / pathloss

        total_interference = torch.sum(rx_power, dim=2, keepdim=True) - rx_power
        sinr = rx_power / (total_interference + self.noise_power)

        capacity = self.bandwidth * torch.log2(1.0 + sinr)
        return capacity
