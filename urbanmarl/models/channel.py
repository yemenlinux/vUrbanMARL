import torch

class VectorizedChannelModel:
    """
    Computes batched path loss, received power, interference, and data rates 
    using vectorized structural tensor manipulations.
    """
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.freq_ghz = config.get("frequency_ghz", 29.0)
        self.bandwidth = config.get("g2a_bandwidth", 10e6)
        self.noise_figure_db = config.get("noise_figure_db", 7.0)
        
        # Compute default internal noise power directly inside native device bounds
        k_b = 1.380649e-23
        t_k = 290.0
        thermal_noise = k_b * t_k * self.bandwidth
        self.noise_power = thermal_noise * (10 ** (self.noise_figure_db / 10.0))

    def compute_data_rates(
        self, 
        tx_pos: torch.Tensor, 
        rx_pos: torch.Tensor, 
        tx_power: torch.Tensor, 
        los_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates achievable throughput mappings leveraging complete path calculations.
        tx_pos: (B, M, 3) [UEs], rx_pos: (B, N, 3) [UAVs]
        tx_power: (B, M)
        los_mask: (B, N, M) -> transposing mapping to match structural processing directions
        Returns: data_rates shape (B, N, M)
        """
        B, M, _ = tx_pos.shape
        _, N, _ = rx_pos.shape
        
        # Calculate pair distances: shape (B, N, M)
        diff = rx_pos.unsqueeze(2) - tx_pos.unsqueeze(1)
        distances = torch.clamp(torch.norm(diff, dim=-1), min=1.0)
        
        # Friis Transmission Equation Constants
        c = 299792458.0
        freq_hz = self.freq_ghz * 1e9
        wavelength = c / freq_hz
        
        # Free Space Pathloss components
        fspl = (4.0 * torch.pi * distances / wavelength) ** 2
        
        # Differentiate propagation quality matching discrete blockage attributes
        pl_exponent = torch.where(los_mask, 2.0, 3.5)
        pathloss = fspl * (distances ** (pl_exponent - 2.0))
        
        # Received Power Profiles: shape (B, N, M)
        rx_power = tx_power.unsqueeze(1) / pathloss
        
        # Approximate co-channel multi-source noise degradation
        total_interference = torch.sum(rx_power, dim=2, keepdim=True) - rx_power
        sinr = rx_power / (total_interference + self.noise_power)
        
        # Map exact channel capacity models directly to continuous execution frames
        capacity = self.bandwidth * torch.log2(1.0 + sinr)
        return capacity 
