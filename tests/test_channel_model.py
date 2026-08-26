import pytest
import torch
from urbanmarl.models.channel import VectorizedChannelModel


@pytest.fixture
def channel_config():
    """Provides standard configuration dictionary for VectorizedChannelModel."""
    return {
        "frequency_ghz": 28.0,
        "g2a_bandwidth": 20e6,
        "noise_figure_db": 6.0,
    }


@pytest.fixture
def channel_model(channel_config):
    """Instantiates a VectorizedChannelModel on CPU."""
    device = torch.device("cpu")
    return VectorizedChannelModel(config=channel_config, device=device)


def test_channel_model_init(channel_config, channel_model):
    """
    Verifies initialization of VectorizedChannelModel, ensuring configuration parameters
    and calculated noise power properties are set correctly.
    """
    assert channel_model.freq_ghz == channel_config["frequency_ghz"]
    assert channel_model.bandwidth == channel_config["g2a_bandwidth"]
    assert channel_model.noise_figure_db == channel_config["noise_figure_db"]
    assert isinstance(channel_model.noise_power, float)
    assert channel_model.noise_power > 0.0


def test_compute_data_rates_shape_and_dtype(channel_model):
    """
    Tests that compute_data_rates outputs a tensor matching expected shape (B, N, M),
    dtype torch.float32, and non-negative data rates when provided mock PyTorch input tensors.
    """
    batch_size = 4
    num_ues = 10
    num_uavs = 3

    # Mock PyTorch input tensors for UEs (transmitters) and UAVs (receivers)
    tx_pos = torch.rand(batch_size, num_ues, 3) * 500.0  # (B, M, 3)
    rx_pos = torch.rand(batch_size, num_uavs, 3) * 500.0  # (B, N, 3)
    rx_pos[..., 2] += 50.0  # Elevate UAVs
    tx_power = torch.full((batch_size, num_ues), 2.0)  # (B, M)
    los_mask = torch.rand(batch_size, num_uavs, num_ues) > 0.5  # (B, N, M)

    data_rates = channel_model.compute_data_rates(tx_pos, rx_pos, tx_power, los_mask)

    assert isinstance(data_rates, torch.Tensor)
    assert data_rates.shape == (batch_size, num_uavs, num_ues)
    assert data_rates.dtype == torch.float32
    assert (data_rates >= 0.0).all()
    assert not torch.isnan(data_rates).any()
    assert not torch.isinf(data_rates).any()


def test_los_vs_nlos_data_rates(channel_model):
    """
    Verifies that Line-of-Sight (LoS) path condition yields higher achievable data rates
    compared to Non-Line-of-Sight (NLoS) at identical positions and transmit power.
    """
    batch_size = 1
    num_ues = 1
    num_uavs = 1

    tx_pos = torch.tensor([[[100.0, 100.0, 0.0]]])  # (1, 1, 3)
    rx_pos = torch.tensor([[[100.0, 100.0, 50.0]]])  # (1, 1, 3)
    tx_power = torch.tensor([[1.0]])  # (1, 1)

    los_mask_true = torch.tensor([[[True]]])
    los_mask_false = torch.tensor([[[False]]])

    rate_los = channel_model.compute_data_rates(tx_pos, rx_pos, tx_power, los_mask_true)
    rate_nlos = channel_model.compute_data_rates(tx_pos, rx_pos, tx_power, los_mask_false)

    assert rate_los.item() > rate_nlos.item()


def test_zero_transmit_power(channel_model):
    """
    Verifies that zero transmit power results in zero achievable data rate.
    """
    batch_size = 2
    num_ues = 4
    num_uavs = 2

    tx_pos = torch.rand(batch_size, num_ues, 3) * 100.0
    rx_pos = torch.rand(batch_size, num_uavs, 3) * 100.0 + 20.0
    tx_power = torch.zeros(batch_size, num_ues)
    los_mask = torch.ones(batch_size, num_uavs, num_ues, dtype=torch.bool)

    data_rates = channel_model.compute_data_rates(tx_pos, rx_pos, tx_power, los_mask)

    assert torch.allclose(data_rates, torch.zeros_like(data_rates))


def test_distance_clamping_behavior(channel_model):
    """
    Tests that compute_data_rates safely handles co-located transmitter and receiver
    positions by clamping distance to a minimum threshold of 1.0 meter.
    """
    batch_size = 1
    num_ues = 1
    num_uavs = 1

    # Co-located UE and UAV positions
    tx_pos = torch.tensor([[[50.0, 50.0, 10.0]]])
    rx_pos = torch.tensor([[[50.0, 50.0, 10.0]]])
    tx_power = torch.tensor([[1.0]])
    los_mask = torch.tensor([[[True]]])

    data_rates = channel_model.compute_data_rates(tx_pos, rx_pos, tx_power, los_mask)

    assert not torch.isnan(data_rates).any()
    assert data_rates.item() > 0.0
