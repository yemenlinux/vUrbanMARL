import pytest
import torch
from urbanmarl.models.mec_queue import VectorizedMECQueue


@pytest.fixture
def mec_queue():
    """Provides a VectorizedMECQueue instance targeting CPU device."""
    device = torch.device("cpu")
    return VectorizedMECQueue(device=device, max_cores=64)


def test_mec_queue_init(mec_queue):
    """
    Verifies VectorizedMECQueue initialization and log-factorial precomputation array.
    """
    assert mec_queue.max_cores == 64
    assert isinstance(mec_queue.log_fact, torch.Tensor)
    assert mec_queue.log_fact.shape == (65,)
    # log(1!) = 0
    assert torch.isclose(mec_queue.log_fact[1], torch.tensor(0.0))


def test_compute_delays_output_dictionary(mec_queue):
    """
    Tests that compute_delays returns a dictionary with all required queue metrics
    ('utilization', 'avg_queue_length', 'avg_waiting_time', 'avg_system_time')
    and expected tensor shapes (B, N).
    """
    batch_size = 3
    num_uavs = 4

    arrival_rates = torch.tensor([[5.0, 10.0, 15.0, 20.0],
                                  [2.0,  4.0,  6.0,  8.0],
                                  [1.0,  2.0,  3.0,  4.0]])  # (B, N)
    service_rates = torch.full((batch_size, num_uavs), 10.0)  # (B, N)
    num_cores = torch.full((batch_size, num_uavs), 4, dtype=torch.long)  # (B, N)

    metrics = mec_queue.compute_delays(arrival_rates, service_rates, num_cores)

    required_keys = {"utilization", "avg_queue_length", "avg_waiting_time", "avg_system_time"}
    assert set(metrics.keys()) == required_keys

    for key, tensor in metrics.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (batch_size, num_uavs)
        assert not torch.isnan(tensor).any()


def test_stable_traffic_metrics(mec_queue):
    """
    Verifies queue metrics calculation for stable traffic conditions (rho < 0.99),
    ensuring average system time is greater than pure service time (1/mu).
    """
    batch_size = 1
    num_uavs = 1

    arrival_rates = torch.tensor([[10.0]])
    service_rates = torch.tensor([[20.0]])
    num_cores = torch.tensor([[2]], dtype=torch.long)

    metrics = mec_queue.compute_delays(arrival_rates, service_rates, num_cores)

    rho = metrics["utilization"].item()
    w_s = metrics["avg_system_time"].item()
    w_q = metrics["avg_waiting_time"].item()

    assert 0.0 < rho < 0.99
    assert w_q >= 0.0
    assert w_s > (1.0 / service_rates.item())


def test_unstable_traffic_penalty(mec_queue):
    """
    Tests that unstable traffic conditions (rho >= 0.99) trigger the instability penalty (1e6)
    for queue length and system time metrics.
    """
    batch_size = 1
    num_uavs = 1

    # Overload: arrival rate exceeds service capacity (lambda > c * mu)
    arrival_rates = torch.tensor([[100.0]])
    service_rates = torch.tensor([[10.0]])
    num_cores = torch.tensor([[2]], dtype=torch.long)  # c * mu = 20 < 100

    metrics = mec_queue.compute_delays(arrival_rates, service_rates, num_cores)

    assert metrics["avg_queue_length"].item() == 1e6
    assert metrics["avg_system_time"].item() == 1e6


def test_multicore_scaling(mec_queue):
    """
    Verifies that increasing available processor cores reduces queue utilization and waiting times.
    """
    batch_size = 1
    num_uavs = 1

    arrival_rates = torch.tensor([[15.0]])
    service_rates = torch.tensor([[10.0]])

    single_core = torch.tensor([[2]], dtype=torch.long)
    multi_core = torch.tensor([[4]], dtype=torch.long)

    metrics_single = mec_queue.compute_delays(arrival_rates, service_rates, single_core)
    metrics_multi = mec_queue.compute_delays(arrival_rates, service_rates, multi_core)

    assert metrics_multi["utilization"].item() < metrics_single["utilization"].item()
    assert metrics_multi["avg_waiting_time"].item() < metrics_single["avg_waiting_time"].item()
