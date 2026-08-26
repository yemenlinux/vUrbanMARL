"""UrbanMARL Mobile Edge Computing (MEC) M/M/c Queue Model.

Pure PyTorch implementation of M/M/c queuing theory calculating server utilization,
queue lengths, waiting times, and total system processing delay across parallel environments.
"""

import torch


class VectorizedMECQueue:
    """PyTorch tensor M/M/c queuing model for edge servers.

    Attributes:
        device (torch.device): Compute device.
        max_cores (int): Maximum server processing core capacity.
        log_fact (torch.Tensor): Precomputed log-factorial array for numerical stability.
    """

    def __init__(self, device: torch.device, max_cores: int = 64) -> None:
        """Initializes the VectorizedMECQueue model.

        Args:
            device (torch.device): Compute device for PyTorch operations.
            max_cores (int): Maximum core count upper bound (default: 64).
        """
        self.device = device
        self.max_cores = max_cores

        n_vals = torch.arange(max_cores + 1, device=device, dtype=torch.float32)
        self.log_fact = torch.lgamma(n_vals + 1.0)

    def compute_delays(
        self,
        arrival_rates: torch.Tensor,
        service_rates: torch.Tensor,
        num_cores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Computes M/M/c queuing delays and utilization metrics for a batch of UAVs.

        Args:
            arrival_rates (torch.Tensor): Task arrival rates (lambda) of shape (B, N).
            service_rates (torch.Tensor): Service rates per core (mu) of shape (B, N).
            num_cores (torch.Tensor): Number of allocated server cores (c) of shape (B, N).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensors for:
                - 'utilization': server traffic intensity rho of shape (B, N).
                - 'avg_queue_length': expected number of queued tasks of shape (B, N).
                - 'avg_waiting_time': expected queue waiting time in seconds of shape (B, N).
                - 'avg_system_time': total system response time (wait + service) of shape (B, N).
        """
        B, N = arrival_rates.shape

        mu_safe = torch.clamp(service_rates, min=1e-6)
        c_safe = torch.clamp(num_cores.float(), min=1.0)
        lam_safe = torch.clamp(arrival_rates, min=1e-6)

        a = lam_safe / mu_safe
        rho = a / c_safe

        stable_mask = rho < 0.99
        rho_clamped = torch.clamp(rho, max=0.99)

        a_exp = a.unsqueeze(2).expand(B, N, self.max_cores)
        n_tensor = (
            torch.arange(self.max_cores, device=self.device)
            .view(1, 1, self.max_cores)
            .expand(B, N, self.max_cores)
        )

        c_exp = c_safe.unsqueeze(2)
        valid_n_mask = n_tensor < c_exp

        log_terms = n_tensor * torch.log(a_exp + 1e-9) - self.log_fact[
            : self.max_cores
        ].view(1, 1, self.max_cores)
        sum_terms = torch.exp(log_terms) * valid_n_mask.float()
        sum1 = torch.sum(sum_terms, dim=2)

        c_idx = c_safe.long()
        log_fact_c = torch.gather(self.log_fact, 0, c_idx.flatten()).view(B, N)
        log_term_c = c_safe * torch.log(a + 1e-9) - log_fact_c
        sum2 = torch.exp(log_term_c) / (1.0 - rho_clamped)

        p0 = 1.0 / (sum1 + sum2 + 1e-9)
        erlang_c = p0 * sum2

        l_q = erlang_c * rho_clamped / (1.0 - rho_clamped + 1e-9)
        w_q = l_q / lam_safe
        w_s = w_q + (1.0 / mu_safe)

        l_q = torch.where(
            stable_mask, l_q, torch.tensor(1e6, device=self.device)
        )
        w_s = torch.where(
            stable_mask, w_s, torch.tensor(1e6, device=self.device)
        )

        return {
            "utilization": rho,
            "avg_queue_length": l_q,
            "avg_waiting_time": w_q,
            "avg_system_time": w_s,
        }
