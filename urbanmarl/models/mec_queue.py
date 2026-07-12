import torch

class VectorizedMECQueue:
    """
    Tensor-accelerated M/M/c queuing theory model.
    Computes utilization, wait times, and system delays in pure PyTorch.
    """
    def __init__(self, device: torch.device, max_cores: int = 64):
        self.device = device
        self.max_cores = max_cores
        
        # Precompute log-factorials for the summation series: ln(n!) = lgamma(n + 1)
        n_vals = torch.arange(max_cores + 1, device=device, dtype=torch.float32)
        self.log_fact = torch.lgamma(n_vals + 1.0)

    def compute_delays(self, arrival_rates: torch.Tensor, service_rates: torch.Tensor, num_cores: torch.Tensor) -> dict:
        """
        Computes system delays for a batch of UAVs.
        arrival_rates (lambda): shape (B, N)
        service_rates (mu): shape (B, N)
        num_cores (c): shape (B, N)
        """
        B, N = arrival_rates.shape
        
        # Avoid division by zero
        mu_safe = torch.clamp(service_rates, min=1e-6)
        c_safe = torch.clamp(num_cores.float(), min=1.0)
        lam_safe = torch.clamp(arrival_rates, min=1e-6)
        
        # Traffic intensity (rho)
        a = lam_safe / mu_safe
        rho = a / c_safe
        
        # Heavily loaded penalty masking (rho >= 1 is unstable in M/M/c)
        stable_mask = rho < 0.99
        rho_clamped = torch.clamp(rho, max=0.99)
        
        # 1. Compute Summation term for P0
        # Expand shapes for sum: (B, N, max_cores)
        a_exp = a.unsqueeze(2).expand(B, N, self.max_cores)
        n_tensor = torch.arange(self.max_cores, device=self.device).view(1, 1, self.max_cores).expand(B, N, self.max_cores)
        
        # Mask out terms where n >= c
        c_exp = c_safe.unsqueeze(2)
        valid_n_mask = n_tensor < c_exp
        
        # Compute terms: (a^n) / n! using log space for numerical stability
        log_terms = n_tensor * torch.log(a_exp + 1e-9) - self.log_fact[:self.max_cores].view(1, 1, self.max_cores)
        sum_terms = torch.exp(log_terms) * valid_n_mask.float()
        sum1 = torch.sum(sum_terms, dim=2)
        
        # 2. Compute the residual term for P0
        c_idx = c_safe.long()
        # Gather the log factorial for the specific c of each UAV
        log_fact_c = torch.gather(self.log_fact, 0, c_idx.flatten()).view(B, N)
        log_term_c = c_safe * torch.log(a + 1e-9) - log_fact_c
        sum2 = (torch.exp(log_term_c) / (1.0 - rho_clamped))
        
        # 3. Probabilities and Erlang C
        p0 = 1.0 / (sum1 + sum2 + 1e-9)
        erlang_c = p0 * sum2
        
        # 4. Queue metrics
        l_q = erlang_c * rho_clamped / (1.0 - rho_clamped + 1e-9)
        w_q = l_q / lam_safe
        w_s = w_q + (1.0 / mu_safe)
        
        # Apply instability penalty
        l_q = torch.where(stable_mask, l_q, torch.tensor(1e6, device=self.device))
        w_s = torch.where(stable_mask, w_s, torch.tensor(1e6, device=self.device))
        
        return {
            "utilization": rho,
            "avg_queue_length": l_q,
            "avg_waiting_time": w_q,
            "avg_system_time": w_s
        }
