"""QuorumSensor: population-density-gated behavior switching.

In biology, bacteria release autoinducer molecules. When concentration
reaches a threshold, the population switches from individual to collective
behavior. Here, each host emits a signal proportional to its training
progress, and the global average gates conjugation bursts.
"""

import torch


class QuorumSensor:
    """Monitors population training progress and triggers conjugation bursts.

    Each host's autoinducer signal qᵢ ∈ [0, 1] grows with training epochs:
        qᵢ = σ(tᵢ / T_max)

    When the global mean q̄ exceeds θ_quorum, the population enters
    Conjugation Burst mode.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        t_max: int = 100,
        cooldown_epochs: int = 5,
    ):
        """
        Args:
            threshold: Quorum threshold θ—when q̄ > θ, burst triggers.
            t_max: Max epochs for sigmoid saturation.
            cooldown_epochs: Minimum epochs between consecutive bursts.
        """
        self.threshold = threshold
        self.t_max = t_max
        self.cooldown_epochs = cooldown_epochs

        self.epochs_since_last_burst: int = cooldown_epochs  # start ready
        self.burst_count: int = 0
        self.current_q_values: list[float] = []

    def update(self, epoch_numbers: list[int]) -> tuple[bool, float]:
        """Compute quorum state given each host's current training epoch.

        Args:
            epoch_numbers: Current training epoch for each host.

        Returns:
            (burst_triggered, global_q_mean)
        """
        self.epochs_since_last_burst += 1

        self.current_q_values = [
            torch.sigmoid(torch.tensor(t / self.t_max)).item()
            for t in epoch_numbers
        ]
        q_mean = sum(self.current_q_values) / len(self.current_q_values)

        if (
            q_mean > self.threshold
            and self.epochs_since_last_burst >= self.cooldown_epochs
        ):
            self.epochs_since_last_burst = 0
            self.burst_count += 1
            return True, q_mean

        return False, q_mean

    def host_signal(self, epochs_trained: int) -> float:
        """Get a single host's autoinducer level."""
        return torch.sigmoid(torch.tensor(epochs_trained / self.t_max)).item()
