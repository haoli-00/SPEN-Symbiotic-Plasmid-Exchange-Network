"""Plasmid: self-contained, transferable neural sub-network module.

Inspired by bacterial plasmids—small circular DNA molecules that replicate
independently and transfer horizontally between bacteria via conjugation.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PlasmidConfig:
    """Configuration for a Plasmid module.

    Attributes:
        dim: Base dimension of the host network.
        rank: Low-rank decomposition rank (LoRA-style).
        dropout: Dropout rate applied within the plasmid.
        mutation_rate: Probability of applying Gaussian noise mutation per epoch.
        mutation_scale: Standard deviation of mutation noise.
        max_consecutive_mutations: Consecutive low-performing mutations before apoptosis.
    """
    dim: int = 128
    rank: int = 8
    dropout: float = 0.1
    mutation_rate: float = 0.3
    mutation_scale: float = 0.01
    max_consecutive_mutations: int = 3


class Plasmid(nn.Module):
    """A low-rank adapter module that acts as a transferable plasmid.

    Each plasmid is a self-contained LoRA-style module that can be:
    - Inserted into any host's forward pass via gating
    - Transferred horizontally between hosts (conjugation)
    - Independently mutated
    - Marked for apoptosis when vitality drops

    Architecture:
        y = A @ (x @ B)  where A: (dim, rank), B: (rank, dim)
        Equivalent to a rank-r update to the host's weight matrix.
    """

    _next_id: int = 0

    def __init__(self, config: PlasmidConfig):
        super().__init__()
        self.config = config
        self.plasmid_id = Plasmid._next_id
        Plasmid._next_id += 1

        # Low-rank matrices: A ∈ R^{dim×rank}, B ∈ R^{rank×dim}
        self.A = nn.Parameter(torch.randn(config.dim, config.rank) * 0.02)
        self.B = nn.Parameter(torch.randn(config.rank, config.dim) * 0.02)
        self.dropout = nn.Dropout(config.dropout)

        # --- Metadata (non-trainable) ---
        self.gain: float = 0.0           # ΔLoss when this plasmid is active
        self.vitality: float = 1.0       # Decaying fitness tracker
        self.mutation_count: int = 0     # Total mutations applied
        self.consecutive_bad_mutations: int = 0
        self.origin_host_id: Optional[int] = None
        self.transfer_history: list[int] = field(default_factory=list)
        self.niche_profile: Optional[torch.Tensor] = None  # Task type embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply plasmid transformation: dropout(A @ (x @ B))."""
        return self.dropout(self.A @ (x @ self.B))

    def mutate(self) -> None:
        """Apply Gaussian noise mutation to plasmid weights."""
        with torch.no_grad():
            self.A.add_(torch.randn_like(self.A) * self.config.mutation_scale)
            self.B.add_(torch.randn_like(self.B) * self.config.mutation_scale)
        self.mutation_count += 1

    def update_vitality(self, gamma: float, current_gain: float) -> None:
        """Exponential moving average of fitness.

        Args:
            gamma: Decay factor in [0, 1].
            current_gain: Most recent ΔLoss from this plasmid.
        """
        self.vitality = self.vitality * gamma + current_gain * (1 - gamma)
        self.gain = current_gain

    def should_apoptose(self, threshold: float, consecutive_rounds: int) -> bool:
        """Check if this plasmid should trigger apoptosis.

        Returns True if vitality < threshold for the required consecutive rounds.
        """
        if self.vitality < threshold:
            self.consecutive_bad_mutations += 1
        else:
            self.consecutive_bad_mutations = 0

        return self.consecutive_bad_mutations >= consecutive_rounds

    def fingerprint(self) -> torch.Tensor:
        """Compute a low-dimensional hash of plasmid weights for CRISPR storage."""
        with torch.no_grad():
            # Flatten and random-projection to fixed-size fingerprint
            flat = torch.cat([self.A.flatten(), self.B.flatten()])
            # Use first 32 dims as fingerprint (can be replaced with real hashing)
            if flat.shape[0] > 32:
                indices = torch.linspace(0, flat.shape[0] - 1, 32, dtype=torch.long)
                return flat[indices]
            return F.pad(flat, (0, 32 - flat.shape[0]))

    def clone(self) -> "Plasmid":
        """Create a deep copy (for transfer)."""
        new_plasmid = Plasmid(self.config)
        with torch.no_grad():
            new_plasmid.A.copy_(self.A)
            new_plasmid.B.copy_(self.B)
        new_plasmid.gain = self.gain
        new_plasmid.vitality = self.vitality
        new_plasmid.transfer_history = self.transfer_history.copy()
        new_plasmid.origin_host_id = self.origin_host_id
        return new_plasmid

    def extra_repr(self) -> str:
        return (
            f"id={self.plasmid_id}, rank={self.config.rank}, "
            f"gain={self.gain:.4f}, vitality={self.vitality:.4f}"
        )
