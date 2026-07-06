"""HostCell: neural network instance carrying a population of plasmids.

Each host trains independently on local data, evaluates its plasmid population,
and participates in quorum-gated conjugation events.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from .plasmid import Plasmid, PlasmidConfig


@dataclass
class HostConfig:
    """Configuration for a HostCell.

    Attributes:
        base_dim: Input/output dimension of the base network.
        hidden_dim: Hidden layer dimension.
        num_plasmids: Number of plasmids this host can carry.
        plasmid_capacity: Maximum plasmid slots (for endosymbiosis expansion).
        vitality_decay: Gamma for plasmid vitality EMA.
        apoptosis_threshold: Vitality below which apoptosis is considered.
        apoptosis_consecutive: Rounds of sub-threshold vitality to trigger apoptosis.
    """
    base_dim: int = 128
    hidden_dim: int = 256
    num_plasmids: int = 3
    plasmid_capacity: int = 6
    vitality_decay: float = 0.9
    apoptosis_threshold: float = 0.05
    apoptosis_consecutive: int = 3


class BaseNetwork(nn.Module):
    """Simple feed-forward base network that plasmids augment."""

    def __init__(self, config: HostConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.base_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.base_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HostCell(nn.Module):
    """A neural network host that carries and manages a population of plasmids.

    The host's forward pass is:
        y = base_network(x) + Σᵢ αᵢ · plasmidᵢ(projectᵢ(x))

    where αᵢ are learnable gating weights (initialized to small positive values).
    """

    _next_id: int = 0

    def __init__(self, host_config: HostConfig, plasmid_config: PlasmidConfig):
        super().__init__()
        self.host_id = HostCell._next_id
        HostCell._next_id += 1
        self.host_config = host_config
        self.plasmid_config = plasmid_config

        # Base network
        self.base_network = BaseNetwork(host_config)

        # Plasmids
        self.plasmids: list[Plasmid] = [
            Plasmid(plasmid_config) for _ in range(host_config.num_plasmids)
        ]
        for p in self.plasmids:
            p.origin_host_id = self.host_id

        # Projectors: map base dim to plasmid input dim
        self.projectors = nn.ModuleList([
            nn.Linear(host_config.base_dim, plasmid_config.dim)
            for _ in range(host_config.num_plasmids)
        ])

        # Gating weights αᵢ (learnable, initialized small)
        self.gate_logits = nn.Parameter(
            torch.zeros(host_config.num_plasmids) - 2.0  # softmax → ~uniform small
        )

        # Training progress tracker (for quorum sensing)
        self.epochs_trained: int = 0

        # Niche embedding in mycorrhizal space
        self.niche_embedding: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Host forward: base + gated plasmid contributions."""
        base_out = self.base_network(x)

        if not self.plasmids:
            return base_out

        gates = torch.softmax(self.gate_logits, dim=0)  # (num_plasmids,)

        plasmid_out = torch.zeros_like(base_out)
        for i, (plasmid, proj) in enumerate(zip(self.plasmids, self.projectors)):
            projected = proj(x)
            plasmid_out = plasmid_out + gates[i] * plasmid(projected)

        return base_out + plasmid_out

    def add_plasmid(self, plasmid: Plasmid) -> None:
        """Add a foreign plasmid (received via conjugation).

        Expands plasmid slots if under capacity. Plasmid is frozen initially
        and evaluated before being allowed to train.
        """
        if len(self.plasmids) >= self.host_config.plasmid_capacity:
            # Evict lowest-vitality plasmid
            worst_idx = min(range(len(self.plasmids)),
                            key=lambda i: self.plasmids[i].vitality)
            removed = self.plasmids.pop(worst_idx)
            self.projectors.__delitem__(worst_idx)
            # Adjust gate_logits
            self.gate_logits = nn.Parameter(
                torch.cat([
                    self.gate_logits[:worst_idx],
                    self.gate_logits[worst_idx + 1:]
                ])
            )

        plasmid.transfer_history.append(self.host_id)
        self.plasmids.append(plasmid)
        self.projectors.append(
            nn.Linear(self.host_config.base_dim, self.plasmid_config.dim)
        )
        # Expand gate_logits
        self.gate_logits = nn.Parameter(
            torch.cat([self.gate_logits, torch.zeros(1) - 2.0])
        )

    def remove_plasmid(self, index: int) -> Optional[Plasmid]:
        """Remove and return a plasmid (for apoptosis or transfer)."""
        if index < 0 or index >= len(self.plasmids):
            return None
        removed = self.plasmids.pop(index)
        self.projectors.__delitem__(index)
        self.gate_logits = nn.Parameter(
            torch.cat([
                self.gate_logits[:index],
                self.gate_logits[index + 1:]
            ])
        )
        return removed

    def evaluate_plasmid_gain(
        self, plasmid: Plasmid, x: torch.Tensor, y_true: torch.Tensor,
        loss_fn: nn.Module
    ) -> float:
        """Compute loss delta when this plasmid is active vs inactive.

        Returns negative value if plasmid improves performance (lower loss = better).
        This is the 'gain' metric used for vitality tracking and conjugation decisions.
        """
        with torch.no_grad():
            # Loss without plasmid
            y_base = self.base_network(x)
            loss_base = loss_fn(y_base, y_true).item()

            # Loss with plasmid
            proj = nn.Linear(self.host_config.base_dim, self.plasmid_config.dim)
            proj.load_state_dict(self.projectors[0].state_dict())  # use first proj
            plasmid_out = plasmid(proj(x))
            y_with = y_base + plasmid_out
            loss_with = loss_fn(y_with, y_true).item()

            # Negative delta = improvement
            return loss_base - loss_with

    def apoptosis_check(self) -> list[int]:
        """Check all plasmids for apoptosis. Returns indices of removed plasmids.

        Before removal, performs knowledge distillation: extracts the dominant
        direction of the dying plasmid's weight and adds it as a bias to the
        base network's last layer.
        """
        to_remove = []
        for i, p in enumerate(self.plasmids):
            if p.should_apoptose(
                self.host_config.apoptosis_threshold,
                self.host_config.apoptosis_consecutive
            ):
                self._distill_plasmid(p)
                to_remove.append(i)

        # Remove in reverse order to preserve indices
        for i in reversed(to_remove):
            self.remove_plasmid(i)

        return to_remove

    def _distill_plasmid(self, plasmid: Plasmid) -> None:
        """Distill essential knowledge from a dying plasmid into the base network.

        Extracts the dominant singular vector of A @ B and adds it as a
        small bias to the last linear layer of the base network.
        """
        with torch.no_grad():
            combined = plasmid.A @ plasmid.B  # (dim, dim)
            U, S, V = torch.svd(combined)
            # Take the dominant direction
            dominant = U[:, 0] * S[0] * 0.01  # scaled down

            # Add to last layer bias
            last_linear = self.base_network.net[-1]
            if last_linear.bias is not None:
                last_linear.bias.add_(dominant[:last_linear.bias.shape[0]])

    def best_plasmids(self, k: int = 2) -> list[Plasmid]:
        """Return top-k plasmids sorted by vitality."""
        sorted_plasmids = sorted(self.plasmids, key=lambda p: p.vitality, reverse=True)
        return sorted_plasmids[:k]

    def extra_repr(self) -> str:
        return f"host_id={self.host_id}, plasmids={len(self.plasmids)}/{self.host_config.plasmid_capacity}"
