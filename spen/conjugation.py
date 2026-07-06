"""ConjugationBridge: horizontal plasmid transfer between host cells.

In biology, bacteria extend a pilus (conjugation bridge) to transfer plasmids.
Here, high-fitness hosts push their best plasmids to paired recipients, who
evaluate and optionally retain them.
"""

from typing import Optional
import torch

from .host import HostCell
from .plasmid import Plasmid


class ConjugationBridge:
    """Manages plasmid transfer between pairs of host cells.

    The bridge is quorum-gated—only active during Conjugation Burst phases
    when the global quorum sensor threshold is exceeded.

    Transfer flow:
        1. Donor (high fitness) selects top-k plasmids
        2. Plasmids are cloned and pushed to recipient
        3. Recipient evaluates each plasmid's gain on its local data
        4. Plasmids with gain > threshold are permanently retained
    """

    def __init__(
        self,
        gain_threshold: float = 0.01,
        top_k_donor: int = 2,
    ):
        """
        Args:
            gain_threshold: Minimum ΔLoss for a recipient to retain a plasmid.
            top_k_donor: Number of best plasmids a donor pushes per transfer.
        """
        self.gain_threshold = gain_threshold
        self.top_k_donor = top_k_donor
        self.transfer_count: int = 0
        self.successful_transfers: int = 0

    def transfer(
        self,
        donor: HostCell,
        recipient: HostCell,
        x_eval: torch.Tensor,
        y_eval: torch.Tensor,
        loss_fn: torch.nn.Module,
    ) -> list[Plasmid]:
        """Execute a single plasmid transfer from donor to recipient.

        Args:
            donor: Source host (higher fitness).
            recipient: Destination host.
            x_eval: Evaluation inputs for gain computation.
            y_eval: Evaluation targets.
            loss_fn: Loss function for gain computation.

        Returns:
            List of plasmids that were successfully retained by the recipient.
        """
        accepted = []

        for plasmid in donor.best_plasmids(self.top_k_donor):
            gain = recipient.evaluate_plasmid_gain(
                plasmid, x_eval, y_eval, loss_fn
            )

            self.transfer_count += 1

            if gain > self.gain_threshold:
                # Clone and transfer
                clone = plasmid.clone()
                clone.origin_host_id = donor.host_id
                clone.transfer_history = plasmid.transfer_history.copy()
                clone.transfer_history.append(recipient.host_id)
                clone.vitality = 0.5  # Reset vitality for fair evaluation

                recipient.add_plasmid(clone)
                accepted.append(clone)
                self.successful_transfers += 1

        return accepted

    def burst_transfer(
        self,
        hosts: list[HostCell],
        fitness_scores: list[float],
        x_eval_batches: list[torch.Tensor],
        y_eval_batches: list[torch.Tensor],
        loss_fn: torch.nn.Module,
        donor_fraction: float = 0.3,
    ) -> int:
        """Batch transfer during a Conjugation Burst.

        Top donor_fraction of hosts (by fitness) push plasmids to all
        other hosts. This is the efficient quorum-gated mode.

        Args:
            hosts: All host cells.
            fitness_scores: Fitness score per host (higher = better).
            x_eval_batches: One evaluation batch per host.
            y_eval_batches: One evaluation target batch per host.
            loss_fn: Loss function.
            donor_fraction: Fraction of hosts that act as donors.

        Returns:
            Total number of successful plasmid transfers.
        """
        n = len(hosts)
        n_donors = max(1, int(n * donor_fraction))

        # Select top hosts as donors
        ranked = sorted(
            range(n), key=lambda i: fitness_scores[i], reverse=True
        )
        donor_indices = ranked[:n_donors]
        recipient_indices = ranked[n_donors:]

        total_accepted = 0
        for d_idx in donor_indices:
            donor = hosts[d_idx]
            for r_idx in recipient_indices:
                if r_idx == d_idx:
                    continue
                recipient = hosts[r_idx]
                accepted = self.transfer(
                    donor, recipient,
                    x_eval_batches[r_idx],
                    y_eval_batches[r_idx],
                    loss_fn,
                )
                total_accepted += len(accepted)

        return total_accepted

    @property
    def success_rate(self) -> float:
        """Fraction of transfer attempts that resulted in plasmid retention."""
        if self.transfer_count == 0:
            return 0.0
        return self.successful_transfers / self.transfer_count
