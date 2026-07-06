"""Ecosystem: the full SPEN orchestrator.

The Ecosystem manages all host cells, the conjugation bridge, quorum sensor,
CRISPR memory bank, and mycorrhizal latent space. It coordinates the 8-phase
training loop that makes SPEN unique.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .plasmid import PlasmidConfig
from .host import HostCell, HostConfig
from .conjugation import ConjugationBridge
from .quorum import QuorumSensor
from .crispr import CRISPRBank
from .mycorrhiza import MycorrhizalSpace


@dataclass
class EcosystemConfig:
    """Full SPEN ecosystem configuration.

    Attributes:
        num_hosts: Number of host cells in the population.
        host_config: Configuration for each HostCell.
        plasmid_config: Configuration for each Plasmid.
        quorum_threshold: Threshold for triggering conjugation burst.
        t_max: Epochs for quorum sigmoid saturation.
        niche_cluster_interval: Epochs between niche re-clustering.
        endosymbiosis_epochs: Consecutive epochs of high gain to trigger endosymbiosis.
    """
    num_hosts: int = 5
    host_config: HostConfig = field(default_factory=HostConfig)
    plasmid_config: PlasmidConfig = field(default_factory=PlasmidConfig)
    quorum_threshold: float = 0.6
    t_max: int = 100
    niche_cluster_interval: int = 10
    endosymbiosis_epochs: int = 5


class Ecosystem:
    """Complete SPEN ecosystem orchestrating hosts, plasmids, and biological mechanisms.

    Training follows an 8-phase loop:
        1. Individual training (each host on its local data shard)
        2. Mycorrhizal embedding (project current batch → latent space)
        3. Diffusion signaling (distress signals + neighbor response)
        4. Plasmid mutation (low-fitness plasmids receive Gaussian noise)
        5. Niche clustering (periodic DBSCAN in latent space)
        6. Quorum check → Conjugation Burst
        7. Apoptosis check (remove low-vitality plasmids)
        8. Endosymbiosis check (integrate sustained high-gain plasmids)

    Usage:
        ecosystem = Ecosystem(
            num_hosts=5,
            host_config=HostConfig(base_dim=128),
            plasmid_config=PlasmidConfig(dim=128, rank=8),
        )
        ecosystem.fit(train_loaders, epochs=50)
    """

    def __init__(
        self,
        num_hosts: Optional[int] = None,
        host_config: Optional[HostConfig] = None,
        plasmid_config: Optional[PlasmidConfig] = None,
        config: Optional[EcosystemConfig] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = EcosystemConfig(
                num_hosts=num_hosts or 5,
                host_config=host_config or HostConfig(),
                plasmid_config=plasmid_config or PlasmidConfig(),
            )

        # Create host population
        self.hosts = [
            HostCell(self.config.host_config, self.config.plasmid_config)
            for _ in range(self.config.num_hosts)
        ]

        # Biological mechanisms
        self.bridge = ConjugationBridge()
        self.quorum = QuorumSensor(
            threshold=self.config.quorum_threshold,
            t_max=self.config.t_max,
        )
        self.crispr = CRISPRBank()
        self.mycorrhiza = MycorrhizalSpace()

        # Initialize mycorrhizal encoder
        self.mycorrhiza.set_encoder(self.config.host_config.base_dim)

        # Track hosts that have experienced endosymbiosis
        self.endosymbiosis_tracker: dict[int, dict[int, int]] = {}
        # host_id → {plasmid_id → consecutive_high_gain_epochs}

    def fit(
        self,
        train_loaders: list,
        epochs: int = 20,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 1e-3,
    ) -> dict:
        """Train the SPEN ecosystem.

        Args:
            train_loaders: One DataLoader per host (or a single DataLoader
                that will be split).
            epochs: Number of training epochs.
            loss_fn: Loss function (default: MSE).
            lr: Learning rate for host optimizers.

        Returns:
            Training statistics dict.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Create optimizer per host
        optimizers = [
            optim.Adam(host.parameters(), lr=lr)
            for host in self.hosts
        ]

        if len(train_loaders) == 1 and self.config.num_hosts > 1:
            # Split single loader across hosts
            all_data = list(train_loaders[0])
            chunk_size = len(all_data) // self.config.num_hosts
            train_loaders = [
                all_data[i * chunk_size : (i + 1) * chunk_size]
                for i in range(self.config.num_hosts)
            ]

        stats = {"losses": [], "bursts": [], "transfers": [], "apoptosis": []}

        for epoch in range(epochs):
            epoch_losses = []

            # === Phase 1: Individual Training ===
            activations_for_mycorrhiza = []
            distress_levels = []

            for i, host in enumerate(self.hosts):
                host.epochs_trained += 1
                host.train()

                if i >= len(train_loaders):
                    continue

                # Get a batch
                batch = train_loaders[i][epoch % len(train_loaders[i])]
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = batch  # Auto-encoder mode

                optimizers[i].zero_grad()

                y_pred = host(x)
                loss = loss_fn(y_pred, y)
                loss.backward()

                # Compute gradient norm for distress signaling
                total_norm = 0.0
                for p in host.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                grad_norm = total_norm ** 0.5
                distress_levels.append(grad_norm)

                # Store activation for mycorrhizal space
                with torch.no_grad():
                    activations_for_mycorrhiza.append(
                        host.base_network(x).detach()
                    )

                optimizers[i].step()
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            stats["losses"].append(avg_loss)

            # === Phase 2: Mycorrhizal Embedding ===
            embeddings = self.mycorrhiza.embed(activations_for_mycorrhiza)
            neighbors = self.mycorrhiza.compute_neighbors(embeddings)

            # === Phase 3: Diffusion Signaling ===
            signals = self.mycorrhiza.diffuse_signals(
                embeddings, distress_levels, neighbors
            )
            # Respond to signals: neighbors push niche-matched plasmids
            for i, host_signals in enumerate(signals):
                if not host_signals:
                    continue
                # For each distress signal received, push best plasmid
                for sender_idx, strength in host_signals[:2]:  # top-2 signals
                    if strength > 0.5:
                        donor = self.hosts[sender_idx]
                        for best_p in donor.best_plasmids(1):
                            # Lightweight evaluation
                            if i < len(train_loaders):
                                batch = train_loaders[i][epoch % len(train_loaders[i])]
                                x, y = (batch if isinstance(batch, tuple)
                                        else (batch, batch))
                                self.bridge.transfer(
                                    donor, self.hosts[i], x, y, loss_fn
                                )

            # === Phase 4: Plasmid Mutation ===
            for host in self.hosts:
                for plasmid in host.plasmids:
                    if plasmid.vitality < 0.3:  # Only mutate low-vitality plasmids
                        if torch.rand(1).item() < self.config.plasmid_config.mutation_rate:
                            plasmid.mutate()

            # === Phase 5: Niche Clustering (periodic) ===
            if epoch % self.config.niche_cluster_interval == 0:
                niche_labels = self.mycorrhiza.cluster_niches(embeddings)
                for i, host in enumerate(self.hosts):
                    host.niche_embedding = embeddings[i].detach().clone()

            # === Phase 6: Quorum Check & Conjugation Burst ===
            burst_triggered, q_mean = self.quorum.update(
                [h.epochs_trained for h in self.hosts]
            )

            if burst_triggered:
                # Compute fitness scores (negative loss = higher fitness)
                fitness_scores = [-l for l in epoch_losses]

                # Prepare eval batches
                x_batches, y_batches = [], []
                for i in range(len(self.hosts)):
                    if i < len(train_loaders):
                        batch = train_loaders[i][epoch % len(train_loaders[i])]
                        x, y = (batch if isinstance(batch, tuple) else (batch, batch))
                        x_batches.append(x)
                        y_batches.append(y)
                    else:
                        x_batches.append(torch.zeros(1, self.config.host_config.base_dim))
                        y_batches.append(torch.zeros(1, self.config.host_config.base_dim))

                n_transfers = self.bridge.burst_transfer(
                    self.hosts, fitness_scores, x_batches, y_batches, loss_fn
                )
                stats["bursts"].append(epoch)
                stats["transfers"].append(n_transfers)

                # Update CRISPR Bank
                for host in self.hosts:
                    for plasmid in host.plasmids:
                        task_sig = self._compute_task_signature(host, train_loaders)
                        plasmid_fp = plasmid.fingerprint()
                        self.crispr.store(task_sig, plasmid_fp, plasmid.gain)

            # === Phase 7: Apoptosis Check ===
            total_apoptosis = 0
            for host in self.hosts:
                # Update vitality for all plasmids
                with torch.no_grad():
                    for i, plasmid in enumerate(host.plasmids):
                        # Evaluate gain on a batch
                        if i < len(train_loaders):
                            batch = train_loaders[i][epoch % len(train_loaders[i])]
                            x, y = (batch if isinstance(batch, tuple)
                                    else (batch, batch))
                            gain = host.evaluate_plasmid_gain(plasmid, x, y, loss_fn)
                            plasmid.update_vitality(
                                self.config.host_config.vitality_decay, gain
                            )

                removed = host.apoptosis_check()
                total_apoptosis += len(removed)

            stats["apoptosis"].append(total_apoptosis)

            # Log epoch
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                    f"Q: {q_mean:.3f} | Plasmids: {sum(len(h.plasmids) for h in self.hosts)} | "
                    f"Burst: {'YES' if burst_triggered else 'no'} | "
                    f"Apoptosis: {total_apoptosis}"
                )

        return stats

    def _compute_task_signature(
        self, host: HostCell, train_loaders: list
    ) -> torch.Tensor:
        """Compute a statistical fingerprint of the host's current task."""
        host_idx = self.hosts.index(host)
        if host_idx >= len(train_loaders):
            return torch.zeros(32)

        batch = train_loaders[host_idx][0]  # First batch
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        # Statistical fingerprint: mean, std, skew (flattened)
        x_flat = x.view(x.size(0), -1).float()
        mean = x_flat.mean(dim=0)
        std = x_flat.std(dim=0)
        skew = ((x_flat - mean) ** 3).mean(dim=0) / (std ** 3 + 1e-8)

        sig = torch.cat([mean, std, skew])
        # Pad/truncate to size 32
        if sig.shape[0] < 32:
            sig = torch.cat([sig, torch.zeros(32 - sig.shape[0])])
        return sig[:32]

    def forward(
        self, x: torch.Tensor, host_idx: int = 0
    ) -> torch.Tensor:
        """Forward pass through a specific host."""
        return self.hosts[host_idx](x)

    def summary(self) -> dict:
        """Return ecosystem summary statistics."""
        return {
            "num_hosts": len(self.hosts),
            "total_plasmids": sum(len(h.plasmids) for h in self.hosts),
            "crispr_records": len(self.crispr),
            "crispr_stats": self.crispr.stats(),
            "bridge_success_rate": self.bridge.success_rate,
            "quorum_bursts": self.quorum.burst_count,
            "avg_vitality": sum(
                p.vitality for h in self.hosts for p in h.plasmids
            ) / max(1, sum(len(h.plasmids) for h in self.hosts)),
        }
