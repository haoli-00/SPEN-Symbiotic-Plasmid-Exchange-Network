"""MycorrhizalSpace: underground fungal communication network.

In forest ecosystems, mycorrhizal fungi connect plant roots, enabling
resource sharing and chemical alarm signaling across species. A tree
under insect attack can warn its neighbors through this network.

Here, we implement a shared latent space where hosts project their
current input distribution. Hosts emit distress signals (high gradient
norm) that diffuse with distance decay. Nearby hosts respond by pushing
niche-matched plasmids.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MycorrhizalSpace(nn.Module):
    """Shared latent communication space for inter-host signaling.

    Each host projects its current batch into a fixed-dimensional embedding.
    The space supports two operations:

    1. Distress Signaling: hosts with high gradient norm emit signals
       that diffuse outward with exponential distance decay.

    2. Niche Clustering: hosts are periodically clustered into ecological
       niches using DBSCAN-style density clustering on their embeddings.
    """

    def __init__(
        self,
        dim: int = 64,
        neighbor_radius: float = 1.0,
        decay_lambda: float = 2.0,
        distress_threshold: float = 1.0,
    ):
        """
        Args:
            dim: Dimensionality of the latent space.
            neighbor_radius: Maximum distance for considering two hosts neighbors.
            decay_lambda: Exponential decay rate for signal diffusion.
            distress_threshold: Gradient norm above which a distress signal is emitted.
        """
        super().__init__()
        self.dim = dim
        self.neighbor_radius = neighbor_radius
        self.decay_lambda = decay_lambda
        self.distress_threshold = distress_threshold

        # Simple encoder: linear projection from base_dim to mycorrhizal dim
        self.encoder: Optional[nn.Linear] = None  # Set during ecosystem init

    def set_encoder(self, base_dim: int) -> None:
        """Initialize the encoder after base_dim is known."""
        self.encoder = nn.Linear(base_dim, self.dim)

    def embed(
        self,
        host_activations: list[torch.Tensor],
    ) -> torch.Tensor:
        """Project host activations into the mycorrhizal latent space.

        Args:
            host_activations: List of activation tensors from each host.
                Each tensor is (batch_size, base_dim) — we take the mean
                over the batch dimension to get a single point per host.

        Returns:
            Tensor of shape (num_hosts, dim): embedded positions.
        """
        if self.encoder is None:
            raise RuntimeError(
                "MycorrhizalSpace encoder not initialized. Call set_encoder() first."
            )

        embeddings = []
        for act in host_activations:
            # Average over batch to get a single representation
            mean_act = act.mean(dim=0, keepdim=True)  # (1, base_dim)
            emb = self.encoder(mean_act)               # (1, dim)
            embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings)  # (num_hosts, dim)

    def compute_neighbors(
        self,
        embeddings: torch.Tensor,
    ) -> list[list[int]]:
        """Compute neighbor lists based on distance threshold.

        Args:
            embeddings: (num_hosts, dim) positions.

        Returns:
            neighbors[i]: list of host indices within neighbor_radius of host i.
        """
        n = embeddings.shape[0]
        # Pairwise distances
        dists = torch.cdist(embeddings, embeddings)  # (n, n)
        neighbors = []
        for i in range(n):
            mask = (dists[i] < self.neighbor_radius) & (torch.arange(n) != i)
            neighbors.append(torch.where(mask)[0].tolist())
        return neighbors

    def diffuse_signals(
        self,
        embeddings: torch.Tensor,
        distress_levels: list[float],
        neighbors: list[list[int]],
    ) -> list[list[tuple[int, float]]]:
        """Diffuse distress signals through the mycorrhizal network.

        Each host emits a distress signal with strength proportional to
        its gradient norm. Signals decay exponentially with distance.

        Args:
            embeddings: (num_hosts, dim) host positions.
            distress_levels: Distress level per host (gradient norm).
            neighbors: Precomputed neighbor lists.

        Returns:
            signals_received[i]: list of (sender_idx, signal_strength)
            for signals received by host i.
        """
        n = embeddings.shape[0]
        signals_received: list[list[tuple[int, float]]] = [[] for _ in range(n)]

        for i in range(n):
            if distress_levels[i] < self.distress_threshold:
                continue  # Not distressed, no signal emitted

            signal_strength = distress_levels[i]

            for j in neighbors[i]:
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
                attenuated = signal_strength * torch.exp(
                    torch.tensor(-self.decay_lambda * dist)
                ).item()
                signals_received[j].append((i, attenuated))

        return signals_received

    def cluster_niches(
        self,
        embeddings: torch.Tensor,
        eps: float = 0.5,
        min_samples: int = 2,
    ) -> list[int]:
        """DBSCAN-style clustering of hosts into ecological niches.

        Args:
            embeddings: (num_hosts, dim) host positions.
            eps: Neighborhood radius for clustering.
            min_samples: Minimum hosts to form a niche cluster.

        Returns:
            niche_labels: list of niche IDs per host (-1 = noise/unclustered).
        """
        n = embeddings.shape[0]
        dists = torch.cdist(embeddings, embeddings)

        # Find core points
        core_mask = (dists < eps).sum(dim=1) >= min_samples

        # BFS to assign clusters
        visited = torch.zeros(n, dtype=torch.bool)
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if visited[i] or not core_mask[i]:
                continue

            # BFS from core point i
            queue = [i]
            visited[i] = True
            labels[i] = cluster_id

            while queue:
                curr = queue.pop(0)
                neighbors = torch.where(
                    (dists[curr] < eps) & (~visited)
                )[0]

                for nb in neighbors:
                    nb_idx = nb.item()
                    visited[nb_idx] = True
                    labels[nb_idx] = cluster_id
                    if core_mask[nb_idx]:
                        queue.append(nb_idx)

            cluster_id += 1

        return labels
