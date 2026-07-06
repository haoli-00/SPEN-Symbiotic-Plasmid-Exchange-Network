"""Minimal working example of the SPEN ecosystem.

This script demonstrates the complete lifecycle:
1. Create an ecosystem with multiple hosts
2. Train on synthetic data
3. Observe plasmid dynamics (conjugation, mutation, apoptosis)
4. Query CRISPR memory bank
"""

import torch
from spen import (
    Ecosystem,
    HostConfig,
    PlasmidConfig,
)


def generate_synthetic_data(n_samples: int = 200, dim: int = 128, n_tasks: int = 5):
    """Generate synthetic regression tasks with different distributions.

    Each task has a different mean and variance, simulating heterogeneous
    data across hosts.
    """
    loaders = []
    for i in range(n_tasks):
        mean = torch.randn(dim) * (i + 1) * 0.5
        std = 0.1 + torch.rand(1).item() * 0.3

        x = torch.randn(n_samples, dim) * std + mean
        # Target: non-linear function of x
        y = torch.tanh(x @ torch.randn(dim, dim) * 0.1) + torch.randn(n_samples, dim) * 0.05

        dataset = list(zip(x, y))
        loaders.append(dataset)

    return loaders


def main():
    print("=" * 60)
    print("SPEN: Symbiotic Plasmid Exchange Network - Demo")
    print("=" * 60)

    # Configuration
    host_config = HostConfig(
        base_dim=128,
        hidden_dim=256,
        num_plasmids=3,
        plasmid_capacity=6,
        vitality_decay=0.9,
        apoptosis_threshold=0.02,
        apoptosis_consecutive=5,
    )

    plasmid_config = PlasmidConfig(
        dim=128,
        rank=8,
        dropout=0.1,
        mutation_rate=0.3,
        mutation_scale=0.01,
    )

    # Create ecosystem
    print("\n[1] Creating ecosystem: 5 hosts, each with 3 plasmids...")
    ecosystem = Ecosystem(
        num_hosts=5,
        host_config=host_config,
        plasmid_config=plasmid_config,
    )
    ecosystem.quorum.threshold = 0.3  # Lower threshold for demo

    # Generate data
    print("[2] Generating synthetic multi-task data...")
    train_loaders = generate_synthetic_data(
        n_samples=200, dim=128, n_tasks=5
    )

    # Train
    print("[3] Training for 30 epochs...\n")
    stats = ecosystem.fit(
        train_loaders,
        epochs=30,
        lr=1e-3,
    )

    # Summary
    print("\n" + "=" * 60)
    print("[4] Training complete. Ecosystem summary:")
    print("=" * 60)

    summary = ecosystem.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n  Final loss: {stats['losses'][-1]:.6f}")
    print(f"  Conjugation bursts: {len(stats['bursts'])}")
    print(f"  Total successful plasmid transfers: {sum(stats['transfers'])}")
    print(f"  Total apoptosis events: {sum(stats['apoptosis'])}")

    # Show plasmid details
    print("\n[5] Plasmid inventory per host:")
    for host in ecosystem.hosts:
        print(f"  Host {host.host_id}: {len(host.plasmids)} plasmids")
        for p in host.plasmids:
            print(
                f"    Plasmid {p.plasmid_id}: "
                f"vitality={p.vitality:.4f}, "
                f"gain={p.gain:.4f}, "
                f"mutations={p.mutation_count}, "
                f"origin=Host{p.origin_host_id}"
            )

    # CRISPR bank
    print(f"\n[6] CRISPR Bank: {len(ecosystem.crispr)} records stored")
    crispr_stats = ecosystem.crispr.stats()
    print(f"  Avg gain: {crispr_stats['avg_gain']:.4f}")
    print(f"  Max gain: {crispr_stats['max_gain']:.4f}")

    print("\n[7] Demo complete!")


if __name__ == "__main__":
    main()
