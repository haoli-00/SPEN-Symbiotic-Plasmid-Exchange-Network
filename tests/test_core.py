"""Tests for SPEN core components."""

import torch
import pytest

from spen import (
    Plasmid,
    PlasmidConfig,
    HostCell,
    HostConfig,
    ConjugationBridge,
    QuorumSensor,
    CRISPRBank,
    MycorrhizalSpace,
)


class TestPlasmid:
    def test_creation(self):
        config = PlasmidConfig(dim=64, rank=4)
        p = Plasmid(config)
        assert p.A.shape == (64, 4)
        assert p.B.shape == (4, 64)
        assert p.vitality == 1.0

    def test_forward(self):
        config = PlasmidConfig(dim=64, rank=4)
        p = Plasmid(config)
        x = torch.randn(8, 64)
        out = p(x)
        assert out.shape == (8, 64)

    def test_mutation(self):
        config = PlasmidConfig(dim=64, rank=4, mutation_scale=0.1)
        p = Plasmid(config)
        original_A = p.A.clone()
        p.mutate()
        assert not torch.allclose(original_A, p.A)
        assert p.mutation_count == 1

    def test_vitality_update(self):
        config = PlasmidConfig(dim=64, rank=4)
        p = Plasmid(config)
        p.update_vitality(gamma=0.9, current_gain=0.5)
        assert 0.9 < p.vitality < 1.0

    def test_apoptosis(self):
        config = PlasmidConfig(dim=64, rank=4)
        p = Plasmid(config)
        p.vitality = 0.01
        # Should trigger after 3 consecutive rounds
        assert not p.should_apoptose(threshold=0.05, consecutive_rounds=3)
        assert not p.should_apoptose(threshold=0.05, consecutive_rounds=3)
        assert p.should_apoptose(threshold=0.05, consecutive_rounds=3)

    def test_fingerprint(self):
        config = PlasmidConfig(dim=64, rank=4)
        p = Plasmid(config)
        fp = p.fingerprint()
        assert fp.shape == (32,)

    def test_clone(self):
        config = PlasmidConfig(dim=64, rank=4)
        p1 = Plasmid(config)
        p1.gain = 0.5
        p2 = p1.clone()
        assert torch.allclose(p1.A, p2.A)
        assert p2.gain == 0.5


class TestHostCell:
    def test_creation(self):
        host_config = HostConfig(base_dim=64, num_plasmids=3)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)
        assert len(host.plasmids) == 3

    def test_forward(self):
        host_config = HostConfig(base_dim=64, hidden_dim=128, num_plasmids=3)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)
        x = torch.randn(8, 64)
        out = host(x)
        assert out.shape == (8, 64)

    def test_add_plasmid(self):
        host_config = HostConfig(base_dim=64, num_plasmids=2, plasmid_capacity=4)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)
        new_p = Plasmid(plasmid_config)
        host.add_plasmid(new_p)
        assert len(host.plasmids) == 3

    def test_remove_plasmid(self):
        host_config = HostConfig(base_dim=64, num_plasmids=3)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)
        removed = host.remove_plasmid(0)
        assert removed is not None
        assert len(host.plasmids) == 2

    def test_best_plasmids(self):
        host_config = HostConfig(base_dim=64, num_plasmids=5)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)
        for i, p in enumerate(host.plasmids):
            p.vitality = i * 0.1
        best = host.best_plasmids(2)
        assert len(best) == 2
        assert best[0].vitality > best[1].vitality

    def test_apoptosis_check(self):
        host_config = HostConfig(
            base_dim=64, num_plasmids=3,
            apoptosis_threshold=0.05, apoptosis_consecutive=3,
        )
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        host = HostCell(host_config, plasmid_config)

        # Force first plasmid to very low vitality
        host.plasmids[0].vitality = 0.01
        for _ in range(3):
            host.plasmids[0].should_apoptose(0.05, 3)

        removed = host.apoptosis_check()
        assert len(removed) > 0


class TestConjugationBridge:
    def test_transfer(self):
        host_config = HostConfig(base_dim=64, num_plasmids=3)
        plasmid_config = PlasmidConfig(dim=64, rank=4)
        donor = HostCell(host_config, plasmid_config)
        recipient = HostCell(host_config, plasmid_config)

        donor.plasmids[0].vitality = 0.9
        bridge = ConjugationBridge(gain_threshold=-999.0)  # Accept everything

        x = torch.randn(4, 64)
        y = torch.randn(4, 64)
        accepted = bridge.transfer(donor, recipient, x, y, torch.nn.MSELoss())
        assert len(accepted) > 0


class TestQuorumSensor:
    def test_below_threshold(self):
        qs = QuorumSensor(threshold=0.6, t_max=100)
        triggered, q_mean = qs.update([10, 10, 10])
        assert not triggered

    def test_above_threshold(self):
        qs = QuorumSensor(threshold=0.6, t_max=100)
        triggered, q_mean = qs.update([80, 80, 80])
        assert triggered

    def test_cooldown(self):
        qs = QuorumSensor(threshold=0.6, t_max=100, cooldown_epochs=3)
        # First burst
        triggered1, _ = qs.update([80, 80, 80])
        assert triggered1
        # Immediately after: should not trigger (cooldown)
        triggered2, _ = qs.update([80, 80, 80])
        assert not triggered2


class TestCRISPRBank:
    def test_store_and_recall(self):
        bank = CRISPRBank()
        task_sig = torch.randn(32)
        plasmid_fp = torch.randn(32)
        bank.store(task_sig, plasmid_fp, gain=0.5)

        # Exact same signature should recall
        results = bank.recall(task_sig)
        assert len(results) == 1
        assert results[0][1] == 0.5

    def test_no_recall_for_different_task(self):
        bank = CRISPRBank()
        task_sig_a = torch.ones(32)
        task_sig_b = -torch.ones(32)
        plasmid_fp = torch.randn(32)
        bank.store(task_sig_a, plasmid_fp, gain=0.5)

        results = bank.recall(task_sig_b)
        assert len(results) == 0  # Completely different

    def test_max_records(self):
        bank = CRISPRBank(max_records=5)
        for i in range(10):
            bank.store(torch.randn(32), torch.randn(32), gain=float(i))
        assert len(bank) == 5


class TestMycorrhizalSpace:
    def test_embed(self):
        space = MycorrhizalSpace(dim=64)
        space.set_encoder(base_dim=128)
        acts = [torch.randn(8, 128) for _ in range(5)]
        emb = space.embed(acts)
        assert emb.shape == (5, 64)

    def test_neighbors(self):
        space = MycorrhizalSpace(dim=16, neighbor_radius=2.0)
        emb = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        neighbors = space.compute_neighbors(emb)
        assert 1 in neighbors[0]
        assert 2 not in neighbors[0]

    def test_diffuse_signals(self):
        space = MycorrhizalSpace(dim=16, neighbor_radius=2.0, distress_threshold=0.5)
        emb = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        neighbors = [[1], [0]]
        distress = [2.0, 0.1]  # Host 0 is distressed, Host 1 is not
        signals = space.diffuse_signals(emb, distress, neighbors)
        # Host 1 should receive signal from Host 0
        assert len(signals[1]) == 1
        # Host 0 should not receive (Host 1 not distressed)
        assert len(signals[0]) == 0

    def test_cluster_niches(self):
        space = MycorrhizalSpace(dim=16)
        # Two tight clusters
        emb = torch.zeros(4, 16)
        emb[0, 0] = 0.0
        emb[1, 0] = 0.1  # Close to 0
        emb[2, 0] = 5.0
        emb[3, 0] = 5.1  # Close to 2
        labels = space.cluster_niches(emb, eps=0.5, min_samples=2)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
