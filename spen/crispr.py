"""CRISPRBank: adaptive immune memory for plasmid-task pairings.

In biology, the CRISPR-Cas system stores fragments of viral DNA as
"mugshots." When the same virus returns, Cas proteins use these
records to precisely cleave the invader's DNA.

Here, we store (task_signature, plasmid_fingerprint, gain) tuples.
When a new task arrives, we query the bank by task similarity and
auto-inject historically successful plasmids.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class CRISPRRecord:
    """A single CRISPR memory entry."""
    task_signature: torch.Tensor      # Statistical fingerprint of the task
    plasmid_fingerprint: torch.Tensor # Low-dimensional hash of plasmid weights
    gain: float                       # Performance gain on this task
    access_count: int = 0             # How many times this record was recalled


class CRISPRBank:
    """Persistent memory bank mapping task signatures to successful plasmids.

    Operations:
        store(task_sig, plasmid_fp, gain): Record a successful pairing.
        recall(task_sig, top_k): Retrieve best plasmid fingerprints for a task.
        query_similar(task_sig, threshold): Find historically similar tasks.
    """

    def __init__(self, max_records: int = 1000, similarity_threshold: float = 0.85):
        """
        Args:
            max_records: Maximum number of records to store.
            similarity_threshold: Cosine similarity threshold for task matching.
        """
        self.max_records = max_records
        self.similarity_threshold = similarity_threshold
        self.records: list[CRISPRRecord] = []

    def store(
        self,
        task_signature: torch.Tensor,
        plasmid_fingerprint: torch.Tensor,
        gain: float,
    ) -> None:
        """Store a successful plasmid-task pairing.

        If a similar record already exists, update it if the new gain is better.
        """
        # Check for existing similar record
        for record in self.records:
            sim = F.cosine_similarity(
                task_signature.unsqueeze(0),
                record.task_signature.unsqueeze(0),
            ).item()
            fp_sim = F.cosine_similarity(
                plasmid_fingerprint.unsqueeze(0),
                record.plasmid_fingerprint.unsqueeze(0),
            ).item()

            if sim > self.similarity_threshold and fp_sim > self.similarity_threshold:
                if gain > record.gain:
                    record.gain = gain
                    record.task_signature = task_signature.clone()
                    record.plasmid_fingerprint = plasmid_fingerprint.clone()
                return

        # Add new record
        record = CRISPRRecord(
            task_signature=task_signature.clone(),
            plasmid_fingerprint=plasmid_fingerprint.clone(),
            gain=gain,
        )
        self.records.append(record)

        # Evict oldest if over capacity (simple FIFO)
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def recall(
        self,
        task_signature: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[torch.Tensor, float]]:
        """Retrieve top-k plasmid fingerprints for a given task.

        Args:
            task_signature: Current task's statistical fingerprint.
            top_k: Number of records to return.

        Returns:
            List of (plasmid_fingerprint, gain) sorted by gain descending.
        """
        if not self.records:
            return []

        scored = []
        for record in self.records:
            sim = F.cosine_similarity(
                task_signature.unsqueeze(0),
                record.task_signature.unsqueeze(0),
            ).item()
            if sim > self.similarity_threshold:
                scored.append((record.plasmid_fingerprint, record.gain, sim))
                record.access_count += 1

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(fp, gain) for fp, gain, _ in scored[:top_k]]

    def query_similar_tasks(
        self,
        task_signature: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> list[float]:
        """Find gains from historically similar tasks (for transfer confidence)."""
        threshold = threshold or self.similarity_threshold
        gains = []
        for record in self.records:
            sim = F.cosine_similarity(
                task_signature.unsqueeze(0),
                record.task_signature.unsqueeze(0),
            ).item()
            if sim > threshold:
                gains.append(record.gain)
        return gains

    def __len__(self) -> int:
        return len(self.records)

    def stats(self) -> dict:
        """Return summary statistics of the CRISPR bank."""
        if not self.records:
            return {"total": 0, "avg_gain": 0.0, "most_accessed": None}

        gains = [r.gain for r in self.records]
        accesses = [r.access_count for r in self.records]
        most_accessed = self.records[accesses.index(max(accesses))]

        return {
            "total": len(self.records),
            "avg_gain": sum(gains) / len(gains),
            "max_gain": max(gains),
            "most_accessed_gain": most_accessed.gain,
            "most_accessed_count": most_accessed.access_count,
        }
