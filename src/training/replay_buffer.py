"""Replay buffer for continual learning in DSI."""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Fixed-size buffer that stores (query, docid_tokenids) pairs from past periods.

    Uses reservoir sampling to maintain a uniform distribution across all past
    periods regardless of individual period sizes.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: list[tuple[str, list[int]]] = []
        self._seen: int = 0  # total items offered (for reservoir sampling)
        self.period_counts: dict[str, int] = {}

    def add_from_dataset(self, dataset, period_name: str) -> None:
        """Add examples from a ChronoDocIDDataset using reservoir sampling.

        Each call represents a new period. Reservoir sampling ensures that
        every example seen so far has equal probability of being in the buffer.
        """
        added = 0
        for i in range(len(dataset)):
            query, token_ids = dataset[i]
            self._seen += 1

            if len(self.buffer) < self.max_size:
                self.buffer.append((query, token_ids))
                added += 1
            else:
                j = random.randint(0, self._seen - 1)
                if j < self.max_size:
                    self.buffer[j] = (query, token_ids)
                    added += 1

        self.period_counts[period_name] = self.period_counts.get(period_name, 0) + added
        logger.info(
            "ReplayBuffer: offered %d from '%s', buffer size %d / %d",
            len(dataset), period_name, len(self.buffer), self.max_size,
        )

    def sample(self, n: int) -> list[tuple[str, list[int]]]:
        """Sample n examples from the buffer (with replacement if n > buffer size)."""
        if not self.buffer:
            return []
        if n <= len(self.buffer):
            return random.sample(self.buffer, n)
        return random.choices(self.buffer, k=n)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str | Path) -> None:
        """Save buffer to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_size": self.max_size,
            "seen": self._seen,
            "period_counts": self.period_counts,
            "buffer": [(q, tids) for q, tids in self.buffer],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved replay buffer (%d items) to %s", len(self.buffer), path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """Load buffer from disk."""
        with open(path) as f:
            data = json.load(f)
        buf = cls(max_size=data["max_size"])
        buf._seen = data["seen"]
        buf.period_counts = data["period_counts"]
        buf.buffer = [(q, tids) for q, tids in data["buffer"]]
        logger.info("Loaded replay buffer (%d items) from %s", len(buf.buffer), path)
        return buf
