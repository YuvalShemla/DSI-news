"""LoRA merging and composition strategies for continual learning."""

import json
import logging
from pathlib import Path

import torch
from peft import PeftModel, set_peft_model_state_dict

logger = logging.getLogger(__name__)


def merge_lora_weights(
    old_merged_state: dict[str, torch.Tensor],
    new_lora_state: dict[str, torch.Tensor],
    alpha: float = 0.9,
) -> dict[str, torch.Tensor]:
    """Weighted merge: merged = alpha * old_merged + (1 - alpha) * new_lora.

    Strategy B: Merge-Before-Forget. Accumulates knowledge from successive
    periods while gradually decaying older information.

    Both state dicts should contain the same keys (LoRA weight matrices).
    """
    merged = {}
    for key in new_lora_state:
        if key in old_merged_state:
            merged[key] = alpha * old_merged_state[key] + (1 - alpha) * new_lora_state[key]
        else:
            merged[key] = new_lora_state[key]

    # Keep any keys only in old state (shouldn't happen but be safe)
    for key in old_merged_state:
        if key not in merged:
            merged[key] = old_merged_state[key]

    return merged


def save_merged_lora(model: PeftModel, output_path: str | Path) -> None:
    """Save the merged LoRA weights (adapter only)."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    logger.info("Saved merged LoRA to %s", output_path)


def compose_loras(
    base_model,
    lora_paths: list[str | Path],
    weights: list[float] | None = None,
) -> PeftModel:
    """Load and compose multiple LoRA adapters.

    Strategy C: LoRA Composition with Temporal Routing. Each period's LoRA is
    loaded as a named adapter. At inference, the chrono prefix routes to the
    appropriate adapter.

    Args:
        base_model: The base model (not yet wrapped in PEFT).
        lora_paths: List of paths to saved LoRA adapters, ordered chronologically.
        weights: Optional per-adapter weights for weighted combination.

    Returns:
        PeftModel with all adapters loaded.
    """
    if weights is None:
        weights = [1.0] * len(lora_paths)

    model = None
    for i, path in enumerate(lora_paths):
        adapter_name = f"period_{i}"
        if model is None:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=adapter_name)
        else:
            model.load_adapter(str(path), adapter_name=adapter_name)
        logger.info("Loaded adapter '%s' from %s (weight=%.2f)", adapter_name, path, weights[i])

    return model


def prune_loras_fifo(
    lora_paths: list[str | Path],
    max_loras: int = 8,
) -> list[Path]:
    """FIFO pruning: remove oldest LoRAs when over capacity.

    Returns the pruned list (most recent max_loras entries).
    """
    paths = [Path(p) for p in lora_paths]
    if len(paths) <= max_loras:
        return paths
    pruned = paths[:len(paths) - max_loras]
    kept = paths[len(paths) - max_loras:]
    for p in pruned:
        logger.info("FIFO pruning: dropping adapter at %s", p)
    return kept


def prune_loras_lra(
    lora_paths: list[str | Path],
    usage_counts: dict[str, int],
    max_loras: int = 8,
) -> list[Path]:
    """LRA pruning: remove least recently activated LoRAs.

    Args:
        lora_paths: All adapter paths.
        usage_counts: Mapping from adapter path string to activation count.
        max_loras: Maximum number of adapters to keep.

    Returns:
        The pruned list sorted by usage (most used first).
    """
    paths = [Path(p) for p in lora_paths]
    if len(paths) <= max_loras:
        return paths
    scored = sorted(paths, key=lambda p: usage_counts.get(str(p), 0), reverse=True)
    kept = scored[:max_loras]
    dropped = scored[max_loras:]
    for p in dropped:
        logger.info("LRA pruning: dropping adapter at %s (usage=%d)", p, usage_counts.get(str(p), 0))
    return kept


class LoRAMerger:
    """Stateful merger that tracks merged weights across CL periods.

    Used by Strategy B (Merge-Before-Forget). After each period, the new LoRA
    is merged into the accumulated state with exponential decay.
    """

    def __init__(self, alpha_decay: float = 0.9):
        self.merged_state: dict[str, torch.Tensor] | None = None
        self.alpha = alpha_decay
        self.num_merges: int = 0

    def merge(self, new_lora_path: str | Path) -> None:
        """Merge a new period's LoRA weights into the accumulated state."""
        new_state = {}
        state_dict = torch.load(Path(new_lora_path) / "adapter_model.bin", map_location="cpu", weights_only=True)
        for key, value in state_dict.items():
            new_state[key] = value.float()

        if self.merged_state is None:
            self.merged_state = new_state
        else:
            self.merged_state = merge_lora_weights(
                self.merged_state, new_state, alpha=self.alpha
            )

        self.num_merges += 1
        logger.info(
            "Merged period %d LoRA (alpha=%.3f). Total merges: %d",
            self.num_merges, self.alpha, self.num_merges,
        )

    def apply_to_model(self, model: PeftModel) -> None:
        """Apply merged weights to a PEFT model's active adapter."""
        if self.merged_state is None:
            raise ValueError("No merged state available. Call merge() first.")

        # Convert to model dtype
        target_dtype = next(model.parameters()).dtype
        state = {k: v.to(target_dtype) for k, v in self.merged_state.items()}
        set_peft_model_state_dict(model, state)
        logger.info("Applied merged state (%d merges) to model", self.num_merges)

    def save(self, path: str | Path) -> None:
        """Save merger state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.merged_state, path / "merged_lora_state.pt")
        meta = {"alpha": self.alpha, "num_merges": self.num_merges}
        with open(path / "merger_meta.json", "w") as f:
            json.dump(meta, f)
        logger.info("Saved LoRAMerger state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LoRAMerger":
        """Load merger state from disk."""
        path = Path(path)
        with open(path / "merger_meta.json") as f:
            meta = json.load(f)
        merger = cls(alpha_decay=meta["alpha"])
        merger.num_merges = meta["num_merges"]
        merger.merged_state = torch.load(path / "merged_lora_state.pt", map_location="cpu", weights_only=True)
        logger.info("Loaded LoRAMerger state (%d merges) from %s", merger.num_merges, path)
        return merger
