"""Synthetic copy/reverse dataset generation and deterministic splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A deterministic in-memory dataset of variable-length token sequences."""

    def __init__(
        self,
        size: int,
        length_range: Sequence[int],
        vocabulary_size: int,
        task: str,
        seed: int,
    ) -> None:
        if task not in {"copy", "reverse"}:
            raise ValueError(f"Unsupported task: {task!r}")
        if len(length_range) != 2 or length_range[0] < 1 or length_range[0] > length_range[1]:
            raise ValueError("length_range must contain [minimum, maximum]")
        if vocabulary_size < 5:
            raise ValueError("vocabulary_size must include PAD, BOS, EOS and data tokens")

        generator = torch.Generator().manual_seed(seed)
        lengths = torch.randint(
            int(length_range[0]), int(length_range[1]) + 1, (size,), generator=generator
        )
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for length in lengths.tolist():
            content = torch.randint(3, vocabulary_size, (length,), generator=generator)
            transformed = content.clone() if task == "copy" else content.flip(0)
            source = torch.cat((content, torch.tensor([EOS_TOKEN])))
            target = torch.cat((transformed, torch.tensor([EOS_TOKEN])))
            self.examples.append((source, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def pad_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch and return source, target and a True-for-valid-token mask."""
    max_length = max(source.numel() for source, _ in batch)
    source_batch = torch.full((len(batch), max_length), PAD_TOKEN, dtype=torch.long)
    target_batch = torch.full_like(source_batch, PAD_TOKEN)
    valid_mask = torch.zeros_like(source_batch, dtype=torch.bool)
    for row, (source, target) in enumerate(batch):
        length = source.numel()
        source_batch[row, :length] = source
        target_batch[row, :length] = target
        valid_mask[row, :length] = True
    return source_batch, target_batch, valid_mask


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    generalization: DataLoader


def create_dataloaders(config: dict, seed: int) -> DataLoaders:
    """Create independent deterministic splits and a seeded training order."""
    common = {
        "vocabulary_size": int(config["vocabulary_size"]),
        "task": str(config["task"]),
    }
    split_specs = {
        "train": (int(config["train_size"]), config["train_length"], seed + 101),
        "validation": (int(config["validation_size"]), config["train_length"], seed + 202),
        "test": (int(config["test_size"]), config["train_length"], seed + 303),
        "generalization": (
            int(config["test_size"]),
            config["generalization_length"],
            seed + 404,
        ),
    }
    datasets = {
        name: SequenceDataset(size, lengths, seed=split_seed, **common)
        for name, (size, lengths, split_seed) in split_specs.items()
    }
    batch_size = int(config["batch_size"])
    loader_args = {"batch_size": batch_size, "collate_fn": pad_batch, "num_workers": 0}
    train_generator = torch.Generator().manual_seed(seed + 505)
    return DataLoaders(
        train=DataLoader(datasets["train"], shuffle=True, generator=train_generator, **loader_args),
        validation=DataLoader(datasets["validation"], shuffle=False, **loader_args),
        test=DataLoader(datasets["test"], shuffle=False, **loader_args),
        generalization=DataLoader(datasets["generalization"], shuffle=False, **loader_args),
    )
