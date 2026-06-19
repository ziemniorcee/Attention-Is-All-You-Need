"""Evaluation metrics for in-distribution and length-generalization splits."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autoregressive: bool = True,
) -> dict[str, float]:
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    for source, target, valid_mask in loader:
        source, target, valid_mask = (
            source.to(device),
            target.to(device),
            valid_mask.to(device),
        )
        target_input = torch.cat(
            (torch.ones_like(target[:, :1]), target[:, :-1]), dim=1
        )
        logits = model(source, valid_mask, target_input)
        loss_sum += criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1)).item()
        predictions = (
            model.generate(source, valid_mask, target.size(1))
            if autoregressive
            else logits.argmax(dim=-1)
        )
        matches = (predictions == target) | ~valid_mask
        correct_tokens += ((predictions == target) & valid_mask).sum().item()
        total_tokens += valid_mask.sum().item()
        correct_sequences += matches.all(dim=1).sum().item()
        total_sequences += source.size(0)

    return {
        "loss": loss_sum / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "sequence_accuracy": correct_sequences / max(total_sequences, 1),
    }
