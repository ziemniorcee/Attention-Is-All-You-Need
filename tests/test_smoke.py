"""Fast smoke tests for data shapes, masks, models, and one training step."""

import torch
from torch import nn

from tiny_transformer.data import SequenceDataset, create_dataloaders
from tiny_transformer.models import build_model
from tiny_transformer.train import train_epoch


def test_reverse_dataset_is_deterministic() -> None:
    first = SequenceDataset(5, [3, 6], 16, "reverse", seed=7)
    second = SequenceDataset(5, [3, 6], 16, "reverse", seed=7)
    for (source_a, target_a), (source_b, target_b) in zip(first, second):
        assert torch.equal(source_a, source_b)
        assert torch.equal(target_a, target_b)
        assert source_a[-1].item() == 2
        assert target_a[-1].item() == 2
        assert torch.equal(target_a[:-1], source_a[:-1].flip(0))


def test_models_produce_token_logits() -> None:
    source = torch.tensor([[1, 2, 3], [4, 5, 0]])
    valid_mask = source.ne(0)
    target_input = torch.tensor([[1, 3, 2], [1, 5, 0]])
    config = {"d_model": 16, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}
    for name in ("transformer", "lstm"):
        model = build_model(name, config, vocabulary_size=12, num_heads=2)
        assert model(source, valid_mask, target_input).shape == (2, 3, 12)
        assert model.generate(source, valid_mask, max_length=3).shape == (2, 3)


def test_one_training_epoch() -> None:
    data_config = {
        "task": "reverse",
        "vocabulary_size": 12,
        "train_size": 16,
        "validation_size": 8,
        "test_size": 8,
        "train_length": [3, 5],
        "generalization_length": [6, 8],
        "batch_size": 8,
    }
    loader = create_dataloaders(data_config, seed=3).train
    model_config = {"d_model": 16, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}
    model = build_model("transformer", model_config, vocabulary_size=12, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(ignore_index=0), torch.device("cpu"), 1.0)
    assert loss > 0
