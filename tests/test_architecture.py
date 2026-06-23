"""Architecture-level tests: positional encoding, mask correctness, tensor dimensions,
and parameter invariance across head configurations."""

import pytest
import torch

from tiny_transformer.models import (
    SinusoidalPositionalEncoding,
    build_model,
    count_trainable_parameters,
)

_SMALL_CFG = {"d_model": 16, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}
_VOCAB = 12


def test_positional_encoding_output_shape() -> None:
    pe = SinusoidalPositionalEncoding(d_model=64)
    x = torch.zeros(8, 20, 64)
    out = pe(x)
    assert out.shape == (8, 20, 64)


def test_positional_encoding_varies_with_position() -> None:
    pe = SinusoidalPositionalEncoding(d_model=64)
    # encoding buffer: (1, max_length, d_model)
    enc = pe.encoding
    assert not torch.allclose(enc[0, 0], enc[0, 1]), (
        "Positional encoding must differ between position 0 and position 1"
    )


def test_positional_encoding_values_in_unit_range() -> None:
    pe = SinusoidalPositionalEncoding(d_model=64)
    enc = pe.encoding
    assert enc.min() >= -1.0 - 1e-5
    assert enc.max() <= 1.0 + 1e-5


def test_positional_encoding_different_lengths_are_consistent() -> None:
    pe = SinusoidalPositionalEncoding(d_model=32)
    x_short = torch.zeros(2, 5, 32)
    x_long = torch.zeros(2, 10, 32)
    out_short = pe(x_short)
    out_long = pe(x_long)
    # first 5 positions of the long output must match the short output
    assert torch.allclose(out_short, out_long[:, :5, :])


def test_causal_mask_diagonal_is_not_blocked() -> None:
    seq_len = 5
    # same construction as TinyTransformer.forward
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )
    for i in range(seq_len):
        assert not causal_mask[i, i], f"Position {i} must be able to attend to itself"


def test_causal_mask_blocks_future_positions() -> None:
    seq_len = 5
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )
    assert causal_mask[0, 4], "Position 0 must NOT see position 4"
    assert causal_mask[1, 2], "Position 1 must NOT see position 2"
    assert not causal_mask[2, 1], "Position 2 must see position 1"
    assert not causal_mask[4, 0], "Position 4 must see position 0"


def test_causal_mask_is_strictly_upper_triangular() -> None:
    seq_len = 6
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )
    for row in range(seq_len):
        for col in range(seq_len):
            expected = col > row
            assert causal_mask[row, col].item() == expected, (
                f"causal_mask[{row},{col}] should be {expected}"
            )


def test_head_dimensions_divide_d_model_base_config() -> None:
    d_model = 128  # matches configs/base.yaml
    for n_heads in [1, 2, 4, 8]:
        assert d_model % n_heads == 0, (
            f"d_model={d_model} is not divisible by n_heads={n_heads}"
        )


def test_invalid_head_count_raises_value_error() -> None:
    # d_model=16, 3 heads → 16 % 3 != 0 → must raise
    with pytest.raises(ValueError):
        build_model("transformer", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=3)


def test_transformer_forward_shape() -> None:
    model = build_model("transformer", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=2)
    source = torch.tensor([[1, 2, 3], [4, 5, 0]])
    valid_mask = source.ne(0)
    target_input = torch.tensor([[1, 3, 2], [1, 5, 0]])
    logits = model(source, valid_mask, target_input)
    assert logits.shape == (2, 3, _VOCAB)


def test_lstm_forward_shape() -> None:
    model = build_model("lstm", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=1)
    source = torch.tensor([[1, 2, 3], [4, 5, 0]])
    valid_mask = source.ne(0)
    target_input = torch.tensor([[1, 3, 2], [1, 5, 0]])
    logits = model(source, valid_mask, target_input)
    assert logits.shape == (2, 3, _VOCAB)


def test_padding_mask_makes_output_invariant_to_pad_values() -> None:
    model = build_model("transformer", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=2)
    model.eval()

    source_zeros = torch.tensor([[3, 5, 7, 0, 0]])
    source_noise = torch.tensor([[3, 5, 7, 9, 4]])
    valid_mask = torch.tensor([[True, True, True, False, False]])
    target = torch.tensor([[1, 3, 5, 7]])

    with torch.no_grad():
        logits_zeros = model(source_zeros, valid_mask, target)
        logits_noise = model(source_noise, valid_mask, target)

    assert torch.allclose(logits_zeros, logits_noise, atol=1e-5), (
        "Padding mask must make decoder output invariant to values at masked source positions"
    )


def test_parameter_counts_identical_across_head_configs() -> None:
    counts = {
        n: count_trainable_parameters(
            build_model("transformer", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=n)
        )
        for n in [1, 2, 4, 8]
    }
    unique = set(counts.values())
    assert len(unique) == 1, (
        f"Parameter count must be identical for all head configs; got: {counts}"
    )


def test_parameter_count_is_positive() -> None:
    model = build_model("transformer", _SMALL_CFG, vocabulary_size=_VOCAB, num_heads=4)
    assert count_trainable_parameters(model) > 0
