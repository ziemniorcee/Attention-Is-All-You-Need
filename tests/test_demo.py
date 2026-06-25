"""Tests for the interactive terminal demo helpers."""

import pytest

from tiny_transformer.demo import checkpoint_path, parse_numbers


def test_parse_numbers_accepts_spaces_and_commas() -> None:
    assert parse_numbers("3 4, 5", vocabulary_size=8) == [3, 4, 5]


def test_parse_numbers_rejects_reserved_tokens() -> None:
    with pytest.raises(ValueError, match="zakresie 3..7"):
        parse_numbers("2 4", vocabulary_size=8)


def test_checkpoint_path_matches_training_run_layout(tmp_path) -> None:
    path = checkpoint_path(tmp_path, "transformer", 4, seed=13)
    assert path == tmp_path / "transformer_heads-4_seed-13" / "best.pt"


def test_lstm_checkpoint_path_uses_na_heads_label(tmp_path) -> None:
    path = checkpoint_path(tmp_path, "lstm", None, seed=13)
    assert path == tmp_path / "lstm_heads-na_seed-13" / "best.pt"
