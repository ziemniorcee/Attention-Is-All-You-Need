"""Tiny Transformer and LSTM baseline model definitions."""

from __future__ import annotations

import math

import torch
from torch import nn

from tiny_transformer.data import BOS_TOKEN


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.size(1)]


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.scale = math.sqrt(d_model)
        self.vocabulary_size = vocabulary_size
        self.scale = math.sqrt(d_model)
        self.source_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.target_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.source_position = SinusoidalPositionalEncoding(d_model)
        self.target_position = SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.output = nn.Linear(d_model, vocabulary_size)

    def forward(
        self,
        source: torch.Tensor,
        source_valid_mask: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden = self.source_position(self.source_embedding(source) * self.scale)
        target_hidden = self.target_position(self.target_embedding(target_input) * self.scale)
        target_valid_mask = target_input.ne(0)
        target_length = target_input.size(1)
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, dtype=torch.bool, device=source.device),
            diagonal=1,
        )
        hidden = self.transformer(
            source_hidden,
            target_hidden,
            tgt_mask=causal_mask,
            src_key_padding_mask=~source_valid_mask,
            tgt_key_padding_mask=~target_valid_mask,
            memory_key_padding_mask=~source_valid_mask,
        )
        return self.output(hidden)

    @torch.inference_mode()
    def generate(
        self, source: torch.Tensor, source_valid_mask: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        generated = torch.full(
            (source.size(0), 1), BOS_TOKEN, dtype=torch.long, device=source.device
        )
        for _ in range(max_length):
            logits = self(source, source_valid_mask, generated)
            generated = torch.cat((generated, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        return generated[:, 1:]


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.target_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, vocabulary_size)

    def encode(
        self, source: torch.Tensor, source_valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = source_valid_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.source_embedding(source), lengths, batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        return state

    def forward(
        self,
        source: torch.Tensor,
        source_valid_mask: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode(source, source_valid_mask)
        hidden, _ = self.decoder(self.target_embedding(target_input), state)
        return self.output(hidden)

    @torch.inference_mode()
    def generate(
        self, source: torch.Tensor, source_valid_mask: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        state = self.encode(source, source_valid_mask)
        current = torch.full(
            (source.size(0), 1), BOS_TOKEN, dtype=torch.long, device=source.device
        )
        outputs = []
        for _ in range(max_length):
            hidden, state = self.decoder(self.target_embedding(current), state)
            current = self.output(hidden[:, -1:]).argmax(dim=-1)
            outputs.append(current)
        return torch.cat(outputs, dim=1)


def build_model(model_name: str, model_config: dict, vocabulary_size: int, num_heads: int) -> nn.Module:
    common = {
        "vocabulary_size": vocabulary_size,
        "d_model": int(model_config["d_model"]),
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    if model_name == "transformer":
        return TinyTransformer(
            **common,
            num_heads=num_heads,
            dim_feedforward=int(model_config["dim_feedforward"]),
        )
    if model_name == "lstm":
        return LSTMBaseline(**common)
    raise ValueError(f"Unsupported model: {model_name!r}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
