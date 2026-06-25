"""Interactive terminal demo for trained LSTM and Transformer variants."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import yaml

from tiny_transformer.data import EOS_TOKEN, PAD_TOKEN
from tiny_transformer.models import build_model
from tiny_transformer.train import select_device, set_seed

MODEL_VARIANTS = [
    ("lstm", None),
    ("transformer", 1),
    ("transformer", 2),
    ("transformer", 4),
    ("transformer", 8),
]


def silence_pytorch_nested_tensor_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of nested tensors is in prototype stage.*",
        category=UserWarning,
    )


def parse_numbers(text: str, vocabulary_size: int) -> list[int]:
    normalized = text.replace(",", " ")
    try:
        numbers = [int(part) for part in normalized.split()]
    except ValueError as error:
        raise ValueError(
            "wpisz tylko liczby calkowite, oddzielone spacjami lub przecinkami"
        ) from error

    if not numbers:
        raise ValueError("podaj przynajmniej jedna liczbe")

    lowest_data_token = EOS_TOKEN + 1
    invalid = [
        number for number in numbers if number < lowest_data_token or number >= vocabulary_size
    ]
    if invalid:
        raise ValueError(
            f"liczby musza byc w zakresie {lowest_data_token}..{vocabulary_size - 1}; "
            f"poza zakresem: {invalid}"
        )
    return numbers


def checkpoint_path(output_dir: Path, model_name: str, num_heads: int | None, seed: int) -> Path:
    heads_label = str(num_heads) if model_name == "transformer" else "na"
    return output_dir / f"{model_name}_heads-{heads_label}_seed-{seed}" / "best.pt"


def build_demo_models(
    config: dict, seed: int, device: torch.device
) -> list[tuple[str, torch.nn.Module, str]]:
    set_seed(seed)
    output_dir = Path(config["experiment"]["output_dir"])
    vocabulary_size = int(config["data"]["vocabulary_size"])
    loaded_models = []

    for model_name, num_heads in MODEL_VARIANTS:
        model_heads = num_heads if num_heads is not None else int(config["model"]["num_heads"][0])
        model = build_model(model_name, config["model"], vocabulary_size, model_heads).to(device)
        path = checkpoint_path(output_dir, model_name, num_heads, seed)
        status = "bez wag treningowych"
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            status = f"checkpoint: {path.as_posix()}"
        model.eval()
        label = "LSTM" if model_name == "lstm" else f"Transformer {num_heads} head(s)"
        loaded_models.append((label, model, status))
    return loaded_models


def predict(model: torch.nn.Module, numbers: list[int], device: torch.device) -> list[int]:
    source_tokens = numbers + [EOS_TOKEN]
    source = torch.tensor([source_tokens], dtype=torch.long, device=device)
    valid_mask = source.ne(PAD_TOKEN)
    generated = model.generate(source, valid_mask, max_length=len(source_tokens))[0].tolist()
    if EOS_TOKEN in generated:
        generated = generated[: generated.index(EOS_TOKEN)]
    return [token for token in generated if token != PAD_TOKEN]


def print_prediction_table(
    models: list[tuple[str, torch.nn.Module, str]],
    numbers: list[int],
    device: torch.device,
) -> None:
    expected = list(reversed(numbers))
    print(f"\nWejscie:              {numbers}")
    print(f"Oczekiwane reverse:   {expected}")
    print("-" * 72)
    for label, model, status in models:
        prediction = predict(model, numbers, device)
        print(f"{label:<24} {str(prediction):<24} ({status})")
    print()


def main() -> None:
    silence_pytorch_nested_tensor_warning()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", help="cpu, cuda albo auto; domyslnie z pliku config")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = select_device(args.device or str(config["training"]["device"]))
    models = build_demo_models(config, args.seed, device)
    vocabulary_size = int(config["data"]["vocabulary_size"])

    print("Terminal demo: reverse sequence")
    print(f"Wpisuj liczby od {EOS_TOKEN + 1} do {vocabulary_size - 1}.")
    print("Przyklad: 3 7 9 12")
    print("Koniec: pusta linia, q albo quit.\n")

    while True:
        raw = input("liczby> ").strip()
        if raw.lower() in {"", "q", "quit", "exit"}:
            break
        try:
            numbers = parse_numbers(raw, vocabulary_size)
        except ValueError as error:
            print(f"Blad: {error}\n")
            continue
        print_prediction_table(models, numbers, device)


if __name__ == "__main__":
    main()
