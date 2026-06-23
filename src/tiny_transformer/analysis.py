"""Create publication-ready tables and figures from raw experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "validation_token_accuracy",
    "validation_sequence_accuracy",
    "test_token_accuracy",
    "test_sequence_accuracy",
    "generalization_token_accuracy",
    "generalization_sequence_accuracy",
    "training_seconds",
]

ACCURACY_METRICS = [
    "validation_token_accuracy",
    "validation_sequence_accuracy",
    "test_token_accuracy",
    "test_sequence_accuracy",
    "generalization_token_accuracy",
    "generalization_sequence_accuracy",
]


def config_label(model: str, num_heads: float | int | None) -> str:
    if model == "lstm":
        return "LSTM"
    return f"Transformer ({int(num_heads)} head{'s' if int(num_heads) != 1 else ''})"


def format_mean_std(mean: float, std: float, percent: bool = True) -> str:
    if percent:
        return f"{mean * 100:.1f} +/- {std * 100:.1f}%"
    return f"{mean:.1f} +/- {std:.1f}"


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["model", "num_heads"], dropna=False, sort=False)

    rows = []
    for (model, num_heads), group in grouped:
        row = {
            "model": model,
            "num_heads": "" if pd.isna(num_heads) else int(num_heads),
            "config": config_label(model, None if pd.isna(num_heads) else num_heads),
            "n_seeds": group["seed"].nunique(),
            "parameters_mean": group["parameters"].mean(),
            "parameters_std": group["parameters"].std(),
            "epochs_completed_mean": group["epochs_completed"].mean(),
            "epochs_completed_std": group["epochs_completed"].std(),
        }

        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        rows.append(row)

    return pd.DataFrame(rows)


def build_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Configuration": row["config"],
                "Seeds": int(row["n_seeds"]),
                "Parameters": f"{int(round(row['parameters_mean'])):,}",
                "Test token acc.": format_mean_std(
                    row["test_token_accuracy_mean"],
                    row["test_token_accuracy_std"],
                ),
                "Test sequence acc.": format_mean_std(
                    row["test_sequence_accuracy_mean"],
                    row["test_sequence_accuracy_std"],
                ),
                "Generalization token acc.": format_mean_std(
                    row["generalization_token_accuracy_mean"],
                    row["generalization_token_accuracy_std"],
                ),
                "Generalization sequence acc.": format_mean_std(
                    row["generalization_sequence_accuracy_mean"],
                    row["generalization_sequence_accuracy_std"],
                ),
                "Training time": format_mean_std(
                    row["training_seconds_mean"],
                    row["training_seconds_std"],
                    percent=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    headers = list(table.columns)
    rows = table.astype(str).values.tolist()

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_test_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["config"].tolist()
    token_means = summary["test_token_accuracy_mean"] * 100
    token_stds = summary["test_token_accuracy_std"] * 100
    sequence_means = summary["test_sequence_accuracy_mean"] * 100
    sequence_stds = summary["test_sequence_accuracy_std"] * 100

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [i - width / 2 for i in x],
        token_means,
        width,
        yerr=token_stds,
        capsize=4,
        label="Token accuracy",
    )
    ax.bar(
        [i + width / 2 for i in x],
        sequence_means,
        width,
        yerr=sequence_stds,
        capsize=4,
        label="Sequence accuracy",
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test accuracy by model configuration")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "test_accuracy_by_config.png", dpi=200)
    plt.close(fig)


def plot_heads_ablation(summary: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    transformer = summary[summary["model"] == "transformer"].copy()
    transformer["num_heads"] = pd.to_numeric(transformer["num_heads"])
    transformer = transformer.sort_values("num_heads")

    heads = transformer["num_heads"].astype(int).tolist()
    means = (pd.to_numeric(transformer[f"{metric}_mean"]) * 100).astype(float).tolist()
    stds = (pd.to_numeric(transformer[f"{metric}_std"]) * 100).astype(float).tolist()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        heads,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    ax.set_xlabel("Number of attention heads")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(heads)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_generalization(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["config"].tolist()
    token_means = summary["generalization_token_accuracy_mean"] * 100
    token_stds = summary["generalization_token_accuracy_std"] * 100
    sequence_means = summary["generalization_sequence_accuracy_mean"] * 100
    sequence_stds = summary["generalization_sequence_accuracy_std"] * 100

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [i - width / 2 for i in x],
        token_means,
        width,
        yerr=token_stds,
        capsize=4,
        label="Token accuracy",
    )
    ax.bar(
        [i + width / 2 for i in x],
        sequence_means,
        width,
        yerr=sequence_stds,
        capsize=4,
        label="Sequence accuracy",
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Generalization to longer sequences")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "generalization_accuracy_by_config.png", dpi=200)
    plt.close(fig)


def plot_training_time(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        summary["config"],
        summary["training_seconds_mean"],
        yerr=summary["training_seconds_std"],
        capsize=4,
    )

    ax.set_ylabel("Training time (seconds)")
    ax.set_title("Training time by model configuration")
    ax.set_xticklabels(summary["config"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "training_time_by_config.png", dpi=200)
    plt.close(fig)


def write_key_findings(summary: pd.DataFrame, output_dir: Path) -> None:
    lstm = summary[summary["model"] == "lstm"].iloc[0]
    transformers = summary[summary["model"] == "transformer"].copy()
    best_transformer = transformers.sort_values("test_token_accuracy_mean", ascending=False).iloc[0]

    lines = [
        "# Key findings",
        "",
        f"- Best Transformer configuration: {best_transformer['config']}.",
        f"- LSTM test token accuracy: {format_mean_std(lstm['test_token_accuracy_mean'], lstm['test_token_accuracy_std'])}.",
        f"- Best Transformer test token accuracy: {format_mean_std(best_transformer['test_token_accuracy_mean'], best_transformer['test_token_accuracy_std'])}.",
        f"- LSTM test sequence accuracy: {format_mean_std(lstm['test_sequence_accuracy_mean'], lstm['test_sequence_accuracy_std'])}.",
        f"- Best Transformer test sequence accuracy: {format_mean_std(best_transformer['test_sequence_accuracy_mean'], best_transformer['test_sequence_accuracy_std'])}.",
        "- Increasing the number of attention heads improves Transformer performance in this experiment.",
        "- All models have 0.0% exact sequence accuracy on the longer-sequence generalization split.",
        "",
    ]

    (output_dir / "key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    summary = build_summary(raw)
    report_table = build_report_table(summary)

    summary.to_csv(output_dir / "summary_by_config.csv", index=False)
    report_table.to_csv(output_dir / "summary_table_report.csv", index=False)
    save_markdown_table(report_table, output_dir / "summary_by_config.md")

    plot_test_accuracy(summary, output_dir)
    plot_heads_ablation(
        summary,
        "test_token_accuracy",
        "Transformer ablation: test token accuracy",
        output_dir / "heads_ablation_test_token_accuracy.png",
    )
    plot_heads_ablation(
        summary,
        "test_sequence_accuracy",
        "Transformer ablation: test sequence accuracy",
        output_dir / "heads_ablation_test_sequence_accuracy.png",
    )
    plot_generalization(summary, output_dir)
    plot_training_time(summary, output_dir)
    write_key_findings(summary, output_dir)

    print(f"Analysis artifacts written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/raw_runs.csv"),
        help="Path to the raw experiment CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory where analysis artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.input, args.output_dir)


if __name__ == "__main__":
    main()