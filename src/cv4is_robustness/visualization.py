from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def show_image_batch(images, labels, class_names, n=12) -> None:
    n = min(n, len(images))
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, image, label in zip(axes, images[:n], labels[:n]):
        ax.imshow(image)
        ax.set_title(class_names[label])
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_corruption_grid(image, corruptions, severities) -> None:
    rows = len(corruptions)
    cols = len(severities)

    fig, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 2.5 * rows), squeeze=False
    )

    for row, (name, corruption_fn) in enumerate(corruptions.items()):
        for col, severity in enumerate(severities):
            rng = np.random.default_rng(100 + severity)
            corrupted = corruption_fn(image, severity, rng)
            axes[row, col].imshow(corrupted)
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(f"severity {severity}")
            if col == 0:
                axes[row, col].set_ylabel(name.replace("_", " "))

    plt.tight_layout()
    plt.show()


def plot_robustness_curves(summary_df, metric="accuracy") -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for corruption_name, group in summary_df.groupby("corruption"):
        group = group.sort_values("severity")
        ax.plot(
            group["severity"],
            group[metric],
            marker="o",
            label=corruption_name.replace("_", " "),
        )

    ax.set_xlabel("Severity")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Robustness curves: {metric.replace('_', ' ')}")
    ax.set_xticks(sorted(summary_df["severity"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(summary_df, metric="accuracy") -> None:
    pivot = summary_df.pivot(index="corruption", columns="severity", values=metric)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([name.replace("_", " ") for name in pivot.index])
    ax.set_xlabel("Severity")
    ax.set_title(f"{metric.replace('_', ' ').title()} heatmap")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
    plt.tight_layout()
    plt.show()


def show_failure_examples(examples) -> None:
    if not examples:
        print("No clean-correct / corrupted-wrong examples were found.")
        return

    rows = len(examples)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3.2 * rows), squeeze=False)

    for row, example in enumerate(examples):
        axes[row, 0].imshow(example["clean_image"])
        axes[row, 0].set_title(
            f"Clean\ntrue: {example['true_label']}\n"
            f"pred: {example['clean_prediction']}"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(example["corrupted_image"])
        axes[row, 1].set_title(
            f"Corrupted\npred: {example['corrupted_prediction']}\n"
            f"confidence: {example['corrupted_confidence']:.2f}"
        )
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_monitoring_table(table, value_columns, title) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for column in value_columns:
        ax.plot(
            table["window"],
            table[column],
            marker="o",
            label=column.replace("_", " "),
        )

    ax.set_xlabel("Window")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
