from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score


def entropy_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=1)


@torch.no_grad()
def evaluate_corrupted_subset(
    model,
    raw_dataset,
    indices,
    corruption_fn,
    severity,
    eval_transform,
    device,
    batch_size=128,
    seed=42,
) -> dict:
    model.eval()

    labels_all = []
    predictions_all = []
    confidence_all = []
    entropy_all = []

    selected_indices = list(indices)

    for start in range(0, len(selected_indices), batch_size):
        batch_indices = selected_indices[start:start + batch_size]
        tensors = []
        labels = []

        for dataset_index in batch_indices:
            image, label = raw_dataset[dataset_index]
            rng = np.random.default_rng(seed + 1009 * int(dataset_index))
            corrupted = corruption_fn(image, severity, rng)
            tensors.append(eval_transform(corrupted))
            labels.append(label)

        images_tensor = torch.stack(tensors).to(device)
        logits = model(images_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        entropy = entropy_from_probabilities(probabilities)

        labels_all.extend(labels)
        predictions_all.extend(predictions.cpu().tolist())
        confidence_all.extend(confidence.cpu().tolist())
        entropy_all.extend(entropy.cpu().tolist())

    labels_array = np.asarray(labels_all)
    predictions_array = np.asarray(predictions_all)

    return {
        "accuracy": float(np.mean(labels_array == predictions_array)),
        "macro_f1": float(
            f1_score(labels_array, predictions_array, average="macro", zero_division=0)
        ),
        "mean_confidence": float(np.mean(confidence_all)),
        "mean_entropy": float(np.mean(entropy_all)),
        "labels": labels_array,
        "predictions": predictions_array,
    }


@torch.no_grad()
def collect_failure_examples(
    model,
    raw_dataset,
    indices,
    corruption_fn,
    severity,
    eval_transform,
    class_names,
    device,
    max_examples=6,
    seed=42,
) -> list[dict]:
    model.eval()
    examples = []

    for dataset_index in indices:
        clean_image, label = raw_dataset[dataset_index]
        rng = np.random.default_rng(seed + 1009 * int(dataset_index))
        corrupted_image = corruption_fn(clean_image, severity, rng)

        batch = torch.stack(
            [eval_transform(clean_image), eval_transform(corrupted_image)]
        ).to(device)

        probabilities = torch.softmax(model(batch), dim=1)
        predictions = probabilities.argmax(dim=1).cpu().tolist()
        clean_prediction, corrupted_prediction = predictions

        if clean_prediction == label and corrupted_prediction != label:
            examples.append(
                {
                    "clean_image": clean_image,
                    "corrupted_image": corrupted_image,
                    "true_label": class_names[label],
                    "clean_prediction": class_names[clean_prediction],
                    "corrupted_prediction": class_names[corrupted_prediction],
                    "corrupted_confidence": float(probabilities[1].max().cpu().item()),
                }
            )

        if len(examples) >= max_examples:
            break

    return examples
