"""Integrated Gradients explanations for tabular IDS models."""

from __future__ import annotations

import time

import numpy as np

from src.models.factory import reshape_for_model


def _positive_class_tensor(predictions):
    """Return the positive-class score tensor from Keras predictions."""
    import tensorflow as tf

    if predictions.shape.rank == 1:
        return predictions
    if predictions.shape[-1] == 1:
        return tf.reshape(predictions, [-1])
    return predictions[:, -1]


def _gradient_batch(model, inputs):
    """Compute gradients of the positive score with respect to model inputs."""
    import tensorflow as tf

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        scores = _positive_class_tensor(predictions)
    gradients = tape.gradient(scores, inputs)
    if gradients is None:
        return tf.zeros_like(inputs)
    return gradients


def explain_integrated_gradients(
    model,
    baseline: np.ndarray,
    X_samples: np.ndarray,
    model_name: str,
    steps: int = 32,
    batch_size: int = 256,
) -> tuple[np.ndarray, float]:
    """Generate dense Integrated Gradients attributions.

    The returned matrix has shape n_samples x n_features and attributes the
    positive-class score relative to the supplied baseline.
    """
    import tensorflow as tf

    start = time.time()
    samples = np.asarray(X_samples, dtype="float32")
    baseline = np.asarray(baseline, dtype="float32").reshape(-1)
    if baseline.shape[0] != samples.shape[1]:
        raise ValueError(
            f"Integrated Gradients baseline has {baseline.shape[0]} features, "
            f"but samples have {samples.shape[1]}."
        )

    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype="float32")
    explanations = np.zeros_like(samples, dtype="float32")

    for sample_idx, sample in enumerate(samples):
        path = baseline[None, :] + alphas[:, None] * (sample[None, :] - baseline[None, :])
        gradients = []
        for start_idx in range(0, len(path), batch_size):
            batch = path[start_idx : start_idx + batch_size]
            batch_model = tf.convert_to_tensor(reshape_for_model(batch, model_name), dtype=tf.float32)
            batch_gradients = _gradient_batch(model, batch_model).numpy().reshape(len(batch), -1)
            gradients.append(batch_gradients)
        path_gradients = np.concatenate(gradients, axis=0)
        avg_gradients = ((path_gradients[:-1] + path_gradients[1:]) / 2.0).mean(axis=0)
        explanations[sample_idx] = (sample - baseline) * avg_gradients

    return explanations.astype("float32"), time.time() - start
