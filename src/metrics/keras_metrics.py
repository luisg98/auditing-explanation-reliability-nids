"""Keras metrics used during IDS model training."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="IDS")
class BinaryF2Score(tf.keras.metrics.Metric):
    """Binary F2 score at a fixed decision threshold."""

    def __init__(self, threshold: float = 0.5, name: str = "f2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), self.dtype)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, self.dtype)
        if sample_weight is None:
            weights = tf.ones_like(y_true, dtype=self.dtype)
        else:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)

        self.tp.assign_add(tf.reduce_sum(weights * y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum(weights * (1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(weights * y_true * (1.0 - y_pred)))

    def result(self):
        beta_sq = tf.constant(4.0, dtype=self.dtype)
        numerator = (1.0 + beta_sq) * self.tp
        denominator = numerator + self.fp + beta_sq * self.fn
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        for value in (self.tp, self.fp, self.fn):
            value.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def binary_classification_metrics() -> list[object]:
    """Return metrics for binary IDS training."""
    return [
        "accuracy",
        BinaryF2Score(name="f2"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
