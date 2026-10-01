"""Numerical and selection invariants for the directed RQ2 recheck."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts import review_binary_diagnostics as review
from src.leakage_free_tabular import epistemic_audit as audit
from src.leakage_free_tabular_ext import study2_dependency_perturbations as ops


def test_tie_policies_preserve_strict_ranks_and_do_not_drop_constant_accounts():
    a = np.array([[4, 2, -2, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
    stable = review.ranking(a, "stable", ["a", "b"])
    reverse = review.ranking(a, "reverse", ["a", "b"])
    assert stable.tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    assert reverse.tolist() == [[0, 2, 1, 4, 3], [4, 3, 2, 1, 0]]
    random = review.ranking(a, "random_0", ["a", "b"])
    np.testing.assert_array_equal(random, review.ranking(a, "random_0", ["a", "b"]))
    for order in [reverse, random]:
        assert order[0, 0] == 0
        assert set(order[0, :3]) == {0, 1, 2}
        np.testing.assert_array_equal(np.sort(order, axis=1), stable * 0 + np.arange(5))


def test_batched_quadrature_matches_sequential_rule_and_linear_completeness():
    model = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(1)])
    model.layers[0].set_weights([np.array([[.1], [.2], [-.15]], np.float32), np.array([.5], np.float32)])
    X = np.array([[.2, .1, .3], [.4, .1, .2]], dtype=np.float32)
    reference = np.zeros_like(X)
    targets = np.array([0, 1])
    actual = review.batched_ig(model, X, targets, reference, 64)
    expected = audit.explain_integrated_gradients(model, X, targets, reference, steps=64, batch_size=2)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    delta = audit.target_probabilities(model, X, targets) - audit.target_probabilities(model, reference, targets)
    np.testing.assert_allclose(actual.sum(axis=1), delta, atol=1e-7)


def test_batched_functional_calculation_matches_original_operator_runner():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(3, 8)).astype(np.float32)
    pool = rng.normal(size=(20, 8)).astype(np.float32)
    covariance = np.cov(pool, rowvar=False) + 1e-6 * np.eye(8)
    context = ops.OperatorContext(SimpleNamespace(X_test=X), pool, np.arange(20)%2,
        pool.mean(axis=0), covariance, np.sqrt(np.diag(covariance)), np.median(pool, axis=0),
        np.stack([np.median(pool, axis=0)]*2), np.arange(8))
    identity = pd.DataFrame({key: [0]*3 for key in audit.IDENTITY_COLUMNS})
    identity = identity.assign(task_id="test", seed=42, flow_id=["a", "b", "c"],
        sample_order=np.arange(3), target_class_id=[0, 1, 1], true_class_id=[0, 1, 1],
        target_class="x", confidence=.8, model_margin=.6, class_specific_inference_eligible=True)
    def model(x, training=False):
        return 1/(1+np.exp(-np.asarray(x).sum(axis=1)))
    attrs = rng.normal(size=X.shape)
    actual = review.evaluate(model, context, identity, review.ranking(attrs, "stable", identity.flow_id),
                             "test", "mlp", 42, "shap")
    study = dict(operators=[dict(operator_id=o, family=o) for o in review.OPERATORS], random_feature_repetitions=50)
    _, expected = ops._evaluate_operators(model, X, identity.target_class_id.to_numpy(),
        identity.true_class_id.to_numpy(), attrs, identity, context, study=study,
        operator_ids=review.OPERATORS, fractions=review.FRACTIONS, bins=5,
        task_id="test", model_name="mlp", seed=42, method="shap", nearest_cap=20)
    keys = ["flow_id", "operator", "functional_check"]
    merged = actual.merge(expected, on=keys, validate="one_to_one")
    np.testing.assert_allclose(merged.value, merged.attribution_minus_random_aopc, atol=1e-7)
