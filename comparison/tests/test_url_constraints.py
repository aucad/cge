import numpy as np
import tensorflow as tf

from comparison.constraints.constraints_executor import \
    TensorFlowConstraintsExecutor
from comparison.constraints.relation_constraint import AndConstraint
from comparison.tests.url_constraints import get_url_constraints


def test_tf_constraints():
    constraints = get_url_constraints()
    x_clean = np.load(
        "./comparison/resources/baseline_X_test_candidates.npy")
    x_clean = tf.convert_to_tensor(x_clean[:2], dtype=tf.float32)
    executor = TensorFlowConstraintsExecutor(
        AndConstraint(constraints.relation_constraints)
    )
    c_eval = executor.execute(x_clean)
    assert tf.reduce_all(c_eval == 0)
