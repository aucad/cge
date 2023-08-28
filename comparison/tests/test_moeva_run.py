# import joblib
# import numpy as np
# import pytest
# from sklearn.pipeline import Pipeline
# from tensorflow.keras.models import load_model
#
# from comparison.attacks import Moeva2
# from comparison.classifier.tensorflow_classifier import (
#     TensorflowClassifier,
# )
# from comparison.objective_calculator.objective_calculator import (
#     ObjectiveCalculator,
# )
# from comparison.tests.url_constraints import get_url_constraints
#
#
# @pytest.mark.parametrize(
#     "model",
#     [
#         (joblib.load("./comparison/resources/baseline_rf.model")),
#         (
#             TensorflowClassifier(
#                 load_model("./comparison/resources/baseline_nn.model")
#             )
#         ),
#     ],
# )
# def test_run(model):
#     constraints = get_url_constraints()
#     x_clean = np.load(
#         "./comparison/resources/baseline_X_test_candidates.npy")[
#         :10
#     ]
#     y_clean = np.load(
#         "./comparison/resources/baseline_y_test_candidates.npy")[
#         :10
#     ]
#     model = joblib.load("./comparison/resources/baseline_rf.model")
#     preprocessing_pipeline = joblib.load(
#         "./comparison/resources/baseline_scaler.joblib"
#     )
#     model_pipeline = Pipeline(
#         steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
#     )
#
#     attack = Moeva2(
#         model_pipeline,
#         constraints,
#         2,
#         preprocessing_pipeline.transform,
#         n_gen=10,
#         save_history="full",
#         seed=42,
#         n_jobs=1,
#     )
#     out = attack.generate(x_clean, y_clean)
#     assert len(out) == 2
#
#
# def test_objective_calculation():
#     constraints = get_url_constraints()
#     x_clean = np.load(
#         "./comparison/resources/baseline_X_test_candidates.npy")[
#         :10
#     ]
#     y_clean = np.load(
#         "./comparison/resources/baseline_y_test_candidates.npy")[
#         :10
#     ]
#     model = joblib.load("./comparison/resources/baseline_rf.model")
#     preprocessing_pipeline = joblib.load(
#         "./comparison/resources/baseline_scaler.joblib"
#     )
#     model_pipeline = Pipeline(
#         steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
#     )
#
#     x_adv = np.repeat(x_clean[:, np.newaxis, :], 5, axis=1)
#     objective_calculator = ObjectiveCalculator(
#         model_pipeline,
#         constraints,
#         thresholds={"misclassification": 0.5, "distance": 0.2},
#         norm=2,
#         fun_distance_preprocess=preprocessing_pipeline.transform,
#     )
#     success_rate = objective_calculator.success_rate_many(
#         x_clean, y_clean, x_adv
#     )
#     for i in range(7):
#         assert 0 <= success_rate[i] and success_rate[i] <= 1.0
#
#     assert success_rate[0] == 1.0
#     assert success_rate[2] == 1.0
#     assert success_rate[3] == success_rate[1]
#     assert success_rate[4] == 1.0
#     assert success_rate[5] == success_rate[1]
#     assert success_rate[6] == success_rate[1]
#
#     # Computed manually
#     assert success_rate[1] == 0.1
