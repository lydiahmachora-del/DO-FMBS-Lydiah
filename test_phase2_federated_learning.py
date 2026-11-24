from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import phase2_federated_learning as phase2


class FakeModel:
    def __init__(self) -> None:
        self.weights = [np.array([0.0])]
        self.fit_payload = None

    def set_weights(self, weights) -> None:
        self.weights = [np.array(value, copy=True) for value in weights]

    def get_weights(self):
        return self.weights

    def fit(self, inputs, outputs, **kwargs):
        self.fit_payload = (inputs, outputs, kwargs)
        self.weights = [np.array([1.0])]
        return SimpleNamespace(history={
            "face_output_accuracy": [0.5],
            "finger_output_accuracy": [0.5],
            "face_output_loss": [0.7],
            "finger_output_loss": [0.8],
            "loss": [1.5],
        })


class FederatedBatchPropagationTests(unittest.TestCase):
    def test_federated_train_passes_each_client_and_seed_to_batcher(self) -> None:
        clients = []
        for offset in range(2):
            clients.append({
                "face_X": np.arange(6, dtype=np.float32).reshape(3, 2) + offset,
                "face_y": np.array([0, 1, 2]),
                "finger_X": np.arange(4, dtype=np.float32).reshape(2, 2) + offset,
                "finger_y": np.array([0, 1]),
            })
        models = []

        def make_model():
            model = FakeModel()
            models.append(model)
            return model

        with patch.object(phase2, "compile_dual_branch_model", side_effect=lambda model, **_: model):
            trained, history = phase2.federated_train(
                make_model, clients, rounds=1, local_epochs=1, batch_size=2,
                learning_rate=0.01, seed=7,
            )

        self.assertIs(trained, models[0])
        self.assertEqual(history["round"], [1])
        self.assertEqual(len(models), 3)
        for client_id, model in enumerate(models[1:]):
            inputs, outputs, _ = model.fit_payload
            expected = phase2.balanced_batch_records(
                clients[client_id], 100_000 + (7 - 42) + client_id,
            )
            np.testing.assert_array_equal(inputs["face_input"], expected[0])
            np.testing.assert_array_equal(outputs["face_output"], expected[1])
            np.testing.assert_array_equal(inputs["finger_input"], expected[2])
            np.testing.assert_array_equal(outputs["finger_output"], expected[3])


if __name__ == "__main__":
    unittest.main()
