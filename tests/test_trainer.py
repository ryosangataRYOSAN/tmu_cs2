import unittest
import pickle
from anomaly_detection.trainer import (
    Trainer,
    InsufficientTrainingDataError,
    TooMuchComponentError,
)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        with open("data/train.pickle", mode="rb") as f:
            self.train_data = pickle.load(f)

    def test_train_and_save(self):
        trainer = Trainer()
        trainer.train(self.train_data["features"])
        trainer.save("tmp.model")

    def test_train_with_insufficient_data(self):
        trainer = Trainer()
        with self.assertRaises(InsufficientTrainingDataError):
            trainer.train([])

    def test_train_with_small_data(self):
        trainer = Trainer()
        with self.assertRaises(TooMuchComponentError):
            trainer.train([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_train_with_current_size_data(self):
        trainer = Trainer()
        test_data = []
        for i in range(20):
            test_data.append([i, i + 1, i + 2])
        trainer.train(test_data)
