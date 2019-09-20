import pickle
import math

class PredictionResult(dict):
    def __init__(self, score, is_error, message):
        dict.__init__(
            self, is_anomaly=self.check_is_anomaly(score), score=score, is_error=is_error, message=message
        )
    def check_is_anomaly(self,score):
        if score < 50:
            return True
        else:
            return False


class Predictor(object):
    def __init__(self) -> None:
        self.trainer = None

    def load(self, filename):
        with open(filename, mode="rb") as f:
            self.trainer = pickle.load(f)


    def predict(self, features):
        # trainerを使って賢く分類するようにしましょう
        predition = self.trainer.score(features)
        predition_score = math.exp(predition)

        return PredictionResult(
            score=predition_score, is_error=False, message=None
        )
