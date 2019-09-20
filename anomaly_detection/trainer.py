import pickle
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors


class InsufficientTrainingDataError(Exception):
    def __init__(self, data) -> None:
        message = (
            "Number of training data (%d examples) is insufficient to train parameters"
            % len(data)
        )
        super(Exception, self).__init__(message)

    def error_type(self):
        return "InsufficientTrainingDataError"


class TooMuchComponentError(Exception):
    def __init__(self, data,n_components) -> None:
        message = (
            "Number of training data (%d examples) is insufficient to train parameters"
            % len(data)
        )
        super(Exception, self).__init__(message)

    def error_type(self):
        return "TooMuchComponentError"

class Trainer(object):
    def __init__(self):
        self.model = None

    def train(self, data):
        if len(data) == 0:
            raise InsufficientTrainingDataError(data)
        #self.model = OneClassSVM(nu=0.003, kernel='rbf', gamma='auto')
        #self.model = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
        self.model = GaussianMixture(n_components=5)
        if len(data) < self.model.n_components:
            raise TooMuchComponentError(data,self.model.n_components)
        self.model.fit(data)

    def save(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self.model, f)

    def SVM_train(self,data):
        if len(data) == 0:
            raise InsufficientTrainingDataError(data)
        self.model = OneClassSVM(nu=0.003, kernel='rbf', gamma='auto')
        self.data.fit(data)