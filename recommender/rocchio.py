from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestCentroid


class RocchioRecommender(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha = 0.5, beta = 0.8):
        self.centroids = None
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = 1 - beta

    def fit(self, X, y):
        NC = NearestCentroid(metric='cosine')
        NC.fit(X, y)

        if NC.classes_[0] == 0:
            self.centroids = NC.centroids_
        else:
            self.centroids = [NC.centroids_[1], NC.centroids_[0]]
        return self

    def predict(self, X):
        distance = list(enumerate(cosine_similarity(self.centroids, X)))
        len = distance[0][1].size
        scores = []

        for i in range(0, len):
            score = distance[1][1][i] * self.BETA - distance[0][1][i] * self.GAMMA
            scores.append(1 if score > self.ALPHA else 0)

        return scores

    def predict_proba(self, X):
        distance = list(enumerate(cosine_similarity(self.centroids, X)))
        len = distance[0][1].size
        scores = []

        for i in range(0, len):
            score = distance[1][1][i] * self.BETA - distance[0][1][i] * self.GAMMA
            scores.append(score)

        return scores


    def predict_log_proba(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        pass