import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iter=100, distance_func='euclidean'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_func = distance_func
        self.labels_ = None
        self.inertia_ = None
        self.medoids = None
  
    def _distance(self, x1, x2):
        if self.distance_func == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance_func == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError('Invalid distance function')
        
    def fit(self, X):
        n_samples, n_features = X.shape
        self.medoids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            # Step 1: Assign each sample to the nearest medoid
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = [self._distance(X[k], self.medoids[j]) for k in range(n_samples)]
            labels = np.argmin(distances, axis=1)
            # Step 2: Update each medoid to be the sample with the lowest total distance to all other samples in its cluster
            for j in range(self.n_clusters):
                indices = np.where(labels == j)[0]
                medoid_candidates = X[indices]
                medoid_distances = np.zeros(len(medoid_candidates))
                for k in range(len(medoid_candidates)):
                    medoid_distances[k] = np.sum([self._distance(medoid_candidates[k], X[l]) for l in indices])
                best_medoid = medoid_candidates[np.argmin(medoid_distances)]
                self.medoids[j] = best_medoid
        self.labels_ = labels
        self.inertia_ = sum(np.min(distances, axis=1))

    def fit_predict(self,X):
        self.fit(X)
        return self.labels_