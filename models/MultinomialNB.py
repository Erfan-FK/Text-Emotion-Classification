import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._class_counts = np.zeros(n_classes, dtype=np.float64)
        self._feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        self._class_log_priors = np.zeros(n_classes, dtype=np.float64)
        self._feature_log_probs = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._class_counts[idx] = X_c.shape[0]
            self._feature_counts[idx, :] = X_c.sum(axis=0) + self.alpha
            self._class_log_priors[idx] = np.log(self._class_counts[idx] / n_samples)

        self._feature_log_probs = np.log(self._feature_counts / self._feature_counts.sum(axis=1, keepdims=True))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        log_posteriors = []

        for idx, c in enumerate(self._classes):
            log_prior = self._class_log_priors[idx]
            log_likelihood = np.sum(x * self._feature_log_probs[idx, :])
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)

        return self._classes[np.argmax(log_posteriors)]