import numpy as np

class LogisticRegression:
    # training X and y
    def __init__(self, X=None, y=None, lr=0.001, num_epochs=10000):
        self.lr = lr
        self.num_epochs = num_epochs
        self.X = X
        self.y = y
        self.theta = None
        self.theta_0 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        n_samples = len(self.X)
        n_features = len(self.X[0])

        self.theta = np.zeros(n_features)
        self.theta_0 = 0

        for i in range(self.num_epochs):
            pred = self.sigmoid(self.theta_0 + np.dot(self.X, self.theta))
            dTheta = (1/n_samples) * np.dot(self.X.T, (self.y - pred))  # Compute the gradient with respect to theta
            dTheta0 = (1/n_samples) * np.sum(self.y - pred)  # Compute the gradient with respect to theta_0

            self.theta = self.theta + self.lr * dTheta  # Update theta using gradient ascent
            self.theta_0 = self.theta_0 + self.lr * dTheta0  # Update theta_0 using gradient ascent

    def get_params(self):
        return (self.theta, self.theta_0)

    # use precomputed theta values
    def set_params(self, theta, theta_0):
        self.theta = theta
        self.theta_0 = theta_0

    def predict(self, X):
        y_hat = self.sigmoid(self.theta_0 + np.dot(X, self.theta))
        return [0 if y <= 0.5 else 1 for y in y_hat]

    def get_confidence(self, X):
        return self.sigmoid(self.theta_0 + np.dot(X, self.theta))
