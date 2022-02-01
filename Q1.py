# Q1_graded
# Do not change the above line.

class Perceptron:
    def __init__(self, lr, epochs, sgd):
        self.learning_rate = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.sgd = sgd


    def activation(self, x):
        # replaces each item in array with its activation function value
        return np.where(x >= 0, 1, 0)

    
    def fit(self, X, Y):
        np.random.seed(1)
        # initializing weights and bias
        self.weights = np.random.random((X.shape[0], 1))
        self.bias = np.zeros((1, 1))
        # stochastic
        if self.sgd:
            for _ in range(self.epochs):
                # calculate predicted and error values
                predicted = self.activation(np.dot(self.weights.T, X))
                error = Y - predicted
                # update weights and bias using delta rule
                for i in range(X.shape[1]):
                    self.weights += (X[:,i].reshape(X.shape[0], 1) * error[0, i]) * self.learning_rate
                    self.bias += error[0, i] * self.learning_rate
        # batch
        else:
            m = X.shape[1]
            for _ in range(self.epochs):
                # calculate predicted and error values
                predicted = self.activation(np.dot(self.weights.T, X))
                error = (1/m) *  np.sum(Y - predicted)
                # update weights and bias using delta rule
                for i in range(m):
                    self.weights += (X[:,i].reshape(X.shape[0], 1) * error) * self.learning_rate
                    self.bias += error * self.learning_rate


    def predict(self, X):
        return self.activation(np.dot(self.weights.T, X.T))

# Q1_graded
# Do not change the above line.

learning_rate = 0.01
epochs = 100
sgd = True
model = Perceptron(learning_rate, epochs, sgd)
model.fit(X, Y)

# Q1_graded
# Do not change the above line.

plot_decision_boundary(lambda x: model.predict(x), X, Y)
plt.title("Decision Boundry")

