# Q2_graded
# Do not change the above line.

# Remove this comment and type your codes here
class MLP:
    def __init__(self, lr, epochs, layerdims):
        np.random.seed(1)
        self.learning_rate = lr
        self.epochs = epochs
        # init params
        self.w = [] # weights
        n = len(layerdims)
        for i in range(n - 1):
            self.w.append(np.random.random((layerdims[i], layerdims[i + 1])))
        
        self.b = [] # bias
        for i in range(1, n):
            self.b.append(np.random.random(layerdims[i]))

        self.a = [] # each layer output
        for i in range(n):
            self.a.append(np.zeros(layerdims[i]))
        
        self.d = [] # each layer delta
        for i in range(1, n):
            self.d.append(np.zeros(layerdims[i]))

        self.activation_funcs = [] # each layer activation function
        self.derivative_funcs = [] # each layer derivative of activation function

        for i in range(1, n - 1):
            self.activation_funcs.append('relu')
            self.derivative_funcs.append('relu_der')
        
        self.activation_funcs.append('sigmoid')
        self.derivative_funcs.append('sigmoid_der')

    def fit(self, X, Y):
        n = len(self.w) + 1
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                target = Y[i,:]
                self.a[0] = X[i, :]
                # feed forward
                for j in range(n - 1):
                    self.a[j + 1] = self.linear_forward_activation(self.a[j], self.w[j], self.b[j], self.activation_funcs[j])
                # calculate last layer delta value
                self.d[-1] = np.multiply((target - self.a[-1]), self.linear_activation_backward(self.a[-1], self.derivative_funcs[-1]))

                # calculate hidden layers delta values
                for j in range(n - 2, 0, -1):
                    self.d[j - 1] = np.multiply(self.d[j].dot(self.w[j].T), self.linear_activation_backward(self.a[j], self.derivative_funcs[j]))
                # update parameters using delta rule
                for j in range(n - 1):
                    self.w[j] = self.w[j] + self.learning_rate * np.outer(self.a[j], self.d[j])
                    self.b[j] = self.b[j] + self.learning_rate * self.d[j]

    def predict(self, X):
        n = len(self.w) + 1
        A = X.T
        for j in range(n - 1):
            A_prev = A
            A = self.linear_forward_activation(A_prev, self.w[j], self.b[j].reshape(-1, 1), self.activation_funcs[j])
        return np.where(A >= 0.5, 1, 0)

    def linear_forward(self, X, W, B):
        Z = W.T.dot(X) + B
        return Z

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_der(self, X):
        Z = self.sigmoid(X)
        return Z * (1 - Z)

    def relu(self, X):
        A = np.maximum(0, X)
        return A

    def relu_der(self, X):
        X[X <= 0] = 0
        X[X > 0] = 1
        return X

    def linear_forward_activation(self, A_prev, W, b, activation):
        Z = self.linear_forward(A_prev, W, b)
        if activation == 'sigmoid':
            return self.sigmoid(Z)
        elif activation == 'relu':
            return self.relu(Z)

    def linear_activation_backward(self, X, activation):
        dZ = None
        if activation == "relu_der":
            dZ = self.relu_der(X)
            
        elif activation == "sigmoid_der":
            dZ = self.sigmoid_der(X)
        
        return dZ    


# Q2_graded
# Do not change the above line.
learning_rate = 0.001
epochs = 100
# first item is number of features and last layer is number of output neurons
layerdims = [2, 2, 1]
model = MLP(learning_rate, epochs, layerdims)
model.fit(X.T, Y.T)

# Q2_graded
# Do not change the above line.

plot_decision_boundary(lambda x: model.predict(x), X, Y)
plt.title("Decision Boundry")

