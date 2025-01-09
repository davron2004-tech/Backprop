import numpy as np
from Backprop import *

# 1. Generate Training Data
np.random.seed(42)
X = np.random.rand(1000, 3) * 10 - 5  # 1000 samples, range [-5, 5]
# Define a function for y
def cubic_function(x):
    return (
        x[:, 0]**3 + 3 * x[:, 1]**2 - 2 * x[:, 2]
    )

y = cubic_function(X).reshape(-1, 1)  # Ensure y is (1000, 1)

model = Backprop(3, [Layer(64, 'relu'), Layer(1)])
model.train(X, y, 'mse', 100, 32, 0.01)

model.predict(X)