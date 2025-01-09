# 🔥 **Backpropagation from Scratch in Python**

A lightweight, modular implementation of backpropagation and neural networks in Python, built from the ground up without the use of high-level deep learning libraries like TensorFlow or PyTorch. This project demonstrates a clear understanding of the inner workings of neural networks, including forward propagation, backward propagation, and gradient descent optimization.

---

## 📜 **Features**
- Customizable fully connected (dense) layers.
- Support for popular activation functions:
  - ReLU (Rectified Linear Unit)
  - Sigmoid
  - Softmax
- Support for loss functions:
  - Mean Squared Error (MSE)
  - Categorical Cross-Entropy
- Batch training with gradient descent optimization.
- Modular design for easy extension and customization.

---

## 🏗️ **Project Structure**

- Backprop.py - Core implementation of the Backpropagation framework
- example_1.py - Example 1: Regression on cubic function data
- example_2.py - Example 2: Classification with spiral dataset
- README.md - Project documentation

---

## 🚀 **Getting Started**

### 🔧 **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/davron2004-tech/Backprop.git
   cd Backprop
   ```
___

## 🔌 **Example usage**

```python
from Backprop import *

# Define input data and target function
X = np.random.rand(1000, 3) * 10 - 5  # Random input in range [-5, 5]
y = (X[:, 0]**3 + 3 * X[:, 1]**2 - 2 * X[:, 2]).reshape(-1, 1)

# Build and train the model
model = Backprop(input_size=3, layers=[
    Layer(64, 'relu'),
    Layer(1)
])
model.train(X, y, loss_function='mse', epochs=100, batch_size=32, learning_rate=0.01)

# Make predictions
predictions = model.predict(X)
```

```python
from nnfs.datasets import spiral_data
from Backprop import *

# Load dataset
X, y = spiral_data(samples=100, classes=3)

# Build and train the model
model = Backprop(input_size=2, layers=[
    Layer(64, 'relu'),
    Layer(3, 'softmax')
])
model.train(X, y, loss_function='categorical_crossentropy', epochs=100, batch_size=32, learning_rate=0.01)

# Make predictions
predictions = model.predict(X)
```
