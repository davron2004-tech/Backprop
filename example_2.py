import numpy as np
from nnfs.datasets import spiral_data
from Backprop import *

X,y = spiral_data(samples=100, classes=3)
model = Backprop(2, [Layer(64, 'relu'), Layer(3, 'softmax')])
model.train(X, y, 'categorical_crossentropy', 100, 32, 0.01)

model.predict(X)