import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for use in backward pass
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues, learning_rate):
        # Gradients w.r.t weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        
        # Update weights and biases
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        # Gradient is 1 for positive inputs, 0 for negative
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten single_output (column vector)
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Apply the chain rule
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        self.loss = np.mean(np.square(y_pred - y_true.reshape(-1, 1)), axis=0)
        return self.loss[0]
    def backward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_pred)
        
        # Reshape y_true if necessary
        y_true = y_true.reshape(y_pred.shape)
        
        # Gradient of MSE w.r.t predictions
        self.dinputs = (2 / samples) * (y_pred - y_true)
        
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        self.loss = np.mean(negative_log_likelihoods)
        return self.loss
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        
        # If labels are sparse integers, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        
        # Gradient of Categorical Cross-Entropy Loss with respect to predictions
        self.dinputs = (y_pred - y_true) / samples


class Layer:
    def __init__(self, neurons, activation = None):
        self.neurons = neurons
        self.activation = activation
    
class Backprop:
    def __init__(self, input_size ,layers):
        self.input_size = input_size
        self.layers = []
        self.activations = []
        self.make_layers_activations(layers)
    def add_layer(self, layer):
        self.layers.append(layer)
    def make_layers_activations(self, layers):
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(Layer_Dense(self.input_size, layers[i].neurons))
            else:
                self.layers.append(Layer_Dense(layers[i-1].neurons, layers[i].neurons))

            if layers[i].activation == 'relu':
                self.activations.append(Activation_ReLU())
            elif layers[i].activation == 'sigmoid':
                self.activations.append(Activation_Sigmoid())
            elif layers[i].activation == 'softmax':
                self.activations.append(Activation_Softmax())
            else:
                self.activations.append(None)
    def train(self, X, y, loss_function, epochs, batch_size, learning_rate):
        num_samples = X.shape[0]
    
        # Normalize input data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
        # Select Loss Function
        if loss_function == 'mse':
            loss = Loss_MeanSquaredError()
        elif loss_function == 'categorical_crossentropy':
            loss = Loss_CategoricalCrossentropy()
        else:
            raise ValueError("Unsupported loss function")
    
        # Training Loop
        for epoch in range(epochs):
            epoch_loss = 0  # Reset loss for each epoch
        
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
            
                # === Forward Pass ===
                output = X_batch
                for layer, activation in zip(self.layers, self.activations):
                    layer.forward(output)
                    output = layer.output
                    if activation is not None:
                        activation.forward(output)
                        output = activation.output
            
                # Compute Loss (Average Loss for this Batch)
                batch_loss = loss.forward(output, y_batch)
                epoch_loss += batch_loss * len(X_batch)  # Sum weighted loss by batch size
            
                # === Backward Pass ===
                loss.backward(output, y_batch)
                dinputs = loss.dinputs
            
                for layer, activation in reversed(list(zip(self.layers, self.activations))):
                    if activation is not None:
                        activation.backward(dinputs)
                        dinputs = activation.dinputs
                    layer.backward(dinputs, learning_rate)
                    dinputs = layer.dinputs
        
            # Average Loss per Epoch
            epoch_loss /= num_samples
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
    def predict(self, X):
        output = X
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(output)
            output = layer.output
            if activation is not None:
                activation.forward(output)
                output = activation.output
        return output