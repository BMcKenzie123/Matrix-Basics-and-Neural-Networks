"""
Multilayer Perceptron (MLP) Neural Network Implementation
Built from scratch using NumPy for the NBA Optimal Team Selection problem.

This module implements:
- Forward propagation
- Backpropagation
- Various activation functions
- Cost functions
- Weight initialization and updates
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


class ActivationFunctions:
    """Collection of activation functions and their derivatives."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        σ(z) = 1 / (1 + e^(-z))

        Args:
            z: Input array

        Returns:
            Activated output in range (0, 1)
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid function.
        σ'(z) = σ(z) * (1 - σ(z))

        Args:
            a: Already activated output (sigmoid(z))

        Returns:
            Derivative value
        """
        return a * (1 - a)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU (Rectified Linear Unit) activation function.
        ReLU(z) = max(0, z)

        Args:
            z: Input array

        Returns:
            Activated output
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU function.
        ReLU'(z) = 1 if z > 0, else 0

        Args:
            z: Pre-activation input

        Returns:
            Derivative value
        """
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent activation function.
        tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Args:
            z: Input array

        Returns:
            Activated output in range (-1, 1)
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh function.
        tanh'(z) = 1 - tanh²(z)

        Args:
            a: Already activated output (tanh(z))

        Returns:
            Derivative value
        """
        return 1 - np.power(a, 2)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for multi-class classification.
        softmax(z)_i = e^(z_i) / Σ e^(z_j)

        Args:
            z: Input array

        Returns:
            Probability distribution (sums to 1)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        """Linear activation (identity function)."""
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of linear activation."""
        return np.ones_like(z)


class CostFunctions:
    """Collection of cost/loss functions."""

    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                              epsilon: float = 1e-15) -> float:
        """
        Binary cross-entropy loss.
        L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

        Args:
            y_pred: Predicted probabilities
            y_true: True labels (0 or 1)
            epsilon: Small value to prevent log(0)

        Returns:
            Cost value
        """
        m = y_true.shape[1]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -1/m * np.sum(y_true * np.log(y_pred) +
                            (1 - y_true) * np.log(1 - y_pred))
        return float(np.squeeze(cost))

    @staticmethod
    def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Squared Error loss.
        MSE = 1/m * Σ(y - ŷ)²

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Cost value
        """
        m = y_true.shape[1]
        cost = 1/(2*m) * np.sum(np.power(y_pred - y_true, 2))
        return float(np.squeeze(cost))

    @staticmethod
    def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                                   epsilon: float = 1e-15) -> float:
        """
        Categorical cross-entropy loss for multi-class classification.
        L = -1/m * ΣΣ y_ij * log(ŷ_ij)

        Args:
            y_pred: Predicted probabilities (softmax output)
            y_true: One-hot encoded true labels
            epsilon: Small value to prevent log(0)

        Returns:
            Cost value
        """
        m = y_true.shape[1]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -1/m * np.sum(y_true * np.log(y_pred))
        return float(np.squeeze(cost))


class MultilayerPerceptron:
    """
    Multilayer Perceptron Neural Network.

    A fully-connected feedforward neural network with customizable
    architecture, activation functions, and training parameters.

    Attributes:
        layer_dims: List of layer dimensions [input, hidden1, ..., output]
        parameters: Dictionary containing weights and biases
        activations: List of activation functions for each layer
        learning_rate: Learning rate for gradient descent
        cost_history: List of cost values during training
    """

    def __init__(self,
                 layer_dims: List[int],
                 activations: List[str] = None,
                 learning_rate: float = 0.01,
                 random_seed: int = 42):
        """
        Initialize the MLP.

        Args:
            layer_dims: List defining network architecture
                       [n_input, n_hidden1, n_hidden2, ..., n_output]
            activations: List of activation functions for each layer
                        Options: 'sigmoid', 'relu', 'tanh', 'softmax', 'linear'
            learning_rate: Step size for gradient descent
            random_seed: Seed for reproducible weight initialization
        """
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of layers (excluding input)
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.cost_history = []
        self.accuracy_history = []

        # Set default activations (ReLU for hidden, sigmoid for output)
        if activations is None:
            self.activations = ['relu'] * (self.L - 1) + ['sigmoid']
        else:
            self.activations = activations

        # Initialize parameters
        self.parameters = {}
        self._initialize_parameters()

        # Cache for storing intermediate values during forward/backward prop
        self.cache = {}

        # Activation function mappings
        self.activation_funcs = {
            'sigmoid': ActivationFunctions.sigmoid,
            'relu': ActivationFunctions.relu,
            'tanh': ActivationFunctions.tanh,
            'softmax': ActivationFunctions.softmax,
            'linear': ActivationFunctions.linear
        }

        self.activation_derivatives = {
            'sigmoid': ActivationFunctions.sigmoid_derivative,
            'relu': ActivationFunctions.relu_derivative,
            'tanh': ActivationFunctions.tanh_derivative,
            'linear': ActivationFunctions.linear_derivative
        }

    def _initialize_parameters(self) -> None:
        """
        Initialize weights and biases using He initialization for ReLU
        and Xavier initialization for sigmoid/tanh.

        He initialization: W ~ N(0, sqrt(2/n_prev))
        Xavier initialization: W ~ N(0, sqrt(1/n_prev))
        """
        np.random.seed(self.random_seed)

        for l in range(1, self.L + 1):
            n_prev = self.layer_dims[l - 1]
            n_curr = self.layer_dims[l]

            # Use He initialization for ReLU, Xavier for others
            if l <= len(self.activations) and self.activations[l-1] == 'relu':
                scale = np.sqrt(2.0 / n_prev)
            else:
                scale = np.sqrt(1.0 / n_prev)

            self.parameters[f'W{l}'] = np.random.randn(n_curr, n_prev) * scale
            self.parameters[f'b{l}'] = np.zeros((n_curr, 1))

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        For each layer l:
            Z[l] = W[l] · A[l-1] + b[l]
            A[l] = g[l](Z[l])

        Args:
            X: Input data of shape (n_features, m_samples)

        Returns:
            AL: Output of the final layer (predictions)
        """
        self.cache = {'A0': X}
        A = X

        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            # Linear transformation: Z = W·A + b
            Z = np.dot(W, A) + b
            self.cache[f'Z{l}'] = Z

            # Apply activation function
            activation = self.activations[l - 1]
            A = self.activation_funcs[activation](Z)
            self.cache[f'A{l}'] = A

        return A

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray,
                     cost_function: str = 'binary_cross_entropy') -> float:
        """
        Compute the cost/loss.

        Args:
            AL: Predicted output from forward propagation
            Y: True labels
            cost_function: Type of cost function to use

        Returns:
            Cost value
        """
        cost_funcs = {
            'binary_cross_entropy': CostFunctions.binary_cross_entropy,
            'mse': CostFunctions.mean_squared_error,
            'categorical_cross_entropy': CostFunctions.categorical_cross_entropy
        }

        return cost_funcs[cost_function](AL, Y)

    def backward_propagation(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform backward propagation to compute gradients.

        Uses the chain rule to compute:
            dZ[l] = dA[l] * g'[l](Z[l])
            dW[l] = 1/m * dZ[l] · A[l-1].T
            db[l] = 1/m * Σ dZ[l]
            dA[l-1] = W[l].T · dZ[l]

        Args:
            Y: True labels of shape (n_output, m_samples)

        Returns:
            gradients: Dictionary containing gradients for all parameters
        """
        gradients = {}
        m = Y.shape[1]

        # Get the output layer activation
        AL = self.cache[f'A{self.L}']

        # Initialize backpropagation
        # For sigmoid output with binary cross-entropy: dAL = -(y/a - (1-y)/(1-a))
        # Simplifies to: dZL = AL - Y (when combined with sigmoid derivative)
        if self.activations[-1] == 'sigmoid':
            dZ = AL - Y
        elif self.activations[-1] == 'softmax':
            # For softmax with categorical cross-entropy
            dZ = AL - Y
        else:
            # General case
            dA = -(np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))
            dZ = dA * self.activation_derivatives[self.activations[-1]](
                self.cache[f'Z{self.L}']
            )

        # Compute gradients for output layer
        A_prev = self.cache[f'A{self.L - 1}']
        gradients[f'dW{self.L}'] = (1/m) * np.dot(dZ, A_prev.T)
        gradients[f'db{self.L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # Backpropagate through hidden layers
        for l in reversed(range(1, self.L)):
            # Compute dA for current layer
            W_next = self.parameters[f'W{l + 1}']
            dA = np.dot(W_next.T, dZ)

            # Compute dZ based on activation function
            activation = self.activations[l - 1]
            if activation == 'relu':
                dZ = dA * self.activation_derivatives['relu'](self.cache[f'Z{l}'])
            elif activation == 'sigmoid':
                dZ = dA * self.activation_derivatives['sigmoid'](self.cache[f'A{l}'])
            elif activation == 'tanh':
                dZ = dA * self.activation_derivatives['tanh'](self.cache[f'A{l}'])
            else:
                dZ = dA * self.activation_derivatives['linear'](self.cache[f'Z{l}'])

            # Compute gradients
            A_prev = self.cache[f'A{l - 1}']
            gradients[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        return gradients

    def update_parameters(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Update parameters using gradient descent.

        W[l] = W[l] - α * dW[l]
        b[l] = b[l] - α * db[l]

        Args:
            gradients: Dictionary containing gradients
        """
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * gradients[f'db{l}']

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              epochs: int = 1000,
              cost_function: str = 'binary_cross_entropy',
              print_cost: bool = True,
              print_interval: int = 100,
              early_stopping: bool = False,
              patience: int = 50,
              min_delta: float = 1e-6) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Args:
            X: Training data of shape (n_features, m_samples)
            Y: Labels of shape (n_output, m_samples)
            epochs: Number of training iterations
            cost_function: Cost function to use
            print_cost: Whether to print cost during training
            print_interval: Interval for printing cost
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement

        Returns:
            History dictionary with cost and accuracy values
        """
        self.cost_history = []
        self.accuracy_history = []
        best_cost = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Forward propagation
            AL = self.forward_propagation(X)

            # Compute cost
            cost = self.compute_cost(AL, Y, cost_function)
            self.cost_history.append(cost)

            # Compute accuracy
            predictions = (AL > 0.5).astype(int)
            accuracy = np.mean(predictions == Y)
            self.accuracy_history.append(accuracy)

            # Backward propagation
            gradients = self.backward_propagation(Y)

            # Update parameters
            self.update_parameters(gradients)

            # Print progress
            if print_cost and epoch % print_interval == 0:
                print(f"Epoch {epoch}: Cost = {cost:.6f}, Accuracy = {accuracy:.4f}")

            # Early stopping check
            if early_stopping:
                if cost < best_cost - min_delta:
                    best_cost = cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        return {
            'cost': self.cost_history,
            'accuracy': self.accuracy_history
        }

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input data of shape (n_features, m_samples)
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        AL = self.forward_propagation(X)
        predictions = (AL > threshold).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Input data of shape (n_features, m_samples)

        Returns:
            Probability predictions
        """
        return self.forward_propagation(X)

    def score(self, X: np.ndarray, Y: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate prediction accuracy.

        Args:
            X: Input data
            Y: True labels
            threshold: Classification threshold

        Returns:
            Accuracy score
        """
        predictions = self.predict(X, threshold)
        return np.mean(predictions == Y)

    def get_architecture_summary(self) -> str:
        """Get a string summary of the network architecture."""
        summary = "MLP Architecture:\n"
        summary += "=" * 50 + "\n"
        summary += f"Input Layer: {self.layer_dims[0]} neurons\n"

        for l in range(1, self.L):
            summary += f"Hidden Layer {l}: {self.layer_dims[l]} neurons "
            summary += f"(activation: {self.activations[l-1]})\n"

        summary += f"Output Layer: {self.layer_dims[-1]} neurons "
        summary += f"(activation: {self.activations[-1]})\n"
        summary += "=" * 50 + "\n"

        # Count total parameters
        total_params = 0
        for l in range(1, self.L + 1):
            w_params = self.parameters[f'W{l}'].size
            b_params = self.parameters[f'b{l}'].size
            total_params += w_params + b_params

        summary += f"Total Parameters: {total_params:,}\n"

        return summary


class TeamSelectionMLP(MultilayerPerceptron):
    """
    Specialized MLP for NBA team selection problem.

    This class extends the base MLP with specific functionality
    for evaluating and selecting optimal basketball team combinations.
    """

    def __init__(self,
                 n_features: int,
                 hidden_layers: List[int] = [64, 32, 16],
                 learning_rate: float = 0.01,
                 random_seed: int = 42):
        """
        Initialize the Team Selection MLP.

        Args:
            n_features: Number of input features (player statistics)
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for training
            random_seed: Random seed for reproducibility
        """
        # Construct layer dimensions: input -> hidden layers -> output (1 for binary)
        layer_dims = [n_features] + hidden_layers + [1]

        # Use ReLU for hidden layers and sigmoid for output
        activations = ['relu'] * len(hidden_layers) + ['sigmoid']

        super().__init__(
            layer_dims=layer_dims,
            activations=activations,
            learning_rate=learning_rate,
            random_seed=random_seed
        )

    def evaluate_player(self, player_features: np.ndarray) -> float:
        """
        Evaluate a single player's suitability score.

        Args:
            player_features: Feature vector for one player

        Returns:
            Suitability score between 0 and 1
        """
        if player_features.ndim == 1:
            player_features = player_features.reshape(-1, 1)
        return float(self.predict_proba(player_features)[0, 0])

    def rank_players(self, X: np.ndarray, player_names: List[str]) -> List[Tuple[str, float]]:
        """
        Rank all players by their suitability scores.

        Args:
            X: Feature matrix (n_features, n_players)
            player_names: List of player names

        Returns:
            List of (player_name, score) tuples, sorted by score descending
        """
        scores = self.predict_proba(X).flatten()
        ranked = list(zip(player_names, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
