# NBA Optimal Team Selection using MLP Neural Network

AIT-204 Topic 2 Assignment: Building an Artificial Neural Network for Basketball Team Optimization

## Overview

This project implements a **Multilayer Perceptron (MLP)** neural network from scratch to identify an optimal NBA basketball team of 5 players from a pool of 100 players. The implementation includes:

- Forward propagation
- Backpropagation
- Gradient descent optimization
- Interactive Streamlit web application

## Project Structure

```
.
├── app.py                    # Streamlit web application
├── mlp/
│   ├── __init__.py          # Package initialization
│   ├── neural_network.py    # MLP implementation (forward/backprop)
│   └── data_processing.py   # NBA data processing and team building
├── DOCUMENTATION.md         # Complete technical report
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── README.md               # This file
```

## Features

### Neural Network Implementation

- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax
- **Cost Functions**: Binary Cross-Entropy, MSE, Categorical Cross-Entropy
- **Weight Initialization**: He (for ReLU) and Xavier (for Sigmoid/Tanh)
- **Customizable Architecture**: Any number of hidden layers

### Player Evaluation

- 9 input features covering offensive, defensive, and efficiency metrics
- Composite scoring for player quality assessment
- Position-balanced team selection

### Streamlit Application

- Interactive network architecture visualization
- Real-time training progress
- Player rankings and team analysis
- Radar charts and performance metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Using the Application

1. **Select Data Source**: Use sample data or upload NBA CSV
2. **Configure Architecture**: Adjust hidden layers and neurons
3. **Set Training Parameters**: Learning rate and epochs
4. **Train Model**: Click "Train MLP" button
5. **View Results**: Explore team selection and analysis

## MLP Architecture

```
Input Layer (9 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Hidden Layer 3 (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

## Input Features

| Feature | Description |
|---------|-------------|
| pts | Points per game |
| reb | Rebounds per game |
| ast | Assists per game |
| net_rating | Team +/- per 100 possessions |
| oreb_pct | Offensive rebound percentage |
| dreb_pct | Defensive rebound percentage |
| usg_pct | Usage rate |
| ts_pct | True shooting percentage |
| ast_pct | Assist percentage |

## Key Equations

### Forward Propagation
```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = g(Z[l])
```

### Backpropagation
```
dZ[L] = A[L] - Y
dW[l] = (1/m) * dZ[l] · A[l-1].T
db[l] = (1/m) * Σ dZ[l]
```

### Gradient Descent
```
W[l] := W[l] - α * dW[l]
b[l] := b[l] - α * db[l]
```

## Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for the complete technical report including:

- Problem statement
- Algorithm details
- Implementation code
- Analysis of findings
- References

## Dataset

NBA Players Dataset from Kaggle:
https://www.kaggle.com/datasets/justinas/nba-players-data

## License

This project is for educational purposes as part of AIT-204 coursework.

## References

1. Rumelhart, D. E., et al. (1986). Learning representations by back-propagating errors.
2. He, K., et al. (2015). Delving Deep into Rectifiers.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
