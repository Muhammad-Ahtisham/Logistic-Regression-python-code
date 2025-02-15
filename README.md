# Logistic-Regression-python-code

## 1. Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
numpy: A library used for working with arrays and mathematical functions. It's widely used in machine learning for operations on datasets and matrices.
matplotlib.pyplot: A plotting library used to create visualizations. Here, it’s used to plot the cost over training epochs.
```
## 2. Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
The sigmoid function is a common activation function in logistic regression.
It maps any input value to a range between 0 and 1, making it useful for binary classification (outputs probabilities).
The function returns the sigmoid of z, which is calculated as 1 / (1 + e^(-z)), where e is the base of the natural logarithm.
## 3. Cost Function (Binary Cross-Entropy Loss)
```python
def compute_cost(X, y, weights):
    m = len(y)  # Number of data points
    predictions = sigmoid(np.dot(X, weights))  # Predicted values (probabilities)
    cost = -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost
```
Binary Cross-Entropy Loss (also called log loss) is used as the cost function for binary classification problems. It measures how well the predicted probabilities match the actual labels.
X: The input data (features).
y: The true labels (0 or 1).
weights: The model weights.
np.dot(X, weights): This computes the dot product of the input features and weights, which gives the linear combination used as input to the sigmoid function.
The loss formula is calculated using the negative log-likelihood:
For each example, it calculates y * log(prediction) + (1 - y) * log(1 - prediction).
The formula penalizes wrong predictions more heavily than correct ones.
## 4. Gradient Descent
```python
def gradient_descent(X, y, weights, learning_rate, epochs):
    m = len(y)
    cost_history = []

    for _ in range(epochs):
        predictions = sigmoid(np.dot(X, weights))
        gradient = (1 / m) * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient

        # Store the cost at each iteration
        cost_history.append(compute_cost(X, y, weights))

    return weights, cost_history
```
Gradient Descent is an optimization algorithm used to minimize the cost function.
The function updates the weights iteratively to reduce the loss (or cost).
X: Input features.
y: True labels.
weights: Initial weights (random or zero).
learning_rate: Controls how much the weights are adjusted after each iteration.
epochs: Number of iterations to run the gradient descent.
The algorithm proceeds as follows:
Predictions are computed using the sigmoid function.
The gradient is calculated as (1/m) * X.T.dot(predictions - y), which measures the rate of change of the cost with respect to each weight.
Weights are updated using the gradient: weights -= learning_rate * gradient.
The cost for each epoch is calculated and stored in cost_history for visualization later.
## 5. Logistic Regression Function
```python
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    # Add a column of ones for the intercept term (bias)
    X = np.c_[np.ones(X.shape[0]), X]

    # Initialize weights randomly
    weights = np.zeros(X.shape[1])

    # Perform gradient descent
    weights, cost_history = gradient_descent(X, y, weights, learning_rate, epochs)

    return weights, cost_history
```
This function trains a logistic regression model.
Intercept Term (Bias): A column of ones is added to the feature matrix X to account for the bias term (b) in the logistic regression equation.
Initial Weights: The weights are initialized to zeros. X.shape[1] ensures there are enough weights for each feature (including the bias).
The gradient descent function is called to update the weights and minimize the cost function.
The function returns the learned weights and the history of the cost function over the epochs.
## 6. Prediction Function
```python
def predict(X, weights):
    # Add a column of ones for the intercept term (bias)
    X = np.c_[np.ones(X.shape[0]), X]

    # Compute probabilities using sigmoid
    probabilities = sigmoid(np.dot(X, weights))

    # Convert probabilities to binary predictions (0 or 1)
    predictions = (probabilities >= 0.5).astype(int)

    return predictions
```
The predict function uses the learned weights to make predictions on new input data.
Similar to the logistic regression training, a column of ones is added to the input features for the bias term.
The sigmoid function is applied to the dot product of the features and weights to compute the probabilities.
These probabilities are then converted to binary predictions (0 or 1) by applying a threshold of 0.5 (if the probability is >= 0.5, the prediction is 1, else it's 0).
## 7. Example Usage
```python
if __name__ == "__main__":
    # Example data (X: features, y: labels)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    # Train logistic regression model
    weights, cost_history = logistic_regression(X, y, learning_rate=0.1, epochs=5)

    print("Trained Weights:", weights)

    # Predictions on the training data
    predictions = predict(X, weights)
    print("Predictions:", predictions)

    # Plot cost over epochs (optional)

    plt.plot(cost_history)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs Epochs')
    plt.show()
```
Training Data: The feature matrix X contains 5 examples, each with 2 features, and the label vector y contains the corresponding binary labels.
The logistic regression model is trained with the data using learning_rate=0.1 and epochs=5.
The trained weights are printed.
The predictions for the training data are printed.
The cost history is plotted to visualize how the cost function decreases over epochs as the model learns.
Key Concepts in Logistic Regression:
Sigmoid function: Used to convert a linear combination of inputs into a probability.
Cost function: Measures how well the model’s predictions match the actual labels.
Gradient Descent: Optimization algorithm to minimize the cost by updating weights.
Binary Classification: Predicting one of two possible classes (0 or 1).
