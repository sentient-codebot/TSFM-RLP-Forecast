import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# Generate multi-dimensional output data (e.g., modify the dataset as needed)
# Here, using make_friedman2 and repeating y to simulate multi-dimensional output
X, y_single = make_friedman2(n_samples=500, noise=0, random_state=0)
y = np.stack([y_single, y_single * 1.5 + 10], axis=1)  # Example multi-dimensional output (shape: (500, 2))

# Define kernel and prepare the list for storing models
kernel = DotProduct() + WhiteKernel()
models = []

# Fit a separate GPR model for each output dimension
for i in range(y.shape[1]):
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y[:, i])
    models.append(gpr)

# Make predictions and plot for the first dimension
X_test = X[:100, :]  # Using 100 samples from X for demonstration
mean_preds = []
std_preds = []

# Generate predictions for each output dimension
for model in models:
    mean_pred, std_pred = model.predict(X_test, return_std=True)
    mean_preds.append(mean_pred)
    std_preds.append(std_pred)

# Plot for the first dimension (y[:, 0])
plt.figure(figsize=(12, 6))

# Sort data to create a smooth curve
sorted_idx = np.argsort(X_test[:, 0])  # Sorting by the first feature for visualization
X_sorted = X_test[sorted_idx]
mean_sorted = mean_preds[0][sorted_idx]
std_sorted = std_preds[0][sorted_idx]

# Mean prediction for the first dimension
plt.plot(X_sorted[:, 0], mean_sorted, 'r-', label="Mean Prediction (Dim 1)")

# 95% Confidence interval for the first dimension
plt.fill_between(X_sorted[:, 0],
                 mean_sorted - 1.96 * std_sorted,
                 mean_sorted + 1.96 * std_sorted,
                 alpha=0.2, color='gray', label="95% Confidence Interval (Dim 1)")

# Scatter plot of actual values for comparison (first dimension)
plt.scatter(X_sorted[:, 0], y[:100, 0][sorted_idx], s=10, color='blue', label="Actual Values (Dim 1)", alpha=0.6)

# Labels and title
plt.xlabel("Feature 0")
plt.ylabel("Target (Dim 1)")
plt.title("Gaussian Process Regression with Probabilistic Predictions (Dimension 1)")
plt.legend()
plt.show()
