import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

dataset = fetch_california_housing()
features = dataset.data
labels   = dataset.target

train_x, test_x, train_y, test_y = train_test_split(
    features, labels, test_size=0.2, random_state=0)

print('Shape of features array is', features.shape)
print('Shape of labels array is', labels.shape)
print('train_x:', train_x.shape)
print('test_x: ', test_x.shape)
print('train_y:', train_y.shape)
print('test_y: ', test_y.shape)
feature_names = dataset.feature_names

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
for i in range(8):
    axes[i // 4, i % 4].hist(train_x[:, i], bins=10, color='skyblue', edgecolor='black')
    axes[i // 4, i % 4].set_title(feature_names[i])
    axes[i // 4, i % 4].set_xlabel('Value')
    axes[i // 4, i % 4].set_ylabel('Frequency')

plt.suptitle('Distribution of Input Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(7, 5))
plt.hist(train_y, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Median House Price in District ($100k)')
plt.ylabel('Frequency')
plt.title('Distribution of Target Labels (Median House Value)')
plt.tight_layout()
plt.show()
def simple_linear_regression_fit(x, y):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate theta1 (slope)
    numerator   = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    theta1 = numerator / denominator

    # Calculate theta0 (intercept)
    theta0 = y_mean - theta1 * x_mean

    return theta0, theta1

# Use MedInc (index 0) as the single input feature
feature_index = 0
theta0, theta1 = simple_linear_regression_fit(train_x[:, feature_index], train_y)

print(f'Slope     (theta1): {theta1:.4f}')
print(f'Intercept (theta0): {theta0:.4f}')

# Predicted values for train and test sets
predicted_y_train = theta0 + theta1 * train_x[:, feature_index]
predicted_y_test  = theta0 + theta1 * test_x[:, feature_index]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Linear Regression', fontsize=14, fontweight='bold')

for ax, X, Y_true, Y_pred, title, label in [
    (axes[0], train_x[:, feature_index], train_y, predicted_y_train, 'Train set', 'Training data'),
    (axes[1], test_x[:, feature_index],  test_y,  predicted_y_test,  'Test set',  'Test data'),
]:
    ax.scatter(X, Y_true, color='blue', s=5, alpha=0.4, label=label)
    si = np.argsort(X)
    ax.plot(X[si], Y_pred[si], color='orange', linewidth=2, label='Model predictions')
    ax.set_title(title)
    ax.set_xlabel('Feature (MedInc)')
    ax.set_ylabel('Target')
    ax.legend()

plt.tight_layout()
plt.show()