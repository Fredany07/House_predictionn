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
from sklearn.linear_model import LinearRegression

# Fit model on all 8 features
multi_lr = LinearRegression()
multi_lr.fit(train_x, train_y)

print('Multivariable Linear Regression Coefficients:')
for name, coef in zip(dataset.feature_names, multi_lr.coef_):
    print(f'  {name:15s}: {coef:+.4f}')
print(f'  {"Intercept":15s}: {multi_lr.intercept_:.4f}')

# Predicted y values for the test set
multi_predicted_y_test = multi_lr.intercept_ + test_x @ multi_lr.coef_
plt.tight_layout()
plt.show()
plt.figure(figsize=(7, 7))
plt.scatter(test_y, multi_predicted_y_test, color='red', s=8, alpha=0.5, label='Predictions')
mn = min(test_y.min(), multi_predicted_y_test.min())
mx = max(test_y.max(), multi_predicted_y_test.max())
plt.plot([mn, mx], [mn, mx], color='blue', linewidth=1.5, label='Perfect fit (y=x)')
plt.xlabel('True house price ($100k)')
plt.ylabel('Predicted house price ($100k)')
plt.title('Multivariable Linear Regression: Predicted vs True')
plt.legend()
plt.tight_layout()
plt.show()
## Part C k-nearest neighbors regression
train_y_class = (train_y > 2.0).astype(int)
test_y_class  = (test_y  > 2.0).astype(int)

print(f'Training - low price: {np.sum(train_y_class==0)}, high price: {np.sum(train_y_class==1)}')
print(f'Test     - low price: {np.sum(test_y_class==0)}, high price: {np.sum(test_y_class==1)}')
at_idx, lon_idx = 6, 7
lat_idx, lon_idx = 6, 7

plt.figure(figsize=(7, 7))
low  = train_y_class == 0
high = train_y_class == 1
plt.scatter(train_x[low,  lat_idx], train_x[low,  lon_idx],
            color='yellow', edgecolors='gray', s=12, label='low price')
plt.scatter(train_x[high, lat_idx], train_x[high, lon_idx],
            color='black', marker='+', s=20, label='high price')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Training Labels: Low vs High Price')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 7))
low  = train_y_class == 0
high = train_y_class == 1
plt.scatter(train_x[low,  lat_idx], train_x[low,  lon_idx],
            color='yellow', edgecolors='gray', s=12, label='low price')
plt.scatter(train_x[high, lat_idx], train_x[high, lon_idx],
            color='black', marker='+', s=20, label='high price')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Training Labels: Low vs High Price')
plt.legend()
plt.tight_layout()
plt.show()
def knn_classify(train_features, train_labels, test_features, k=5):
    n_test = test_features.shape[0]
    predicted_labels = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        diff      = train_features - test_features[i]
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        k_idx     = np.argsort(distances)[:k]
        predicted_labels[i] = np.bincount(train_labels[k_idx]).argmax()
    return predicted_labels

def get_accuracy(predicted_labels, true_labels):
    correct = np.sum(predicted_labels == true_labels)
    return correct / len(true_labels)

predicted_test_class = knn_classify(train_x, train_y_class, test_x, k=5)
accuracy = get_accuracy(predicted_test_class, test_y_class)
print(f'K-Nearest-Neighbor accuracy is {accuracy}')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('KNN Classification Results', fontsize=14, fontweight='bold')

for ax, labs, title in [
    (axes[0], predicted_test_class, 'Predicted house price'),
    (axes[1], test_y_class,         'Actual house price'),
]:
    lm = labs == 0
    hm = labs == 1
    ax.scatter(test_x[lm, lat_idx], test_x[lm, lon_idx],
               color='yellow', edgecolors='gray', s=12, label='low price')
    ax.scatter(test_x[hm, lat_idx], test_x[hm, lon_idx],
               color='black', marker='+', s=20, label='high price')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
