import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="California Housing", page_icon="🏠", layout="wide")
 
st.title("🏠 California Housing Prediction ")
st.markdown("Exploring house prices")
 
# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    dataset = fetch_california_housing()
    features = dataset.data
    labels   = dataset.target
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=0)
    return dataset, features, labels, train_x, test_x, train_y, test_y
 
dataset, features, labels, train_x, test_x, train_y, test_y = load_data()
feature_names = dataset.feature_names
 
# ── Dataset overview ─────────────────────────────────────────────────────────
st.header("📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", features.shape[0])
col2.metric("Features", features.shape[1])
col3.metric("Training Samples", train_x.shape[0])
col4.metric("Test Samples", test_x.shape[0])
 
# ── Feature distributions ────────────────────────────────────────────────────
st.header("📊 Feature Distributions")
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6))
for i in range(8):
    axes[i // 4, i % 4].hist(train_x[:, i], bins=10, color='skyblue', edgecolor='black')
    axes[i // 4, i % 4].set_title(feature_names[i])
    axes[i // 4, i % 4].set_xlabel('Value')
    axes[i // 4, i % 4].set_ylabel('Frequency')
plt.suptitle('Distribution of Input Features', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()
 
# Target distribution
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.hist(train_y, bins=10, color='skyblue', edgecolor='black')
ax2.set_xlabel('Median House Price in District ($100k)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Target Labels (Median House Value)')
plt.tight_layout()
st.pyplot(fig2)
plt.close()
 
# ── Simple Linear Regression ──────────────────────────────────────────────────
st.header("📈 Part A — Simple Linear Regression")
 
def simple_linear_regression_fit(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator   = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    theta1 = numerator / denominator
    theta0 = y_mean - theta1 * x_mean
    return theta0, theta1
 
feature_index = 0
theta0, theta1 = simple_linear_regression_fit(train_x[:, feature_index], train_y)
 
col1, col2 = st.columns(2)
col1.metric("Slope (theta1)", f"{theta1:.4f}")
col2.metric("Intercept (theta0)", f"{theta0:.4f}")
 
predicted_y_train = theta0 + theta1 * train_x[:, feature_index]
predicted_y_test  = theta0 + theta1 * test_x[:, feature_index]
 
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Simple Linear Regression', fontsize=14, fontweight='bold')
for ax, X, Y_true, Y_pred, title, label in [
    (axes3[0], train_x[:, feature_index], train_y, predicted_y_train, 'Train set', 'Training data'),
    (axes3[1], test_x[:, feature_index],  test_y,  predicted_y_test,  'Test set',  'Test data'),
]:
    ax.scatter(X, Y_true, color='blue', s=5, alpha=0.4, label=label)
    si = np.argsort(X)
    ax.plot(X[si], Y_pred[si], color='orange', linewidth=2, label='Model predictions')
    ax.set_title(title)
    ax.set_xlabel('Feature (MedInc)')
    ax.set_ylabel('Target')
    ax.legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close()
 
# ── Multivariable Linear Regression ──────────────────────────────────────────
st.header("📉 Part B — Multivariable Linear Regression")
 
multi_lr = LinearRegression()
multi_lr.fit(train_x, train_y)
multi_predicted_y_test = multi_lr.intercept_ + test_x @ multi_lr.coef_
 
st.subheader("Model Coefficients")
coef_data = {name: round(coef, 4) for name, coef in zip(feature_names, multi_lr.coef_)}
coef_data["Intercept"] = round(multi_lr.intercept_, 4)
st.table(coef_data)
 
fig4, ax4 = plt.subplots(figsize=(7, 6))
ax4.scatter(test_y, multi_predicted_y_test, color='red', s=8, alpha=0.5, label='Predictions')
mn = min(test_y.min(), multi_predicted_y_test.min())
mx = max(test_y.max(), multi_predicted_y_test.max())
ax4.plot([mn, mx], [mn, mx], color='blue', linewidth=1.5, label='Perfect fit (y=x)')
ax4.set_xlabel('True house price ($100k)')
ax4.set_ylabel('Predicted house price ($100k)')
ax4.set_title('Multivariable Linear Regression: Predicted vs True')
ax4.legend()
plt.tight_layout()
st.pyplot(fig4)
plt.close()
 
# ── KNN Classification ────────────────────────────────────────────────────────
st.header("📍 Part C — K-Nearest Neighbors Classification")
 
train_y_class = (train_y > 2.0).astype(int)
test_y_class  = (test_y  > 2.0).astype(int)
 
col1, col2 = st.columns(2)
col1.metric("Training - Low Price", int(np.sum(train_y_class == 0)))
col2.metric("Training - High Price", int(np.sum(train_y_class == 1)))
 
lat_idx, lon_idx = 6, 7
 
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
    return np.sum(predicted_labels == true_labels) / len(true_labels)
 
k_value = st.slider("Select K value for KNN", min_value=1, max_value=20, value=5)
 
with st.spinner("Running KNN classifier..."):
    predicted_test_class = knn_classify(train_x, train_y_class, test_x, k=k_value)
    accuracy = get_accuracy(predicted_test_class, test_y_class)
 
st.metric("KNN Accuracy", f"{accuracy:.4f}")
 
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle('KNN Classification Results', fontsize=14, fontweight='bold')
for ax, labs, title in [
    (axes5[0], predicted_test_class, 'Predicted house price'),
    (axes5[1], test_y_class,         'Actual house price'),
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
st.pyplot(fig5)
plt.close()
 
st.success("✅ Analysis complete!")