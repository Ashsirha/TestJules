import nbformat

# Read the existing notebook
with open('data_science_learning_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# --- Part 3: Venturing into Machine Learning ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""## Part 3: Venturing into Machine Learning"""
))

# --- Section 6: Fundamental Machine Learning Concepts ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 6: Fundamental Machine Learning Concepts"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Differentiate between major types of ML.
    *   Understand the basic workflow of building an ML model."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **What is Machine Learning?**
        *   Definition: Algorithms that allow computers to learn from data without being explicitly programmed."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Types of Machine Learning:**
        *   Supervised Learning: Learning from labeled data.
            *   Regression: Predicting continuous values (e.g., house prices).
            *   Classification: Predicting discrete categories (e.g., spam vs. not spam).
        *   Unsupervised Learning: Learning from unlabeled data.
            *   Clustering: Grouping similar data points (e.g., customer segmentation).
            *   Association Rule Mining: Finding relationships between items (e.g., market basket analysis - brief mention).
        *   Reinforcement Learning: Learning through trial and error with rewards/penalties (very brief concept, e.g., game playing AI)."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Key Terminology:**
        *   Features (Input Variables): The data attributes used for prediction.
        *   Labels/Targets (Output Variables): The value or category to be predicted in supervised learning.
        *   Training Data: Data used to "teach" the model.
        *   Test Data: Data used to evaluate the model's performance on unseen data."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **The Modeling Process (Conceptual Overview):**
        *   Data Splitting: Dividing data into training and testing sets.
        *   Model Training: The algorithm learns patterns from the training data.
        *   Prediction/Inference: Using the trained model to make predictions on new data.
        *   Model Evaluation: Measuring how well the model performs (e.g., accuracy for classification, error metrics for regression - conceptual)."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Overfitting and Underfitting:**
        *   Intuitive explanation: Model too complex (overfitting - memorizes training data) vs. model too simple (underfitting - fails to capture patterns)."""
))

# --- Section 7: A Gentle Introduction to a Simple Algorithm: k-Nearest Neighbors (k-NN) ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 7: A Gentle Introduction to a Simple Algorithm: k-Nearest Neighbors (k-NN)"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Understand the intuition behind k-NN.
    *   Implement a basic k-NN for classification."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **How k-NN Works:**
        *   Analogy (e.g., "birds of a feather flock together").
        *   Importance of a distance metric.
        *   The role of 'k' (number of neighbors) and how it affects the prediction."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Step-by-Step k-NN for Classification (from scratch):**
        1.  Choose a value for `k`.
        2.  For a new, unseen data point:
            a.  Calculate the distance (e.g., Euclidean) from the new point to every point in the training dataset.
            b.  Select the `k` training data points that are closest (the "neighbors").
            c.  Determine the most frequent class among these `k` neighbors.
            d.  Assign this majority class as the prediction for the new data point."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import euclidean_distance, mode
# Alternatively, for mode with non-numeric labels, collections.Counter is good.
# pymath.mode is designed for list[float], so for class labels, Counter is more general.
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np # For sample data generation

def knn_predict(training_features, training_labels, query_point, k):
    \"\"\"Predicts the class of a query point using k-NN.\"\"\"
    distances = []
    for i, train_point in enumerate(training_features):
        dist = euclidean_distance(query_point, train_point)
        distances.append((dist, training_labels[i]))

    # Sort by distance and take the k nearest
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # Get the labels of the k nearest neighbors
    neighbor_labels = [neighbor[1] for neighbor in neighbors]

    # Find the most common class among neighbors
    # Using collections.Counter for flexibility with label types
    most_common = Counter(neighbor_labels).most_common(1)
    return most_common[0][0]

# Sample 2D training data
# Features: [X, Y]
# Labels: 0 or 1
X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 6]]
y_train = [0, 0, 0, 1, 1, 1] # Two classes

# New point to classify
query = [4, 4]
k_value = 3

predicted_class = knn_predict(X_train, y_train, query, k_value)
print(f"The query point {query} is predicted as class: {predicted_class}")

# Visualization (Optional but good for 2D)
plt.figure(figsize=(6,4))
# Plot training data
for i, point in enumerate(X_train):
    plt.scatter(point[0], point[1], color='blue' if y_train[i] == 0 else 'red', marker='o', label=f'Class {y_train[i]}' if i == 0 or y_train[i] != y_train[i-1] else "")

plt.scatter(query[0], query[1], color='green', marker='x', s=100, label='Query Point')
plt.title(f'k-NN Classification (k={k_value}) - Predicted: Class {predicted_class}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# Create legend entries manually for classes if not all present or to avoid duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # Remove duplicate labels
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.show()"""
))

# --- Section 8: Connecting `pymath` Functions to ML Ideas ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 8: Connecting `pymath` Functions to ML Ideas"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   See how some `pymath` functions relate to ML concepts."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Activation Functions in Neural Networks (Conceptual):**
        *   Briefly explain what an activation function is (introduces non-linearity)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import sigmoid, relu
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x_vals = np.linspace(-5, 5, 100)

# Sigmoid
y_sigmoid = [sigmoid(x) for x in x_vals]
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_sigmoid)
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)

# ReLU
y_relu = [relu(x) for x in x_vals]
plt.subplot(1, 2, 2)
plt.plot(x_vals, y_relu)
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Sigmoid: Used in output layers for binary classification (outputs probabilities).")
print("ReLU: Common in hidden layers of neural networks due to its simplicity and efficiency.")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Output Transformation for Multi-class Classification:**
        *   `softmax(vector)`: Explain its role in converting raw scores (logits) from a neural network's output layer into a probability distribution over multiple classes."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import softmax
import matplotlib.pyplot as plt

sample_scores = [2.0, 1.0, 0.1, 3.0] # Example logits
softmax_probs = softmax(sample_scores)

print(f"Original Scores (Logits): {sample_scores}")
print(f"Softmax Probabilities: {softmax_probs}")
print(f"Sum of Probabilities: {sum(softmax_probs):.2f}") # Should be close to 1.0

plt.figure(figsize=(6,3))
plt.bar([f'Class {i+1}' for i in range(len(sample_scores))], softmax_probs, color='skyblue')
plt.title('Softmax Output Probabilities')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Feature Scaling:**
        *   `normalize(data)` (Min-Max Scaling): Explain why feature scaling is important for distance-based algorithms (like k-NN) and gradient-based algorithms. Show an example of its application."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import normalize

data_for_scaling = [0, 10, 20, 50, 100, 200]
normalized_scaled_data = normalize(data_for_scaling)

print(f"Original Data: {data_for_scaling}")
print(f"Min-Max Normalized Data (to [0,1]): {normalized_scaled_data}")

# Example of how it might affect distances (conceptual)
# Consider points (0, 1) and (100, 10)
# Original Euclidean distance: sqrt((100-0)^2 + (10-1)^2) = sqrt(10000 + 81) = sqrt(10081) approx 100.4
# If we normalize [0, 100] to [0,1] and [1,10] to [0,1] (independently)
# Normalized points: (0,0) and (1,1)
# New Euclidean distance: sqrt((1-0)^2 + (1-0)^2) = sqrt(2) approx 1.41
# This shows how scaling can change the influence of features in distance calculations."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Information Theory in Machine Learning (Conceptual):**
        *   `entropy(probabilities)`: Briefly link to its use in decision trees (information gain) or as a component of loss functions.
        *   `cross_entropy(p, q)`: Conceptually explain its role as a common loss function in classification tasks, measuring the difference between predicted and true probability distributions.
        *   `kl_divergence(p, q)`: Briefly mention its use in comparing probability distributions, e.g., in generative models or reinforcement learning (conceptual)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import entropy, cross_entropy, kl_divergence

# Example for Entropy
prob_dist_fair_coin = [0.5, 0.5]
print(f"Entropy of a fair coin flip: {entropy(prob_dist_fair_coin):.2f} bits")

prob_dist_certain = [1.0, 0.0, 0.0]
print(f"Entropy of a certain event: {entropy(prob_dist_certain):.2f} bits")

# Example for Cross-Entropy & KL-Divergence (conceptual illustration)
p_true = [0.1, 0.6, 0.3]
q_pred1 = [0.2, 0.5, 0.3] # A good prediction
q_pred2 = [0.7, 0.1, 0.2] # A poor prediction

print(f"\\nTrue distribution p: {p_true}")
print(f"Prediction 1 q1: {q_pred1}")
print(f"Prediction 2 q2: {q_pred2}")

ce_pq1 = cross_entropy(p_true, q_pred1)
ce_pq2 = cross_entropy(p_true, q_pred2)
print(f"Cross-Entropy H(p, q1): {ce_pq1:.2f} (lower is better)")
print(f"Cross-Entropy H(p, q2): {ce_pq2:.2f} (higher means worse prediction)")

kl_pq1 = kl_divergence(p_true, q_pred1)
kl_pq2 = kl_divergence(p_true, q_pred2)
print(f"KL Divergence D_KL(p || q1): {kl_pq1:.2f} (lower means distributions are more similar)")
print(f"KL Divergence D_KL(p || q2): {kl_pq2:.2f} (higher means distributions are less similar)")"""
))


# Write the updated notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 3 appended successfully.")
print(f"Total cells in notebook now: {len(notebook.cells)}")
