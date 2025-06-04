import nbformat

# Read the existing notebook
with open('data_science_learning_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# --- Part 4: Putting It Together & Next Steps ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""## Part 4: Putting It Together & Next Steps"""
))

# --- Section 9: Mini-Project: Classifying Simple Data ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 9: Mini-Project: Classifying Simple Data"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Apply the learned EDA and k-NN concepts to a slightly larger (but still small) dataset."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Introduction to the Project:**
        *   In this mini-project, we'll create a simple dataset of imaginary fruits based on two features: 'width' and 'height'.
        *   Our goal is to classify these fruits into two categories (e.g., 'Type A' and 'Type B') using our k-NN algorithm."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Dataset Creation"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""import random
import numpy as np # For potential use in data generation or manipulation if needed later

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Features: [width, height]
# Labels: 0 (Type A), 1 (Type B)

# Generate data for Type A (generally smaller width, smaller height)
features_a = [[random.uniform(2, 5), random.uniform(3, 6)] for _ in range(15)]
labels_a = [0] * 15

# Generate data for Type B (generally larger width, larger height)
features_b = [[random.uniform(5, 8), random.uniform(6, 9)] for _ in range(15)]
labels_b = [1] * 15

# Combine datasets
all_features = features_a + features_b
all_labels = labels_a + labels_b

print("Sample of the dataset (first 5 entries):")
for i in range(5):
    print(f"Features: {all_features[i]}, Label: {all_labels[i]}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### EDA on the Project Data"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import mean, median, standard_deviation
import matplotlib.pyplot as plt
%matplotlib inline

# Extract features for easier analysis
widths = [f[0] for f in all_features]
heights = [f[1] for f in all_features]

print("\\nDescriptive Statistics for Features:")
print(f"Widths - Mean: {mean(widths):.2f}, Median: {median(widths):.2f}, StdDev: {standard_deviation(widths):.2f}")
print(f"Heights - Mean: {mean(heights):.2f}, Median: {median(heights):.2f}, StdDev: {standard_deviation(heights):.2f}")

# Scatter plot of features colored by class label
plt.figure(figsize=(6,4))
colors = ['blue' if label == 0 else 'red' for label in all_labels]
plt.scatter(widths, heights, c=colors)
import matplotlib.patches as mpatches # For legend
blue_patch = mpatches.Patch(color='blue', label='Type A (0)')
red_patch = mpatches.Patch(color='red', label='Type B (1)')
plt.legend(handles=[blue_patch, red_patch])

plt.title('Fruit Dataset: Width vs. Height')
plt.xlabel('Width')
plt.ylabel('Height')
plt.grid(True)
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Training/Test Split (Conceptual & Manual)
*   To evaluate our k-NN model, we need to train it on one portion of the data and test it on another, unseen portion.
*   For this small dataset, we'll do a simple manual split: the first 70% for training and the remaining 30% for testing."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""split_ratio = 0.7
split_index = int(len(all_features) * split_ratio)

X_train_proj = all_features[:split_index]
y_train_proj = all_labels[:split_index]

X_test_proj = all_features[split_index:]
y_test_proj = all_labels[split_index:]

print(f"Training samples: {len(X_train_proj)}")
print(f"Test samples: {len(X_test_proj)}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Applying k-NN"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Assuming knn_predict function from Part 3, Section 7 is available in the notebook's execution context.
# If running cells out of order, or if this part is run separately,
# knn_predict and its dependencies (euclidean_distance, Counter/mode) would need to be redefined or re-imported here.
# from pymath.lib.math import euclidean_distance
# from collections import Counter
# def knn_predict(training_features, training_labels, query_point, k): ... (definition from Part 3)

k_project = 3
predictions_proj = []
for test_point in X_test_proj:
    pred = knn_predict(X_train_proj, y_train_proj, test_point, k_project)
    predictions_proj.append(pred)

print("Predictions vs Actual Labels for Test Set:")
for i in range(len(X_test_proj)):
    print(f"  Point: {X_test_proj[i]}, Predicted: {predictions_proj[i]}, Actual: {y_test_proj[i]}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Simple Evaluation
*   A common way to evaluate a classification model is **accuracy**: the proportion of correct predictions out of the total predictions made."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""correct_predictions = 0
for i in range(len(predictions_proj)):
    if predictions_proj[i] == y_test_proj[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(y_test_proj) if len(y_test_proj) > 0 else 0
print(f"Number of correct predictions: {correct_predictions} out of {len(y_test_proj)}")
print(f"Accuracy on the test set: {accuracy*100:.2f}%""")
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Mini-Project Conclusion
*   In this mini-project, we created a synthetic dataset, performed basic EDA, manually split our data, applied our k-NN classifier, and calculated its accuracy. This demonstrates a simplified end-to-end flow of a classification task."""
))

# --- Section 10: Your Data Science Journey Continues! ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 10: Your Data Science Journey Continues!"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Know what to learn next.
    *   Find resources for continued learning."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Recap of What Was Learned:**
        *   Brief summary of Python basics, descriptive statistics, visualization, introductory ML concepts, and the mini-project covered in this notebook."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Key Areas for Further Study:**
        *   **Core Data Science Libraries:**
            *   NumPy: For numerical computing (arrays, matrices).
            *   Pandas: For data manipulation and analysis (DataFrames).
            *   Scikit-learn: Comprehensive library for machine learning algorithms, preprocessing, and evaluation.
        *   **More Advanced ML Algorithms:** Decision Trees, Random Forests, Support Vector Machines, Gradient Boosting, Neural Networks, etc.
        *   **Data Engineering Basics:** Data collection, storage, cleaning pipelines, and database concepts.
        *   **Specialized Fields:** Natural Language Processing (NLP), Computer Vision (CV), Time Series Analysis, Recommender Systems.
        *   **Statistics and Probability:** Deeper dive into inferential statistics, hypothesis testing, probability distributions, Bayesian methods."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Recommended Learning Resources (Examples):**
        *   Online Courses:
            *   Coursera (e.g., Andrew Ng's Machine Learning Specialization, IBM Data Science Professional Certificate)
            *   edX (e.g., Microsoft Professional Program in Data Science)
            *   DataCamp (Interactive courses on Python, R, SQL, and data science topics)
            *   fast.ai (Practical deep learning courses)
            *   Khan Academy (For foundational math and statistics)
        *   Documentation:
            *   [Python Official Documentation](https://docs.python.org/3/)
            *   [NumPy Documentation](https://numpy.org/doc/)
            *   [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
            *   [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
            *   [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
        *   Books:
            *   "Python for Data Analysis" by Wes McKinney
            *   "Introduction to Machine Learning with Python" by Andreas C. Müller & Sarah Guido
            *   "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
        *   Communities & Practice:
            *   Kaggle (Competitions, datasets, and notebooks)
            *   Stack Overflow (Q&A for programming and data science questions)
            *   Towards Data Science (Medium publication with many articles)
            *   Local meetups and data science groups."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Encouragement and Final Thoughts:**
        *   Data science is a vast and rapidly evolving field. Continuous learning and hands-on practice are key.
        *   Start with small projects, explore datasets that interest you, and don't be afraid to experiment.
        *   Good luck on your data science journey!"""
))

# Write the updated notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 4 appended successfully (v2).")
print(f"Total cells in notebook now: {len(notebook.cells)}")

# For verification, let's print the source of the last cell
print("\\n--- Last cell for verification ---")
if notebook.cells:
    print(notebook.cells[-1].source)
else:
    print("Notebook is empty!")
print("------------")
