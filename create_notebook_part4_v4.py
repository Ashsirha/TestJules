import nbformat

# Read the existing notebook
with open('data_science_learning_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# --- Part 4: Putting It Together & Next Steps ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""## Part 4: Putting It Together & Next Steps"""
))

# --- Section 9: Mini-Project: Classifying Simple Data ---
# (Keeping Section 9 as is, assuming it's okay for now)
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
import numpy as np

random.seed(42)
np.random.seed(42)

features_a = [[random.uniform(2, 5), random.uniform(3, 6)] for _ in range(15)]
labels_a = [0] * 15
features_b = [[random.uniform(5, 8), random.uniform(6, 9)] for _ in range(15)]
labels_b = [1] * 15
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

widths = [f[0] for f in all_features]
heights = [f[1] for f in all_features]

print("\\nDescriptive Statistics for Features:")
print(f"Widths - Mean: {mean(widths):.2f}, Median: {median(widths):.2f}, StdDev: {standard_deviation(widths):.2f}")
print(f"Heights - Mean: {mean(heights):.2f}, Median: {median(heights):.2f}, StdDev: {standard_deviation(heights):.2f}")

plt.figure(figsize=(6,4))
colors = ['blue' if label == 0 else 'red' for label in all_labels]
plt.scatter(widths, heights, c=colors)
import matplotlib.patches as mpatches
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
    source="""# Assuming knn_predict function from Part 3, Section 7 is available
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
        *   Placeholder for key areas. This content is shortened for debugging."""
)) # Shortened content for "Key Areas"
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Recommended Learning Resources (Examples):**
        *   Placeholder for resources list. This cell content is intentionally shortened for debugging."""
)) # Shortened content for "Recommended Resources"
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Encouragement and Final Thoughts:**
        *   Data science is a vast and rapidly evolving field. Continuous learning and hands-on practice are key.
        *   Start with small projects, explore datasets that interest you, and don't be afraid to experiment.
        *   Good luck on your data science journey!"""
))

# Write the updated notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 4 appended successfully (v4).")
print(f"Total cells in notebook now: {len(notebook.cells)}")
if notebook.cells:
    print("\\n--- Last cell source for verification ---")
    print(notebook.cells[-1].source)
    print("------------")
else:
    print("Notebook is empty after script execution!")

print("\\n--- Source of cell suspected of error (Key Areas - now shortened) ---")
# Find the "Key Areas" cell (now should be the 3rd to last)
for cell in notebook.cells:
    if "Placeholder for key areas" in cell.source:
        print(cell.source)
        break
print("------------")
