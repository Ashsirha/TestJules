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
    source="""# Simplified Dataset Creation
all_features = [[2,3],[3,4],[4,5],[5,6],[6,7]]
all_labels = [0,0,0,1,1]
print("Sample dataset created.")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### EDA on the Project Data"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Simplified EDA
# from pymath.lib.math import mean
# import matplotlib.pyplot as plt
# %matplotlib inline
print("EDA would go here. For debugging, this cell is simplified.")
widths = [f[0] for f in all_features]
# plt.scatter(widths, [f[1] for f in all_features], c=all_labels)
# plt.title('Simplified Fruit Dataset')
# plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Training/Test Split (Conceptual & Manual)
*   Simple manual split."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Simplified Train/Test Split
X_train_proj = all_features[:3]
y_train_proj = all_labels[:3]
X_test_proj = all_features[3:]
y_test_proj = all_labels[3:]
print(f"Training: {len(X_train_proj)}, Test: {len(X_test_proj)}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Applying k-NN"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Simplified k-NN Application
# Assuming knn_predict is defined from Part 3
# For debugging, we'll simulate predictions.
# def knn_predict(train_f, train_l, query, k_val): return random.choice([0,1]) # Dummy
predictions_proj = [0, 1] # Dummy predictions
print(f"Dummy Predictions: {predictions_proj}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Simple Evaluation
*   Calculating accuracy."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Simplified Evaluation
correct_predictions = 0
if len(predictions_proj) == len(y_test_proj):
    for i in range(len(predictions_proj)):
        if predictions_proj[i] == y_test_proj[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_test_proj) if len(y_test_proj) > 0 else 0
    print(f"Accuracy: {accuracy*100:.2f}% (based on dummy predictions)")
else:
    print("Prediction and actual label lists differ in length.")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Mini-Project Conclusion
*   Summary of the simplified project."""
))

# --- Section 10: Your Data Science Journey Continues! ---
# (Keeping Section 10 cells very short as in v4)
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
        *   Brief summary."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Key Areas for Further Study:**
        *   Placeholder for key areas. This content is shortened for debugging."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Recommended Learning Resources (Examples):**
        *   Placeholder for resources list. This cell content is intentionally shortened for debugging."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Encouragement and Final Thoughts:**
        *   Keep learning!"""
))

# Write the updated notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 4 appended successfully (v5).")
print(f"Total cells in notebook now: {len(notebook.cells)}")
if notebook.cells:
    print("\\n--- Last cell source for verification ---")
    print(notebook.cells[-1].source)
    print("------------")
else:
    print("Notebook is empty after script execution!")

print("\\n--- Source of cell suspected of error (Mini-Project Conclusion - now last cell of Sec 9) ---")
# Find the "Mini-Project Conclusion" cell
for cell in notebook.cells:
    if "Summary of the simplified project" in cell.source:
        print(cell.source)
        break
print("------------")
