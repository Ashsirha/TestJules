import nbformat

# Read the existing notebook
with open('data_science_learning_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# --- Part 2: Working with Data & Initial Analysis ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""## Part 2: Working with Data & Initial Analysis"""
))

# --- Section 3: Understanding Data with Descriptive Statistics ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 3: Understanding Data with Descriptive Statistics"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Learn how to summarize data using central tendency measures.
    *   Understand data spread/dispersion.
    *   Grasp the concept of relationships between variables (covariance/correlation)."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Introduction to Descriptive Statistics:**
        *   What are they and why are they important?"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Measures of Central Tendency:**
        *   Mean: Average value (theory, formula, example).
        *   Median: Middle value (theory, how to find it for odd/even datasets, example).
        *   Mode: Most frequent value(s) (theory, example)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from pymath.lib.math import mean, median, mode, variance, standard_deviation, covariance, correlation
import random # For generating sample data
# Note: Ensure pymath.lib.math is accessible in your Python environment when running the notebook.

# Sample data
sample_data_ct = [random.randint(0, 20) for _ in range(15)]
sample_data_mode = [random.choice([5,10,10,15,15,15,20,20,20,20]) for _ in range(30)] # Data likely to have modes

print(f"Sample Data for Central Tendency: {sample_data_ct}")
print(f"Mean: {mean(sample_data_ct)}")
print(f"Median: {median(sample_data_ct)}")
print(f"Sample Data for Mode: {sample_data_mode}")
print(f"Mode: {mode(sample_data_mode)}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Measures of Dispersion (Spread):**
        *   Range: Difference between max and min (theory, example).
        *   Variance: Average of squared differences from the Mean (theory, population vs. sample distinction, example).
        *   Standard Deviation: Square root of variance (theory, population vs. sample, example)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""sample_data_disp = [random.uniform(0, 50) for _ in range(20)]

print(f"Sample Data for Dispersion: {sample_data_disp}")
# Range
data_range = max(sample_data_disp) - min(sample_data_disp) if sample_data_disp else 0
print(f"Range: {data_range}")

# Variance
pop_variance = variance(sample_data_disp, is_sample=False)
sample_var = variance(sample_data_disp, is_sample=True)
print(f"Population Variance: {pop_variance}")
print(f"Sample Variance: {sample_var}")

# Standard Deviation
pop_std_dev = standard_deviation(sample_data_disp, is_sample=False)
sample_std_dev = standard_deviation(sample_data_disp, is_sample=True)
print(f"Population Standard Deviation: {pop_std_dev}")
print(f"Sample Standard Deviation: {sample_std_dev}")"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Understanding Relationships Between Variables:**
        *   Covariance: How two variables change together (theory, positive/negative interpretation, example).
        *   Pearson Correlation Coefficient: Standardized measure of linear relationship (-1 to 1) (theory, interpretation, example)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Sample data for relationships
data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data2_pos_corr = [x + random.uniform(-1, 1) for x in data1] # Positively correlated
data2_neg_corr = [10 - x + random.uniform(-1, 1) for x in data1] # Negatively correlated
data2_no_corr = [random.uniform(0,10) for _ in range(len(data1))] # Likely no strong correlation

print(f"Data1: {data1}")
print(f"Data2 (Positive Correlation): {data2_pos_corr}")
print(f"Data2 (Negative Correlation): {data2_neg_corr}")
print(f"Data2 (No Correlation): {data2_no_corr}")

# Covariance
cov_pos = covariance(data1, data2_pos_corr)
cov_neg = covariance(data1, data2_neg_corr)
cov_no = covariance(data1, data2_no_corr)
print(f"Covariance (Positive): {cov_pos}")
print(f"Covariance (Negative): {cov_neg}")
print(f"Covariance (No real): {cov_no}")

# Correlation
corr_pos = correlation(data1, data2_pos_corr)
corr_neg = correlation(data1, data2_neg_corr)
corr_no = correlation(data1, data2_no_corr)
print(f"Correlation (Positive): {corr_pos}")
print(f"Correlation (Negative): {corr_neg}")
print(f"Correlation (No real): {corr_no}")"""
))

# --- Section 4: Basic Data Visualization ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 4: Visualizing Data with Matplotlib"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Learn the basics of plotting with Matplotlib.
    *   Create common chart types for data exploration."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Introduction to Matplotlib:**
        *   Brief overview of the library.
        *   Importing `matplotlib.pyplot as plt`.
        *   The `%matplotlib inline` magic command for Jupyter (ensures plots appear in the notebook)."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""import matplotlib.pyplot as plt
from pymath.lib.math import fibonacci, mode, mean, median, correlation # For examples
import numpy as np # For histogram example data

# This magic command is typically used in Jupyter itself.
# In a script generating a notebook, it's a standard line of code in a cell.
%matplotlib inline"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Common Plots for Data Exploration:**
        *   Line Plot: `plt.plot()`. Useful for showing trends over time or sequence.
        *   Bar Chart: `plt.bar()`. Good for comparing categorical data or frequencies.
        *   Histogram: `plt.hist()`. Visualizing the distribution of a single numerical variable.
        *   Scatter Plot: `plt.scatter()`. Showing the relationship between two numerical variables."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Line Plot Example (Fibonacci Sequence)"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""fib_sequence = [fibonacci(n) for n in range(1, 16)] # First 15 Fibonacci numbers
plt.figure(figsize=(8, 4)) # Adjusted figure size
plt.plot(range(1, 16), fib_sequence, marker='o', linestyle='--')
plt.title('First 15 Fibonacci Numbers')
plt.xlabel('n-th Number')
plt.ylabel('Fibonacci Value F(n)')
plt.grid(True)
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Bar Chart Example (Mode Frequencies)"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""from collections import Counter # For frequency counting

data_for_mode_plot = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
modes = mode(data_for_mode_plot) # Using pymath.mode
print(f"Mode(s): {modes}")

counts = Counter(data_for_mode_plot)
plt.figure(figsize=(8, 4)) # Adjusted figure size
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('Frequency of Numbers (Mode Example)')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.xticks(list(counts.keys())) # Ensure all numbers are shown as ticks
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Histogram Example (Data Distribution)"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Generate some sample data (e.g., normally distributed)
np.random.seed(42) # for reproducibility
hist_data = np.random.normal(loc=20, scale=5, size=100)

mean_val = mean(hist_data.tolist()) # Use pymath.mean
median_val = median(hist_data.tolist()) # Use pymath.median

plt.figure(figsize=(8, 4)) # Adjusted figure size
plt.hist(hist_data, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
plt.title('Distribution of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, axis='y')
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(source="""#### Scatter Plot Example (Relationship for Correlation)"""))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""scatter_data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scatter_data2_pos = [x + random.uniform(-2, 2) for x in scatter_data1]
# corr_val = correlation(scatter_data1, scatter_data2_pos) # Using pymath.correlation

plt.figure(figsize=(8, 4)) # Adjusted figure size
plt.scatter(scatter_data1, scatter_data2_pos, color='coral')
# plt.title(f'Scatter Plot of Two Variables (Correlation: {corr_val:.2f})')
plt.title(f'Scatter Plot of Two Variables') # Title without correlation for now
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.grid(True)
plt.show()"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Customizing Plots:**
        *   Adding titles: `plt.title()`.
        *   Labeling axes: `plt.xlabel()`, `plt.ylabel()`.
        *   Adding legends: `plt.legend()`.
        *   Changing colors and markers.
        *   Adjusting figure size: `plt.figure(figsize=(width, height))`."""
))

# --- Section 5: Exploratory Data Analysis (EDA) in Action ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 5: Exploratory Data Analysis (EDA) in Action"""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Understand the goals and process of EDA.
    *   Apply statistical and visualization techniques to explore a dataset."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **What is EDA?**
        *   Its importance in the data science lifecycle.
        *   Goals: Understanding data properties, finding patterns, identifying anomalies, formulating hypotheses."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Key Steps in EDA (Simplified for this notebook):**
        *   Data Representation: We'll use Python lists and dictionaries to represent our small datasets.
        *   Initial Data Inspection:
            *   Understanding the "shape" (number of entries, number of features - conceptual).
            *   Understanding data types (numbers, categories - conceptual).
        *   Handling Missing Data (Conceptual Discussion):
            *   Briefly explain why it's an issue.
            *   Common strategies: Imputation (e.g., with mean/median), removal. (No implementation here).
        *   Identifying Outliers (Conceptual Discussion & Visual Inspection):
            *   What are outliers?
            *   Visual inspection using plots like box plots (mention) or scatter plots.
        *   Univariate Analysis: Analyzing single variables.
            *   Using descriptive statistics (`mean`, `median`, `mode`, `variance`, `std_dev`).
            *   Using visualizations (histograms, bar charts for categorical data).
        *   Bivariate Analysis: Analyzing relationships between two variables.
            *   Using statistics (`correlation`, `covariance`).
            *   Using visualizations (scatter plots)."""
))
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Mini Case Study:**
        *   Let's create a small dataset representing scores of students in two subjects and their study hours."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Mini Case Study Dataset
students_data = {
    'StudentID': list(range(1, 11)),
    'StudyHours': [2, 3, 1, 5, 4, 3, 2, 6, 5, 4], # Hours per week
    'MathScore': [60, 65, 50, 85, 75, 70, 55, 90, 80, 72],
    'ScienceScore': [65, 70, 55, 90, 80, 72, 60, 95, 85, 78]
}

print("Student Data (First 3 students):")
for i in range(3):
    print(f"ID: {students_data['StudentID'][i]}, StudyHours: {students_data['StudyHours'][i]}, Math: {students_data['MathScore'][i]}, Science: {students_data['ScienceScore'][i]}")

# Univariate Analysis: StudyHours
study_hours = students_data['StudyHours']
print(f"\\nStudy Hours Stats:")
print(f"  Mean: {mean(study_hours):.2f}")
print(f"  Median: {median(study_hours):.2f}")
print(f"  Mode: {mode(study_hours)}")
print(f"  Std Dev: {standard_deviation(study_hours):.2f}")

plt.figure(figsize=(6,3))
plt.hist(study_hours, bins=5, edgecolor='black', alpha=0.7)
plt.title('Distribution of Study Hours')
plt.xlabel('Hours')
plt.ylabel('Number of Students')
plt.show()

# Bivariate Analysis: StudyHours vs. MathScore
math_scores = students_data['MathScore']
corr_study_math = correlation(study_hours, math_scores)

plt.figure(figsize=(6,3))
plt.scatter(study_hours, math_scores)
plt.title(f'Study Hours vs. Math Score (Correlation: {corr_study_math:.2f})')
plt.xlabel('Study Hours')
plt.ylabel('Math Score')
plt.grid(True)
plt.show()"""
))


# Write the updated notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 2 appended successfully.")
