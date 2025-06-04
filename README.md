# PyMath: A Versatile Math Library

PyMath is a Python library providing a collection of mathematical functions for various calculations. It includes modules for core mathematical operations, statistical analysis, probability and information theory, basic linear algebra, and data utility functions. The library is designed to be simple to use and extend.

## Features

*   Core Mathematical Operations (Factorial, Fibonacci, GCD, LCM, Prime Checks)
*   Statistical Analysis Functions (Mean, Median, Mode, Variance, Standard Deviation, Covariance, Correlation)
*   Probability, Information Theory, and Activation Functions (Sigmoid, ReLU, Softmax, Entropy, Cross-Entropy, KL-Divergence)
*   Basic Linear Algebra Operations (Dot Product, Vector Add/Subtract, Scalar/Matrix-Vector Multiplication)
*   Data Normalization and Distance Metrics (Min-Max Normalization, Euclidean/Manhattan Distance)
*   Interactive Visualizations via Jupyter Notebook

## Function Reference

Below is a reference to the functions available in the `pymath.lib.math` module.

### Core Mathematical Operations

*   `factorial(n: int) -> int`: Calculates the factorial of `n`.
*   `fibonacci(n: int) -> int`: Calculates the nth Fibonacci number.
*   `gcd(a: int, b: int) -> int`: Calculates the Greatest Common Divisor of `a` and `b`.
*   `lcm(a: int, b: int) -> int`: Calculates the Least Common Multiple of `a` and `b`.
*   `is_perfect_square(n: int) -> bool`: Checks if `n` is a perfect square.
*   `is_prime(n: int) -> bool`: Checks if `n` is a prime number.

### Statistical Functions

*   `mean(data: list[float]) -> float`: Calculates the arithmetic mean of a list of numbers.
*   `median(data: list[float]) -> float`: Calculates the median of a list of numbers.
*   `mode(data: list[float]) -> list[float]`: Finds the mode(s) of a list of numbers. Returns a list, as there can be multiple modes.
*   `variance(data: list[float], is_sample: bool = False) -> float`: Calculates the variance (population or sample) of a list of numbers.
*   `standard_deviation(data: list[float], is_sample: bool = False) -> float`: Calculates the standard deviation (population or sample) of a list of numbers.
*   `covariance(data1: list[float], data2: list[float], is_sample: bool = False) -> float`: Calculates the covariance between two lists of numbers.
*   `correlation(data1: list[float], data2: list[float]) -> float`: Calculates the Pearson correlation coefficient between two lists of numbers.

### Probability, Information Theory, and Activation Functions

*   `sigmoid(x: float) -> float`: Calculates the sigmoid (logistic) function: `1 / (1 + exp(-x))`.
*   `relu(x: float) -> float`: Calculates the Rectified Linear Unit (ReLU) function: `max(0, x)`.
*   `softmax(vector: list[float]) -> list[float]`: Computes the softmax function, converting a vector of numbers into a probability distribution.
*   `entropy(probabilities: list[float]) -> float`: Calculates Shannon entropy for a discrete probability distribution. `H(X) = -sum(p * log2(p))`.
*   `cross_entropy(p: list[float], q: list[float]) -> float`: Calculates cross-entropy between two discrete probability distributions `p` (true) and `q` (predicted). `H(p, q) = -sum(p_i * log2(q_i))`.
*   `kl_divergence(p: list[float], q: list[float]) -> float`: Calculates Kullback-Leibler divergence `D_KL(p || q)` between two discrete probability distributions.

### Basic Linear Algebra Functions

*   `dot_product(vec1: list[float], vec2: list[float]) -> float`: Calculates the dot product of two vectors.
*   `vector_add(vec1: list[float], vec2: list[float]) -> list[float]`: Performs element-wise addition of two vectors.
*   `vector_subtract(vec1: list[float], vec2: list[float]) -> list[float]`: Performs element-wise subtraction of two vectors (`vec1 - vec2`).
*   `scalar_multiply(scalar: float, vector: list[float]) -> list[float]`: Multiplies each element of a vector by a scalar.
*   `matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]`: Multiplies a matrix (list of lists) by a vector.

### Utility Functions (Normalization and Distance)

*   `normalize(data: list[float]) -> list[float]`: Performs min-max normalization on a list of numbers to scale them to the range [0, 1].
*   `euclidean_distance(point1: list[float], point2: list[float]) -> float`: Calculates the Euclidean distance between two points (vectors).
*   `manhattan_distance(point1: list[float], point2: list[float]) -> float`: Calculates the Manhattan (L1) distance between two points (vectors).

## Installation / Setup

This library is currently self-contained within this repository. To use the functions:

1.  Clone or download the repository.
2.  Ensure the Python interpreter can find the `pymath` directory (e.g., by having your script in the root directory or by adjusting `PYTHONPATH`).
3.  Import functions directly from the `pymath.lib.math` module.

Example:
```python
from pymath.lib.math import mean, softmax

# Your code here that uses mean() or softmax()
```

## Usage Examples

Here are a few examples demonstrating how to use some of the library's functions:

```python
from pymath.lib.math import mean, factorial, softmax, dot_product, normalize

# Example 1: Basic statistics
data = [10.0, 12.5, 13.0, 14.5, 12.0]
print(f"Mean of data: {mean(data)}") # Output: 12.4

# Example 2: Core math
print(f"Factorial of 5: {factorial(5)}") # Output: 120

# Example 3: Softmax for probability distribution
scores = [1.0, 3.0, 0.5, 2.0]
probabilities = softmax(scores)
# Output: e.g., [0.085..., 0.628..., 0.052..., 0.231...] (summing to 1.0)
print(f"Softmax probabilities: {probabilities}")

# Example 4: Dot product
vec1 = [1, 2, 3]
vec2 = [4, 0, -1]
print(f"Dot product: {dot_product(vec1, vec2)}") # Output: 1.0

# Example 5: Normalizing data
original_data = [0, 10, 50, 100]
normalized_data = normalize(original_data)
# Output: [0.0, 0.1, 0.5, 1.0]
print(f"Normalized data: {normalized_data}")
```

## Running Tests

Unit tests are provided in the `pymath/tests/test_math.py` file. To run the tests, navigate to the root directory of the repository in your terminal and execute:

```bash
python -m unittest pymath.tests.test_math.py
```
This will run all defined tests and report their status.

## Visualization Notebook

An interactive Jupyter Notebook, `pymath_visualization.ipynb`, is available in the root directory of this project. It showcases visual examples of several library functions, including:

*   Activation functions like Sigmoid and ReLU
*   The output of the Softmax function
*   The effect of Min-Max Normalization on a dataset
*   A comparison of Euclidean and Manhattan distances

To explore these visualizations:
1.  Ensure you have Jupyter Notebook or JupyterLab installed. If not, you can install it via pip:
    ```bash
    pip install notebook matplotlib
    ```
    (`matplotlib` is also required for the plots in the notebook).
2.  Navigate to the repository's root directory and launch Jupyter:
    ```bash
    jupyter notebook
    ```
3.  Open the `pymath_visualization.ipynb` file from the Jupyter interface.

[View the Notebook](./pymath_visualization.ipynb)

## Data Science for Beginners - A Learning Notebook

Beyond the `pymath` library itself, this project now also includes a comprehensive Jupyter Notebook designed as a learning guide for individuals new to Data Science:

*   **[`data_science_learning_notebook.ipynb`](./data_science_learning_notebook.ipynb)**

This notebook, titled "A Beginner's Journey into Data Science," provides a structured learning path covering:
*   Core concepts of Data Science and its lifecycle.
*   Python essentials for data tasks.
*   Descriptive statistics and data exploration (EDA) techniques, often using examples from the `pymath` library.
*   Fundamentals of data visualization with Matplotlib.
*   An introduction to Machine Learning concepts, including a hands-on example of the k-Nearest Neighbors algorithm.
*   A mini-project to apply learned skills.
*   Guidance on further learning and resources.

This notebook is intended to be an interactive, educational resource. To use it, ensure you have Jupyter Notebook or JupyterLab installed, along with libraries like `matplotlib` and `numpy`.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue to report bugs or suggest new features, or submit a pull request with your enhancements.

## License

This project is provided as is, for demonstration purposes. No specific license is attached at this time. (Placeholder)
"""
