# PyMath: A Simple Math Library

## Introduction
PyMath is a lightweight Python library that provides a collection of simple mathematical functions. It aims to offer clear and efficient implementations for common mathematical operations.

## Features/Functions
The library currently includes the following functions:

*   **`factorial(n)`**: Calculates the factorial of a non-negative integer `n`.
*   **`fibonacci(n)`**: Calculates the nth Fibonacci number.
*   **`gcd(a, b)`**: Computes the Greatest Common Divisor (GCD) of two integers `a` and `b`.
*   **`lcm(a, b)`**: Computes the Least Common Multiple (LCM) of two integers `a` and `b`.
*   **`is_perfect_square(n)`**: Checks if a non-negative integer `n` is a perfect square.
*   **`is_prime(n)`**: Determines if an integer `n` is a prime number.

All functions are located in the `pymath.lib.math` module.

## Usage Examples
Here's how you can import and use functions from PyMath:

```python
from pymath.lib.math import factorial, is_prime, fibonacci

# Calculate the factorial of 5
print(f"Factorial of 5: {factorial(5)}") # Output: 120

# Check for primality
print(f"Is 7 prime? {is_prime(7)}")     # Output: True
print(f"Is 10 prime? {is_prime(10)}")   # Output: False

# Calculate a Fibonacci number
print(f"The 10th Fibonacci number is: {fibonacci(10)}") # Output: 55
```

## Testing
The library comes with a suite of unit tests to ensure correctness. The tests are located in `pymath/tests/test_math.py`.

To run the tests, navigate to the root directory of the project and execute the following command:
```bash
python -m unittest pymath/tests/test_math.py
```
This will run all the defined tests and report their status.

## Visualizations and Examples Notebook
A Jupyter Notebook, `pymath_visualization.ipynb`, is available in the root directory. This notebook provides visual examples and demonstrations of some of the library's functions, including plots for the Fibonacci sequence, factorial growth, and prime number identification.

You can view it by opening it with Jupyter Notebook or JupyterLab. The file is located [here](./pymath_visualization.ipynb).

To run the notebook and its visualizations, ensure you have Jupyter Notebook or JupyterLab installed, along with the `matplotlib` library. You can typically install these with pip:
```bash
pip install jupyter matplotlib
```
