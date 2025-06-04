import math
from collections import Counter

def factorial(n):
  """Calculates the factorial of a non-negative integer n."""
  if n < 0:
    raise ValueError("Factorial is not defined for negative numbers")
  elif n == 0:
    return 1
  else:
    return n * factorial(n - 1)


def fibonacci(n):
  """Calculates the nth Fibonacci number."""
  if n < 0:
    raise ValueError("Fibonacci sequence is not defined for negative numbers")
  elif n == 0:
    return 0
  elif n == 1:
    return 1  # Corrected base case
  else:
    return fibonacci(n - 1) + fibonacci(n - 2)


def gcd(a, b):
  """Calculates the Greatest Common Divisor (GCD) of two integers."""
  while b:
    a, b = b, a % b
  return abs(a) # GCD is usually non-negative


def lcm(a, b):
  """Calculates the Least Common Multiple (LCM) of two integers."""
  if a == 0 or b == 0:
    return 0
  return abs(a * b) // gcd(a, b) if gcd(a,b) != 0 else 0


def is_perfect_square(n):
  """Checks if a number is a perfect square."""
  if n < 0:
    return False
  if n == 0:
    return True
  root = int(n**0.5)
  return root * root == n


def is_prime(n):
  """Checks if a number is a prime number.

  Args:
    n: An integer.

  Returns:
    True if n is a prime number, False otherwise.
  """
  if n < 2:
    return False  # Numbers less than 2 are not prime
  if n == 2:
    return True  # 2 is the smallest prime number
  if n % 2 == 0:
    return False  # Even numbers other than 2 are not prime
  # Check for divisibility by odd numbers up to the square root of n
  for i in range(3, int(n**0.5) + 1, 2):
    if n % i == 0:
      return False
  return True


def mean(data: list[float]) -> float:
    """Calculates the arithmetic mean of a list of numbers."""
    if not data:
        raise ValueError("Input list cannot be empty to calculate mean.")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calculates the median of a list of numbers."""
    if not data:
        raise ValueError("Input list cannot be empty to calculate median.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        # Odd number of elements
        return sorted_data[mid]
    else:
        # Even number of elements
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2


def mode(data: list[float]) -> list[float]:
    """Finds the mode(s) of a list of numbers. Returns a list of modes."""
    if not data:
        raise ValueError("Input list cannot be empty to find mode.")
    counts = Counter(data)
    max_count = 0
    # Find the maximum frequency first
    for item in counts:
        if counts[item] > max_count:
            max_count = counts[item]

    # In case all elements appear once (no mode) or all elements are modes
    # A common convention is that if all items have the same frequency, there is no mode,
    # unless there's only one distinct item.
    # However, another convention (often used in simple implementations) is to return all items
    # that have the highest frequency. We'll follow the latter for simplicity here.
    # If max_count is 1 and len(counts) > 1 (multiple unique items, each once), some might say no mode.
    # But returning all items with max_count covers this and multimodal cases.

    modes = [item for item, count in counts.items() if count == max_count]
    return sorted(list(set(modes))) # Return sorted unique modes


def variance(data: list[float], is_sample: bool = False) -> float:
    """Calculates the variance (population or sample) of a list of numbers."""
    if not data:
        raise ValueError("Input list cannot be empty to calculate variance.")

    n = len(data)
    if is_sample:
        if n < 2:
            raise ValueError("Sample variance requires at least two data points.")
        divisor = n - 1
    else:
        if n == 0: # Should be caught by the first check, but for clarity
             raise ValueError("Population variance requires at least one data point.")
        divisor = n

    m = mean(data)
    sum_sq_diff = sum((x - m) ** 2 for x in data)
    return sum_sq_diff / divisor


def standard_deviation(data: list[float], is_sample: bool = False) -> float:
    """Calculates the standard deviation (population or sample) of a list of numbers."""
    # ValueError conditions are handled by the variance function
    var = variance(data, is_sample)
    return math.sqrt(var)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid (logistic) function."""
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """Calculates the Rectified Linear Unit (ReLU) function."""
    return max(0, x)


def softmax(vector: list[float]) -> list[float]:
    """Computes the softmax function, converting a vector of numbers into a probability distribution."""
    if not vector:
        raise ValueError("Input list cannot be empty for softmax.")

    max_val = max(vector) # For numerical stability
    exp_values = [math.exp(val - max_val) for val in vector]
    sum_exp_values = sum(exp_values)

    if sum_exp_values == 0: # Should not happen if vector is not all -inf, but good to check
        # This case can lead to division by zero.
        # Depending on desired behavior, could return uniform distribution or raise error.
        # For now, let it lead to potential division by zero if not handled by caller,
        # or handle as uniform if appropriate (e.g. return [1/len(vector)] * len(vector))
        # However, with max_val subtraction, exp_values should have at least one non-zero positive value unless original vector was empty or all were -infinity
        # Given float inputs, -infinity is possible. Let's assume standard floats.
        # If all elements are very small negative numbers, exp_values could underflow to 0.
        # A robust softmax might add a small epsilon or handle this case explicitly.
        # For this version, we'll proceed and let division by zero occur if sum_exp_values is 0.
        # This implies the input was problematic (e.g. all elements were so negative they underflowed).
        pass

    probabilities = [ev / sum_exp_values for ev in exp_values]
    return probabilities


def _is_probability_distribution(probabilities: list[float], tolerance: float = 1e-9) -> bool:
    """Helper to check if a list forms a valid probability distribution."""
    if not probabilities: # entropy/cross_entropy/kl_divergence will check this first
        return False
    if any(p < 0 for p in probabilities):
        return False
    return math.isclose(sum(probabilities), 1.0, abs_tol=tolerance)


def entropy(probabilities: list[float]) -> float:
    """Calculates Shannon entropy for a discrete probability distribution."""
    if not probabilities:
        raise ValueError("Input list cannot be empty for entropy calculation.")
    if not _is_probability_distribution(probabilities):
        raise ValueError("Probabilities must be non-negative and sum to 1.")

    H = 0.0
    for p in probabilities:
        if p > 0:
            H -= p * math.log2(p)
    return H


def cross_entropy(p: list[float], q: list[float]) -> float:
    """Calculates cross-entropy between two discrete probability distributions p (true) and q (predicted)."""
    if not p or not q:
        raise ValueError("Input lists cannot be empty for cross-entropy calculation.")
    if len(p) != len(q):
        raise ValueError("Probability distributions must have the same length.")
    if not _is_probability_distribution(p) or not _is_probability_distribution(q):
        raise ValueError("Both lists must be valid probability distributions (non-negative, sum to 1).")

    H_pq = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            if q[i] == 0:
                # This implies infinite cross-entropy.
                # Some libraries return math.inf, others raise error. We'll raise error.
                raise ValueError("Cross-entropy is undefined if true probability p[i] > 0 and predicted q[i] == 0.")
            H_pq -= p[i] * math.log2(q[i])
    return H_pq


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Calculates Kullback-Leibler divergence between two discrete probability distributions p and q."""
    # ValueError checks for empty lists, length mismatch, and valid distributions
    # will be implicitly handled by calls to cross_entropy and entropy.
    # However, we need to explicitly check for p[i] > 0 and q[i] == 0 for D_KL definition
    # before calling, or ensure cross_entropy and entropy handle all checks.
    # The definition H(p,q) - H(p) is simpler if those functions are robust.

    if not p or not q:
        raise ValueError("Input lists cannot be empty for KL divergence.")
    if len(p) != len(q):
        raise ValueError("Probability distributions must have the same length.")
    if not _is_probability_distribution(p, tolerance=1e-7) or not _is_probability_distribution(q, tolerance=1e-7): # Using slightly looser for intermediate sum
        raise ValueError("Both lists must be valid probability distributions for KL divergence.")

    # Direct calculation to handle p_i > 0 and q_i == 0 case specifically for KL divergence definition
    # D_KL(p || q) = sum(p_i * log2(p_i / q_i))
    # This is equivalent to sum(p_i * (log2(p_i) - log2(q_i)))
    # = sum(p_i * log2(p_i)) - sum(p_i * log2(q_i))
    # = -entropy(p) + cross_entropy(p,q) -- if using base 2 for entropy.

    # Check for q_i == 0 where p_i > 0 before calculating terms
    for i in range(len(p)):
        if p[i] > 0 and q[i] == 0:
            raise ValueError("KL divergence is infinite if p[i] > 0 and q[i] == 0.")

    # Using the formula: cross_entropy(p, q) - entropy(p)
    # Need to ensure the tolerance for sum checks in _is_probability_distribution is handled.
    # It's generally better to calculate directly if specific conditions like q[i]=0 for p[i]>0 lead to infinity.

    divergence = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            # q[i] must be > 0 here due to the check above.
            divergence += p[i] * (math.log2(p[i]) - math.log2(q[i]))
            # or divergence += p[i] * math.log2(p[i] / q[i])
    return divergence


# --- Basic Linear Algebra Functions ---

def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Calculates the dot product of two vectors."""
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty for dot product.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length for dot product.")
    return sum(x * y for x, y in zip(vec1, vec2))


def vector_add(vec1: list[float], vec2: list[float]) -> list[float]:
    """Performs element-wise addition of two vectors."""
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty for vector addition.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length for vector addition.")
    return [x + y for x, y in zip(vec1, vec2)]


def vector_subtract(vec1: list[float], vec2: list[float]) -> list[float]:
    """Performs element-wise subtraction of two vectors (vec1 - vec2)."""
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty for vector subtraction.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length for vector subtraction.")
    return [x - y for x, y in zip(vec1, vec2)]


def scalar_multiply(scalar: float, vector: list[float]) -> list[float]:
    """Multiplies each element of a vector by a scalar."""
    if not vector: # As per instruction, return empty list for empty vector
        return []
    return [scalar * x for x in vector]


def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Multiplies a matrix (list of lists) by a vector."""
    if not matrix or not matrix[0]: # Check if matrix or its first row is empty
        raise ValueError("Matrix cannot be empty or have empty rows.")
    if not vector:
        raise ValueError("Vector cannot be empty for matrix-vector multiplication.")

    num_matrix_cols = len(matrix[0])
    if any(len(row) != num_matrix_cols for row in matrix):
        raise ValueError("All rows in the matrix must have the same number of columns.")

    if num_matrix_cols != len(vector):
        raise ValueError("Number of columns in matrix must equal length of vector.")

    result_vector = []
    for row in matrix:
        # dot_product will raise ValueError if row and vector lengths mismatch,
        # but the check above (num_matrix_cols != len(vector)) should cover this
        # if all rows indeed have num_matrix_cols.
        result_vector.append(dot_product(row, vector))
    return result_vector


# --- Other Utility/Statistical Functions ---

def normalize(data: list[float]) -> list[float]:
    """Performs min-max normalization on a list of numbers to scale them to the range [0, 1]."""
    if not data or len(data) < 2:
        raise ValueError("Input list must contain at least two elements for normalization.")

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        # All elements are the same, return a list of zeros
        return [0.0 for _ in data]

    return [(x - min_val) / (max_val - min_val) for x in data]


def euclidean_distance(point1: list[float], point2: list[float]) -> float:
    """Calculates the Euclidean distance between two points (vectors)."""
    if not point1 or not point2:
        raise ValueError("Input points (vectors) cannot be empty.")
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions.")

    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def manhattan_distance(point1: list[float], point2: list[float]) -> float:
    """Calculates the Manhattan (L1) distance between two points (vectors)."""
    if not point1 or not point2:
        raise ValueError("Input points (vectors) cannot be empty.")
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions.")

    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))


def covariance(data1: list[float], data2: list[float], is_sample: bool = False) -> float:
    """Calculates the covariance between two lists of numbers."""
    if not data1 or not data2:
        raise ValueError("Input lists cannot be empty for covariance calculation.")
    if len(data1) != len(data2):
        raise ValueError("Lists must have the same length for covariance calculation.")

    n = len(data1)
    if is_sample:
        if n < 2:
            raise ValueError("Sample covariance requires at least two data points.")
        divisor = n - 1
    else: # Population covariance
        if n == 0: # Should be caught by initial check, but for clarity
            raise ValueError("Population covariance requires at least one data point.")
        divisor = n

    mean1 = mean(data1)
    mean2 = mean(data2)

    sum_prod_diff = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))

    return sum_prod_diff / divisor


def correlation(data1: list[float], data2: list[float]) -> float:
    """Calculates the Pearson correlation coefficient between two lists of numbers."""
    # ValueErrors for empty lists or mismatched lengths are handled by covariance and standard_deviation.
    # Standard deviation functions will also handle lists with < 2 elements for sample calculations if needed.

    # For Pearson correlation, we typically use population standard deviations
    # and population covariance, or sample versions for all.
    # Let's assume population statistics (is_sample=False for internal calls).

    cov = covariance(data1, data2, is_sample=False)

    std_dev1 = standard_deviation(data1, is_sample=False)
    std_dev2 = standard_deviation(data2, is_sample=False)

    if std_dev1 == 0 or std_dev2 == 0:
        # If one variable is constant, correlation is undefined or 0.
        # Common practice is to return 0 if one std dev is 0 but not both,
        # or NaN/raise error if both are 0 (or if one is 0 and covariance is non-zero, implies issue).
        # For simplicity, if either std dev is 0, the denominator is 0.
        # If covariance is also 0, it could be 0/0 -> NaN. If cov is non-zero, it's انف / 0.
        # Let's raise an error for division by zero if std_dev1 * std_dev2 is zero.
        # A more nuanced handling might return 0.0 if covariance is also 0.
        # If covariance is non-zero and a std_dev is zero, it's an odd case, usually implying data is not varying.
        if cov == 0: # If covariance is 0, and at least one std_dev is 0, result is 0 or undefined.
                     # If both std_devs are 0, all points are constant, cov is 0.
            return 0.0 # Or one might argue it should be NaN or error. Let's return 0.0 for this case.
        else: # Covariance is non-zero but a std_dev is zero. This is problematic.
            raise ValueError("Correlation is undefined if one variable is constant (std_dev=0) and covariance is non-zero.")

    return cov / (std_dev1 * std_dev2)
