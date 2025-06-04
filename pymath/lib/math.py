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
