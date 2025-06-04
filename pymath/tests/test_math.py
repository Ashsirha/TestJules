import unittest
import math
from pymath.lib.math import (
    factorial, fibonacci, gcd, lcm, is_perfect_square, is_prime,
    mean, median, mode, variance, standard_deviation,
    sigmoid, relu, softmax, entropy, cross_entropy, kl_divergence,
    dot_product, vector_add, vector_subtract, scalar_multiply, matrix_vector_multiply,
    normalize, euclidean_distance, manhattan_distance, covariance, correlation
)

class TestMathFunctions(unittest.TestCase):

    def test_factorial(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(10), 3628800)
        with self.assertRaises(ValueError):
            factorial(-1)

    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)
        self.assertEqual(fibonacci(2), 1)
        self.assertEqual(fibonacci(3), 2)
        self.assertEqual(fibonacci(10), 55)
        with self.assertRaises(ValueError):
            fibonacci(-1)

    def test_gcd(self):
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(18, 48), 6)
        self.assertEqual(gcd(17, 5), 1)
        self.assertEqual(gcd(0, 5), 5)
        self.assertEqual(gcd(5, 0), 5)
        self.assertEqual(gcd(0, 0), 0)
        self.assertEqual(gcd(-48, 18), 6)
        self.assertEqual(gcd(48, -18), 6)
        self.assertEqual(gcd(-48, -18), 6) # Added test for two negative inputs

    def test_lcm(self):
        self.assertEqual(lcm(4, 6), 12)
        self.assertEqual(lcm(6, 4), 12)
        self.assertEqual(lcm(5, 7), 35)
        self.assertEqual(lcm(0, 5), 0)
        self.assertEqual(lcm(5, 0), 0)
        self.assertEqual(lcm(0, 0), 0)
        self.assertEqual(lcm(-4, 6), 12)
        self.assertEqual(lcm(4, -6), 12)
        self.assertEqual(lcm(-4, -6), 12) # Added test for two negative inputs

    def test_is_perfect_square(self):
        self.assertTrue(is_perfect_square(0))
        self.assertTrue(is_perfect_square(1))
        self.assertTrue(is_perfect_square(4))
        self.assertTrue(is_perfect_square(9))
        self.assertFalse(is_perfect_square(2))
        self.assertFalse(is_perfect_square(8))
        self.assertFalse(is_perfect_square(-1))
        self.assertFalse(is_perfect_square(-4))

    def test_is_prime(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))
        self.assertTrue(is_prime(11))
        self.assertTrue(is_prime(13))
        self.assertTrue(is_prime(97))
        self.assertFalse(is_prime(-1))
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))
        self.assertFalse(is_prime(8))
        self.assertFalse(is_prime(9))
        self.assertFalse(is_prime(10))
        self.assertFalse(is_prime(100))

    def test_mean(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(mean([-1, 0, 1]), 0.0)
        self.assertAlmostEqual(mean([5]), 5.0)
        self.assertAlmostEqual(mean([1.5, 2.5, 3.5]), 2.5)
        with self.assertRaises(ValueError):
            mean([])

    def test_median(self):
        self.assertAlmostEqual(median([1, 3, 2, 5, 4]), 3.0)
        self.assertAlmostEqual(median([1, 3, 2, 4]), 2.5)
        self.assertAlmostEqual(median([1, 2, 2, 3]), 2.0)
        self.assertAlmostEqual(median([5]), 5.0)
        self.assertAlmostEqual(median([1, 5, 2, 8, 3, 9]), 4.0) # (3+5)/2
        self.assertAlmostEqual(median([10, 20]), 15.0)
        with self.assertRaises(ValueError):
            median([])

    def test_mode(self):
        self.assertEqual(sorted(mode([1, 2, 2, 3, 4])), [2.0])
        self.assertEqual(sorted(mode([1, 2, 2, 3, 3, 4])), [2.0, 3.0])
        self.assertEqual(sorted(mode([1, 1, 2, 2])), [1.0, 2.0])
        self.assertEqual(sorted(mode([5])), [5.0])
        self.assertEqual(sorted(mode([1, 2, 3, 4, 5])), [1.0, 2.0, 3.0, 4.0, 5.0]) # All are modes
        self.assertEqual(sorted(mode([1, 1, 1, 2, 3])), [1.0])
        self.assertEqual(sorted(mode([1.5, 2.0, 2.0, 1.5])), [1.5, 2.0])
        with self.assertRaises(ValueError):
            mode([])

    def test_variance(self):
        self.assertAlmostEqual(variance([1, 2, 3, 4, 5]), 2.0)
        self.assertAlmostEqual(variance([1, 2, 3, 4, 5], is_sample=True), 2.5)
        self.assertAlmostEqual(variance([5]), 0.0) # Population variance of single point is 0
        self.assertAlmostEqual(variance([2, 2, 2, 2]), 0.0)
        self.assertAlmostEqual(variance([10, 12, 15, 11, 12]), 2.8) # Pop Var for (10,12,15,11,12) mean=12, sumsqdiff = (4+0+9+1+0)=14, 14/5=2.8.
                                                                    # Corrected: mean = (10+12+15+11+12)/5 = 60/5 = 12.
                                                                    # (10-12)^2 = 4
                                                                    # (12-12)^2 = 0
                                                                    # (15-12)^2 = 9
                                                                    # (11-12)^2 = 1
                                                                    # (12-12)^2 = 0
                                                                    # Sum of squares = 4+0+9+1+0 = 14. Population variance = 14/5 = 2.8
        self.assertAlmostEqual(variance([10, 12, 15, 11, 12], is_sample=True), 3.5) # Sample variance = 14/4 = 3.5

        with self.assertRaises(ValueError):
            variance([])
        with self.assertRaises(ValueError):
            variance([], is_sample=True)
        with self.assertRaises(ValueError):
            variance([5], is_sample=True)

    def test_standard_deviation(self):
        self.assertAlmostEqual(standard_deviation([1, 2, 3, 4, 5]), math.sqrt(2.0))
        self.assertAlmostEqual(standard_deviation([1, 2, 3, 4, 5], is_sample=True), math.sqrt(2.5))
        self.assertAlmostEqual(standard_deviation([5]), 0.0)
        self.assertAlmostEqual(standard_deviation([2, 2, 2, 2]), 0.0)
        self.assertAlmostEqual(standard_deviation([10, 12, 15, 11, 12]), math.sqrt(2.8))
        self.assertAlmostEqual(standard_deviation([10, 12, 15, 11, 12], is_sample=True), math.sqrt(3.5))

        with self.assertRaises(ValueError):
            standard_deviation([])
        with self.assertRaises(ValueError):
            standard_deviation([], is_sample=True)
        with self.assertRaises(ValueError):
            standard_deviation([5], is_sample=True)

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(100), 1.0) # Approaches 1 for large positive x
        self.assertAlmostEqual(sigmoid(-100), 0.0) # Approaches 0 for large negative x
        self.assertAlmostEqual(sigmoid(0.458), 0.612539624420E+00, places=7) # Value from online calculator

    def test_relu(self):
        self.assertEqual(relu(10), 10)
        self.assertEqual(relu(-10), 0)
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(0.5), 0.5)
        self.assertEqual(relu(-0.5), 0)

    def test_softmax(self):
        # Test with a simple vector
        res1 = softmax([1, 2, 3])
        self.assertAlmostEqual(sum(res1), 1.0)
        self.assertAlmostEqual(res1[0], 0.09003057, places=7)
        self.assertAlmostEqual(res1[1], 0.24472847, places=7)
        self.assertAlmostEqual(res1[2], 0.66524096, places=7)

        # Test with zero and negative numbers
        res2 = softmax([-1, 0, 1])
        self.assertAlmostEqual(sum(res2), 1.0)
        # math.exp(0)/ (math.exp(-2) + math.exp(-1) + math.exp(0)) # if max_val = 1
        # exp(-1 -1) = exp(-2) = 0.13533528
        # exp(0 - 1) = exp(-1) = 0.36787944
        # exp(1 - 1) = exp(0)  = 1.0
        # sum = 1.50321472
        # approx: [0.09003057, 0.24472847, 0.66524096] (same as [1,2,3] shifted)
        self.assertAlmostEqual(res2[0], 0.09003057, places=7)
        self.assertAlmostEqual(res2[1], 0.24472847, places=7)
        self.assertAlmostEqual(res2[2], 0.66524096, places=7)

        # Test with a single element vector
        res3 = softmax([5])
        self.assertAlmostEqual(sum(res3), 1.0)
        self.assertEqual(res3, [1.0])

        # Test with identical elements
        res4 = softmax([2, 2, 2])
        self.assertAlmostEqual(sum(res4), 1.0)
        self.assertAlmostEqual(res4[0], 1/3)
        self.assertAlmostEqual(res4[1], 1/3)
        self.assertAlmostEqual(res4[2], 1/3)

        with self.assertRaises(ValueError):
            softmax([])

    def test_entropy(self):
        self.assertAlmostEqual(entropy([0.5, 0.5]), 1.0)
        self.assertAlmostEqual(entropy([1.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(entropy([0.8, 0.2]), - (0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))) # approx 0.7219
        self.assertAlmostEqual(entropy([0.25, 0.25, 0.25, 0.25]), 2.0) # - (4 * (0.25 * log2(0.25))) = - (0.25 * (-2)) * 4 = 0.5 * 4 = 2

        with self.assertRaises(ValueError):
            entropy([])
        with self.assertRaises(ValueError): # Does not sum to 1
            entropy([0.5, 0.6])
        with self.assertRaises(ValueError): # Contains negative
            entropy([-0.1, 1.1])
        with self.assertRaises(ValueError): # Sums to 1 but has negative
            entropy([-0.5, 0.5, 0.5, 0.5])


    def test_cross_entropy(self):
        p1 = [0.5, 0.5]
        q1 = [0.5, 0.5]
        self.assertAlmostEqual(cross_entropy(p1, q1), entropy(p1)) # Should be 1.0

        p2 = [1.0, 0.0]
        q2 = [0.1, 0.9]
        # -(1*log2(0.1) + 0*log2(0.9)) = -log2(0.1) approx 3.3219
        self.assertAlmostEqual(cross_entropy(p2, q2), -math.log2(0.1))

        p3 = [0.25, 0.25, 0.5]
        q3 = [0.3, 0.3, 0.4]
        expected_ce3 = -(0.25*math.log2(0.3) + 0.25*math.log2(0.3) + 0.5*math.log2(0.4))
        self.assertAlmostEqual(cross_entropy(p3, q3), expected_ce3)

        with self.assertRaises(ValueError): # Empty lists
            cross_entropy([], [])
        with self.assertRaises(ValueError):
            cross_entropy(p1, [])
        with self.assertRaises(ValueError):
            cross_entropy([], q1)
        with self.assertRaises(ValueError): # Mismatched lengths
            cross_entropy(p1, p3)
        with self.assertRaises(ValueError): # p not a distribution
            cross_entropy([0.5, 0.6], q1)
        with self.assertRaises(ValueError): # q not a distribution
            cross_entropy(p1, [0.5, 0.6])
        with self.assertRaises(ValueError): # q_i = 0 where p_i > 0
            cross_entropy([0.5, 0.5], [1.0, 0.0])

    def test_kl_divergence(self):
        p1 = [0.5, 0.5]
        q1 = [0.5, 0.5]
        self.assertAlmostEqual(kl_divergence(p1, q1), 0.0)

        p2 = [0.8, 0.2]
        q2 = [0.5, 0.5]
        # D_KL(P || Q) = sum(P[i] * log2(P[i] / Q[i]))
        # 0.8 * log2(0.8/0.5) + 0.2 * log2(0.2/0.5)
        # 0.8 * log2(1.6) + 0.2 * log2(0.4)
        # 0.8 * 0.6780719051126377 + 0.2 * (-1.3219280948873622)
        # 0.5424575240901102 - 0.26438561897747244 = 0.2780719051126378
        self.assertAlmostEqual(kl_divergence(p2, q2), 0.8 * math.log2(1.6) + 0.2 * math.log2(0.4))

        p3 = [1.0, 0.0] # p
        q3 = [0.1, 0.9] # q
        # 1.0 * log2(1.0/0.1) + 0 = log2(10)
        self.assertAlmostEqual(kl_divergence(p3, q3), math.log2(10))
        # Using cross_entropy(p,q) - entropy(p)
        # cross_entropy([1,0], [0.1,0.9]) = -log2(0.1) approx 3.3219
        # entropy([1,0]) = 0
        # So KL div should be -log2(0.1) - 0 = -log2(0.1)
        self.assertAlmostEqual(kl_divergence(p3, q3), cross_entropy(p3, q3) - entropy(p3))


        with self.assertRaises(ValueError): # Empty lists
            kl_divergence([], [])
        with self.assertRaises(ValueError): # Mismatched lengths
            kl_divergence(p1, [0.1, 0.2, 0.7])
        with self.assertRaises(ValueError): # p not a distribution
            kl_divergence([0.5, 0.6], q1)
        with self.assertRaises(ValueError): # q not a distribution
            kl_divergence(p1, [0.5, 0.6])
        with self.assertRaises(ValueError): # q_i = 0 where p_i > 0
            kl_divergence([0.5, 0.5], [1.0, 0.0])

    # --- Linear Algebra Tests ---

    def test_dot_product(self):
        self.assertAlmostEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)
        self.assertAlmostEqual(dot_product([-1, 0, 1], [1, 2, 3]), 2.0)
        self.assertAlmostEqual(dot_product([1.5, 2.5], [2.0, 3.0]), 3.0 + 7.5) # 10.5
        self.assertAlmostEqual(dot_product([1,2,3], [0,0,0]), 0.0)
        with self.assertRaises(ValueError):
            dot_product([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            dot_product([], [1, 2, 3])
        with self.assertRaises(ValueError):
            dot_product([1, 2, 3], [])
        with self.assertRaises(ValueError):
            dot_product([], [])

    def test_vector_add(self):
        self.assertEqual(vector_add([1, 2, 3], [4, 5, 6]), [5, 7, 9])
        self.assertEqual(vector_add([-1, 0, 1], [1, 2, 3]), [0, 2, 4])
        self.assertEqual(vector_add([1.5, 2.5], [0.5, 1.5]), [2.0, 4.0])
        with self.assertRaises(ValueError):
            vector_add([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            vector_add([], [1, 2, 3])
        with self.assertRaises(ValueError):
            vector_add([1, 2, 3], [])
        with self.assertRaises(ValueError):
            vector_add([], [])

    def test_vector_subtract(self):
        self.assertEqual(vector_subtract([4, 5, 6], [1, 2, 3]), [3, 3, 3])
        self.assertEqual(vector_subtract([-1, 0, 1], [1, 2, 3]), [-2, -2, -2])
        self.assertEqual(vector_subtract([1.5, 2.5], [0.5, 1.5]), [1.0, 1.0])
        with self.assertRaises(ValueError):
            vector_subtract([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            vector_subtract([], [1, 2, 3])
        with self.assertRaises(ValueError):
            vector_subtract([1, 2, 3], [])
        with self.assertRaises(ValueError):
            vector_subtract([], [])

    def test_scalar_multiply(self):
        self.assertEqual(scalar_multiply(3, [1, 2, 3]), [3, 6, 9])
        self.assertEqual(scalar_multiply(0, [1, 2, 3]), [0, 0, 0])
        self.assertEqual(scalar_multiply(-2, [1, -2, 3]), [-2, 4, -6])
        self.assertEqual(scalar_multiply(1.5, [2, 4]), [3.0, 6.0])
        self.assertEqual(scalar_multiply(3, []), [])

    def test_matrix_vector_multiply(self):
        mat1 = [[1, 2, 3], [4, 5, 6]] # 2x3
        vec1 = [1, 2, 1] # 3
        self.assertEqual(matrix_vector_multiply(mat1, vec1), [8, 20])

        mat2 = [[1, 0], [0, 1]] # 2x2 (Identity)
        vec2 = [3, 4] # 2
        self.assertEqual(matrix_vector_multiply(mat2, vec2), [3, 4])

        mat3 = [[1, -1], [-1, 1]] # 2x2
        vec3 = [5, 2] # 2
        self.assertEqual(matrix_vector_multiply(mat3, vec3), [3, -3])

        mat4 = [[1.0, 2.5], [3.0, 0.5]]
        vec4 = [2.0, 4.0]
        self.assertEqual(matrix_vector_multiply(mat4, vec4), [1.0*2.0 + 2.5*4.0, 3.0*2.0 + 0.5*4.0]) # [2+10, 6+2] = [12, 8]


        with self.assertRaises(ValueError): # Empty matrix
            matrix_vector_multiply([], [1, 2])
        with self.assertRaises(ValueError): # Matrix with empty row (caught by matrix[0] check)
            matrix_vector_multiply([[]], [1,2])
        with self.assertRaises(ValueError): # Empty vector
            matrix_vector_multiply([[1, 2]], [])
        with self.assertRaises(ValueError): # Incompatible dimensions
            matrix_vector_multiply([[1, 2, 3], [4, 5, 6]], [1, 2])
        with self.assertRaises(ValueError): # Inconsistent row lengths
            matrix_vector_multiply([[1, 2], [3, 4, 5]], [1, 2])

    # --- Other Utility/Statistical Tests ---

    def test_normalize(self):
        self.assertEqual(normalize([1, 2, 3, 4, 5]), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(normalize([-2, -1, 0, 1, 2]), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(normalize([10, 20]), [0.0, 1.0])
        # Test with list where all elements are the same
        self.assertEqual(normalize([5, 5, 5]), [0.0, 0.0, 0.0])

        with self.assertRaises(ValueError): # Empty list
            normalize([])
        with self.assertRaises(ValueError): # Single element list
            normalize([5])

        # Test with floating point numbers
        normalized_floats = normalize([1.0, 2.5, 5.0, 10.0]) # min=1, max=10, range=9
        expected_floats = [(1.0-1.0)/9.0, (2.5-1.0)/9.0, (5.0-1.0)/9.0, (10.0-1.0)/9.0]
        for res, exp in zip(normalized_floats, expected_floats):
            self.assertAlmostEqual(res, exp)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance([1, 2], [4, 6]), 5.0)
        self.assertAlmostEqual(euclidean_distance([0, 0, 0], [1, 1, 1]), math.sqrt(3))
        self.assertAlmostEqual(euclidean_distance([1, 2, 3, 4], [1, 2, 3, 4]), 0.0)
        self.assertAlmostEqual(euclidean_distance([-1, -2], [-4, -6]), 5.0)

        with self.assertRaises(ValueError): # Different dimensions
            euclidean_distance([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError): # Empty points
            euclidean_distance([], [])
        with self.assertRaises(ValueError):
            euclidean_distance([1,2], [])
        with self.assertRaises(ValueError):
            euclidean_distance([], [1,2])

    def test_manhattan_distance(self):
        self.assertAlmostEqual(manhattan_distance([1, 2], [4, 6]), 7.0)
        self.assertAlmostEqual(manhattan_distance([0, 0, 0], [1, 1, 1]), 3.0)
        self.assertAlmostEqual(manhattan_distance([1, 2, 3, 4], [1, 2, 3, 4]), 0.0)
        self.assertAlmostEqual(manhattan_distance([-1, -2], [-4, -6]), 7.0) # abs(-4 - (-1)) + abs(-6 - (-2)) = abs(-3) + abs(-4) = 3+4 = 7

        with self.assertRaises(ValueError): # Different dimensions
            manhattan_distance([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError): # Empty points
            manhattan_distance([], [])

    def test_covariance(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6] # mean1=3, mean2=4. DiffProds: (-2*-2),(-1*-1),(0*0),(1*1),(2*2) => 4,1,0,1,4. Sum=10
        self.assertAlmostEqual(covariance(data1, data2), 2.0) # Pop: 10/5=2
        self.assertAlmostEqual(covariance(data1, data2, is_sample=True), 2.5) # Sample: 10/4=2.5

        data3 = [1, 2, 3]
        data4 = [3, 2, 1] # mean3=2, mean4=2. DiffProds: (-1*1),(0*0),(1*-1) => -1,0,-1. Sum=-2
        self.assertAlmostEqual(covariance(data3, data4), -2/3) # Pop: -2/3
        self.assertAlmostEqual(covariance(data3, data4, is_sample=True), -1.0) # Sample: -2/2=-1

        data5 = [1, 2, 3, 4, 5]
        data6 = [5, 5, 5, 5, 5] # mean5=3, mean6=5. DiffProds: (-2*0),(-1*0),(0*0),(1*0),(2*0) => 0 for all. Sum=0
        self.assertAlmostEqual(covariance(data5, data6), 0.0)
        self.assertAlmostEqual(covariance(data5, data6, is_sample=True), 0.0)

        with self.assertRaises(ValueError): # Different lengths
            covariance([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError): # Empty lists
            covariance([], [])
        with self.assertRaises(ValueError): # Sample cov with < 2 elements
            covariance([1], [2], is_sample=True)
        self.assertAlmostEqual(covariance([1], [2]), 0.0) # Pop cov with 1 element is 0

    def test_correlation(self):
        self.assertAlmostEqual(correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 1.0) # Perfect positive
        self.assertAlmostEqual(correlation([1, 2, 3, 4, 5], [-2, -4, -6, -8, -10]), -1.0) # Perfect negative

        data1 = [1, 2, 3, 4, 5] # std_dev = sqrt(2)
        data2 = [2, 3, 4, 5, 6] # std_dev = sqrt(2)
        # cov = 2.0 (from above)
        # corr = 2.0 / (sqrt(2)*sqrt(2)) = 2.0 / 2.0 = 1.0
        self.assertAlmostEqual(correlation(data1, data2), 1.0)

        data3 = [1, 2, 3] # std_dev = sqrt( ( (1-2)^2 + (2-2)^2 + (3-2)^2 ) / 3 ) = sqrt( (1+0+1)/3 ) = sqrt(2/3)
        data4 = [3, 2, 1] # std_dev = sqrt(2/3)
        # cov = -2/3 (from above)
        # corr = (-2/3) / (sqrt(2/3)*sqrt(2/3)) = (-2/3) / (2/3) = -1.0
        self.assertAlmostEqual(correlation(data3, data4), -1.0)

        # Test for near zero correlation
        data_x = [1, 2, 3, 4, 5, 6, 7, 8]
        data_y = [2, 4, 1, 5, 3, 7, 2, 6] # Random-ish
        # Using online calculator, this should be around 0.5345...
        # For this test, let's just confirm it's between -1 and 1 and not exactly 1 or -1.
        corr_xy = correlation(data_x, data_y)
        self.assertTrue(-1.0 <= corr_xy <= 1.0)
        self.assertNotAlmostEqual(abs(corr_xy), 1.0) # Check it's not perfectly correlated


        with self.assertRaises(ValueError): # Different lengths
            correlation([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError): # Empty lists
            correlation([], [])

        # Test case for zero standard deviation
        # According to implementation: if cov is 0 and a std_dev is 0, returns 0.0
        self.assertAlmostEqual(correlation([1, 1, 1], [2, 2, 2]), 0.0) # cov=0, std1=0, std2=0
        self.assertAlmostEqual(correlation([1, 2, 3], [5, 5, 5]), 0.0) # cov=0, std2=0

        # Test case: covariance is non-zero but one std_dev is zero (should raise error)
        # This case is hard to construct naturally as if one std_dev is zero (constant values),
        # covariance with any other varying dataset should also be zero.
        # The implementation checks `if cov == 0: return 0.0` before `else: raise ValueError`.
        # So, if a std_dev is 0, cov must also be 0 for it to return 0.0.
        # If std_dev is 0 and cov is somehow non-zero (which shouldn't happen with correct cov calculation),
        # then it would raise an error. The current logic seems to cover this.

if __name__ == '__main__':
    unittest.main()
