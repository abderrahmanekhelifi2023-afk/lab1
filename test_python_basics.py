"""Примеры и граничные случаи для задач на базовый Python."""

import unittest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


class PythonBasicsTests(unittest.TestCase):
    def test_count_vowels_mixed_case(self):
        self.assertEqual(count_vowels(TextInput("Hello WORLD")), 3)
        self.assertEqual(count_vowels(TextInput("aEiOuAEIOU")), 10)

    def test_count_vowels_empty_and_without_vowels(self):
        self.assertEqual(count_vowels(TextInput("")), 0)
        self.assertEqual(count_vowels(TextInput("rhythm")), 0)

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("abc !")))
        self.assertFalse(has_unique_characters(TextInput("hello")))

    def test_unique_characters_empty_and_case(self):
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertTrue(has_unique_characters(TextInput("aA")))
        self.assertFalse(has_unique_characters(TextInput("  ")))

    def test_one_bits(self):
        for number, expected in [(1, 1), (8, 1), (13, 3), (255, 8)]:
            with self.subTest(number=number):
                self.assertEqual(count_one_bits(PositiveIntegerInput(number)), expected)

    def test_one_bits_large_number(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput((1 << 1000) + 1)), 2)

    def test_persistence_examples(self):
        for number, expected in [(39, 3), (4, 0), (999, 4)]:
            with self.subTest(number=number):
                self.assertEqual(multiplicative_persistence(PositiveIntegerInput(number)), expected)

    def test_persistence_single_digit_and_zero_digit(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(1)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(10)), 1)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(101)), 1)

    def test_mse(self):
        self.assertEqual(mse(VectorPairInput([1, 2, 3], [1, 2, 3])), 0.0)
        self.assertAlmostEqual(mse(VectorPairInput([1, 2, 3], [2, 2, 5])), 5 / 3)

    def test_mse_single_value_and_negative_values(self):
        self.assertEqual(mse(VectorPairInput([3], [-1])), 16.0)
        self.assertEqual(mse(VectorPairInput((-1, -2), (1, 2))), 10.0)

    def test_mse_invalid_lengths(self):
        for predicted, expected in [([], []), ([1], []), ([], [1]), ([1], [1, 2])]:
            with self.subTest(predicted=predicted, expected=expected):
                with self.assertRaises(ValueError):
                    mse(VectorPairInput(predicted, expected))

    def test_factorization_example(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(86240)), "(2**5)(5)(7**2)(11)")

    def test_factorization_boundary_and_primes(self):
        for number, expected in [(1, ""), (2, "(2)"), (97, "(97)"), (49, "(7**2)")]:
            with self.subTest(number=number):
                self.assertEqual(prime_factorization(PositiveIntegerInput(number)), expected)

    def test_factorization_powers_and_mixed_factors(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(1024)), "(2**10)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(90)), "(2)(3**2)(5)")

    def test_pyramid_exact_heights(self):
        for count, height in [(1, 1), (5, 2), (14, 3), (30, 4), (55, 5)]:
            with self.subTest(count=count):
                self.assertEqual(pyramid(PositiveIntegerInput(count)), height)

    def test_pyramid_impossible(self):
        for count in [2, 4, 6, 13, 15]:
            with self.subTest(count=count):
                self.assertEqual(pyramid(PositiveIntegerInput(count)), "It is impossible")

    def test_pyramid_large_count(self):
        height = 10**12
        count = height * (height + 1) * (2 * height + 1) // 6
        self.assertEqual(pyramid(PositiveIntegerInput(count)), height)
        self.assertEqual(pyramid(PositiveIntegerInput(count + 1)), "It is impossible")

    def test_balanced_examples(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))

    def test_balanced_small_numbers(self):
        for number in [1, 9, 10, 12, 99]:
            with self.subTest(number=number):
                self.assertTrue(is_balanced_number(PositiveIntegerInput(number)))

    def test_balanced_middle_digits_are_ignored(self):
        for number, expected in [(191, True), (192, False), (1991, True), (1992, False), (120030, True)]:
            with self.subTest(number=number):
                self.assertEqual(is_balanced_number(PositiveIntegerInput(number)), expected)

    def test_positive_integer_validation(self):
        for function in [count_one_bits, multiplicative_persistence, prime_factorization, pyramid, is_balanced_number]:
            for number in [0, -1]:
                with self.subTest(function=function.__name__, number=number):
                    with self.assertRaises(ValueError):
                        function(PositiveIntegerInput(number))


if __name__ == "__main__":
    unittest.main()
