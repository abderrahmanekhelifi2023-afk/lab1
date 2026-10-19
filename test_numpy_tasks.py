"""Собственные тесты NumPy-задач и их граничных случаев.

Запуск из директории lab1: python -m unittest test_numpy_tasks -v
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import numpy_tasks as tasks
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


class MatrixOperationsTests(unittest.TestCase):
    def test_sum_prod_adds_corresponding_products(self):
        matrices = np.array([[[1, 2], [3, 4]], [[0, 1], [1, 0]]])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        result = tasks.sum_prod(MatrixVectorBatchInput(matrices, vectors))
        self.assertEqual(result.shape, (2, 1))
        assert_array_equal(result, [[9], [14]])

    def test_sum_prod_single_scalar_matrix(self):
        result = tasks.sum_prod(MatrixVectorBatchInput([[[0.5]]], [[[3.0]]]))
        self.assertEqual(result.shape, (1, 1))
        assert_allclose(result, [[1.5]])

    def test_sum_prod_empty_batch_preserves_vector_shape(self):
        data = MatrixVectorBatchInput(np.empty((0, 2, 2)), np.empty((0, 2, 1)))
        result = tasks.sum_prod(data)
        self.assertEqual(result.shape, (2, 1))
        assert_array_equal(result, [[0], [0]])

    def test_binarize_uses_strict_threshold_and_keeps_input(self):
        matrix = np.array([[0.2, 0.5], [0.6, -1.0]])
        original = matrix.copy()
        result = tasks.binarize(BinarizeInput(matrix, threshold=0.5))
        assert_array_equal(result, [[0, 0], [1, 0]])
        assert_array_equal(matrix, original)
        self.assertFalse(np.shares_memory(matrix, result))

    def test_binarize_negative_threshold_and_empty_matrix(self):
        result = tasks.binarize(BinarizeInput([[-3, -2, 0]], threshold=-2))
        assert_array_equal(result, [[0, 0, 1]])
        empty = tasks.binarize(BinarizeInput(np.empty((0, 3))))
        self.assertEqual(empty.shape, (0, 3))

    def test_unique_rows_returns_sorted_individual_lists(self):
        matrix = np.array([[3, 1, 3], [2, 2, 1]])
        result = tasks.unique_rows(MatrixInput(matrix))
        self.assertEqual(result, [[1, 3], [1, 2]])
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(row, list) for row in result))

    def test_unique_columns_uses_columns_instead_of_rows(self):
        matrix = np.array([[2, 0], [2, 1], [1, 0]])
        result = tasks.unique_columns(MatrixInput(matrix))
        self.assertEqual(result, [[1, 2], [0, 1]])
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(column, list) for column in result))

    def test_unique_rows_empty_axes(self):
        self.assertEqual(tasks.unique_rows(MatrixInput(np.empty((0, 3)))), [])
        self.assertEqual(tasks.unique_rows(MatrixInput(np.empty((2, 0)))), [[], []])

    def test_unique_columns_empty_axes(self):
        self.assertEqual(tasks.unique_columns(MatrixInput(np.empty((0, 3)))), [[], [], []])
        self.assertEqual(tasks.unique_columns(MatrixInput(np.empty((2, 0)))), [])


class RandomMatrixTests(unittest.TestCase):
    def test_seed_generates_expected_matrix_and_population_statistics(self):
        result = tasks.matrix_statistics(RandomMatrixInput(2, 3, seed=42))
        expected = np.array([
            [0.30471707975443135, -1.0399841062404955, 0.7504511958064572],
            [0.9405647163912139, -1.9510351886538364, -1.302179506862318],
        ])
        self.assertIsInstance(result, MatrixStatistics)
        assert_allclose(result.matrix, expected)
        row_means = [sum(row) / 3 for row in expected]
        column_means = [sum(column) / 2 for column in expected.T]
        row_variances = [
            sum((value - mean) ** 2 for value in row) / 3
            for row, mean in zip(expected, row_means)
        ]
        column_variances = [
            ((column[0] - column[1]) / 2) ** 2 for column in expected.T
        ]
        assert_allclose(result.row_means, row_means)
        assert_allclose(result.column_means, column_means)
        assert_allclose(result.row_variances, row_variances)
        assert_allclose(result.column_variances, column_variances)

    def test_same_seed_repeats_and_different_seed_changes_values(self):
        first = tasks.matrix_statistics(RandomMatrixInput(3, 2, seed=7))
        repeated = tasks.matrix_statistics(RandomMatrixInput(3, 2, seed=7))
        other = tasks.matrix_statistics(RandomMatrixInput(3, 2, seed=8))
        assert_array_equal(first.matrix, repeated.matrix)
        self.assertFalse(np.array_equal(first.matrix, other.matrix))

    def test_zero_standard_deviation_and_single_column(self):
        result = tasks.matrix_statistics(RandomMatrixInput(2, 1, mean=4.5, std=0))
        assert_array_equal(result.matrix, [[4.5], [4.5]])
        assert_array_equal(result.row_means, [4.5, 4.5])
        assert_array_equal(result.column_means, [4.5])
        assert_array_equal(result.row_variances, [0, 0])
        assert_array_equal(result.column_variances, [0])


class ChessTests(unittest.TestCase):
    def test_non_square_chess_matrix_starts_with_first_value(self):
        result = tasks.chess(ChessInput(3, 4, 2, 9))
        assert_array_equal(result, [[2, 9, 2, 9], [9, 2, 9, 2], [2, 9, 2, 9]])

    def test_single_row_and_single_column_preserve_fractional_values(self):
        row = tasks.chess(ChessInput(1, 3, 0.25, 0.75))
        column = tasks.chess(ChessInput(3, 1, 0.25, 0.75))
        assert_allclose(row, [[0.25, 0.75, 0.25]])
        assert_allclose(column, [[0.25], [0.75], [0.25]])


class DrawingTests(unittest.TestCase):
    shape_color = (200, 50, 10)
    background_color = (2, 4, 6)

    def assert_image_mask(self, result, expected_mask):
        expected_mask = np.asarray(expected_mask, dtype=bool)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, expected_mask.shape + (3,))
        assert_array_equal(result[expected_mask], np.tile(self.shape_color, (expected_mask.sum(), 1)))
        assert_array_equal(
            result[~expected_mask], np.tile(self.background_color, ((~expected_mask).sum(), 1))
        )

    def test_rectangle_has_exact_dimensions_and_centering(self):
        result = tasks.draw_rectangle(RectangleInput(
            width=3, height=2, image_height=4, image_width=5,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])

    def test_rectangle_larger_than_canvas_is_clipped(self):
        result = tasks.draw_rectangle(RectangleInput(
            width=7, height=1, image_height=4, image_width=5,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

    def test_ellipse_includes_boundary_pixels(self):
        result = tasks.draw_ellipse(EllipseInput(
            semi_axis_x=2, semi_axis_y=1, image_height=5, image_width=5,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])

    def test_ellipse_is_symmetric_on_even_non_square_canvas(self):
        result = tasks.draw_ellipse(EllipseInput(
            semi_axis_x=2, semi_axis_y=1, image_height=4, image_width=6,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ])

    def test_ellipse_keeps_exact_boundary_pixels_for_radius_thirteen(self):
        result = tasks.draw_ellipse(EllipseInput(
            semi_axis_x=13, semi_axis_y=13, image_height=27, image_width=27,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        for offset_x, offset_y in [(5, 12), (12, 5)]:
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    x = 13 + sign_x * offset_x
                    y = 13 + sign_y * offset_y
                    with self.subTest(x=x, y=y):
                        assert_array_equal(result[y, x], self.shape_color)
        assert_array_equal(result[25, 19], self.background_color)

    def test_ellipse_large_axis_avoids_integer_overflow(self):
        result = tasks.draw_ellipse(EllipseInput(
            semi_axis_x=10 ** 12, semi_axis_y=1, image_height=3, image_width=3,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def test_ellipse_on_single_pixel_and_invalid_semi_axes(self):
        result = tasks.draw_ellipse(EllipseInput(
            semi_axis_x=1, semi_axis_y=1, image_height=1, image_width=1,
            shape_color=self.shape_color, background_color=self.background_color,
        ))
        self.assert_image_mask(result, [[1]])
        for axis_x, axis_y in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
            with self.subTest(axis_x=axis_x, axis_y=axis_y):
                with self.assertRaises(ValueError):
                    tasks.draw_ellipse(EllipseInput(
                        axis_x, axis_y, 3, 3, self.shape_color, self.background_color,
                    ))


class TimeSeriesTests(unittest.TestCase):
    def test_integer_series_preserves_small_values_and_cancellation(self):
        values = np.array([2 ** 54, 1, 0], dtype=np.int64)
        result = tasks.analyze_time_series(TimeSeriesInput(values, 1))
        assert_array_equal(result.moving_average, values.astype(float))
        result = tasks.analyze_time_series(TimeSeriesInput([2 ** 54, -(2 ** 54), 1], 3))
        self.assertAlmostEqual(result.mean, 1 / 3)
        assert_allclose(result.moving_average, [1 / 3])
        values = np.array([-(2 ** 63), 2 ** 63 - 1], dtype=np.int64)
        result = tasks.analyze_time_series(TimeSeriesInput(values, 2))
        self.assertEqual(result.mean, -0.5)
        assert_array_equal(result.moving_average, [-0.5])
        self.assertEqual(result.variance, ((2 ** 64 - 1) ** 2) / 4)

    def test_large_integer_extrema_preserve_exact_comparisons_and_input(self):
        base = 2 ** 53
        cases = [
            ([base, base + 1, base], [1], []),
            ([base + 1, base, base + 1], [], [1]),
        ]
        for values, maxima, minima in cases:
            with self.subTest(values=values):
                values = np.array(values, dtype=np.int64)
                original = values.copy()
                result = tasks.analyze_time_series(TimeSeriesInput(values, 1))
                assert_array_equal(result.local_maxima_indices, maxima)
                assert_array_equal(result.local_minima_indices, minima)
                self.assertAlmostEqual(result.variance, 2 / 9)
                self.assertAlmostEqual(result.std, (2 / 9) ** 0.5)
                assert_array_equal(result.moving_average, values.astype(float))
                assert_array_equal(values, original)

    def test_statistics_extrema_and_moving_average(self):
        result = tasks.analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 0], 3))
        self.assertIsInstance(result, TimeSeriesStatistics)
        self.assertAlmostEqual(result.mean, 2.0)
        self.assertAlmostEqual(result.variance, 2.0)
        self.assertAlmostEqual(result.std, 2 ** 0.5)
        assert_array_equal(result.local_maxima_indices, [1, 3])
        assert_array_equal(result.local_minima_indices, [2])
        assert_allclose(result.moving_average, [2, 3, 2])

    def test_plateaus_are_not_strict_extrema_and_window_one_preserves_values(self):
        values = np.array([2, 2, 1, 1, 3, 3])
        result = tasks.analyze_time_series(TimeSeriesInput(values, 1))
        assert_array_equal(result.local_maxima_indices, [])
        assert_array_equal(result.local_minima_indices, [])
        assert_allclose(result.moving_average, values)

    def test_endpoints_are_excluded_and_full_window_has_one_value(self):
        result = tasks.analyze_time_series(TimeSeriesInput([9, 1, 9], 3))
        assert_array_equal(result.local_maxima_indices, [])
        assert_array_equal(result.local_minima_indices, [1])
        assert_allclose(result.moving_average, [19 / 3])

    def test_short_series_have_no_extrema(self):
        for values, window, average in [([4], 1, [4]), ([1, 2], 2, [1.5])]:
            with self.subTest(values=values):
                result = tasks.analyze_time_series(TimeSeriesInput(values, window))
                assert_array_equal(result.local_maxima_indices, [])
                assert_array_equal(result.local_minima_indices, [])
                assert_allclose(result.moving_average, average)

    def test_invalid_series_and_windows_raise_value_error(self):
        for values, window in [([], 1), ([[1, 2]], 1), ([1, 2, 3], 0), ([1, 2, 3], 4)]:
            with self.subTest(values=values, window=window):
                with self.assertRaises(ValueError):
                    tasks.analyze_time_series(TimeSeriesInput(values, window))


class OneHotTests(unittest.TestCase):
    def test_inferred_classes_include_unused_labels(self):
        result = tasks.one_hot(OneHotInput([0, 2, 3, 0]))
        assert_array_equal(result, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

    def test_explicit_class_count_preserves_extra_columns(self):
        result = tasks.one_hot(OneHotInput(np.array([1, 1]), class_count=4))
        self.assertEqual(result.shape, (2, 4))
        assert_array_equal(result, [[0, 1, 0, 0], [0, 1, 0, 0]])

    def test_empty_labels_preserve_requested_class_count(self):
        for class_count, shape in [(None, (0, 0)), (0, (0, 0)), (3, (0, 3))]:
            with self.subTest(class_count=class_count):
                result = tasks.one_hot(OneHotInput([], class_count))
                self.assertEqual(result.shape, shape)

    def test_invalid_labels_raise_value_error(self):
        for labels in [[0, -1], [0, 1.5], [[0, 1]], [np.nan]]:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    tasks.one_hot(OneHotInput(labels))

    def test_invalid_class_count_raises_value_error(self):
        for labels, class_count in [([0], -1), ([0], 0), ([0, 2], 2), ([0], 1.5)]:
            with self.subTest(labels=labels, class_count=class_count):
                with self.assertRaises(ValueError):
                    tasks.one_hot(OneHotInput(labels, class_count))


if __name__ == "__main__":
    unittest.main()
