"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    """Сложить произведения соответствующих матриц и векторов."""
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Матрицы должны иметь форму (p, n, n)")
    if vectors.shape != (matrices.shape[0], matrices.shape[1], 1):
        raise ValueError("Векторы должны иметь форму (p, n, 1)")
    return (matrices @ vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    """Вернуть новую матрицу: единица только при строгом превышении порога."""
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    """Найти отсортированные уникальные значения отдельно в каждой строке."""
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    """Найти отсортированные уникальные значения отдельно в каждом столбце."""
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    """Создать нормальную случайную матрицу и вычислить статистики по осям."""
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if rows <= 0 or columns <= 0:
        raise ValueError("Размеры матрицы должны быть положительными")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    """Заполнить матрицу двумя значениями в шахматном порядке."""
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    """Нарисовать центральный прямоугольник, обрезав его по краям изображения."""
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if width < 0 or height < 0 or image_height <= 0 or image_width <= 0:
        raise ValueError("Размеры фигуры неотрицательны, размеры изображения положительны")
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    image[
        max(top, 0):min(top + height, image_height),
        max(left, 0):min(left + width, image_width),
    ] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    """Нарисовать закрашенный эллипс с центром в геометрическом центре кадра."""
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if min(semi_axis_x, semi_axis_y, image_height, image_width) <= 0:
        raise ValueError("Полуоси и размеры изображения должны быть положительными")
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    # Удвоенные координаты позволяют включить границу без ошибок округления.
    dx = 2 * x - (image_width - 1)
    dy = 2 * y - (image_height - 1)
    axis_x_squared = int(semi_axis_x) ** 2
    axis_y_squared = int(semi_axis_y) ** 2
    limit = 4 * axis_x_squared * axis_y_squared
    max_sum = (
        (int(image_width) - 1) ** 2 * axis_y_squared
        + (int(image_height) - 1) ** 2 * axis_x_squared
    )
    if max(max_sum, limit) > np.iinfo(np.int64).max:
        dx, dy = dx.astype(object), dy.astype(object)
    mask = dx ** 2 * axis_y_squared + dy ** 2 * axis_x_squared <= limit
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    """Вычислить статистики, строгие экстремумы и скользящее среднее ряда."""
    values, window = data.values, data.window
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Временной ряд должен быть непустым одномерным массивом")
    if not isinstance(window, (int, np.integer)) or not 1 <= window <= values.size:
        raise ValueError("Размер окна должен быть целым числом от 1 до длины ряда")
    middle = values[1:-1]
    maxima = (middle > values[:-2]) & (middle > values[2:])
    minima = (middle < values[:-2]) & (middle < values[2:])
    if np.issubdtype(values.dtype, np.integer):
        # Точные целые суммы предотвращают переполнение и потерю малых различий.
        integers = values.astype(object)
        size = values.size
        total = integers.sum()
        mean = float(total / size)
        variance = float((size * (integers ** 2).sum() - total ** 2) / size ** 2)
        cumulative = np.concatenate((np.array([0], dtype=object), np.cumsum(integers)))
        moving_average = np.asarray(
            (cumulative[window:] - cumulative[:-window]) / window, dtype=float
        )
    else:
        numeric_values = np.asarray(values, dtype=float)
        mean = float(numeric_values.mean())
        variance = float(numeric_values.var())
        moving_average = np.convolve(numeric_values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=float(np.sqrt(variance)),
        local_maxima_indices=np.flatnonzero(maxima) + 1,
        local_minima_indices=np.flatnonzero(minima) + 1,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    """Преобразовать неотрицательные целочисленные метки в one-hot матрицу."""
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Метки должны быть одномерным вектором")
    if labels.size and (
        not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
    ):
        raise ValueError("Метки должны быть неотрицательными целыми числами")
    required_classes = int(labels.max()) + 1 if labels.size else 0
    if class_count is None:
        class_count = required_classes
    if not isinstance(class_count, (int, np.integer)) or class_count < required_classes:
        raise ValueError("Количество классов должно покрывать все метки")
    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1
    return result
