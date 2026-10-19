"""Гистограммы строк и столбцов случайной матрицы из задания 4."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def _save_histograms(
    series: np.ndarray, title: str, item_name: str, destination: Path
) -> None:
    """Размещает отдельную гистограмму каждого ряда на общей фигуре."""
    count = len(series)
    columns = min(2, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False
    )
    figure.suptitle(title, fontsize=14)
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.set_visible(False)
            continue
        plt.sca(axis)
        plt.hist(series[index], bins="auto", color="#3978b8", edgecolor="white")
        axis.set_title(f"{item_name} {index + 1}")
        axis.set_xlabel("Значение")
        axis.set_ylabel("Частота")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.yaxis.get_major_locator().set_params(integer=True)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_histograms(
    data: RandomMatrixInput, output_dir: str | Path = "plots"
) -> tuple[Path, Path]:
    """Создаёт одну матрицу и сохраняет гистограммы её строк и столбцов."""
    statistics = matrix_statistics(data)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    row_path = destination / "row_histograms.png"
    column_path = destination / "column_histograms.png"
    _save_histograms(
        statistics.matrix, "Распределение значений по строкам", "Строка", row_path
    )
    _save_histograms(
        statistics.matrix.T,
        "Распределение значений по столбцам",
        "Столбец",
        column_path,
    )
    return row_path, column_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots"), help="Папка для PNG"
    )
    arguments = parser.parse_args()
    data = RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=42)
    for path in save_histograms(data, arguments.output_dir):
        print(f"Сохранено: {path.resolve()}")


if __name__ == "__main__":
    main()
