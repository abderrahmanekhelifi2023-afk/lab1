# Решение лабораторной работы № 1

Условие сохранено в исходном `README.md`. Решения первой части находятся в
`python_basics.py`, второй — в `numpy_tasks.py`. Проверки примеров и граничных
случаев собраны в тестах. Код построения гистограмм вынесен в
`plot_histograms.py`.

## Запуск

Нужен Python 3.10 или новее. Откройте терминал в папке `lab1` и выполните:

```console
python -m pip install -r requirements.txt
python -m unittest discover -v
python plot_histograms.py
```

Последняя команда создаёт случайную матрицу 3 × 4 со средним 0, стандартным
отклонением 1 и `seed=42`. Гистограммы каждой строки и каждого столбца
сохраняются отдельно на двух общих фигурах:

- `plots/row_histograms.png` — три гистограммы строк;
- `plots/column_histograms.png` — четыре гистограммы столбцов.

Используется `matplotlib.pyplot.hist`; backend `Agg` позволяет создавать PNG
без графического интерфейса. Импорт модуля сам по себе не создаёт файлы.
Другую папку можно указать командой:

```console
python plot_histograms.py --output-dir my_plots
```

## Вызов функций

Входные параметры передаются одним объектом `dataclass` из
`grader_contracts`. Поля такого объекта именованы; `frozen=True` запрещает
переприсваивать его поля. Сами контракты и имена функций сохранены.

```python
from grader_contracts.python_basics import PositiveIntegerInput, TextInput
from grader_contracts.numpy_tasks import RandomMatrixInput
from python_basics import count_vowels, prime_factorization
from numpy_tasks import matrix_statistics

print(count_vowels(TextInput(value="Hello")))  # 2
print(prime_factorization(PositiveIntegerInput(value=86240)))
# (2**5)(5)(7**2)(11)

result = matrix_statistics(RandomMatrixInput(rows=3, columns=4, seed=42))
print(result.matrix)
print(result.row_means)
print(result.column_variances)
```

Функции статистики возвращают объекты `MatrixStatistics` и
`TimeSeriesStatistics`: результаты доступны через именованные поля.

## Принятые соглашения

- Разложение числа 1 — пустая строка: у единицы нет простых множителей.
- Случайные числа создаются локальным генератором
  `numpy.random.default_rng(seed)`. Одинаковый `seed` воспроизводит матрицу;
  глобальное состояние случайных чисел не меняется.
- Дисперсия и стандартное отклонение вычисляются с `ddof=0`, то есть с
  делением на число элементов.
- Центр эллипса в координатах пикселей:
  `x0 = (image_width - 1) / 2`, `y0 = (image_height - 1) / 2`.
- Начальные координаты прямоугольника вычисляются как
  `(размер_холста - размер_фигуры) // 2` по каждой оси. Фигура, выходящая за
  границы изображения, обрезается по границам холста.
- Для пустого набора меток `one_hot` возвращает массив формы `(0, 0)`,
  если число классов не задано, и `(0, class_count)` при явно заданном
  числе классов.

## Проверка решения

Все 50 собственных тестов проходят: 21 тест первой части и 29 тестов второй.
Проверены примеры из задания, граничные случаи и формы возвращаемых массивов.
Скрипт гистограмм выполнен, обе PNG просмотрены. Исходные `README.md` и
`grader_contracts` не изменены.

## Сдача

Разместите решение в своём GitHub-репозитории вместе с `grader_contracts`,
тестами и графиками. Ссылку на свой репозиторий отправьте в
[форму сдачи](https://forms.gle/rLDF8sE3W4LuWFKT8).
Локальные папки `.dependencies`, `.mplconfig`, `.venv` и `__pycache__`
публиковать не нужно. Дедлайн из объявления — 19 октября, 23:59.
