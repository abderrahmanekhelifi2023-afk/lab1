"""Решения задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    """Подсчитывает латинские гласные без учёта регистра."""
    text = data.value
    return sum(character in "aeiou" for character in text.lower())


def has_unique_characters(data: TextInput) -> bool:
    """Проверяет, что все символы строки различны."""
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    """Возвращает количество единиц в двоичной записи числа."""
    number = data.value
    if number <= 0:
        raise ValueError("Число должно быть положительным")
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    """Считает шаги перемножения цифр до однозначного числа."""
    number = data.value
    if number <= 0:
        raise ValueError("Число должно быть положительным")

    steps = 0
    while number >= 10:
        product = 1
        while number:
            number, digit = divmod(number, 10)
            product *= digit
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    """Находит среднюю квадратичную ошибку непустых векторов."""
    predicted, expected = data.predicted, data.expected
    if len(predicted) == 0 or len(predicted) != len(expected):
        raise ValueError("Векторы должны быть непустыми и одинаковой длины")
    return sum((x - y) ** 2 for x, y in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    """Записывает простые множители по возрастанию; для 1 — пустая строка."""
    number = data.value
    if number <= 0:
        raise ValueError("Число должно быть положительным")

    factors = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            factors.append(f"({divisor}**{power})" if power > 1 else f"({divisor})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    """Находит высоту пирамиды точным двоичным поиском суммы квадратов."""
    cube_count = data.value
    if cube_count <= 0:
        raise ValueError("Количество кубиков должно быть положительным")

    lower, upper = 1, cube_count
    while lower <= upper:
        height = (lower + upper) // 2
        required = height * (height + 1) * (2 * height + 1) // 6
        if required == cube_count:
            return height
        if required < cube_count:
            lower = height + 1
        else:
            upper = height - 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    """Сравнивает суммы крайних цифр, исключая одну или две средние."""
    number = data.value
    if number <= 0:
        raise ValueError("Число должно быть положительным")

    digits = str(number)
    side_length = (len(digits) - 1) // 2
    left = digits[:side_length]
    right = digits[len(digits) - side_length:]
    return sum(map(int, left)) == sum(map(int, right))
