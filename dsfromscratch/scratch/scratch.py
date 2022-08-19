import math
from typing import List, Tuple, Callable
from collections import Counter

Vector = List[float]

height_weight_age = [70, 170, 40]

grades = [95, 80, 75, 62]


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    assert vectors

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors)

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def scalar_mulitply(c: float, v: Vector) -> Vector:
    """Multiples every element by c"""
    return [c * v_i for v_i in v]


assert scalar_mulitply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_mulitply(1 / n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 + w_1 + ... + v_n * w_n, ie Dot Product

    Projection of one vector onto another
    """
    assert len(v) == len(w), "vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14


def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))


assert magnitude([3, 4]) == 5


def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> float:
    """Computes the distances between v and w"""
    return magnitude(subtract(v, w))


Matrix = List[List[float]]


def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0
    return num_rows, num_columns


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j] for A_i in A]


def make_matrix(
    num_rows: int,
    num_cols: int,
    entry_fn: Callable[[int, int], float],
) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)=th entry is entry_fn(i,j)
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    """Returns the nx n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(5) == [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    """If len(xs) is even , the the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


def quantile(xs: List[float], p: float) -> float:
    """Rerturns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(x: List[float]) -> List[float]:
    """Returns a list, since there maybe more than one mode"""
    counts = Counter(x)

    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


assert mode([1, 2, 2, 3, 3, 4, 5]) == [2, 3]


def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so ther result has a mean of 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    """The Standard deviation is the square root of variance"""
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75%-ile and the 25%-ile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
    """Measure how much xs and ys var in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0
