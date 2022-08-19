from typing import List
from collections import Counter
from scratch import sum_of_squares, dot
import math


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
