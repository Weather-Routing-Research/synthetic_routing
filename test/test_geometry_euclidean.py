from typing import Tuple

import numpy as np
import pytest

from synthrouting.geometry import DEG2RAD, Euclidean


@pytest.mark.parametrize(
    ("p0", "p1", "d"),
    [
        ((0, 0), (0, 10), 10),
        ((0, 0), (10, 0), 10),
        ((-10, -10), (10, 10), 20 * np.sqrt(2)),
        ((180, 0), (-180, 0), 360),
    ],
)
def test_dist_p0_to_p1(p0: Tuple[float], p1: Tuple[float], d: float):
    dist = Euclidean().dist_p0_to_p1(p0, p1)
    np.testing.assert_allclose(dist, d, rtol=0.1)


@pytest.mark.parametrize(
    ("p0", "p1", "a"),
    [
        ((0, 0), (1, 0), 0),
        ((0, 0), (np.sqrt(3), 1), 30),
        ((0, 0), (1, 1), 45),
        ((0, 0), (1, np.sqrt(3)), 60),
        ((0, 0), (0, 1), 90),
        ((0, 0), (-1, 0), 180),
    ],
)
def test_angle_p0_to_p1(p0: Tuple[float], p1: Tuple[float], a: float):
    ang = Euclidean().angle_p0_to_p1(p0, p1)
    np.testing.assert_allclose(ang, a * DEG2RAD, rtol=0.1)


def test_dist_between_coords():
    lon = np.array([-1, -1, 0, 0, 2, 1])
    lat = np.array([-1, 0, 0, 0, 0, 2])
    d = Euclidean().dist_between_coords(lon, lat)
    expected = np.array([1, 1, 0, 2, 2.24])
    np.testing.assert_allclose(d, expected, rtol=0.1)


def test_angle_between_coords():
    lon = np.array([0, 1, 0, 0, 0])
    lat = np.array([0, 0, 0, 1, 0])
    angle = Euclidean().ang_between_coords(lon, lat)
    expected = np.array([0, 180, 90, -90]) * DEG2RAD
    np.testing.assert_allclose(angle, expected, rtol=0.001)
