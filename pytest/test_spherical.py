from typing import Tuple

import numpy as np

import pytest
from hybrid_routing.geometry.spherical import DEG2RAD, Spherical


@pytest.mark.parametrize(
    ("p0", "p1", "d"),
    [
        ((0, 0), (0, 10), 1113000),
        ((0, 0), (10, 0), 1113000),
        ((-10, -10), (10, 10), 3140000),
        ((180, 0), (-180, 0), 0),
    ],
)
def test_dist_p0_to_p1(p0: Tuple[float], p1: Tuple[float], d: float):
    # To radians
    p0 = [x * DEG2RAD for x in p0]
    p1 = [x * DEG2RAD for x in p1]
    dist = Spherical().dist_p0_to_p1(p0, p1)
    np.testing.assert_allclose(dist, d, rtol=0.1)


@pytest.mark.parametrize(
    ("p0", "p1", "a"),
    [
        ((0, 0), (0, 10), np.pi / 2),
        ((0, 0), (10, 0), 0),
        ((-10, -10), (10, 10), np.pi / 4),
        ((180, 0), (-180, 0), np.pi),
    ],
)
def test_angle_p0_to_p1(p0: Tuple[float], p1: Tuple[float], a: float):
    # To radians
    p0 = [x * DEG2RAD for x in p0]
    p1 = [x * DEG2RAD for x in p1]
    ang = Spherical().angle_p0_to_p1(p0, p1)
    np.testing.assert_allclose(ang, a, rtol=0.1)


def test_dist_between_coords():
    lon = np.array([-1.1, -1, 0, 0, 2, 1]) * DEG2RAD
    lat = np.array([-1, 0, 0, 0, 0, 2]) * DEG2RAD
    d = Spherical().dist_between_coords(lon, lat)
    expected = np.array([120, 111, 0, 222, 249]) * 1000
    np.testing.assert_allclose(d, expected, rtol=0.1)


def test_angle_between_coords():
    lon = np.array([0, 1, 0, 0, 0]) * DEG2RAD
    lat = np.array([0, 0, 0, 1, 0]) * DEG2RAD
    angle = Spherical().ang_between_coords(lon, lat)
    expected = np.array([0, np.pi, np.pi / 2, -np.pi / 2])
    np.testing.assert_allclose(angle, expected, rtol=0.001)
