import numpy as np

from hybrid_routing.geometry.base import Geometry


class Euclidean(Geometry):
    def __str__(self) -> str:
        return "Euclidean"

    def __repr__(self) -> str:
        return "Euclidean"

    def dist_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Compute the distance between two points, where p=(x, y)"""
        return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def angle_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Compute the angle between two points in radians, where p=(x, y)"""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return np.arctan2(dy, dx)
