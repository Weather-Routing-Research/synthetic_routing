from typing import Tuple

import numpy as np

from hybrid_routing.geometry.base import Geometry

RADIUS = 6367449  # meters
RAD2M = RADIUS / (2 * np.pi)  # Radians to meters conversion
DEG2RAD = np.pi / 180


def lonlatunitvector(p: Tuple[float]) -> np.ndarray:
    lon, lat = p[0], p[1]
    return np.stack([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])


class Spherical(Geometry):
    def __str__(self) -> str:
        return "Spherical Geometry"

    def __repr__(self) -> str:
        return "Spherical Geometry"

    def dist_p0_to_p1(self, p0: Tuple[float], p1: Tuple[float]) -> float:
        """Compute the distance between two points, defined in radians. Returns meters."""
        return RADIUS * np.arccos(
            (lonlatunitvector(p0) * lonlatunitvector(p1)).sum(axis=0)
        )

    def angle_p0_to_p1(self, p0: Tuple[float], p1: Tuple[float]) -> float:
        """Return angle (in radians) between two points, w.r.t. X-axis. Returns radians."""
        a1, b1, c1 = lonlatunitvector(p0)
        a2, b2, c2 = lonlatunitvector(p1)
        gvec = np.array(
            [-a2 * b1 + a1 * b2, -(a1 * a2 + b1 * b2) * c1 + (a1**2 + b1**2) * c2]
        )
        gd = self.dist_p0_to_p1(p0, p1)
        vector = np.nan_to_num(gvec * gd / np.sqrt(gvec**2), 0)
        return np.arctan2(vector[1], vector[0])

    def dist_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return euclidean distance (meters) between each set of points (radians)"""
        p = np.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.dist_p0_to_p1(p0, p1)

    def ang_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return angle (in radians) between each set of points (radians)"""
        p = np.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.angle_p0_to_p1(p0, p1)

    def components_to_ang_mod(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        """Gets vector components, returns angle (rad) and module"""
        pass

    def ang_mod_to_components(self, a: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray]:
        """Gets angle (rad) and module, returns vector components"""
        pass
