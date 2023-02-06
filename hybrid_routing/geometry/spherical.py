import numpy as np

from hybrid_routing.geometry.base import Geometry

RADIUS = 6367449  # meters
RAD2M = RADIUS  # Radians to meters conversion
DEG2RAD = np.pi / 180


def lonlatunitvector(p: np.ndarray) -> np.ndarray:
    lon, lat = p[0], p[1]
    return np.stack([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])


class Spherical(Geometry):
    def __str__(self) -> str:
        return "Spherical"

    def __repr__(self) -> str:
        return "Spherical"

    def dist_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Compute the distance between two points, defined in radians,
        where p=(lon, lat). Returns meters."""
        return RADIUS * np.arccos(
            np.clip(
                (lonlatunitvector(p0) * lonlatunitvector(p1)).sum(axis=0),
                a_min=-1,
                a_max=1,
            )
        )

    def angle_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Return angle (in radians) between two points, w.r.t. X-axis,
        where p=(lon, lat). Returns radians."""
        a1, b1, c1 = lonlatunitvector(p0)
        a2, b2, c2 = lonlatunitvector(p1)
        gvec = np.array(
            [-a2 * b1 + a1 * b2, -(a1 * a2 + b1 * b2) * c1 + (a1**2 + b1**2) * c2]
        )
        gd = self.dist_p0_to_p1(p0, p1)
        vector = gvec * gd / np.clip(np.abs(gvec), a_min=1e-20)
        return np.arctan2(vector[1], vector[0])
