from typing import Tuple

import numpy as np

from hybrid_routing.geometry.base import Geometry


class Euclidean(Geometry):

    def __str__(self) -> str:
        return "Euclidean Geometry"

    def __repr__(self) -> str:
        return "Euclidean Geometry"

    def dist_p0_to_p1(self,p0: Tuple[float], p1: Tuple[float]) -> float:
        """Compute the distance between two points."""
        return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def angle_p0_to_p1(self,p0: Tuple[float], p1: Tuple[float]) -> float:
        """Compute the angle between two points in radians"""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return np.arctan2(dy, dx)

    def dist_between_coords(self,x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return euclidean distance between each set of points"""
        return np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)


    def ang_between_coords(self,x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return angle (in radians) between each set of points"""
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        return np.arctan2(dy, dx)

    
    def components_to_ang_mod(self,x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        """Gets vector components, returns angle (rad) and module"""
        a = np.arctan2(x, y)
        m = np.sqrt(x**2 + y**2)
        return a, m

    
    def ang_mod_to_components(self,a: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray]:
        """Gets angle (rad) and module, returns vector components"""
        x = m * np.sin(a)
        y = m * np.cos(a)
        return x, y
