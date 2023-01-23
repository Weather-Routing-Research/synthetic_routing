from abc import abstractmethod
from typing import Tuple

import numpy as np


class Geometry:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def dist_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Compute the distance between two points."""
        pass

    @abstractmethod
    def angle_p0_to_p1(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Compute the angle between two points in radians"""
        pass

    def dist_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return euclidean distance between each set of points"""
        p = np.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.dist_p0_to_p1(p0, p1)

    def ang_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return angle (in radians) between each set of points"""
        p = np.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.angle_p0_to_p1(p0, p1)

    def components_to_ang_mod(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        """Gets vector components, returns angle (rad) and module"""
        a = np.arctan2(x, y)
        m = np.sqrt(x**2 + y**2)
        return a, m

    def ang_mod_to_components(self, a: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray]:
        """Gets angle (rad) and module, returns vector components"""
        x = m * np.sin(a)
        y = m * np.cos(a)
        return x, y
