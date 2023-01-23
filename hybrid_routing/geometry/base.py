from abc import abstractmethod
from typing import Tuple

import numpy as np


class Geometry:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def dist_p0_to_p1(self, p0: Tuple[float], p1: Tuple[float]) -> float:
        """Compute the distance between two points."""
        pass

    @abstractmethod
    def angle_p0_to_p1(self, p0: Tuple[float], p1: Tuple[float]) -> float:
        """Compute the angle between two points in radians"""
        pass

    @abstractmethod
    def dist_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return euclidean distance between each set of points"""
        pass

    @abstractmethod
    def ang_between_coords(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return angle (in radians) between each set of points"""
        pass

    @abstractmethod
    def components_to_ang_mod(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        """Gets vector components, returns angle (rad) and module"""
        pass

    @abstractmethod
    def ang_mod_to_components(self, a: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray]:
        """Gets angle (rad) and module, returns vector components"""
        pass
