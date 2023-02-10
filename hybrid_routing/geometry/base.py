from abc import abstractmethod
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit


class Geometry:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def dist_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Compute the distance between two points."""
        pass

    @abstractmethod
    def angle_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Compute the angle between two points in radians"""
        pass

    @partial(jit, static_argnums=(0,))
    def dist_between_coords(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Return euclidean distance between each set of points"""
        p = jnp.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.dist_p0_to_p1(p0, p1)

    @partial(jit, static_argnums=(0,))
    def ang_between_coords(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Return angle (in radians) between each set of points"""
        p = jnp.stack([x, y])
        p0 = p[..., :-1]
        p1 = p[..., 1:]
        return self.angle_p0_to_p1(p0, p1)

    @partial(jit, static_argnums=(0,))
    def components_to_ang_mod(
        self, x: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        """Gets vector components, returns angle (rad) and module"""
        a = jnp.arctan2(x, y)
        m = jnp.sqrt(x**2 + y**2)
        return a, m

    @partial(jit, static_argnums=(0,))
    def ang_mod_to_components(
        self, a: jnp.ndarray, m: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        """Gets angle (rad) and module, returns vector components"""
        x = m * jnp.sin(a)
        y = m * jnp.cos(a)
        return x, y
