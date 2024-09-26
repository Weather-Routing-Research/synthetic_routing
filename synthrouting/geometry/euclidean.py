from functools import partial

import jax.numpy as jnp
from jax import jit

from synthrouting.geometry.base import Geometry


class Euclidean(Geometry):
    def __str__(self) -> str:
        return "Euclidean"

    def __repr__(self) -> str:
        return "Euclidean"

    @partial(jit, static_argnums=(0,))
    def dist_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Compute the distance between two points, where p=(x, y)"""
        return jnp.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    @partial(jit, static_argnums=(0,))
    def angle_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Compute the angle between two points in radians, where p=(x, y)"""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return jnp.arctan2(dy, dx)
