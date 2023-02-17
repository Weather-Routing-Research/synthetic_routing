from functools import partial

import jax.numpy as jnp
from jax import jit

from hybrid_routing.geometry.base import Geometry

RADIUS = 6367449  # meters
RAD2M = RADIUS  # Radians to meters conversion
DEG2RAD = jnp.pi / 180


@jit
def lonlatunitvector(p: jnp.ndarray) -> jnp.ndarray:
    lon, lat = p[0], p[1]
    return jnp.stack(
        [jnp.cos(lon) * jnp.cos(lat), jnp.sin(lon) * jnp.cos(lat), jnp.sin(lat)]
    )


class Spherical(Geometry):
    def __str__(self) -> str:
        return "Spherical"

    def __repr__(self) -> str:
        return "Spherical"

    @partial(jit, static_argnums=(0,))
    def dist_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Compute the distance between two points, defined in radians,
        where p=(lon, lat). Returns meters."""
        return RADIUS * jnp.arccos(
            jnp.clip(
                (lonlatunitvector(p0) * lonlatunitvector(p1)).sum(axis=0),
                a_min=-1,
                a_max=1,
            )
        )

    @partial(jit, static_argnums=(0,))
    def angle_p0_to_p1(self, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
        """Return angle (in radians) between two points, w.r.t. X-axis,
        where p=(lon, lat). Returns radians."""
        a1, b1, c1 = lonlatunitvector(p0)
        a2, b2, c2 = lonlatunitvector(p1)
        gvec = jnp.array(
            [-a2 * b1 + a1 * b2, -(a1 * a2 + b1 * b2) * c1 + (a1**2 + b1**2) * c2]
        )
        gvec_norm = gvec / jnp.clip(jnp.abs(gvec), a_min=1e-20, a_max=None)
        return jnp.arctan2(gvec_norm[1], gvec_norm[0])
