from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from hybrid_routing.vectorfields.base import Vectorfield


class Circular(Vectorfield):
    """Circular vector field, implements Vectorfield class.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = 0.05 * (y + 1), v(x, y) = 0.05 * (-x - 3)
    with:
        du/dx = 0,      du/dy = 0.05
        dv/dx = -0.05,  dv/dy = 0
    """

    @partial(jit, static_argnums=(0,))
    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return jnp.asarray([-0.05]), jnp.asarray([0])

    @partial(jit, static_argnums=(0,))
    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return jnp.asarray([0]), jnp.asarray([0.05])

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([0.05 * (y + 1), 0.05 * (-x - 3)])
