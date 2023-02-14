from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from hybrid_routing.vectorfields.base import Vectorfield


class Sink(Vectorfield):
    """Sink vector field, implements Vectorfield class.
    Sink coordinates defined by setting u, v to 0 and solve for x, y.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = -1 / 25 * (x - 8), v(x, y) = -1 / 25 * (y - 8)
    with:
        du/dx = -1 / 25,    du/dy = 0
        dv/dx = 0      ,    dv/dy = -1/25
    """

    @partial(jit, static_argnums=(0,))
    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return (jnp.tile(0.0, x.shape), jnp.tile(-1 / 25, y.shape))

    @partial(jit, static_argnums=(0,))
    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return (jnp.tile(-1 / 25, x.shape), jnp.tile(0.0, y.shape))

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([-(x - 8) / 25, -(y - 8) / 25])
