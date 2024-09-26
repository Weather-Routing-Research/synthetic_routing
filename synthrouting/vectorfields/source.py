from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from synthrouting.vectorfields.base import Vectorfield


class Source(Vectorfield):
    @partial(jit, static_argnums=(0,))
    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return (jnp.tile(0.0, x.shape), jnp.tile(1 / 25, y.shape))

    @partial(jit, static_argnums=(0,))
    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return (jnp.tile(1 / 25, x.shape), jnp.tile(0.0, y.shape))

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([(x - 5) / 25, (y - 5) / 25])
