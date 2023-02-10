from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit

from hybrid_routing.vectorfields.base import Vectorfield


class Source(Vectorfield):
    @partial(jit, static_argnums=(0,))
    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return jnp.asarray([0]), jnp.asarray([1 / 75])

    @partial(jit, static_argnums=(0,))
    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        return jnp.asarray([1 / 75]), jnp.asarray([0])

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([(x - 5) / 25, (y - 5) / 25])
