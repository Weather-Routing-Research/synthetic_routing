from functools import partial
from typing import Iterable

import jax.numpy as jnp
from jax import jit

from synthrouting.vectorfields.base import Vectorfield


class HillBowl(Vectorfield):
    """Implements Vectorfield class.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = 1, v(x, y) = sin(x^2 + y^2)
    with:
        du/dx = 0,      du/dy = 0
        dv/dx = 2 * x * cos(x^2 + y^2),  dv/dy = 2 * y * cos(x^2 + y^2)
    """

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([jnp.ones(x.shape), jnp.sin(x**2 + y**2)])

    @partial(jit, static_argnums=(0,))
    def _ode_zermelo_euclidean(
        self,
        p: Iterable[float],
        t: Iterable[float],
        vel: jnp.float16 = jnp.float16(0.1),
    ) -> Iterable[float]:
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]
        # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
        dthetadt = 2 * x * jnp.cos(x**2 + y**2) * jnp.sin(
            theta
        ) ** 2 + -2 * jnp.sin(theta) * jnp.cos(theta) * y * jnp.cos(x**2 + y**2)

        return [dxdt, dydt, dthetadt]
