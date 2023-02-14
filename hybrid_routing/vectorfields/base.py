from abc import ABC, abstractmethod
from functools import partial
from typing import Iterable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd, jacrev, jit

from hybrid_routing.geometry import Euclidean, Geometry, Spherical
from hybrid_routing.geometry.spherical import RAD2M


class Vectorfield(ABC):
    """The parent class of vector fields.

    Methods
    ----------
    get_current : _type_
        pass upon initialization, returns the current in tuples `(u, v)` given
        the position of the boat `(x, y)`
    """

    geometry: Geometry
    rad2m = jnp.float32(RAD2M)  # Radians to meters conversion
    is_discrete: bool = False

    def __init__(self, spherical: bool = False):
        self._dv = jit(jacrev(self.get_current, argnums=1))
        self._du = jit(jacfwd(self.get_current, argnums=0))
        self.spherical = spherical
        if spherical:
            self.ode_zermelo = jit(self._ode_zermelo_spherical)
            self.geometry = Spherical()
        else:
            self.ode_zermelo = jit(self._ode_zermelo_euclidean)
            self.geometry = Euclidean()

    @abstractmethod
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        pass

    @partial(jit, static_argnums=(0,))
    def get_current_rad(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Takes the current values (u,v) at a given point (x,y) on the grid.
        Returns radians per second.

        Parameters
        ----------
        x : jnp.array
            x-coordinate of the ship
        y : jnp.array
            y-coordinate of the ship

        Returns
        -------
        jnp.array
            The current's velocity in x and y direction (u, v)
        """
        u, v = self.get_current(x, y)
        # Meters to radians
        # Velocity component across longitude is affected by latitude
        u = u / self.rad2m / jnp.cos(y)
        v = v / self.rad2m
        return u, v

    """
    Takes the Jacobian (a 2x2 matrix) of the background vectorfield (W) using JAX
    package by Google LLC if it is not specified in the children classes.
    
    Jax docs:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html#jax.jacfwd 
    https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html#jax.jacrev.
    
    `W: R^2 -> R^2, W: (x,y) -> (u,v)`
    Each function below returns a specific linearized partial derivative with respect
    to the variable.

    Parameters
    ----------
    x : x-coordinate of the boat's current location.
    y : y-coordinate of the boat's current location.

    Returns
    -------
    float
        The value of dv/dx, dv/dy, du/dx, du/dy, with respect to the call.
    """

    @partial(jit, static_argnums=(0,))
    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        out = jnp.asarray([self._du(x, y) for x, y in zip(x.ravel(), y.ravel())])
        return out[:, 0].reshape(x.shape), -out[:, 1].reshape(x.shape)

    @partial(jit, static_argnums=(0,))
    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        out = jnp.asarray([self._dv(x, y) for x, y in zip(x.ravel(), y.ravel())])
        return -out[:, 0].reshape(x.shape), out[:, 1].reshape(x.shape)

    @partial(jit, static_argnums=(0,))
    def _ode_zermelo_euclidean(
        self,
        p: Iterable[float],
        t: Iterable[float],
        vel: jnp.float16 = jnp.float16(0.1),
    ) -> Iterable[float]:
        """System of ODE set up for scipy initial value problem method to solve in
        optimize.py

        Parameters
        ----------
        p : Iterable[float]
            Initial position: `(x, y, theta)`. The pair `(x,y)` is the position of the
            boat and `theta` is heading (in radians) of the boat
            (with respect to the x-axis).
        t : Iterable[float]
            Array of time steps, evenly spaced inverval from t_start to t_end, of
            length `n`.
        vel : jnp.float16, optional
            Speed of the boat, by default jnp.float16(0.1)

        Returns
        -------
        Iterable[float]
            A list of coordinates on the locally optimal path of length `n`, same
            format as `p`: `(x, y, theta)`.
            `dxdt`, `dydt` are in m / s, `dthetadt` is in rad / s
        """
        x, y, theta = p
        u, v = self.get_current(x, y)
        st, ct = jnp.sin(theta), jnp.cos(theta)
        dxdt = vel * ct + u
        dydt = vel * st + v
        dvdx, dvdy = self.dv(x, y)
        dudx, dudy = self.du(x, y)
        dthetadt = dvdx * st**2 + st * ct * (dudx - dvdy) - dudy * ct**2

        return [dxdt, dydt, dthetadt]

    @partial(jit, static_argnums=(0,))
    def _ode_zermelo_spherical(
        self,
        p: Iterable[float],
        t: Iterable[float],
        vel: jnp.float16 = jnp.float16(0.1),
    ) -> Iterable[float]:
        """System of ODE set up for scipy initial value problem method to solve in
        optimize.py

        Parameters
        ----------
        p : Iterable[float]
            Initial position: `(x, y, theta)`. The pair `(x,y)` is the position of the
            boat and `theta` is heading (in radians) of the boat
            (with respect to the x-axis)
        t : Iterable[float]
            Array of time steps, evenly spaced inverval from t_start to t_end, of
            length `n`.
        vel : jnp.float16, optional
            Speed of the boat, by default jnp.float16(0.1)

        Returns
        -------
        Iterable[float]
            A list of coordinates on the locally optimal path of length `n`, same
            format as `p`: `(x, y, theta)`.
            `dxdt`, `dydt`, `dthetadt` are in rad / s
        """
        x, y, theta = p
        u, v = self.get_current(x, y)  # m / s
        st, ct = jnp.sin(theta), jnp.cos(theta)  # Assuming theta in radians
        cy = jnp.cos(y)  # Assuming y is in radians
        dxdt = (vel * ct + u) / cy / self.rad2m  # rad / s
        dydt = (vel * st + v) / self.rad2m  # rad / s
        dvdx, dvdy = self.dv(x, y)  # m / (rad * s)
        dudx, dudy = self.du(x, y)  # m / (rad * s)

        dthetadt = (
            dvdx * (st**2) / cy
            + st * ct * (dudx - dvdy * cy) / cy
            - dudy * ct**2
            - ct * jnp.tan(y) * (vel + u * ct + v * st)
        ) / self.rad2m  # rad / s

        return [dxdt, dydt, dthetadt]

    def discretize(
        self,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
        step: float = 1,
    ) -> "VectorfieldDiscrete":
        """Discretizes the vectorfield

        Parameters
        ----------
        x_min : float, optional
            Minimum x-value of the grid, by default 0
        x_max : float, optional
            Maximum x-value of the grid, by default 10
        y_min : float, optional
            Minimum y-value of the grid, by default 0
        y_max : float, optional
            Maximum y_value of the grid, by default 10
        step : float, optional
            "Fineness" of the grid, by default 1

        Returns
        -------
        VectorfieldDiscrete
            Discretized vectorfield
        """
        if self.is_discrete:
            return self
        else:
            return VectorfieldDiscrete.from_vectorfield(
                self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, step=step
            )

    @partial(jit, static_argnums=(0,))
    def is_land(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Just a placeholder function. Indicates that no point has land."""
        return jnp.tile(False, x.shape)

    def plot(
        self,
        x_min: float = -4,
        x_max: float = 4,
        y_min: float = -4,
        y_max: float = 4,
        step: float = 1,
        do_color: bool = False,
        **kwargs
    ):
        """Plots the vector field

        Parameters
        ----------
        x_min : float, optional
            Left limit of X axes, by default 0
        x_max : float, optional
            Right limit of X axes, by default 125
        y_min : float, optional
            Bottom limit of Y axes, by default 0
        y_max : float, optional
            Up limit of Y axes, by default 125
        step : float, optional
            Distance between points to plot, by default .5
        do_color : bool, optional
            Plot a background color indicating the strength of the current
        """
        # Quiver
        x, y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        u, v = self.get_current(x, y)
        # Heatmap
        if do_color:
            # Matshow color is finer than quiver
            x, y = np.meshgrid(
                np.arange(x_min, x_max, step / 5), np.arange(y_min, y_max, step / 5)
            )
            u, v = self.get_current(x, y)
            # Velocity module
            m = (u**2 + v**2) ** (1 / 2)
            plt.matshow(
                m,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                alpha=0.6,
            )
        plt.quiver(x, y, u, v, **kwargs)


class VectorfieldDiscrete(Vectorfield):
    is_discrete: True

    @classmethod
    def from_vectorfield(
        cls,
        vectorfield: Vectorfield,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
        step: float = 1,
    ):
        # Copy all atributes of the original vectorfield into this one
        cls.spherical = vectorfield.spherical
        cls.ode_zermelo = vectorfield.ode_zermelo
        cls.geometry = vectorfield.geometry
        # Compute some other attributes
        cls.arr_x = jnp.arange(x_min, x_max, step)
        cls.arr_y = jnp.arange(y_min, y_max, step)
        mat_x, mat_y = jnp.meshgrid(cls.arr_x, cls.arr_y)
        u, v = vectorfield.get_current(mat_x, mat_y)
        cls.u, cls.v = u, v
        cls.land = jnp.zeros(u.shape)
        # Compute the average step between X and Y coordinates
        # We are assuming this step is constant!
        cls._dx = jnp.abs(jnp.mean(jnp.diff(cls.arr_x)))
        cls._dy = jnp.abs(jnp.mean(jnp.diff(cls.arr_y)))
        # Define methods to get closest indexes
        cls.closest_idx = jnp.vectorize(lambda x: jnp.argmin(jnp.abs(cls.arr_x - x)))
        cls.closest_idy = jnp.vectorize(lambda y: jnp.argmin(jnp.abs(cls.arr_y - y)))
        cls.is_discrete = True
        return cls()

    @partial(jit, static_argnums=(0,))
    def _weight_coordinates(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Weights the influence of each coordinate of the grid at point (x,y)
        on the grid.

        Parameters
        ----------
        x : jnp.array
            x-coordinate of the ship
        y : jnp.array
            y-coordinate of the ship

        Returns
        -------
        jnp.array
            The weight of each coordinate w.r.t given point
        """
        # Reshape arrays
        # P = shape of the point array `x`, `y` (may be multidimensional)
        # X = length of the X-grid, `self.arr_x`
        # Y = length of the Y-grid, `self.arr_x`
        arr_x = jnp.tile(self.arr_x, x.shape + (1,))  # (P, X)
        arr_y = jnp.tile(self.arr_y, y.shape + (1,))  # (P, Y)
        x = jnp.reshape(x, x.shape + (1,))  # (P, 1)
        y = jnp.reshape(y, y.shape + (1,))  # (P, 1)

        # Compute distance from all points to the X grid points
        dx = jnp.abs(arr_x - x) / self._dx  # (P, X)
        # Compute distance from all points to the Y grid points
        dy = jnp.abs(arr_y - y) / self._dy  # (P, Y)

        # Assign a weight relative to its proximity
        # Grid points more that one point away will have zero weight
        wx = 1 - jnp.where(dx < 1, dx, 1)  # (P, X)
        wy = 1 - jnp.where(dy < 1, dy, 1)  # (P, Y)

        # Turn arrays of weights into mesh grids
        wx = jnp.reshape(wx, wx.shape + (1,))  # (P, X, 1)
        wy = jnp.reshape(wy, wy.shape[:-1] + (1, wy.shape[-1]))  # (P, 1, Y)
        # Multiply both matrices to get the final matrix of weights
        w = wx * wy  # (P, X, Y)
        return w

    @partial(jit, static_argnums=(0,))
    def get_current(self, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray]:
        """Takes the current values (u,v) at a given point (x,y) on the grid.
        Returns meter per second.

        Parameters
        ----------
        x : jnp.array
            x-coordinate of the ship
        y : jnp.array
            y-coordinate of the ship

        Returns
        -------
        Tuple[jnp.array]
            The current's velocity in x and y direction (u, v)
        """
        w = self._weight_coordinates(x, y)

        # Use the weights to compute the velocity component
        # relative to those points
        u = (self.u.T * w).sum(axis=(-2, -1))  # (P, )
        v = (self.v.T * w).sum(axis=(-2, -1))  # (P, )

        return u, v

    def plot(
        self,
        x_min: float = -4,
        x_max: float = 4,
        y_min: float = -4,
        y_max: float = 4,
        step: float = 1,
        do_color: bool = False,
        **kwargs
    ):
        """Plots the vector field

        Parameters
        ----------
        x_min : float, optional
            Left limit of X axes, by default 0
        x_max : float, optional
            Right limit of X axes, by default 125
        y_min : float, optional
            Bottom limit of Y axes, by default 0
        y_max : float, optional
            Up limit of Y axes, by default 125
        step : float, optional
            Distance between points to plot (in radians), by default 1
        do_color : bool, optional
            Plot a background color indicating the strength of the current
        """
        # Compute the step for the discrete arrays
        s = int(max(1, step // np.mean(np.abs(np.diff(self.arr_x)))))
        idx = jnp.argwhere((self.arr_x >= x_min) & (self.arr_x <= x_max)).flatten()
        idy = jnp.argwhere((self.arr_y >= y_min) & (self.arr_y <= y_max)).flatten()
        idxx, idyy = idx[::s], idy[::s]
        # Prepare matrices to plot
        x = self.arr_x[idxx]
        y = self.arr_y[idyy]
        xx, yy = np.meshgrid(x, y)
        u = self.u[jnp.ix_(idyy, idxx)]
        v = self.v[jnp.ix_(idyy, idxx)]
        # Heatmap
        if do_color:
            # Velocity module
            m = (self.u**2 + self.v**2) ** (1 / 2)
            # Mask land
            m = np.ma.masked_where(self.land == 1, m)
            plt.matshow(
                m[jnp.ix_(idy, idx)],
                origin="lower",
                extent=[
                    self.arr_x[idx[0]],
                    self.arr_x[idx[-1]],
                    self.arr_y[idy[0]],
                    self.arr_y[idy[-1]],
                ],
                alpha=0.6,
            )
        # Plot the quiver
        plt.quiver(xx, yy, u, v, **kwargs)
