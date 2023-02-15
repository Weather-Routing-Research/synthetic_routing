from importlib import import_module
from typing import Optional, Tuple, Union

import jax.numpy as jnp

from hybrid_routing.geometry import Euclidean, Geometry
from hybrid_routing.vectorfields.base import Vectorfield


class Route:
    def __init__(
        self,
        x: jnp.array,
        y: jnp.array,
        t: Optional[jnp.array] = None,
        theta: Optional[jnp.array] = None,
        geometry: Optional[Union[Geometry, str]] = None,
    ):
        self.x: jnp.ndarray = jnp.atleast_1d(x)
        self.y: jnp.ndarray = jnp.atleast_1d(y)
        self.t: jnp.ndarray = (
            jnp.atleast_1d(t) if t is not None else jnp.arange(0, len(self.x), 1)
        )
        assert len(self.x) == len(self.y) == len(self.t), "Array lengths are not equal"
        # Heading of the vessel
        self.theta: jnp.ndarray = (
            jnp.atleast_1d(theta) if theta is not None else jnp.zeros_like(x)
        )
        # Define the geometry of this route
        if isinstance(geometry, Geometry):
            self.geometry = geometry
        elif isinstance(geometry, str):
            module = import_module("hybrid_routing.geometry")
            self.geometry: Geometry = getattr(module, geometry)()
        else:
            self.geometry = Euclidean()

        # Compute distance
        self.d = self.geometry.dist_between_coords(self.x, self.y)

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return (
            f"Route(x0={self.x[0]:.2f}, y0={self.y[0]:.2f}, "
            f"xN={self.x[-1]:.2f}, yN={self.y[-1]:.2f}, "
            f"length={len(self)})"
        )

    def __str__(self) -> str:
        return (
            f"Route(x0={self.x[0]:.2f}, y0={self.y[0]:.2f}, "
            f"xN={self.x[-1]:.2f}, yN={self.y[-1]:.2f}, "
            f"length={len(self)})"
        )

    def asdict(self) -> dict:
        """Return dictionary with coordinates, times and headings"""
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "t": self.t.tolist(),
            "theta": self.theta.tolist(),  # Heading of the vessel
            "d": self.d.tolist(),
            "geometry": str(self.geometry),
        }

    @property
    def pts(self):
        return jnp.stack([self.x, self.y], axis=1)

    @property
    def dt(self):
        return -jnp.diff(self.t)

    @property
    def dx(self):
        return -jnp.diff(self.x)

    @property
    def dy(self):
        return -jnp.diff(self.y)

    @property
    def dxdt(self):
        return self.dx / self.dt

    @property
    def dydt(self):
        return self.dy / self.dt

    def append_points(
        self,
        x: jnp.array,
        y: jnp.array,
        t: Optional[jnp.array] = None,
        theta: Optional[jnp.array] = None,
    ):
        """Append new points to the end of the route

        Parameters
        ----------
        x : jnp.array
            Coordinates on X-axis, typically longitudes
        y : jnp.array
            Coordinates on X-axis, typically latitudes
        t : jnp.array
            Timestamp of each point, typically in seconds
        """
        x, y = jnp.atleast_1d(x), jnp.atleast_1d(y)
        self.x = jnp.concatenate([self.x, x])
        self.y = jnp.concatenate([self.y, y])
        t = jnp.atleast_1d(t) if t is not None else self.t + jnp.arange(0, len(x), 1)
        self.t = jnp.concatenate([self.t, t])
        theta = (
            jnp.atleast_1d(theta)
            if theta is not None
            else jnp.tile(self.theta[-1], x.shape)
        )
        self.theta = jnp.concatenate([self.theta, theta])
        # Compute distance
        self.d = self.geometry.dist_between_coords(self.x, self.y)

    def append_point_end(self, p: Tuple[float], vel: float):
        """Append an end point to the route and compute its timestamp.
        It does not take into account the effect of vectorfields.

        Parameters
        ----------
        x : float
            Coordinate on X-axis, typically longitude
        y : float
            Coordinate on X-axis, typically latitude
        vel : float
            Vessel velocity, typically in meters per second
        """
        dist = self.geometry.dist_p0_to_p1((self.x[-1], self.y[-1]), p)
        t = dist / vel + self.t[-1]
        self.append_points(p[0], p[1], t)

    def interpolate(self, n: int, vel: float = None):
        """Interpolate route to `n` points"""
        i = jnp.linspace(0, len(self), num=n)
        j = jnp.linspace(0, len(self), num=len(self))
        self.x = jnp.interp(i, j, self.x)
        self.y = jnp.interp(i, j, self.y)
        self.theta = jnp.interp(i, j, self.theta)
        self.d = self.geometry.dist_between_coords(self.x, self.y)
        if vel:
            self.t = self.d / vel + self.t[0]
        else:
            self.t = jnp.interp(i, j, self.t)

    def recompute_times(self, vel: float, vf: Vectorfield):
        """Given a vessel velocity and a vectorfield, recompute the
        times for each coordinate contained in the route

        Parameters
        ----------
        vel : float
            Vessel velocity
        vf : Vectorfield
            Vectorfield
        """
        x, y = self.x, self.y
        # Update distance
        self.d = self.geometry.dist_between_coords(self.x, self.y)
        # Angle over ground between points
        a_g = self.geometry.ang_between_coords(x, y)
        # Componentes of the velocity of vectorfield
        # We loop to avoid memory errors when using GPU
        v_cx = jnp.zeros(len(x) - 1)
        v_cy = jnp.zeros(len(y) - 1)
        for i in range(len(x) - 1):
            a, b = vf.get_current(x[i], y[i])
            v_cx = v_cx.at[i].set(a)
            v_cy = v_cy.at[i].set(b)
        # Angle and module of the velocity of vectorfield
        a_c, v_c = self.geometry.components_to_ang_mod(v_cx, v_cy)
        # Angle of the vectorfield w.r.t. the direction over ground
        a_cg = a_g - a_c  # TODO: Ensure this difference is done right
        # Components of the vectorfield w.r.t. the direction over ground
        v_cg_para, v_cg_perp = self.geometry.ang_mod_to_components(a_cg, v_c)
        # The perpendicular component of the vessel velocity must compensate the
        # vector field
        v_vg_perp = -v_cg_perp
        # Component of the vessel velocity parallel w.r.t. the direction over ground
        v_vg_para = jnp.sqrt(jnp.power(vel, 2) - jnp.power(v_vg_perp, 2))
        # Velocity over ground is the sum of vessel and vectorfield parallel components
        v_g = v_vg_para + v_cg_para
        # Time is distance divided by velocity over ground
        t: jnp.ndarray = jnp.divide(self.d, v_g)
        # Identify NaN and negative values
        mask_nan = jnp.isnan(t)
        mask_neg = t < 0
        if mask_nan.any():
            print(
                f"[WARNING] Negative times found in {mask_nan.sum()} "
                f"out of {len(t)} points. Consider raising vessel velocity over {vel}."
                " NaN values were changed to max."
            )
            t = t.at[mask_nan].set(jnp.nanmax(t))
        if mask_neg.any():
            tneg = t[t < 0]
            print(
                f"[WARNING] Negative times found in {len(tneg)} out of {len(t)} points."
                f" Worst is {min(tneg)}. Consider raising vessel velocity over {vel}."
                " Time values lower than 0 were changed to max."
            )
            t = t.at[mask_neg].set(jnp.nanmax(t))

        # Update route times
        self.t = jnp.concatenate([jnp.asarray([0]), jnp.cumsum(t)])
