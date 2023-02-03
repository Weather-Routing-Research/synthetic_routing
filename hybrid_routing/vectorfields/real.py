from pathlib import Path
from typing import Tuple, Union
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd

from hybrid_routing.geometry.spherical import DEG2RAD
from hybrid_routing.vectorfields.base import Vectorfield


class VectorfieldReal(Vectorfield):
    def __init__(self, df_x: pd.DataFrame, df_y: pd.DataFrame, radians: bool = False):
        """Loads a real vectorfield from two dataframes, containing (u, v) velocities.
        Index must be the Y-coordinates and columns the X-coordinates, both in degrees

        Parameters
        ----------
        df_x : pd.DataFrame
            Velocity component u, in meters per second
        df_y : pd.DataFrame
            Velocity component v, in meters per second
        radians : bool, optional
            Use radians, by default False
        """
        assert df_x.shape == df_y.shape, "Dataframes are not the same shape"
        assert (df_x.index == df_y.index).all(), "Y coordinates do not match"
        assert (df_x.columns == df_y.columns).all(), "X coordinates do not match"

        # Load X, Y coordinates (longitude and latitude in degrees)
        self.arr_x = jnp.asarray(df_x.columns.astype(float).tolist())
        self.arr_y = jnp.asarray(df_x.index.astype(float).tolist())
        if radians:
            self.arr_x = self.arr_x * DEG2RAD
            self.arr_y = self.arr_y * DEG2RAD

        # Load velocity components (in meters per second)
        u = jnp.asarray(df_x.values)
        v = jnp.asarray(df_y.values)

        # Land mask, change NaN to 0's
        self.land = jnp.isnan(u) | jnp.isnan(v)
        self.u = jnp.nan_to_num(u)
        self.v = jnp.nan_to_num(v)

        # Compute the average step between X and Y coordinates
        # We are assuming this step is constant!
        self._dx = jnp.abs(jnp.mean(jnp.diff(self.arr_x)))
        self._dy = jnp.abs(jnp.mean(jnp.diff(self.arr_y)))

        super().__init__(spherical=True)

        # Add the rest of parameters used by the vectorfield
        self.is_discrete = True

    @classmethod
    def from_folder(
        cls, path: Union[str, Path], name: str, radians: bool = False
    ) -> "VectorfieldReal":
        """Load the vectorfield from a folder

        Parameters
        ----------
        path : Union[str, Path]
            Path to the folder
        name : str
            Name of the vectorfield
        radians : bool, optional
            Use radians, by default False

        Returns
        -------
        VectorfieldReal
        """
        path = Path(path) if isinstance(path, str) else path
        df_x = pd.read_csv(path / (name + "-lon.csv"), index_col=0)
        df_y = pd.read_csv(path / (name + "-lat.csv"), index_col=0)
        return cls(df_x, df_y, radians=radians)

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
        u = (self.u * w).sum(axis=(-2, -1))  # (P, )
        v = (self.v * w).sum(axis=(-2, -1))  # (P, )

        return u, v

    def is_land(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Indicates the presence of land at a given point (x,y) on the grid.

        Parameters
        ----------
        x : jnp.array
            x-coordinate of the ship
        y : jnp.array
            y-coordinate of the ship

        Returns
        -------
        jnp.array
            Boolean array
        """
        w = self._weight_coordinates(x, y)
        b: jnp.ndarray = (self.land * w).sum(axis=(-2, -1))  # (P, )
        return b.at[b > 0].set(1).astype(bool)

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
        idx = jnp.argwhere((self.arr_x >= x_min) & (self.arr_x <= x_max)).flatten()[
            ::10
        ]
        idy = jnp.argwhere((self.arr_y >= y_min) & (self.arr_y <= y_max)).flatten()[
            ::10
        ]
        plt.quiver(
            self.arr_x[idx],
            self.arr_y[idy],
            self.u[jnp.ix_(idx, idy)],
            self.v[jnp.ix_(idx, idy)],
            **kwargs
        )
        # Heatmap
        if do_color:
            # Velocity module
            m = (self.u**2 + self.v**2) ** (1 / 2)
            plt.matshow(
                m[jnp.ix_(idx, idy)],
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                alpha=0.6,
            )
