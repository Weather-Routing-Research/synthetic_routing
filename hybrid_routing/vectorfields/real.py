from pathlib import Path
from typing import Union

import jax.numpy as jnp
import pandas as pd

from hybrid_routing.utils.spherical import DEG2RAD
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
        self.u = jnp.asarray(df_x.values)
        self.v = jnp.asarray(df_y.values)

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

    def get_current(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
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
        jnp.array
            The current's velocity in x and y direction (u, v)
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

        # Use the weights to compute the velocity component
        # relative to those points
        u = (self.u * w).sum(axis=(-2, -1))  # (P, )
        v = (self.v * w).sum(axis=(-2, -1))  # (P, )

        return u, v
