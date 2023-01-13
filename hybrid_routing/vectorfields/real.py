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
        # Define methods to get closest indexes
        self.closest_idx = jnp.vectorize(lambda x: jnp.argmin(jnp.abs(self.arr_x - x)))
        self.closest_idy = jnp.vectorize(lambda y: jnp.argmin(jnp.abs(self.arr_y - y)))

        #
        self.spherical = True
        self.is_discrete = True
        self.ode_zermelo = self._ode_zermelo_spherical

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

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        """Takes the current values (u,v) at a given point (x,y) on the grid.

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
        idx, idy = self.closest_idx(x), self.closest_idy(y)
        return jnp.asarray([self.u[idx, idy], self.v[idx, idy]])
