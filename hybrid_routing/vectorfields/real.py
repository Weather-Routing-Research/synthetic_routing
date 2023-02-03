from pathlib import Path
from typing import Tuple, Union

import jax.numpy as jnp
import pandas as pd

from hybrid_routing.geometry.spherical import DEG2RAD
from hybrid_routing.vectorfields.base import VectorfieldDiscrete


class VectorfieldReal(VectorfieldDiscrete):
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
