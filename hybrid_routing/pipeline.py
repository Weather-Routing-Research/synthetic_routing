from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.jax_utils import DNJ, Optimizer, Route
from hybrid_routing.utils.plot import plot_textbox, plot_ticks_radians_to_degrees
from hybrid_routing.vectorfields import VectorfieldReal
from hybrid_routing.vectorfields.base import Vectorfield


class Pipeline:
    _num_dnj: int = 10

    def __init__(
        self,
        p0: Tuple[float],
        pn: Tuple[float],
        key: str,
        path: Optional[Union[str, Path]] = None,
        si_units: bool = False,
    ):
        """Initialize the pipeline with the start and end point, and the vectorfield

        Parameters
        ----------
        p0 : Tuple[float]
            Start point (x, y)
        pn : Tuple[float]
            End point (x, y)
        key : str
            Benchmark key (name of the vectorfield)
        path : Optional[Union[str, Path]], optional
            Path to the data, if the vectorfield is real.
            If None, assume vectorfield is synthetic. By default None
        si_units : bool, optional
            Assume units arrive in SI (degrees, meters), by default False
        """
        self.x0, self.y0 = p0
        self.xn, self.yn = pn
        # Convert to radians
        if si_units:
            self.x0, self.y0 = self.x0 * DEG2RAD, self.y0 * DEG2RAD
            self.xn, self.yn = self.xn * DEG2RAD, self.yn * DEG2RAD
        self.si_units = si_units

        if path is None:
            module = import_module("hybrid_routing.vectorfields")
            self.vectorfield: Vectorfield = getattr(module, key)()
            self.real = False
        else:
            self.vectorfield = VectorfieldReal.from_folder(path, key, radians=True)
            self.real = True
        self.key = key

        # Empty attributes
        self.optimizer: Optimizer = None
        self.vel = 1
        self.route_zivp: Route = None
        self.dnj: DNJ = None
        self.route_dnj: Route = None
        self._routes_dnj: List[Route] = [None] * self._num_dnj

    @property
    def filename(self):
        return f"{self.key.lower().replace(' ', '_')}_{self.vel:03d}"

    def add_route(self, route: Union[Route, Dict], vel: Optional[float] = None):
        """Adds a route, skipping the ZIVP step

        Parameters
        ----------
        route : Union[Route, Dict]
            Route to be added, either in dictionary or Route format
        vel : Optional[float], optional
            Vessel velocity. If not given, it is computed. By default None
        """
        if isinstance(route, Route):
            route = deepcopy(route)
        else:
            route = Route(**route)
        # Compute velocity
        if vel is None:
            d = self.vectorfield.geometry.dist_between_coords(route.x, route.y)
            vel = d / route.dt
            self.vel = np.mean(vel)
        else:
            self.vel = vel
        # Recompute times
        route.recompute_times(vel, self.vectorfield)
        # Update route and optimizer
        self.route_zivp = route
        self.optimizer = Optimizer(self.vectorfield, vel=self.vel)

    def solve_zivp(
        self,
        time_iter: float = 0.5,
        time_step: float = 0.025,
        angle_amplitude: float = np.pi,
        angle_heading: float = np.pi / 2,
        num_angles: int = 20,
        vel: float = 1,
        dist_min: float = 0.1,
        max_iter: int = 2000,
        use_rk: bool = True,
        method: str = "direction",
        num_points: Optional[int] = None,
    ):
        """Solve the Zermelo Initial Value Problem

        Parameters
        ----------
        time_iter : float, optional
            The total amount of time the ship is allowed to travel by at each iteration,
            by default 0.5
        time_step : float, optional
            Number of steps to reach from 0 to time_iter (equivalently, how "smooth"
            each path is), by default 0.025
        angle_amplitude : float, optional
            The search cone range in radians, by default pi
        angle_heading : float, optional
            Maximum deviation allowed when optimizing direction,
            by default pi/2
        num_angles : int, optional
            Number of initial search angles, by default 20
        vel : float, optional
            Speed of the ship (unit unknown), by default 1
        dist_min : float, optional
            Minimum terminating distance around the destination (x_end, y_end),
            by default 0.1
        max_iter : int, optional
            Maximum number of iterations allowed for the optimizer, by default 2000
        use_rk : bool, optional
            Use Runge-Kutta solver instead of odeint solver, by default True
        method: str, optional
            Method to compute the optimal route. Options are:
            - "direction": Keeps the routes which direction points to the goal
            - "closest": Keeps the closest route to the goal
        """
        # Initialize the optimizer
        self.optimizer = Optimizer(
            self.vectorfield,
            time_iter=time_iter,
            time_step=time_step,
            angle_amplitude=angle_amplitude,
            angle_heading=angle_heading,
            num_angles=num_angles,
            vel=vel,
            dist_min=dist_min,
            max_iter=max_iter,
            use_rk=use_rk,
            method=method,
        )
        # Run the optimizer until it converges
        for list_routes in self.optimizer.optimize_route(
            (self.x0, self.y0), (self.xn, self.yn)
        ):
            route = list_routes[0]
            # print(
            #     "  (x, y, t) = "
            #     f"({route.x[-1]:.2f}, {route.y[-1]:.2f}, {route.t[-1]:.0f})"
            # )

        # Take the best route
        route: Route = list_routes[0]
        route.append_point_end(p=(self.xn, self.yn), vel=self.optimizer.vel)

        # Interpolate to `num_points` points
        if num_points:
            route.interpolate(num_points, vel=vel)

        # Recompute times
        route.recompute_times(vel, self.vectorfield)

        # Store parameters
        self.route_zivp = deepcopy(route)
        self.vel = vel

    def solve_dnj(
        self,
        num_iter: int = 5,
        optimize_for: str = "time",
    ):
        """Solve the Discrete Newton-Jacobi, using the route from ZIVP

        Parameters
        ----------
        num_iter : int, optional
            Number of DNJ iterations, by default 5
        optimize_for : str, optional
            Optimization criteria, by default "time"

        Raises
        ------
        AttributeError
            If ZIVP step has not been done yet
        """
        if self.route_zivp is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        # Apply DNJ
        time_step = float(np.mean(np.diff(self.route_zivp.t)))
        self.dnj = DNJ(self.vectorfield, time_step=time_step, optimize_for=optimize_for)
        # Apply DNJ in loop
        num_iter = num_iter // self._num_dnj
        route = deepcopy(self.route_zivp)
        # Intermediate steps
        for n in range(self._num_dnj):
            self.dnj.optimize_route(route, num_iter=num_iter)
            route.recompute_times(self.vel, self.vectorfield)
            self._routes_dnj[n] = deepcopy(route)
            print(f"  DNJ step {n+1} out of 10")
        # Take the one with lowest time
        _, idx = min((route.t[-1], idx) for (idx, route) in enumerate(self._routes_dnj))
        self.route_dnj = self._routes_dnj[idx]

    def to_dict(self) -> Dict:
        if self.route_zivp is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        if self.route_dnj is None:
            raise AttributeError("DNJ step is missing. Run `solve_dnj` first.")
        return {
            "key": self.key,
            "real": self.real,
            "vel": self.vel,
            "route_zivp": self.route_zivp.asdict(),
            "route_dnj": self.route_dnj.asdict(),
            "optimizer": self.optimizer.asdict(),
            "dnj": self.dnj.asdict(),
        }

    def plot(
        self,
        extent: Optional[Tuple[float]] = None,
        textbox_pos: Optional[Tuple[float]] = None,
        textbox_align: str = "bottom",
    ):
        if self.route_zivp is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        if self.route_dnj is None:
            raise AttributeError("DNJ step is missing. Run `solve_dnj` first.")

        # Vectorfield
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        elif self.real:
            xmin = self.vectorfield.arr_x.min()
            xmax = self.vectorfield.arr_x.max()
            ymin = self.vectorfield.arr_y.min()
            ymax = self.vectorfield.arr_y.max()
        else:
            xmin, xmax, ymin, ymax = [
                self.route_dnj.x.min() * 0.9,
                self.route_dnj.x.max() * 1.1,
                self.route_dnj.y.min() * 0.9,
                self.route_dnj.y.max() * 1.1,
            ]

        if self.real:
            self.vectorfield.plot(
                step=1 * DEG2RAD,
                color="grey",
                alpha=0.8,
                do_color=True,
                x_min=xmin,
                x_max=xmax,
                y_min=ymin,
                y_max=ymax,
            )
            plot_ticks_radians_to_degrees(step=5)
            # Times to hours
            times = (self.route_zivp.t[-1] / 3600, self.route_dnj.t[-1] / 3600)
        else:
            self.vectorfield.plot(
                step=0.25,
                color="grey",
                alpha=0.8,
                do_color=False,
                x_min=xmin,
                x_max=xmax,
                y_min=ymin,
                y_max=ymax,
            )
            xticks = np.arange(xmin, xmax, 1)
            plt.xticks(xticks)
            yticks = np.arange(ymin, ymax, 1)
            plt.yticks(yticks)
            # Times in not unit
            times = (self.route_zivp.t[-1], self.route_dnj.t[-1])

        plt.gca().set_aspect("equal")

        # Plot source and destination point
        plt.scatter(self.x0, self.y0, c="green", s=20, zorder=10)
        plt.scatter(self.xn, self.yn, c="green", s=20, zorder=10)
        # Plot route
        plt.plot(
            self.route_zivp.x,
            self.route_zivp.y,
            c="red",
            linewidth=2.5,
            alpha=0.9,
            zorder=5,
        )

        # Plot DNJ intermediate steps and result
        for route in self._routes_dnj:
            plt.plot(route.x, route.y, c="grey", linewidth=1, alpha=0.6, zorder=5)
        plt.plot(
            self.route_dnj.x,
            self.route_dnj.y,
            c="black",
            linewidth=2,
            alpha=0.9,
            zorder=5,
        )

        if self.si_units:
            p0 = (self.x0 / DEG2RAD, self.y0 / DEG2RAD)
            pn = (self.xn / DEG2RAD, self.yn / DEG2RAD)
        else:
            p0 = (self.x0, self.y0)
            pn = (self.xn, self.yn)

        plot_textbox(p0, pn, times, pos=textbox_pos, align=textbox_align)

        # Set plot limits
        if extent is not None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

        plt.tight_layout()
