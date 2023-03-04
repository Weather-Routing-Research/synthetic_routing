import json
import time
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD, Geometry
from hybrid_routing.optimization import DNJ, Optimizer, Route
from hybrid_routing.utils.config import load_config
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
        plot: Optional[Dict] = None,
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
        plot : Dict, optional
            Plot configuration, will be used by default when calling plot function
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
        self.geodesic: Route = None
        self._timeit: Dict[str, int] = {}
        self._dict_plot = plot if plot is not None else {}

    @property
    def filename(self):
        return f"{self.key.lower().replace(' ', '_')}_{self.vel:03d}"

    @property
    def geometry(self) -> Geometry:
        return self.vectorfield.geometry

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
            d = self.geometry.dist_between_coords(route.x, route.y)
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
        tic = time.process_time()  # We want to time this process
        for list_routes in self.optimizer.optimize_route(
            (self.x0, self.y0), (self.xn, self.yn)
        ):
            route = list_routes[0]

        # Take the best route
        route: Route = list_routes[0]
        route.append_point_end(p=(self.xn, self.yn), vel=self.optimizer.vel)

        # Interpolate to `num_points` points
        if num_points:
            route.interpolate(num_points, vel=vel)

        # Recompute times
        route.recompute_times(vel, self.vectorfield)

        # Store parameters, and computation time
        self._timeit.update({"zivp": int(time.process_time() - tic)})
        self.route_zivp = deepcopy(route)
        self.vel = vel

    def compute_geodesic(self):
        """Builds the route of minimum distance between the two points"""
        if self.route_zivp is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        # First compute the total distance to travel
        dist = self.geometry.dist_p0_to_p1((self.x0, self.y0), (self.xn, self.yn))
        # Initialize the optimizer
        optimizer = Optimizer(
            self.vectorfield,
            time_iter=1 / 20,
            time_step=1 / 100,  # Whole travel takes 1 unit of time
            angle_amplitude=20 * DEG2RAD,  # Small angle
            angle_heading=20 * DEG2RAD,
            num_angles=11,  # Allow some variation to avoid land
            vel=dist,  # Very high to ignore currents
            dist_min=dist / 20,
            max_iter=self.optimizer.max_iter,
            use_rk=self.optimizer.use_rk,
            method=self.optimizer.method,
        )
        # Run the optimizer until it converges
        tic = time.process_time()  # We want to time this process
        for list_routes in optimizer.optimize_route(
            (self.x0, self.y0), (self.xn, self.yn)
        ):
            pass
        # Take best route, append final point, interpolate to N points
        # and compute the real velocity and time of this journey
        route: Route = list_routes[0]
        route.append_point_end((self.xn, self.yn), vel=self.vel)
        route.interpolate(n=len(self.route_zivp), vel=self.vel)
        route.recompute_times(vel=self.vel, vf=self.vectorfield)
        # Store the route as the geodesic, and computation time
        self._timeit.update({"geodesic": int(time.process_time() - tic)})
        self.geodesic = route

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
        tic = time.process_time()  # We want to time this process
        num_iter = num_iter // self._num_dnj
        route = deepcopy(self.route_zivp)
        # Intermediate steps
        for n in range(self._num_dnj):
            self.dnj.optimize_route(route, num_iter=num_iter)
            route.recompute_times(self.vel, self.vectorfield)
            self._routes_dnj[n] = deepcopy(route)
            print(f"  DNJ step {n+1} out of 10")
        # Store the computation time
        self._timeit.update({"dnj": int(time.process_time() - tic)})
        # Take the one with lowest time
        _, idx = min((route.t[-1], idx) for (idx, route) in enumerate(self._routes_dnj))
        self.route_dnj = self._routes_dnj[idx]

    def to_dict(self) -> Dict:
        if self.route_zivp is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        if self.route_dnj is None:
            raise AttributeError("DNJ step is missing. Run `solve_dnj` first.")
        dict_return = {
            "key": self.key,
            "real": self.real,
            "vel": float(self.vel),
            "time": {
                "zivp": float(self.route_zivp.t[-1]),
                "dnj": float(self.route_dnj.t[-1]),
            },
            "dist": {
                "zivp": float(self.route_zivp.d.sum()),
                "dnj": float(self.route_dnj.d.sum()),
            },
            "timeit": self._timeit,
            "route": {"zivp": self.route_zivp.asdict(), "dnj": self.route_dnj.asdict()},
            "optimizer": self.optimizer.asdict(),
            "dnj": self.dnj.asdict(),
        }
        if self.geodesic is not None:
            dict_return["route"].update({"geodesic": self.geodesic.asdict()})
            dict_return["time"].update({"geodesic": float(self.geodesic.t[-1])})
            dict_return["dist"].update({"geodesic": float(self.geodesic.d.sum())})

        return dict_return

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

        # The plot parameters may have been predefined in the initial call
        # If that is the case, use those parameters in case no other were provided
        extent = extent if extent is not None else self._dict_plot.get("extent", None)
        textbox_pos = (
            textbox_pos
            if textbox_pos is not None
            else self._dict_plot.get("textbox_pos", None)
        )
        textbox_align = (
            textbox_align
            if textbox_align is not None
            else self._dict_plot.get("textbox_align", None)
        )

        # Vectorfield
        if self.real:
            if extent is None:
                extent = (
                    self.vectorfield.arr_x.min(),
                    self.vectorfield.arr_x.max(),
                    self.vectorfield.arr_y.min(),
                    self.vectorfield.arr_y.max(),
                )
            self.vectorfield.plot(
                extent=extent,
                step=DEG2RAD,
                color="grey",
                alpha=0.8,
                do_color=True,
            )
            plot_ticks_radians_to_degrees(step=5)
            # Times to hours
            times = (self.route_zivp.t[-1] / 3600, self.route_dnj.t[-1] / 3600)
        else:
            if extent is None:
                extent = (
                    self.route_dnj.x.min() - 1,
                    self.route_dnj.x.max() + 1,
                    self.route_dnj.y.min() - 1,
                    self.route_dnj.y.max() + 1,
                )
            self.vectorfield.plot(
                extent=extent,
                step=0.25,
                color="grey",
                alpha=0.8,
                do_color=False,
            )
            xticks = np.arange(extent[0], extent[1], 1)
            plt.xticks(xticks)
            yticks = np.arange(extent[2], extent[3], 1)
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
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])

        plt.tight_layout()


def run_pipelines(
    key: str,
    path_config: str = "data/config.toml",
    path_out: Union[str, Path] = "output",
    max_thread: int = 6,
):
    """Run the benchmarks defined by a configuration `toml` file

    Parameters
    ----------
    key : str
        Group of benchmarks, for instance "synthetic" or "real"
    path_config : str, optional
        Path to the configuration file, by default "data/config.toml"
    path_out : str, optional
        Output path, by default "output"
    max_thread : int, optional
        Maximum number of threads allowed, by default 6
    """
    # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
    matplotlib.use("Agg")

    list_benchmark = load_config(path_config, key).tolist()
    dict_zivp = load_config(path_config, "zivp")[key]
    dict_dnj = load_config(path_config, "dnj")[key]

    list_vel = dict_zivp["vel"]
    list_vel = list_vel if isinstance(list_vel, List) else [list_vel]

    # Custom folder for this date
    path_out = Path(path_out) if isinstance(path_out, str) else path_out
    if not path_out.exists():
        path_out.mkdir()

    # Maximum number of threads cannot be higher than number of processes
    max_thread = min(max_thread, len(list_benchmark) * len(list_vel))

    # Initialize list of pipelines
    list_pipes: List[Pipeline] = [None for n in range(max_thread)]

    def run_pipeline(n_thread: int, dict_pipe: dict, vel: float):
        print(f"Initializing: {dict_pipe['key']}, vel = {vel}")
        pipe = Pipeline(**dict_pipe)

        pipe.solve_zivp(**dict_zivp)
        pipe.compute_geodesic()
        pipe.solve_dnj(**dict_dnj)

        # Append pipeline to list
        list_pipes[n_thread] = pipe

        # Decide filename
        file = path_out / pipe.filename

        # Store in dictionary (json)
        with open(file.with_suffix(".json"), "w") as outfile:
            dict_results = pipe.to_dict()
            json.dump(dict_results, outfile)

        print(f"Done {pipe.filename} vectorfield, {vel} m/s\n---")

    # Initialize list of threads and index
    threads: List[Thread] = [None for i in range(max_thread)]
    n_thread = 0

    for dict_pipe in list_benchmark:
        for vel in list_vel:
            threads[n_thread] = Thread(
                target=run_pipeline, args=(n_thread, dict_pipe, vel)
            )
            threads[n_thread].start()
            n_thread += 1
            # If maximum index is reached, wait for all threads to finish
            if n_thread == max_thread:
                # Plot each thread independently, to avoid overlaps
                for n_thread, t in enumerate(threads):
                    t.join()  # Waits for thread to finish before plotting
                    pipe = list_pipes[n_thread]  # Get the associated pipeline
                    # Decide filename
                    file = path_out / pipe.filename
                    # Plot results and store
                    plt.figure(dpi=120)
                    pipe.plot()
                    plt.savefig(file.with_suffix(".png"))
                    plt.close()
                # Reset thread number
                n_thread = 0
