from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.jax_utils import DNJ, Optimizer, RouteJax
from hybrid_routing.utils.plot import plot_textbox, plot_ticks_radians_to_degrees
from hybrid_routing.vectorfields import VectorfieldReal
from hybrid_routing.vectorfields.base import Vectorfield


class Pipeline:
    def __init__(
        self,
        p0: Tuple[float],
        pn: Tuple[float],
        key: str,
        path: Optional[Union[str, Path]] = None,
        radians: bool = True,
    ):
        self.x0, self.y0 = p0
        self.xn, self.yn = pn

        if path is None:
            module = __import__("hybrid_routing")
            module = getattr(module, "vectorfields")
            self.vectorfield: Vectorfield = getattr(module, key)()
            self.vf_real = False
        else:
            self.vectorfield = VectorfieldReal.from_folder(path, key, radians=radians)
            self.vf_real = True
        self.vf_key = key

        self.route_rk: RouteJax = None
        self.route_dnj: RouteJax = None
        self._routes_dnj: List[RouteJax] = [None] * 4

    def solve_zivp(
        self,
        time_iter: float = 0.5,
        time_step: float = 0.025,
        angle_amplitude: float = np.pi,
        angle_heading: float = np.pi / 2,
        num_angles: int = 20,
        vel: float = 1,
        dist_min: float = 0.1,
        use_rk: bool = True,
        method: str = "direction",
    ):
        # Initialize the optimizer
        optimizer = Optimizer(
            self.vectorfield,
            time_iter=time_iter,
            time_step=time_step,
            angle_amplitude=angle_amplitude,
            angle_heading=angle_heading,
            num_angles=num_angles,
            vel=vel,
            dist_min=dist_min,
            use_rk=use_rk,
            method=method,
        )
        # Run the optimizer until it converges
        for list_routes in optimizer.optimize_route(self.x0, self.y0, self.xn, self.yn):
            pass

        # Take the best route
        route: RouteJax = list_routes[0]
        route.append_point_end(x=self.xn, y=self.yn, vel=optimizer.vel)

        self.route_rk = deepcopy(route)
        self.vel = vel

        # Recompute times
        route.recompute_times(optimizer.vel, self.vectorfield, interp=False)
        self.route_rk2 = deepcopy(route)

    def solve_dnj(
        self,
        num_iter: int = 5,
        time_step: float = 0.01,
        optimize_for: str = "time",
    ):
        if self.route_rk is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        # Apply DNJ
        dnj = DNJ(self.vectorfield, time_step=time_step, optimize_for=optimize_for)
        # Apply DNJ in loop
        num_iter = num_iter // 5
        route = deepcopy(self.route_rk)
        # Intermediate steps
        for n in range(4):
            dnj.optimize_route(route, num_iter=num_iter)
            self._routes_dnj[n] = deepcopy(route)
        # Last DNJ run
        dnj.optimize_route(route, num_iter=num_iter)
        route.recompute_times(self.vel, self.vectorfield, interp=False)
        self.route_dnj = deepcopy(route)

    def to_dict(self) -> Dict:
        if self.route_rk is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        if self.route_dnj is None:
            raise AttributeError("DNJ step is missing. Run `solve_dnj` first.")
        return {
            "key": self.vf_key,
            "real": self.vf_real,
            "vel": self.vel,
            "route_rk": self.route_rk.asdict(),
            "route_rk2": self.route_rk2.asdict(),
            "route_dnj": self.route_dnj.asdict(),
        }

    def plot(
        self,
        extent: Optional[Tuple[float]] = None,
        textbox_pos: Optional[Tuple[float]] = None,
        textbox_align: str = "top",
    ):
        if self.route_rk is None:
            raise AttributeError("ZIVP step is missing. Run `solve_zivp` first.")
        if self.route_dnj is None:
            raise AttributeError("DNJ step is missing. Run `solve_dnj` first.")

        # Initialize figure with vectorfield
        plt.figure(figsize=(5, 5))
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        elif self.vf_real:
            xmin = self.vectorfield.arr_x.min()
            xmax = self.vectorfield.arr_x.max()
            ymin = self.vectorfield.arr_y.min()
            ymax = self.vectorfield.arr_y.max()
        else:
            xmin, xmax, ymin, ymax = [-5, 5, -5, 5]

        if self.vf_real:
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

        plt.gca().set_aspect("equal")

        # Plot source and destination point
        plt.scatter(self.x0, self.y0, c="green", s=20, zorder=10)
        plt.scatter(self.xn, self.yn, c="green", s=20, zorder=10)
        # Plot route
        plt.plot(
            self.route_rk.x,
            self.route_rk.y,
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

        plot_textbox(
            (self.x0, self.y0),
            (self.xn, self.yn),
            (self.route_rk2.t[-1], self.route_dnj.t[-1]),
            pos=textbox_pos,
            align=textbox_align,
        )

        # Set plot limits
        if extent is not None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

        plt.tight_layout()
        plt.tight_layout()
