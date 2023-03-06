from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp

from hybrid_routing.geometry import Euclidean, Geometry, Spherical
from hybrid_routing.optimization.route import Route
from hybrid_routing.optimization.zivp import (
    solve_discretized_zermelo,
    solve_ode_zermelo,
    solve_rk_zermelo,
)
from hybrid_routing.vectorfields.base import Vectorfield


def compute_thetas_in_cone(
    cone_center: float, angle_amplitude: float, num_angles: int
) -> jnp.array:
    # Define the search cone
    delta = 1e-4 if angle_amplitude <= 1e-4 else angle_amplitude / 2
    if num_angles > 1:
        thetas = jnp.linspace(
            cone_center - delta,
            cone_center + delta,
            num_angles,
        )
    else:
        thetas = jnp.array([cone_center])
    return thetas


class Optimizer:
    geometry: Geometry

    def __init__(
        self,
        vectorfield: Vectorfield,
        time_iter: float = 2,
        time_step: float = 0.1,
        angle_amplitude: float = jnp.pi,
        angle_heading: Optional[float] = None,
        num_angles: int = 5,
        vel: float = 5,
        dist_min: Optional[float] = None,
        prop_keep_before_land: float = 0.9,
        max_iter: int = 2000,
        use_rk: bool = False,
        method: str = "direction",
    ):
        """Optimizer class

        Parameters
        ----------
        vectorfield : Vectorfield
            Background vectorfield for the ship to set sail on
        time_iter : float, optional
            The total amount of time the ship is allowed to travel by at each iteration,
            by default 2
        time_step : float, optional
            Number of steps to reach from 0 to time_iter (equivalently, how "smooth"
            each path is), by default 0.1
        angle_amplitude : float, optional
            The search cone range in radians, by default pi
        angle_heading : float, optional
            Maximum deviation allower when optimizing direction,
            by default 1/4 angle amplitude
        num_angles : int, optional
            Number of initial search angles, by default 5
        vel : float, optional
            Speed of the ship (unit unknown), by default 5
        dist_min : float, optional
            Minimum terminating distance around the destination (x_end, y_end),
            by default None
        max_iter : int, optional
            Maximum number of iterations allowed, by default 2000
        prop_keep_before_land : float, optional
            Proportion of the trajectory to keep before touching land, by default 0.9
        use_rk : bool, optional
            Use Runge-Kutta solver instead of odeint solver
        method: str, optional
            Method to compute the optimal route. Options are:
            - "direction": Keeps the routes which direction points to the goal
            - "closest": Keeps the closest route to the goal
        """
        self.vectorfield = vectorfield

        # Define distance metric
        if vectorfield.spherical:
            self.geometry = Spherical()
        else:
            self.geometry = Euclidean()

        # Choose solving method depends on whether the vectorfield is discrete
        if use_rk:
            self.solver = solve_rk_zermelo
        elif vectorfield.is_discrete:
            self.solver = solve_discretized_zermelo
        else:
            self.solver = solve_ode_zermelo
        self.use_rk = use_rk

        # Compute minimum distance as the average distance
        # transversed during one loop
        self.dist_min = vel * time_iter if dist_min is None else dist_min

        # Store the other parameters
        self.time_iter = time_iter
        self.time_step = time_step
        self.angle_amplitude = angle_amplitude
        self.angle_heading = (
            angle_amplitude / 4 if angle_heading is None else angle_heading
        )
        self.num_angles = num_angles
        self.vel = vel
        if method in ["closest", "direction"]:
            self.method = method
        else:
            print("Non recognized method, using 'direction'.")
            self.method = "direction"
        self.max_iter = max_iter
        self.prop_land = prop_keep_before_land
        self.exploration = None

    def asdict(self) -> Dict:
        return {
            "time_iter": self.time_iter,
            "time_step": self.time_step,
            "angle_amplitude": self.angle_amplitude,
            "angle_heading": self.angle_heading,
            "num_angles": self.num_angles,
            "vel": self.vel,
            "dist_min": self.dist_min,
            "use_rk": self.use_rk,
            "method": self.method,
        }

    def min_dist_p0_to_p1(
        self, list_routes: List[Route], pt_goal: Tuple, skip: Optional[List[int]] = None
    ) -> int:
        """Out of a list of routes, returns the index of the route the ends
        at the minimum distance to the goal.

        Parameters
        ----------
        list_routes : List[jnp.array]
            List of routes, defined by (x, y, theta)
        pt_goal : Tuple
            Goal point, defined by (x, y)
        skip : List[int], optional
            If given, skip these indices

        Returns
        -------
        int
            Index of the route that ends at the minimum distance to the goal.
        """
        min_dist = jnp.inf
        skip = [] if skip is None else skip
        for idx, route in enumerate(list_routes):
            if idx in skip:
                continue
            dist = self.geometry.dist_p0_to_p1((route.x[-1], route.y[-1]), pt_goal)
            if dist < min_dist:
                min_dist = dist
                idx_best_point = idx
        return idx_best_point

    def solve_ivp(
        self,
        x: jnp.array,
        y: jnp.array,
        theta: jnp.array,
        t: float = 0,
        time_iter: float = 1,
    ) -> List[Route]:
        """Solve an initial value problem, given arrays of same length for
        x, y and theta (heading, w.r.t. x-axis)

        Parameters
        ----------
        x : jnp.array
            Initial coordinate on x-axis
        y : jnp.array
            Initial coordinate on y-axis
        theta : jnp.array
            Initial heading w.r.t. x-axis, in radians
        t : float, optional
            Initial time, by default 0

        Returns
        -------
        List[Route]
            Routes generated with this IVP
        """
        return self.solver(
            self.vectorfield,
            x,
            y,
            theta,
            time_start=t,
            time_end=t + time_iter,
            time_step=self.time_step,
            vel=self.vel,
        )

    # TODO: Add land condition
    def _optimize_by_closest(self, p0: Tuple[float], pn: Tuple[float]) -> List[Route]:
        """
        System of ODE is from Zermelo's Navigation Problem
        https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#General_solution)
        1) This function first computes the locally optimized paths with Scipy's ODE
        solver. Given the starting coordinates (x_start, y_start), time (t_max),
        speed of the ship (vel), and the direction the ship points in
        (angle_amplitude / num_angles), the ODE solver returns a list of points on the
        locally optimized path.
        2) We then use a loop to compute all locally optimal paths with given angles in
        the angle amplitude and store them in a list.
        3) We next finds the list of paths with an end point (x1, y1) that has the
        smallest Euclidean distance to the destination (x_end, y_end).
        4) We then use the end point (x1, y1) on that path to compute the next set of
        paths by repeating the above algorithm.
        5) This function terminates till the last end point is within a neighbourhood of
        the destination (defaults vel * time_end).

        Parameters
        ----------
        x_start : float
            x-coordinate of the starting position
        y_start : float
            y-coordinate of the starting position
        x_end : float
            x-coordinate of the destinating position
        y_end : float
            y-coordinate of the destinating position

        Yields
        ------
        Iterator[List[Route]]
            Returns a list with all paths generated within the search cone.
            The path that terminates closest to destination is on top.
        """
        # Compute angle between first and last point
        cone_center = self.geometry.angle_p0_to_p1(p0, pn)

        pt = p0  # Position now
        t = 0  # Time now

        # Initialize route best
        route_best: Route = None

        # Distance and number of iterations
        dist = self.geometry.dist_p0_to_p1(pt, pn)
        n_iter = 0
        max_iter = self.max_iter
        # Make a copy of these parameters to tune them in case the optimization
        # does not converge
        dist_min = self.dist_min
        time_iter = self.time_iter

        while (dist > dist_min) and (n_iter <= max_iter):
            # Get arrays of initial coordinates for these segments
            arr_x = jnp.repeat(pt[0], self.num_angles)
            arr_y = jnp.repeat(pt[1], self.num_angles)
            arr_theta = compute_thetas_in_cone(
                cone_center, self.angle_amplitude, self.num_angles
            )

            list_routes = self.solve_ivp(
                arr_x, arr_y, arr_theta, t=t, time_iter=time_iter
            )

            # The routes outputted start at the closest point
            # We append those segments to the best route, if we have it
            if route_best is not None:
                for idx, route_new in enumerate(list_routes):
                    route: Route = deepcopy(route_best)
                    route.append_points(
                        route_new.x[1:], route_new.y[1:], t=route_new.t[1:]
                    )
                    list_routes[idx] = route

            # Update the closest points and best route
            pt_1 = pt
            idx_best = self.min_dist_p0_to_p1(list_routes, pn)
            route_best = deepcopy(list_routes[idx_best])
            pt = route_best.x[-1], route_best.y[-1]
            t = route_best.t[-1]

            # Recompute the cone center
            cone_center = self.geometry.angle_p0_to_p1(pt, pn)

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            # If the optimization does not progress, that means all routes are
            # getting over the goal. Decrease the iteration time or
            # increase the minimum distance
            if pt[0] == pt_1[0] and pt[1] == pt_1[1]:
                if time_iter == self.time_step:
                    dist_min *= 10
                    print(
                        "[WARN] Not converging! "
                        f"Minimum distance increased to {dist_min}. "
                    )
                else:
                    time_iter = max(self.time_step, time_iter / 2)
                    print(
                        "[WARN] Not converging! "
                        f"Iteration time decreased to {time_iter}. "
                    )

            # Update distance and number of iterations
            dist = self.geometry.dist_p0_to_p1(pt, pn)
            n_iter += 1
        else:
            # Message when loop finishes
            print(
                f"Optimization finished after {n_iter} iterations. "
                f"Distance from goal: {dist}"
            )

    def _optimize_by_direction(self, p0: Tuple[float], pn: Tuple[float]) -> List[Route]:
        # Compute angle between first and last point
        cone_center = self.geometry.angle_p0_to_p1(p0, pn)

        pt = p0  # Position now
        t = jnp.float64(0.0)  # Time now

        # Initialize the routes
        # Each one starts with a different angle
        arr_theta = compute_thetas_in_cone(
            cone_center, self.angle_amplitude, self.num_angles
        )
        list_routes: List[Route] = [
            Route(p0[0], p0[1], t, theta, geometry=self.geometry) for theta in arr_theta
        ]
        # Initialize the best route as the middle one (avoids UnboundLocalError)
        route_best = deepcopy(list_routes[len(list_routes) // 2])

        # Initialize list of routes to stop (outside of angle threshold)
        list_stop: List[int] = []
        # Define whether the next step is exploitation or exploration, and the
        # exploitation index
        # We start in the exploration step, so next step is exploitation
        self.exploration = True  # Exploitation step / Exploration step
        idx_refine = 1  # Where the best segment start + 1

        # The loop continues until the algorithm reaches the end
        dist = self.geometry.dist_p0_to_p1(pt, pn)
        n_iter = 0  # Number of iterations
        n_iter_step = 0  # Number of iterations in current step
        max_iter = self.max_iter
        # Make a copy of these parameters to tune them in case the optimization
        # does not converge
        dist_min = self.dist_min
        time_iter = self.time_iter

        while (dist > dist_min) and (n_iter <= max_iter):
            # Get arrays of initial coordinates for these segments
            arr_x = jnp.array([route.x[-1] for route in list_routes])
            arr_y = jnp.array([route.y[-1] for route in list_routes])
            arr_theta = jnp.array([route.theta[-1] for route in list_routes])

            # Compute the new route segments
            list_segments = self.solve_ivp(
                arr_x, arr_y, arr_theta, t=t, time_iter=time_iter
            )

            # Develop each route of our previous iteration,
            # following its current heading
            for idx, route in enumerate(list_routes):
                # If the index is inside the list of stopped routes, skip
                if idx in list_stop:
                    continue
                route_new = list_segments[idx]
                # Compute angle between route and goal
                # We keep routes which heading is inside search cone
                theta_goal = self.geometry.angle_p0_to_p1(
                    (route_new.x[-1], route_new.y[-1]), pn
                )
                delta_theta = abs(route_new.theta[-1] - theta_goal)
                cond_theta = delta_theta <= (self.angle_heading)
                # Check if the route crosses land. If not, we keep it
                is_land = self.vectorfield.is_land(route_new.x, route_new.y)
                cond_land = not is_land.any()
                # Check the trajectory has not gone out of bounds
                is_out = self.vectorfield.out_of_bounds(route_new.x, route_new.y)
                cond_out = not is_out.any()
                # Check both conditions
                if cond_theta and cond_land and cond_out:
                    route.append_points(
                        route_new.x[1:],
                        route_new.y[1:],
                        t=route_new.t[1:],
                        theta=route_new.theta[1:],
                    )
                elif (not cond_land) or (not cond_out):
                    # If the route has been stopped for reaching land
                    # or getting out of bounds, cut the last `prop_land` of it
                    icut = max(1, int(self.prop_land * len(route)))
                    route.x = route.x[:icut]
                    route.y = route.y[:icut]
                    route.t = route.t[:icut]
                    route.theta = route.theta[:icut]
                    # Add the route to the stopped list
                    list_stop.append(idx)
                else:
                    # Add the route to the stopped list
                    list_stop.append(idx)

            # If all routes have been stopped, generate new ones
            if len(list_stop) == len(list_routes):
                # Change next step from exploitation <-> exploration
                self.exploration = not self.exploration
                if n_iter_step == 0:
                    # If the last step had no iterations, that means all routes are
                    # getting over the goal. Decrease the iteration time or
                    # increase the minimum distance
                    if time_iter == self.time_step:
                        dist_min *= 10
                        print(
                            "[WARN] Not converging! "
                            f"Minimum distance increased to {dist_min}. "
                        )
                    else:
                        time_iter = max(self.time_step, time_iter / 2)
                        print(
                            "[WARN] Not converging! "
                            f"Iteration time decreased to {time_iter}. "
                        )
                else:
                    n_iter_step = 0
                if self.exploration:
                    # Exploration step: New routes are generated starting from
                    # the end of the best segment, using a cone centered
                    # around the direction to the goal
                    # Recompute the cone center using best route
                    cone_center = self.geometry.angle_p0_to_p1(pt, pn)
                    # Generate new arr_theta
                    arr_theta = compute_thetas_in_cone(
                        cone_center, self.angle_amplitude, self.num_angles
                    )
                    route_new = deepcopy(route_best)
                    # Set the new exploitation index
                    idx_refine = len(route_new.x)
                else:
                    # Exploitation step: New routes are generated starting from
                    # the beginning of best segment, using a small cone centered
                    # around the direction of the best segment
                    # Recompute the cone center using best route
                    cone_center = route_best.theta[idx_refine - 1]
                    # Generate new arr_theta
                    arr_theta = compute_thetas_in_cone(
                        cone_center, self.angle_amplitude / 5, self.num_angles
                    )
                    route_new = Route(
                        route_best.x[:idx_refine],
                        route_best.y[:idx_refine],
                        t=route_best.t[:idx_refine],
                        theta=route_best.theta[:idx_refine],
                        geometry=self.geometry,
                    )
                # Reinitialize route lists
                list_routes: List[Route] = []
                list_stop: List[int] = []
                # Fill new list of routes
                for theta in arr_theta:
                    route_new.theta = route_new.theta.at[-1].set(theta)
                    list_routes.append(deepcopy(route_new))
                # Update the time of the last point, will go backwards when changing
                # from exploration to exploitation
                t = route_new.t[-1]
            else:
                # The best route will be the one closest to our destination
                # and does not traverse land
                idx_best = self.min_dist_p0_to_p1(list_routes, pn)
                route_best = list_routes[idx_best]
                pt = route_best.x[-1], route_best.y[-1]
                t = max(route.t[-1] for route in list_routes)
                n_iter_step += 1  # Number of iterations in current step increases

            # Yield list of routes with best route in first position
            list_routes_yield = deepcopy(list_routes)
            list_routes_yield.insert(0, list_routes_yield.pop(idx_best))
            yield list_routes_yield

            # Update distance and number of iterations
            dist = self.geometry.dist_p0_to_p1(pt, pn)
            n_iter += 1  # Total number of iterations increases
        else:
            # Message when loop finishes
            print(
                f"Optimization finished after {n_iter} iterations. "
                f"Distance from goal: {dist}"
            )

    def optimize_route(self, p0: Tuple[float], pn: Tuple[float]) -> List[Route]:
        d = self.geometry.dist_p0_to_p1(p0, pn)
        if self.dist_min >= d:
            raise ValueError(
                f"Minimum distance allowed is {self.dist_min} "
                f"and distance to cover is {d}."
            )
        if self.method == "closest":
            return self._optimize_by_closest(p0, pn)
        elif self.method == "direction":
            return self._optimize_by_direction(p0, pn)
        else:
            raise ValueError(f"Method not identified: {self.method}")
