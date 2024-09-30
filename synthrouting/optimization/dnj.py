import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import typer
from jax import grad, jacfwd, jacrev, jit, random, vmap
from matplotlib import pyplot as plt

from synthrouting.optimization.route import Route
from synthrouting.vectorfields import Swirlys
from synthrouting.vectorfields.base import Vectorfield

KEY = random.PRNGKey(42)


# defines the hessian of our functions
def hessian(f: Callable, argnums: int = 0):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


class DNJ:
    def __init__(
        self,
        vectorfield: Vectorfield,
        time_step: Optional[float] = 0.1,
        time_end: Optional[float] = None,
        num_points: Optional[int] = None,
        num_iter: int = 10,
    ):
        """Initialize the DNJ algorithm

        Parameters
        ----------
        vectorfield : Vectorfield
            Vector field where we are optimizing the route
        time_step : float, optional
            Time step between points. For the method to work properly,
            must be the same as the route, by default 0.1
        num_iter : int, optional
            Number of DNJ iterations, by default 10

        Raises
        ------
        ValueError
            When `optimize_for` is not valid
        """
        self.vectorfield = vectorfield
        if time_step is None:
            self.time_step = time_end / num_points
            self.optimize_for = "fuel"
        else:
            self.time_step = time_step
            self.optimize_for = "time"
        self.num_iter = num_iter
        h = self.time_step
        if vectorfield.spherical:
            get_current = vectorfield.get_current_rad
        else:
            get_current = vectorfield.get_current

        if self.optimize_for == "fuel":

            def cost_function(x: jnp.array, xp: jnp.array) -> jnp.array:
                """Cost function that penalizes fuel consumption. It outputs lower
                values the more the vessel velocity is oriented with the vector field

                Parameters
                ----------
                x : jnp.array
                    Vessel coordinates (x, y) in space units (radians)
                xp : jnp.array
                    Veseel velocities (u, v) in space units / time units
                    (radians / second)

                Returns
                -------
                jnp.array
                    Cost of this state
                """
                w = get_current(x[0], x[1])
                cost = h * ((xp[0] - w[0]) ** 2 + (xp[1] - w[1]) ** 2) / 2
                return cost

            def cost_function_discretized(q0: jnp.array, q1: jnp.array) -> jnp.array:
                l1 = cost_function(q0, (q1 - q0) / h)
                l2 = cost_function(q1, (q1 - q0) / h)
                ld = (l1 + l2) / 2  # Skip h because is in cost function
                return ld

        elif self.optimize_for == "time":

            def cost_function(x: jnp.array, xp: jnp.array) -> jnp.array:
                """Cost function that penalizes time.
                It assumes that the vectorfield velocity module is always lower than 1
                Ref.: https://doi.org/10.1016/j.ifacol.2021.11.097

                Parameters
                ----------
                x : jnp.array
                    Vessel coordinates (x, y) in space units (radians)
                xp : jnp.array
                    Veseel velocities (u, v) in space units / time units
                    (radians / second)

                Returns
                -------
                jnp.array
                    Cost of this state
                """
                w = get_current(x[0], x[1])
                alpha = 1 - (w[0] ** 2 + w[1] ** 2)
                cost = (
                    jnp.sqrt(
                        1 / alpha * (xp[0] ** 2 + xp[1] ** 2)
                        + 1 / (alpha**2) * (w[0] * xp[0] + w[1] * xp[1]) ** 2
                    )
                    - 1 / alpha * (w[0] * xp[0] + w[1] * xp[1])
                ) ** 2
                return cost

            def cost_function_discretized(q0: jnp.array, q1: jnp.array) -> jnp.array:
                l1 = cost_function(q0, (q1 - q0) / h)
                l2 = cost_function(q1, (q1 - q0) / h)
                ld = h / 2 * (l1**2 + l2**2)
                return ld

        else:
            raise ValueError("unrecognized cost function")

        d1ld = grad(cost_function_discretized, argnums=0)
        d2ld = grad(cost_function_discretized, argnums=1)
        d11ld = hessian(cost_function_discretized, argnums=0)
        d22ld = hessian(cost_function_discretized, argnums=1)

        def optimize(qkm1: jnp.array, qk: jnp.array, qkp1: jnp.array) -> jnp.array:
            b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
            a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
            return jnp.linalg.solve(a, b)

        self.cost_function = cost_function
        self.cost_function_discretized = cost_function_discretized
        self.optim_vect = vmap(optimize, in_axes=(0, 0, 0), out_axes=0)

    def __hash__(self):
        return hash(())

    def __eq__(self, other):
        return isinstance(other, DNJ)

    def asdict(self) -> Dict:
        return {
            "time_step": self.time_step,
            "time_end": self.time_end,
            "num_points": self.num_points,
        }

    @partial(jit, static_argnums=(0, 2))
    def optimize_distance(self, pts: jnp.array, damping: float = 0.9) -> jnp.array:
        pts_new = jnp.copy(pts)
        q = self.optim_vect(pts[:-2], pts[1:-1], pts[2:])
        return pts_new.at[1:-1].set(damping * q + pts[1:-1])

    def optimize_route(self, route: Route, num_iter: Optional[int] = None):
        """Optimizes a route for any number of iterations

        Parameters
        ----------
        route : Route
            Route to optimize
        num_iter : int, optional
            Number of DNJ iterations, by default 10
        """
        num_iter = self.num_iter if num_iter is None else num_iter
        pts = route.pts
        mask_nan = jnp.isnan(pts)
        # Loop iterations
        for iteration in range(num_iter):
            pts_old = pts
            pts = self.optimize_distance(pts)
            # TODO: Sometimes the DNJ produces NaNs, understand why and fix
            # Temporal Solution: NaNs are replaced with last valid value
            mask_nan = jnp.isnan(pts)
            pts = pts.at[mask_nan].set(pts_old[mask_nan])
        # Warn user if NaNs appeared
        if mask_nan.any():
            warnings.warn(
                "There has been NaNs in the last iteration. "
                "This may prevent optimization."
            )
        # Update the points of the route
        route.x = pts[:, 0]
        route.y = pts[:, 1]
        vel = jnp.diff(pts, axis=0) / self.time_step

        # Update the minimum cost (needs to flip dimensions)
        route.cost = self.cost_function(pts[:-1].T, vel.T).sum()


class DNJRandomGuess:
    def __init__(
        self,
        vectorfield: Vectorfield,
        q0: Tuple[float, float],
        q1: Tuple[float, float],
        time_step: Optional[float] = None,
        time_end: Optional[float] = None,
        angle_amplitude: float = jnp.pi,
        num_points: int = 200,
        num_routes: int = 3,
        num_iter: int = 500,
    ):
        """Initializes a DNJ with random guesses"""
        x_start, y_start = q0
        x_end, y_end = q1
        if time_step is None:
            t = jnp.linspace(0, time_end, num_points)
        else:
            t = jnp.linspace(0, time_end, num_points)
        list_routes: List[Route] = [None] * num_routes
        # Randomly select number of segments per route
        num_segments = random.randint(KEY, (num_routes,), minval=2, maxval=5)
        ls_angs = random.uniform(KEY, (num_routes * 5,), minval=-0.5, maxval=0.5)
        ls_dist = random.uniform(KEY, (num_routes * 5,), minval=0.1, maxval=0.9)
        for idx_route in range(num_routes):
            # We first will choose the bounding points of each segment
            x_pts = [x_start]
            y_pts = [y_start]
            dist = []
            for idx_seg in range(num_segments[idx_route] - 1):
                # The shooting direction is centered on the final destination
                dx = x_end - x_pts[-1]
                dy = y_end - y_pts[-1]
                ang = jnp.arctan2(dy, dx)
                # Randomly select angle deviation
                ang_dev = angle_amplitude * ls_angs[idx_route * 5 + idx_seg]
                # Randomly select the distance travelled
                d = jnp.sqrt(dx**2 + dy**2) * ls_dist[idx_route * 5 + idx_seg]
                # Get the final point of the segment
                x_pts.append(x_pts[-1] + d * jnp.cos(ang + ang_dev))
                y_pts.append(y_pts[-1] + d * jnp.sin(ang + ang_dev))
                dist.append(d)
            # Append final point
            dx = x_end - x_pts[-1]
            dy = y_end - y_pts[-1]
            d = jnp.sqrt(dx**2 + dy**2)
            x_pts.append(x_end)
            y_pts.append(y_end)
            dist.append(d)
            dist = jnp.array(dist).flatten()
            # To ensure the points of the route are equi-distant,
            # the number of points per segment will depend on its distance
            # in relation to the total distance travelled
            num_points_seg = (num_points * dist / dist.sum()).astype(int)
            # Start generating the points
            x = jnp.array([x_start])
            y = jnp.array([y_start])
            for idx_seg in range(num_segments[idx_route]):
                x_new = jnp.linspace(
                    x_pts[idx_seg], x_pts[idx_seg + 1], num_points_seg[idx_seg] + 1
                ).flatten()
                x = jnp.concatenate([x, x_new[1:]])
                y_new = jnp.linspace(
                    y_pts[idx_seg], y_pts[idx_seg + 1], num_points_seg[idx_seg] + 1
                ).flatten()
                y = jnp.concatenate([y, y_new[1:]])
            if len(x) < num_points:
                # Add final point
                x = jnp.concatenate([x, jnp.array([x_end])])
                y = jnp.concatenate([y, jnp.array([y_end])])
            # Add the route to the list
            list_routes[idx_route] = Route(x, y, t=t, geometry=vectorfield.geometry)
        # Store parameters
        self.dnj = DNJ(
            vectorfield=vectorfield,
            time_step=time_step,
            time_end=time_end,
            num_points=num_points,
        )
        self.list_routes = list_routes
        self.num_iter = num_iter
        self.total_iter: int = 0
        self.time_end = time_end

    def __next__(self) -> List[Route]:
        for route in self.list_routes:
            self.dnj.optimize_route(route, num_iter=self.num_iter)
        self.total_iter += self.num_iter
        return self.list_routes


def main(num_iter: int = 22000):
    vectorfield = Swirlys()

    q0 = (0, 0)
    qn = (6, 5)
    tend = 30

    optimizer = DNJRandomGuess(
        vectorfield,
        q0,
        qn,
        time_end=tend,
        num_points=200,
        num_iter=num_iter,
        num_routes=5,
        angle_amplitude=2 * jnp.pi,
    )
    list_routes = next(optimizer)

    # Concatenate values
    x = jnp.concatenate([route.x for route in list_routes])
    y = jnp.concatenate([route.y for route in list_routes])

    # Round limits
    xmin = int(x.min()) - 1
    xmax = int(x.max()) + 1
    ymin = int(y.min()) - 1
    ymax = int(y.max()) + 1

    vectorfield.plot(extent=(xmin, xmax, ymin, ymax), step=0.5)
    for route in list_routes:
        plt.plot(route.x, route.y, label=f"{route.cost:.3f}")
    plt.scatter([q0[0], qn[0]], [q0[1], qn[1]], color="red")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.legend(title="Cost (fuel)")
    plt.title("DNJ")
    plt.savefig("output/dnj.png")

    # Store the route in txt file
    x = list_routes[0].x
    y = list_routes[0].y
    t = list_routes[0].t
    with open("output/dnj.txt", "w") as f:
        f.write("x;y;t\n")
        for xi, yi, ti in zip(x, y, t):
            f.write(f"{xi};{yi};{ti}\n")


if __name__ == "__main__":
    typer.run(main)
