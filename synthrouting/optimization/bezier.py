from typing import Optional, Tuple

import cma
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from jax import random

from synthrouting.optimization import Optimizer, Route
from synthrouting.vectorfields import Swirlys
from synthrouting.vectorfields.base import Vectorfield


def batch_bezier(t: jnp.ndarray, control: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Evaluate a batch of N-dimensional Bézier curves.

    Parameters
    ----------
    t : np.ndarray
        Evaluation points, which should be a vector with a shape that matches the number
        of control points (n_controls). Each value must be between 0 and 1.
    control : np.ndarray
        A batched matrix of control points with shape (B, C, n), where B is the batch size,
        C is the number of control points per curve, and N is the number of dimensions.

    Returns
    -------
    np.ndarray
        A batch of evaluated Bézier curves with shape (B, C, n).
    """
    # kawrgs are used because other bezier functions may have additional parameters

    if t.ndim == 1:
        t = t[None, :]
    assert control.ndim == 3

    control = jnp.tile(control[:, :, None, :], [1, 1, t.shape[1], 1])
    while control.shape[1] > 1:
        control = (1 - t[:, None, :, None]) * control[:, :-1, :, :] + t[
            :, None, :, None
        ] * control[:, 1:, :, :]
    return control[:, 0, ...]


class CMAOptimizer(Optimizer):

    def __init__(
        self,
        vectorfield: Vectorfield,
        num_control_points: int = 6,
        num_segments: int = 64,
        popsize: int = 200,
        sigma0: float = 2.0,
        timeout: Optional[int] = None,
        maxfevals: Optional[int] = None,
        tolfun: float = 1e-4,
        verb_disp: int = 1,
        seed: int = 0,
    ):
        super().__init__(vectorfield=vectorfield)
        self.popsize = popsize
        self.num_control_points = num_control_points
        self.num_segments = num_segments
        self.sigma0 = sigma0
        self.timeout = timeout
        self.maxfevals = maxfevals
        self.tolfun = tolfun
        self.verb_disp = verb_disp
        self.seed = seed

    def evaluate_candidates(self, pts: jnp.array) -> jnp.array:
        # Times for Bézier curve
        t = jnp.linspace(0, 1, self.num_segments)
        # Unflatten control points (popsize, num_control_points, 2)
        pts = jnp.reshape(pts, (self.popsize, self.num_control_points, 2))
        xy = batch_bezier(t[1:-1], pts)  # (popsize, num_segments, 2)
        # Change to (2, num_segments, popsize)
        xy = jnp.transpose(xy, (2, 1, 0))
        # Add p0 and pn to form (2, num_segments + 2, popsize)
        # First replicate p0 and pn
        p0 = jnp.tile(self.p0[:, None], (1, self.popsize))
        pn = jnp.tile(self.pn[:, None], (1, self.popsize))
        xy = jnp.concatenate((p0[:, None, :], xy, pn[:, None, :]), axis=1)
        # Distance of every segment
        dist = self.geometry.dist_p0_to_p1(xy[:, :-1], xy[:, 1:])
        # (num_segments - 1, popsize)

        # Angle of every segment
        theta = self.geometry.angle_p0_to_p1(xy[:, :-1], xy[:, 1:])
        # (num_segments - 1, popsize)

        # Re-scale times
        t = t * self.tend
        # Time delta
        dt = jnp.diff(t)  # (num_segments - 1,)
        dt = jnp.tile(dt[:, None], (1, self.popsize))

        # Speed Over Ground
        sog = dist / dt
        # Components of SOG
        sogx = sog * jnp.cos(theta)
        sogy = sog * jnp.sin(theta)

        # Speed of currents
        wx, wy = self.vectorfield.get_current(xy[0, :-1], xy[1, :-1])
        # (num_segments - 1, popsize)

        # Speed Through Water
        # Components of STW
        stwx = sogx - wx
        stwy = sogy - wy
        # Module of STW
        # stw = jnp.sqrt(stwx**2 + stwy**2)

        # Cost of every candidate
        cost = (dt * (stwx**2 + stwy**2) / 2).sum(axis=0)  # (popsize,)
        # Velocities must be closer to 1
        # cost = ((stw - 1) ** 2).sum(axis=0)
        # Penalize going out of bounds
        # cost += (xy[:, 1:-1] > 6).sum(axis=(0, 1))
        # cost += (xy[:, 1:-1] < -1).sum(axis=(0, 1))
        return xy, cost

    def optimize_route(
        self, p0: Tuple[float], pn: Tuple[float], tend: jnp.float32 = 10
    ) -> Route:
        self.tend = tend
        self.p0 = jnp.array(p0)
        self.pn = jnp.array(pn)

        # Initial guess is random
        pt0 = random.uniform(
            random.PRNGKey(self.seed),
            (2 * self.num_control_points,),
            minval=-10,
            maxval=10,
        )

        es = cma.CMAEvolutionStrategy(
            pt0,
            self.sigma0,
            inopts={
                "popsize": self.popsize,
                "timeout": self.timeout,
                "maxfevals": self.maxfevals,
                "tolfun": self.tolfun,
                "bounds": None,
                "verb_disp": self.verb_disp,
                "seed": self.seed,
            },
        )
        while not es.stop():
            X = es.ask()  # sample `popsize` candidate solutions
            # Score the routes and apply the evolution strategy
            xy, cost = self.evaluate_candidates(jnp.array(X))
            es.tell(X, cost.tolist())
            es.disp()

        # Take the best candidate
        idx_best = jnp.argmin(cost)
        self.cost_min = cost[idx_best]
        x_best = xy[0, :, idx_best]
        y_best = xy[1, :, idx_best]
        t = jnp.linspace(0, tend, self.num_segments)
        route_best = Route(x_best, y_best, t, geometry=self.geometry)
        return route_best


def main():
    vectorfield = Swirlys()

    x_start, y_start = 0, 0
    x_end, y_end = 6, 5
    tend = 30

    optimizer = CMAOptimizer(vectorfield)
    route = optimizer.optimize_route((x_start, y_start), (x_end, y_end), tend=tend)

    # Round limits
    xmin = int(route.x.min()) - 1
    xmax = int(route.x.max()) + 1
    ymin = int(route.y.min()) - 1
    ymax = int(route.y.max()) + 1

    vectorfield.plot(extent=(xmin, xmax, ymin, ymax), step=0.5)
    plt.plot(route.x, route.y, color="green", linestyle="--", alpha=0.7)
    plt.scatter([x_start, x_end], [y_start, y_end], color="red")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.gca().set_aspect("equal")
    plt.title(f"Total fuel: {optimizer.cost_min:.2f}")
    plt.savefig("output/bezier.png")


if __name__ == "__main__":
    typer.run(main)
