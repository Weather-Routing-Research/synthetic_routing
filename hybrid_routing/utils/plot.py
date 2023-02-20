from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pylab import cm

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.optimization import Route
from hybrid_routing.vectorfields.base import Vectorfield


def plot_ticks_radians_to_degrees(
    step: float = 5, extent: Optional[List[float]] = None
):
    """Changes the ticks from radians to degrees

    Parameters
    ----------
    step : float, optional
        Step in degrees, by default 5
    extent : List[float], optional
        Extent of the ticks, by default the same as the graph
    """
    ax: Axes = plt.gca()
    if extent is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    else:
        xmin, xmax, ymin, ymax = extent
    # Radians to degrees
    xticks = np.arange(xmin, xmax, step * DEG2RAD)
    plt.xticks(xticks, labels=[f"{i:.1f}" for i in xticks / DEG2RAD])
    yticks = np.arange(ymin, ymax, step * DEG2RAD)
    plt.yticks(yticks, labels=[f"{i:.1f}" for i in yticks / DEG2RAD])
    plt.yticks(yticks, labels=[f"{i:.1f}" for i in yticks / DEG2RAD])


def plot_textbox(
    p0: Tuple[float],
    pn: Tuple[float],
    t: Tuple[float],
    pos: Optional[Tuple[float]] = None,
    align: str = "top",
):
    """Add a textbox indicating the starting and end position,
    and the times computed by the ZIVP and DNJ.

    Parameters
    ----------
    p0 : Tuple[float]
        Initial position (x, y)
    pn : Tuple[float]
        End position (x, y)
    t : Tuple[float]
        Times for ZIVP and DNJ respectively
    pos : Tuple[float], optional
        Textbox position, by default (0, 0)
    align : str, optional
        Textbox alignment, by default "top"
    """
    x0, y0 = p0
    xn, yn = pn
    topt, tdnj = t
    # Textbox properties
    dict_bbox = dict(boxstyle="round", facecolor="white", alpha=0.95)
    text = (
        r"$\left\langle x_0, y_0 \right\rangle = \left\langle"
        + "{:.1f}".format(x0)
        + ", "
        + "{:.1f}".format(y0)
        + r"\right\rangle$"
        + "\n"
        r"$\left\langle x_T, y_T \right\rangle = \left\langle"
        + "{:.1f}".format(xn)
        + ", "
        + "{:.1f}".format(yn)
        + r"\right\rangle$"
        + "\nOptimized (red):\n"
        + f"  t = {topt:.3f}\n"
        + "Smoothed (black):\n"
        + f"  t = {tdnj:.3f}"
    )
    # Textbox position, by default at the bottom left corner
    if pos is None:
        ax: Axes = plt.gca()
        pos = (ax.get_xlim()[0], ax.get_ylim()[0])
    plt.text(
        pos[0],
        pos[1],
        text,
        fontsize=11,
        verticalalignment=align,
        bbox=dict_bbox,
    )


def plot_routes(
    list_route: List[Route],
    vectorfield: Vectorfield,
    labels: Optional[List[str]] = None,
    vel: Optional[int] = None,
    step: int = 1,
    legend: bool = True,
    **kwargs,
):
    """Plot a list of routes, adding its time and distance to the legend. If the
    vector field is real, both the vector field and the routes are assumed to have
    SI units (meters, seconds, radians).

    Parameters
    ----------
    list_route : List[Route]
        List of routes to plot
    vectorfield : Vectorfield
        Vectorfield where the routes are plotted on
    vel : Optional[int], optional
        Velocity to recompute times, by default None
    legend : bool, optional
        Add the legend with times and distances, by default True
    """
    # If the vector field is real, we will use SI units
    si_units = str(vectorfield.geometry).lower() == "spherical"
    if si_units:
        step = step * DEG2RAD
        prop_time = 1 / 3600
        prop_dist = 1 / 1000
    else:
        step = step
        prop_time = 1
        prop_dist = 1

    # Limits of the vector field
    xmin = min([min(r.x) for r in list_route])
    xmax = max([max(r.x) for r in list_route])
    ymin = min([min(r.y) for r in list_route])
    ymax = max([max(r.y) for r in list_route])

    vectorfield.plot(
        x_min=xmin - 2 * step,
        x_max=xmax + 2 * step,
        y_min=ymin - 2 * step,
        y_max=ymax + 2 * step,
        step=step,
        do_color=True,
        **kwargs,
    )

    # Use a scale of colors distinct from the color map
    colors = cm.gist_heat(np.linspace(0.1, 0.9, len(list_route)))

    # Loop over and plot the routes
    for idx, route in enumerate(list_route):
        if vel:
            route.recompute_times(vel, vectorfield)
        time = route.t[-1] * prop_time
        dist = sum(route.d) * prop_dist

        # Plot route
        if si_units:
            label = f"{dist:.2f} km | {time:.2f} h"
        else:
            label = f"dist = {dist:.2f} | t = {time:.2f}"
        if labels:
            label = labels[idx] + " | " + label
        plt.plot(route.x, route.y, color=colors[idx], linewidth=3, label=label)

    if si_units:
        plot_ticks_radians_to_degrees(step=5)

    if legend:
        legend = plt.legend()
        # get the width of your widest label, since every label will need
        # to shift by this amount after we align to the right
        # https://stackoverflow.com/questions/7936034/text-alignment-in-a-matplotlib-legend
        max_shift = max([t.get_window_extent().width for t in legend.get_texts()])
        for t in legend.get_texts():
            t.set_ha("right")  # ha is alias for horizontalalignment
            temp_shift = max_shift - t.get_window_extent().width
            t.set_position((temp_shift, 0))
