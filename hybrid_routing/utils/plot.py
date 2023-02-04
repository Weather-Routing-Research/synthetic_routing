from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_routing.geometry import DEG2RAD


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
    pos: Tuple[float] = (0, 0),
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
        + "{:.1f}".format(x0 / DEG2RAD)
        + ", "
        + "{:.1f}".format(y0 / DEG2RAD)
        + r"\right\rangle$"
        + "\n"
        r"$\left\langle x_T, y_T \right\rangle = \left\langle"
        + "{:.1f}".format(xn / DEG2RAD)
        + ", "
        + "{:.1f}".format(yn / DEG2RAD)
        + r"\right\rangle$"
        + "\nOptimized (red):\n"
        + f"  t = {topt:.3f}\n"
        + "Smoothed (black):\n"
        + f"  t = {tdnj:.3f}"
    )
    plt.text(
        pos[0],
        pos[1],
        text,
        fontsize=11,
        verticalalignment=align,
        bbox=dict_bbox,
    )
