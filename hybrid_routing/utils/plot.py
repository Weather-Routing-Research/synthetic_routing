from typing import List, Optional

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
