from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.utils.plot import plot_ticks_radians_to_degrees
from hybrid_routing.vectorfields import VectorfieldReal

list_vf = [
    dict(
        p0=(-79.7, 32.7),
        pn=(-29.5, 38.5),
        key="real",
        path="./data",
        si_units=True,
    ),
    dict(
        p0=(43.49, -1.66),
        pn=(98.14, 10.21),
        key="real-land",
        path="./data",
        si_units=True,
    ),
]

path_data = "./data"
path_out = Path("output/")

for dict_vf in list_vf:
    p0 = np.array(dict_vf["p0"]) * DEG2RAD
    pn = np.array(dict_vf["pn"]) * DEG2RAD
    key = dict_vf["key"]
    vf = VectorfieldReal.from_folder(dict_vf["path"], key, radians=dict_vf["si_units"])
    vf.plot(step=DEG2RAD, do_color=True)
    plot_ticks_radians_to_degrees(step=5)
    plt.scatter(p0[0], p0[1], s=20, c="y")
    plt.scatter(pn[0], pn[1], s=20, c="r")
    plt.tight_layout()
    plt.savefig(path_out / f"{key}_vectorfield.png")
    plt.close()
