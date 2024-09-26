from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthrouting.geometry import DEG2RAD
from synthrouting.utils.config import load_config
from synthrouting.utils.plot import plot_ticks_radians_to_degrees
from synthrouting.vectorfields import VectorfieldReal
from synthrouting.vectorfields.base import Vectorfield

path_cfg = "data/config.toml"

list_benchmark = (
    load_config(path_cfg, "synthetic").tolist() + load_config(path_cfg, "real").tolist()
)

path_data = "./data"
path_out = Path("output/")

for dict_vf in list_benchmark:
    key = dict_vf["key"]
    try:
        vf = VectorfieldReal.from_folder(
            dict_vf["path"], key, radians=dict_vf["si_units"]
        )
        p0 = np.array(dict_vf["p0"]) * DEG2RAD
        pn = np.array(dict_vf["pn"]) * DEG2RAD
        vf.plot(step=DEG2RAD, do_color=True)
        plot_ticks_radians_to_degrees(step=6)
    except KeyError:
        module = import_module("hybrid_routing.vectorfields")
        vf: Vectorfield = getattr(module, key)()
        p0 = np.array(dict_vf["p0"])
        pn = np.array(dict_vf["pn"])
        vf.plot(step=0.5, extent=dict_vf["plot"]["extent"])
    plt.scatter(p0[0], p0[1], s=50, c="black")
    plt.scatter(p0[0], p0[1], s=25, c="y")
    plt.scatter(pn[0], pn[1], s=50, c="black")
    plt.scatter(pn[0], pn[1], s=25, c="r")
    plt.tight_layout()
    plt.savefig(path_out / f"{key}_vectorfield.png")
    plt.close()
