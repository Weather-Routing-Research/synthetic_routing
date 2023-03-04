from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.geometry import DEG2RAD
from hybrid_routing.utils.config import load_config
from hybrid_routing.utils.plot import plot_ticks_radians_to_degrees
from hybrid_routing.vectorfields import VectorfieldReal

config = load_config("data/config.toml", "real")
list_benchmark = config.tolist()

path_data = "./data"
path_out = Path("output/")

for dict_vf in list_benchmark:
    p0 = np.array(dict_vf["p0"]) * DEG2RAD
    pn = np.array(dict_vf["pn"]) * DEG2RAD
    key = dict_vf["key"]
    vf = VectorfieldReal.from_folder(dict_vf["path"], key, radians=dict_vf["si_units"])
    vf.plot(step=DEG2RAD, do_color=True)
    plot_ticks_radians_to_degrees(step=6)
    plt.scatter(p0[0], p0[1], s=20, c="y")
    plt.scatter(pn[0], pn[1], s=20, c="r")
    plt.tight_layout()
    plt.savefig(path_out / f"{key}_vectorfield.png")
    plt.close()
