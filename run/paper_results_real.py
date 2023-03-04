"""
Generate all the figures used in the paper. Results section
"""

from hybrid_routing.pipeline import run_pipelines

run_pipelines(
    "real",
    path_config="data/config.toml",
    path_out="output",
    max_thread=6,
)
