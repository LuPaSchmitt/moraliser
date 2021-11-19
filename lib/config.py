# 0 cooperating, 1 defecting
# Map (my_action, other_action) to payoff
PAYOFF_MAP = [
    [3, 0],
    [5, 1],
]

NEIGHBOR_TYPE = 8  # 8, 4, or 2
NEIGHBOR_RADIUS = 1

NUM_SUBSTEPS = 10  # substeps in a generation

# GA parameters
MUT_PROB = 0.7
MUT_STRENGTH = 1

DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 10
TORUS_GRID = False
CANVAS_DX = 30

MESA_SEED = 2
NUMPY_SEED = 3

VISUALIZE_GRID_TYPE = 'agent_type'  # 'agent_type' or 'defecting_ratio'


def config_to_str():
    return '\n'.join([
        f"{PAYOFF_MAP=}",
        f"{NEIGHBOR_TYPE=}",
        f"{NEIGHBOR_RADIUS=}",
        f"{NUM_SUBSTEPS=}",
        f"{MUT_PROB=}",
        f"{MUT_STRENGTH=}",
        f"{DEFAULT_WIDTH=}",
        f"{DEFAULT_HEIGHT=}",
        f"{TORUS_GRID=}",
        f"{CANVAS_DX=}",
        f"{MESA_SEED=}",
        f"{NUMPY_SEED=}",
        f"{VISUALIZE_GRID_TYPE=}",
    ])
