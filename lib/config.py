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
USE_LOCAL_GA = True  # weather to use the localized version of GA
FITNESS_MULTIPLIER = 2  # parameter used for scaling fitness function, see P15 of the java manual
EPS = 1e-3
MUT_PROB = 5e-3
MUT_STRENGTH = 1

MEM_LEN = 3

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
TORUS_GRID = True
CANVAS_DX = 30

MESA_SEED = 3
NUMPY_SEED = 3

VISUALIZE_GRID_TYPE = 'agent_type'  # 'agent_type' or 'defecting_ratio' or 'inherited_attr'


def config_to_str():
    return '\n'.join([
        f"{PAYOFF_MAP=}",
        f"{NEIGHBOR_TYPE=}",
        f"{NEIGHBOR_RADIUS=}",
        f"{NUM_SUBSTEPS=}",
        f"{FITNESS_MULTIPLIER=}",
        f"{MUT_PROB=}",
        f"{MEM_LEN=}",
        f"{MUT_STRENGTH=}",
        f"{DEFAULT_WIDTH=}",
        f"{DEFAULT_HEIGHT=}",
        f"{TORUS_GRID=}",
        f"{CANVAS_DX=}",
        f"{MESA_SEED=}",
        f"{NUMPY_SEED=}",
        f"{VISUALIZE_GRID_TYPE=}",
    ])
