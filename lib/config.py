# 0 cooperating, 1 defecting
# Map (my_action, other_action) to payoff
PAYOFF_MAP = [
    [3, 0],
    [5, 1],
]

NEIGHBOR_TYPE = 4  # 8, 4, or 2
NEIGHBOR_RADIUS = 1

NUM_SUBSTEPS = 10  # substeps in a generation

# GA parameters
MUT_PROB = 0.9
MUT_STRENGTH = 1

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
TORUS_GRID = False
CANVAS_DX = 30

MESA_SEED = 3
NUMPY_SEED = 3

VISUALIZE_GRID_TYPE = 'agent_type'  # 'agent_type' or 'defecting_ratio'
