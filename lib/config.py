# 0 cooperating, 1 defecting
# Map (my_action, other_action) to payoff
PAYOFF_MAP = [
    [3, 0],
    [5, 1],
]

NEIGHBOR_TYPE = 2  # 8, 4, or 2
NEIGHBOR_RADIUS = 1

NUM_SUBSTEPS = 20  # substeps in a generation

# GA parameters
MUT_PROB = 0.2

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 1
TORUS_GRID = True
CANVAS_DX = 30

MESA_SEED = None
NUMPY_SEED = None

VISUALIZE_GRID_TYPE = 'agent_type'  # 'agent_type' or 'defecting_ratio'
