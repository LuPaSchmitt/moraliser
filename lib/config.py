# 0 cooperating, 1 defecting
# Map (my_action, other_action) to payoff
PAYOFF_MAP = [
    [3, 0],
    [5, 1],
]

NEIGHBOR_TYPE = 'moore'
# NEIGHBOR_TYPE = 'von neumann'
NEIGHBOR_RADIUS = 1

NUM_SUBSTEPS = 20  # substeps in a generation

# GA parameters
MUT_PROB = 0.2

DEFAULT_WIDTH = 15
DEFAULT_HEIGHT = 3
TORUS_GRID = True
CANVAS_DX = 40

MESA_SEED = None
NUMPY_SEED = None
