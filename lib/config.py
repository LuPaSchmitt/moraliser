# 0 cooperating, 1 defecting
# Map (my_action, other_action) to payoff
PAYOFF_MAP = [
    [3, 0],
    [5, 1],
]

# Neighbors of each agent: can be 8 (everyone around), 4 (left right top bottom), or 2 (left right)
NEIGHBOR_TYPE = 4
NEIGHBOR_RADIUS = 1  # interaction radius

NUM_SUBSTEPS = 10  # rounds in a generation

# Genetic Algorithm parameters
USE_LOCAL_GA = True  # weather to use the localized version of GA
FITNESS_MULTIPLIER = 2  # parameter used for scaling fitness function, see P15 of the java manual
EPS = 1e-3  # for fitness scaling
MUT_PROB = 0.2  # mutation probability for NeuralAgent and StringAgent
MUT_STRENGTH = 1.0  # how strong to perturb the weights of NeuralAgent
TFT_REPRODUCABLE = False  # does TFT participate in the reproduction
NEURAL_REPRODUCABLE = True  # does NeuralAgent participate in the reproduction
STRING_REPRODUCABLE = True  # does StringAgent participate in the reproduction

# Network topology for NeuralAgent
# e.g. [6, 1] means a network with 6 inputs (i.e. memory size 3), fully connected to the output
DEFAULT_NEURAL_STRUCTURE = [6, 1]

# StringAgent's memory size
MEM_LEN = 2

# Settings for PDModel
DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 10
TORUS_GRID = True
CANVAS_DX = 30

# Random seeds
MESA_SEED = 4
NUMPY_SEED = 3

# The way agents are rendered in the Web demo, must be one of the followings:
# 'agent_type': the colors indicate which type of agent it is, i.e. NeuralAgent, StringAgent, TitForTatAgent, ...
# 'defecting_ratio': visualize how defecting agents are, using a heatmap
# 'inherited_attr': visualize who are the offsprings from whom, a way to understand the genetic algorithm
VISUALIZE_GRID_TYPE = 'agent_type'


def config_to_str():
    return '\n'.join([
        f"{PAYOFF_MAP=}",
        f"{NEIGHBOR_TYPE=}",
        f"{NEIGHBOR_RADIUS=}",
        f"{NUM_SUBSTEPS=}",
        f"{FITNESS_MULTIPLIER=}",
        f"{MUT_PROB=}",
        f"{DEFAULT_NEURAL_STRUCTURE=}",
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
