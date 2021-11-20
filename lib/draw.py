import matplotlib as matplotlib
import matplotlib.cm as cm

from lib.agent import PDAgent
from lib.config import VISUALIZE_GRID_TYPE
from lib.strategies import *


def color_map(value, cmap_name='cool', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def draw_agent(agent: PDAgent):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the agent in its current state.
    :param agent:  the agent in the simulation
    :return: the portrayal dictionary
    """
    if agent.model.f_max <= 0 or agent.fitness <= 0:
        r = 0.8
    else:
        r = agent.fitness / agent.model.f_max

    if VISUALIZE_GRID_TYPE == 'defecting_ratio':
        color = color_map(agent.defecting_ratio)
    elif VISUALIZE_GRID_TYPE == 'agent_type':
        if isinstance(agent, SimpleAgent):
            color = 'Blue'
        elif isinstance(agent, TitForTatAgent):
            color = 'Orange'
        elif isinstance(agent, GoodAgent):
            color = 'Green'
        elif isinstance(agent, BadAgent):
            color = 'Red'
        elif isinstance(agent, NeuralAgent):
            # color = 'Orange'
            color = color_map(agent.feature_vector()[0])
            # color = agent.inherited_attr
        elif isinstance(agent, StringAgent):
            color = color_map(agent.defecting_ratio)
        else:
            color = 'Black'
    elif VISUALIZE_GRID_TYPE == 'inherited_attr':
        color = agent.inherited_attr
    else:
        color = 'Black'

    return {
        "Shape": "circle",
        "r": r,
        "Filled": "true",
        "Layer": 0,
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Color": color,
    }
