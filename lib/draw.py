from agent import PDAgent
import matplotlib.cm as cm
import matplotlib as matplotlib


def color_map_color(value, cmap_name='cool', vmin=0, vmax=1):
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
    if agent.model.max_score > 0:
        r = agent.score / agent.model.max_score
    else:
        r = 0.5
    color = color_map_color(agent.defecting_ratio)
    return {
        "Shape": "circle",
        "r": r,
        "Filled": "true",
        "Layer": 0,
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Color": color,
    }
