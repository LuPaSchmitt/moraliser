from agent import PDAgent


def draw_agent(agent: PDAgent):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the agent in its current state.
    :param agent:  the agent in the simulation
    :return: the portrayal dictionary
    """
    assert agent is not None
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Color": "blue" if agent.is_cooperating else "red",
    }
