import numpy as np
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement

from lib.config import *
from lib.draw import draw_agent
from lib.strategies import *
from lib.model import PDModel


def agent_type_map1(x, y):
    if x == 4 or x == 0:
        return 'tit_for_tat'
    if x == 5 or x == 9:
        return 'bad'
    return 'string'


def agent_type_map2(x, y):
    return 'string', {'starting_action': 0}
    # if (x, y) in [(3, 4), (4, 4), (5, 4), (6, 4)]:
    #     # return 'string_tit_for_tat', {'starting_action': 0 if x < 3 else 1}
    #     return 'string', {'starting_action': '0'}
    # else:
    #     return 'string', {'starting_action': '1'}


model_params = {
    "width": 10,
    "height": 9,
    "seed": MESA_SEED,
    # "agent_type_map": agent_type_map2,
    "num_substeps": UserSettableParameter(
        "slider",
        "Number of steps in each generation",
        value=NUM_SUBSTEPS,
        min_value=1,
        max_value=50,
    ),
    "fitness_type": UserSettableParameter(
        "choice",
        "Fitness function type",
        value="score",
        choices=['score', 'cooperating_ratio', 'defecting_ratio'],
    ),
    "agent_type": UserSettableParameter(
        "choice",
        "Agent type",
        value="neural",
        choices=['neural', 'string', 'tit_for_tat', 'simple', 'mixed'],
    ),
    "neighbor_type": UserSettableParameter(
        "choice",
        "Neighbor type",
        value=NEIGHBOR_TYPE,
        choices=[8, 4, 2],
    ),
}

canvas_element = CanvasGrid(
    draw_agent, model_params['width'], model_params['height'],
    model_params['width'] * CANVAS_DX, model_params['height'] * CANVAS_DX
)


class HistogramModule(VisualizationElement):
    """
    To visualize the defecting ratio distribution
    """
    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]

    def __init__(self, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = [i / 8 for i in range(9)]  # all possible defecting_ratios
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(self.bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        ratios = [agent.defecting_ratio for agent in model.schedule.agents]
        hist = np.histogram(ratios, bins=len(self.bins))[0]
        return [int(x) for x in hist]


class PDElement(TextElement):
    """
    Display a text counts.
    """

    def __init__(self):
        pass

    def render(self, model: PDModel):
        fittest = max(model.agents, key=lambda a: a.fitness)
        appendix = ''
        if isinstance(fittest, StringAgent):
            appendix = ' Fittest chromosome: ' + fittest.chromosome_str()
        if isinstance(fittest, NeuralAgent):
            f = fittest.feature_vector()
            appendix = f' Fittest feature vector: {f[0]:.4f}, {f[1]:.4f}'
        return f"Generation: {model.generations}" + appendix


pd_elem = PDElement()
score_chart = ChartModule([
    # {"Label": "Cooperating_Agents", "Color": "Blue"},
    {"Label": "Mean_Score", "Color": "Orange"},
    {"Label": "Max_Score", "Color": "Red"},
    {"Label": "Min_Score", "Color": "Green"},
], data_collector_name='data_collector')

agent_stat_chart = ChartModule([
    {"Label": "Cooperating_Agents", "Color": "Black"},
    {"Label": "Simple_Agents", "Color": "Blue"},
    {"Label": "Tit_for_tat_Agents", "Color": "Green"},
    {"Label": "Neural_Agents", "Color": "Orange"},
], data_collector_name='data_collector')

hist_elem = HistogramModule(200, 500)

if NUMPY_SEED is not None:
    np.random.seed(NUMPY_SEED)

server = ModularServer(PDModel, [canvas_element, pd_elem, agent_stat_chart, score_chart], "Evolution of Prisoner's Dilemma", model_params)
