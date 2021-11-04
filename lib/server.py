from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from draw import draw_agent
from model import PDModel
from config import *
import numpy as np

model_params = {
    "width": DEFAULT_WIDTH,
    "height": DEFAULT_HEIGHT,
    "seed": MESA_SEED,
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
        return f"Cooperating agents: {model.num_cooperating_agents / len(model.agents) * 100:.2f}%"


pd_elem = PDElement()
score_chart = ChartModule([
    # {"Label": "Cooperating_Agents", "Color": "Blue"},
    # {"Label": "Total_Scores", "Color": "Black"},
    {"Label": "Mean_Score", "Color": "Orange"},
    {"Label": "Max_Score", "Color": "Red"},
    {"Label": "Min_Score", "Color": "Green"},
], data_collector_name='data_collector')

cooperation_chart = ChartModule([
    {"Label": "Cooperating_Agents", "Color": "Black"},
], data_collector_name='data_collector')

hist_elem = HistogramModule(200, 500)

server = ModularServer(PDModel, [canvas_element, pd_elem, score_chart, hist_elem, cooperation_chart], "Evolution of Prisoner's Dilemma", model_params)

if __name__ == '__main__':
    server.launch(port=8080, open_browser=False)
