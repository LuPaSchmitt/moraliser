from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from draw import draw_agent
from model import PDModel
from config import *

model_params = {
    "width": DEFAULT_WIDTH,
    "height": DEFAULT_HEIGHT,
}

canvas_element = CanvasGrid(draw_agent, model_params['width'], model_params['height'], 500, 500)


class PDElement(TextElement):
    """
    Display a text counts.
    """

    def __init__(self):
        pass

    def render(self, model: PDModel):
        return f"Cooperating agents: {model.num_cooperating_agents}"


pd_elem = PDElement()
score_chart = ChartModule([
    {"Label": "Cooperating_Agents", "Color": "Blue"},
    {"Label": "Total_Scores", "Color": "Black"},
], data_collector_name='data_collector')

server = ModularServer(PDModel, [canvas_element, pd_elem, score_chart], "Evolution of Prisoner's Dilemma", model_params)

if __name__ == '__main__':
    server.launch()
