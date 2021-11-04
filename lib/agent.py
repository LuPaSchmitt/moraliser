from abc import abstractmethod

from mesa import Agent

from config import *


class PDAgent(Agent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, unique_id, model):
        """
        Create a new Prisoner's Dilemma agent.
        """
        super().__init__(unique_id, model)

        self.score = 0  # total scores
        self.cur_score = 0
        self.neighbors = []
        self.action = {}  # current action, maps other agent's id to my action with it
        self.next_action = {}  # for advancing
        # TODO: more efficient implementation of action table, currently the dictionary lookup takes too much time

    def initialize(self, starting_action=None):
        """
        Called before model runs
        """
        self.score = 0
        # Assuming that neighbors don't change throughout the game
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=NEIGHBOR_TYPE == 'moore', include_center=False, radius=NEIGHBOR_RADIUS)
        if isinstance(starting_action, int):
            self.action = {other.unique_id: starting_action for other in self.neighbors}
        else:
            actions = self.random.choices([0, 1], k=len(self.neighbors))
            self.action = {other.unique_id: action for other, action in zip(self.neighbors, actions)}

    @abstractmethod
    def clone(self):
        """
        Clone the agent into a new instance.
        Notice that the cloned instance is uninitialized, loses memory and position information of the old, and is not managed by model.scheduler
        This function is useful for genetic algorithm
        """
        pass

    def mutate(self):
        """
        Possibly mutate the agent. Derived classes can have different mutation policies
        This function is needed for genetic algorithm
        """
        pass

    @property
    def defecting_ratio(self):
        return sum(self.action.values()) / len(self.neighbors)

    @property
    def cooperating_ratio(self):
        return 1 - self.defecting_ratio

    @property
    def is_cooperating(self):
        return self.defecting_ratio <= 0.5

    @property
    def is_defecting(self):
        return self.defecting_ratio > 0.5

    @abstractmethod
    def make_action(self, other: Agent):
        """
        Make an action with the opponent `other`
        Derived classes can make actions w.r.t. different strategies, e.g. Tit-for-tat, Neural Networks, ...
        :return: 0 - cooperating or 1 - defecting
        """
        pass

    def step(self):
        """
        Update actions with all my neighbors
        """
        self.next_action = {}
        for other in self.neighbors:
            self.next_action[other.unique_id] = self.make_action(other)

    def advance(self):
        self.action = self.next_action

    def update_scores(self):
        self.cur_score = sum(PAYOFF_MAP[self.action[other.unique_id]][other.action[self.unique_id]] for other in self.neighbors)
        self.score += self.cur_score

    def feedback(self):
        """
        Called after each tournament
        Derived classes can, e.g., update histories or mutate genes in the feedback function
        """
        pass
