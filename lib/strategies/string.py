from collections import deque
from typing import Dict

import numpy as np
from mesa import Agent

from lib.agent import PDAgent
from lib.config import *


class StringAgent(PDAgent):
    """
    String based strategy encoding, refer to https://github.com/aerrity/prisoners-dilemma manual section 4.2
    """

    def __init__(self, unique_id, model, mem_len=MEM_LEN, mut_prob=MUT_PROB, chromosome=None, starting_action=None):
        super().__init__(unique_id, model)
        self.mem_len = mem_len  # Maximum memory length
        self.prev_actions: Dict[int, deque] = {}  # Map other's id to its previous action tuples
        # id -> a ring queue(maxlen = mem_len) of (my action, other action)

        # 1st, 2nd, ..., mem_len(th) move: strategy tree -> 1 + 2 + ... + 2^(mem_len - 1) bits
        # For later moves, the state space is as large as 4^mem_len
        self.init_move_bits = 2 ** mem_len - 1
        self.later_move_bits = 4 ** mem_len
        self.later_move_mask = self.later_move_bits - 1
        if chromosome is not None:
            assert isinstance(chromosome, list)
            assert len(chromosome) == self.init_move_bits + self.later_move_bits
            self.chromosome = chromosome.copy()
        else:
            self.chromosome = [0] * (self.init_move_bits + self.later_move_bits)
            if starting_action is not None:
                assert starting_action in [0, 1]
                self.chromosome[0] = starting_action

        # The agent behaves like an FSM. In each feedback it moves to a new state, and the action is made based on its instant state
        # current_state points to the current action index in the chromosome
        self.current_states: Dict[int, int] = {}

        self.mutate_probability = mut_prob

    def reproducable(self) -> bool:
        return STRING_REPRODUCABLE

    def clone(self):
        return StringAgent(self.unique_id, self.model, self.mem_len)

    def random_chromosome(self):
        self.chromosome = self.random.choices([0, 1], k=len(self.chromosome))

    def initialize(self, neighbor_type, starting_action=None):
        if starting_action is None:  # not random anymore
            starting_action = self.chromosome[0]
        super().initialize(neighbor_type, starting_action)
        self.prev_actions = {other.unique_id: deque(maxlen=self.mem_len) for other in self.neighbors}
        self.current_states = {other.unique_id: 0 for other in self.neighbors}

    def make_action(self, other: Agent):
        return self.chromosome[self.current_states[other.unique_id]]

    def feedback(self):
        # Update history and state
        for other in self.neighbors:
            my_action, other_action = self.action[other.unique_id], other.action[self.unique_id]
            history, state = self.prev_actions[other.unique_id], self.current_states[other.unique_id]
            if len(history) < self.mem_len:
                assert state < self.init_move_bits
                state = state * 2 + 1 + other_action  # walk in the initial strategy tree
            else:
                assert self.init_move_bits <= state < self.init_move_bits + self.later_move_bits
                state -= self.init_move_bits
                state = (state << 2) & self.later_move_mask | ((my_action << 1) | other_action)
                state += self.init_move_bits
            old_hist_len = len(history)
            history.append((my_action, other_action))
            if old_hist_len == self.mem_len - 1:  # into later strategy space
                state = 0
                for m, o in history:
                    state = (state << 2) | ((m << 1) | o)
                state += self.init_move_bits
            self.current_states[other.unique_id] = state

    def mutate(self):
        for position in range(len(self.chromosome)):
            if np.random.random() < self.mutate_probability:
                self.chromosome[position] ^= 1

    def cross(self, other: PDAgent):
        assert type(self) == type(other), f"{type(self)} cannot cross with {type(self)} agent"
        assert len(self.chromosome) == len(other.chromosome)
        c1, c2 = self.clone(), other.clone()
        c1.inherited_attr = self.inherited_attr
        c2.inherited_attr = other.inherited_attr
        position = np.random.randint(0, len(self.chromosome))
        c1.chromosome[:position] = self.chromosome[:position]
        c1.chromosome[position:] = other.chromosome[position:]
        c2.chromosome[:position] = other.chromosome[:position]
        c2.chromosome[position:] = self.chromosome[position:]

        return c1, c2

    def chromosome_str(self):
        s = ''.join(map(str, self.chromosome))
        return s[:self.init_move_bits] + ' ' + s[self.init_move_bits:]

    @staticmethod
    def create_tit_for_tat(unique_id, model, mem_len=MEM_LEN, mut_prob=MUT_PROB, starting_action=None):
        a = StringAgent(unique_id, model, mem_len, mut_prob)
        if starting_action is not None:
            assert starting_action in [0, 1]
            a.chromosome[0] = starting_action
        for i in range(1, a.init_move_bits):
            a.chromosome[i] = int(not (i & 1))
        for i in range(a.init_move_bits, a.init_move_bits + a.later_move_bits):
            a.chromosome[i] = (i - a.init_move_bits) & 1
        return a

    @staticmethod
    def create_always_cooperating(unique_id, model, mem_len=MEM_LEN, mut_prob=MUT_PROB):
        a = StringAgent(unique_id, model, mem_len, mut_prob)
        a.chromosome = [0] * len(a.chromosome)
        return a

    @staticmethod
    def create_always_defecting(unique_id, model, mem_len=MEM_LEN, mut_prob=MUT_PROB):
        a = StringAgent(unique_id, model, mem_len, mut_prob)
        a.chromosome = [1] * len(a.chromosome)
        return a


if __name__ == '__main__':
    from mesa import Model

    m = Model()


    def test1():
        a = StringAgent.create_tit_for_tat(0, m, mem_len=1)
        b = StringAgent(1, m, mem_len=1)
        assert len(a.chromosome) == len(b.chromosome) == 5
        assert a.chromosome_str() == '0 0101'
        b.chromosome = [int(c) for c in '01111']  # bad
        a.neighbors = [b]
        b.neighbors = [a]
        a.prev_actions = {other.unique_id: deque(maxlen=a.mem_len) for other in a.neighbors}
        a.current_states = {other.unique_id: 0 for other in a.neighbors}
        b.prev_actions = {other.unique_id: deque(maxlen=a.mem_len) for other in b.neighbors}
        b.current_states = {other.unique_id: 0 for other in b.neighbors}

        a_actions = [0, 0, 1, 1, 1]
        b_actions = [0, 1, 1, 1, 1]
        a_states = [1, 2, 4, 4, 4]
        b_states = [1, 3, 4, 4, 4]
        for i in range(len(a_actions)):
            a.step()
            b.step()
            a.advance()
            b.advance()
            assert a.action[b.unique_id] == a_actions[i], i
            assert b.action[a.unique_id] == b_actions[i], i
            a.feedback()
            b.feedback()
            assert a.current_states[b.unique_id] == a_states[i], i
            assert b.current_states[a.unique_id] == b_states[i], i


    def test2():
        a = StringAgent.create_tit_for_tat(0, m, mem_len=2)
        b = StringAgent(1, m, mem_len=2)
        assert len(a.chromosome) == len(b.chromosome) == 19
        assert a.chromosome_str() == '001 0101010101010101'  # t4t
        b.chromosome = [int(c) for c in '0001111111111111110']
        a.neighbors = [b]
        b.neighbors = [a]
        a.prev_actions = {other.unique_id: deque(maxlen=a.mem_len) for other in a.neighbors}
        a.current_states = {other.unique_id: 0 for other in a.neighbors}
        b.prev_actions = {other.unique_id: deque(maxlen=a.mem_len) for other in b.neighbors}
        b.current_states = {other.unique_id: 0 for other in b.neighbors}

        a_actions = [0, 0, 0, 1, 1, 1, 0]
        b_actions = [0, 0, 1, 1, 1, 0, 1]
        a_states = [1, 3, 4, 10, 18, 17, 12]
        b_states = [1, 3, 5, 14, 18, 16, 9]
        for i in range(len(a_actions)):
            a.step()
            b.step()
            a.advance()
            b.advance()
            assert a.action[b.unique_id] == a_actions[i], i
            assert b.action[a.unique_id] == b_actions[i], i
            a.feedback()
            b.feedback()
            assert a.current_states[b.unique_id] == a_states[i], i
            assert b.current_states[a.unique_id] == b_states[i], i


    test1()
    test2()
