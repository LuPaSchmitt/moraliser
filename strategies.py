SCORE_MAP = [
    [3, 0],
    [5, 1],
]


def default(our_prev_action, their_prev_action):
    return 0


def tit_for_tat(our_prev_action, their_prev_action):
    our_current_action = their_prev_action
    return our_current_action
