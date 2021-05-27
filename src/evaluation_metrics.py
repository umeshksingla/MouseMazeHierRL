"""

"""

from parameters import NODE_LVL
from collections import defaultdict
import numpy as np


def exploration_efficiency(episodes):
    step = 0
    steps_taken = dict([(2**i, np.nan) for i in range(0, 15)])
    nodes_explored = defaultdict(int)
    for each in episodes:
        for n in each:
            if n in NODE_LVL[6]:
                step += 1
                nodes_explored[n] += 1
                if step in steps_taken:
                    steps_taken[step] = len(nodes_explored)

    print(nodes_explored)
    # print(steps_taken)
    return steps_taken
