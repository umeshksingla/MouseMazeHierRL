import pickle

import numpy as np
from collections import defaultdict

from MM_Maze_Utils import NewMaze
import parameters as p
from utils import nodes2cell, connect_path_node

ma = NewMaze(6)
NODE_CELL_MAPPING = np.array([ma.ru[n][-1] if n != p.HOME_NODE else 0 for n in p.ALL_VISITABLE_NODES])
CELL_XY = np.array([np.array([ma.xc[c], ma.yc[c]]) for c in p.ALL_VISITABLE_CELLS])
# print(NODE_CELL_MAPPING)
# print(NODE_CELL_MAPPING.shape)
# print(CELL_XY)
# print(CELL_XY.shape)


# len(nodes2cell([connect_path_node(self.s, o)])[0][0])

# Takes insanely long!!
# cell_path_lengths_from_l6 = defaultdict(lambda: defaultdict(int))
# for l6n in p.LVL_6_NODES:
#     for n in p.ALL_MAZE_NODES:
#         if l6n == n:
#             cell_path_lengths_from_l6[l6n][n] = 0
#         else:
#             cell_path_lengths_from_l6[l6n][n] = len(nodes2cell([connect_path_node(l6n, n)])[0][0])-1
#         print(l6n, n, cell_path_lengths_from_l6[l6n][n])
# print(cell_path_lengths_from_l6)



