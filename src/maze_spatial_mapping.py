import numpy as np

from MM_Maze_Utils import NewMaze
import parameters as p


ma = NewMaze(6)
NODE_CELL_MAPPING = np.array([ma.ru[n][-1] if n != p.HOME_NODE else 0 for n in p.ALL_VISITABLE_NODES])
CELL_XY = np.array([np.array([ma.xc[c], ma.yc[c]]) for c in p.ALL_VISITABLE_CELLS])
# print(NODE_CELL_MAPPING)
# print(NODE_CELL_MAPPING.shape)
# print(CELL_XY)
# print(CELL_XY.shape)
