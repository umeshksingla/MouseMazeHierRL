# Constants
UNRWD_NAMES = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
RWD_NAMES = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
ALL_NAMES = RWD_NAMES + UNRWD_NAMES
UNRWD_NAMES_SUB = ['B5','B6','B7','D3','D4','D5','D7','D8','D9'] # excluding D6 which barely entered the maze

WATERPORT_NODE = 116
HOME_NODE = 127
RWD_STATE = 128
INVALID_STATE = -1
FRAME_RATE = 30  # Hz

TIME_EACH_MOVE=0.3  # in seconds. Mice take an average of 0.3 per each move

# Define the 3 modes of behavior
LEAVE = 0
DRINK = 1
EXPLORE = 2

# Define cell numbers of end/leaf nodes
ALL_MAZE_NODES = list(range(0, 127))
ALL_VISITABLE_NODES=ALL_MAZE_NODES+[HOME_NODE]
LVL_6_NODES = dict.fromkeys(list(range(63,127)), True)
LVL_5_NODES = dict.fromkeys(list(range(31,63)), True)
LVL_4_NODES = dict.fromkeys(list(range(15,31)), True)
LVL_3_NODES = dict.fromkeys(list(range(7,15)), True)
LVL_2_NODES = dict.fromkeys(list(range(3,7)), True)
LVL_1_NODES = dict.fromkeys(list(range(1,3)), True)
LVL_0_NODES = dict.fromkeys(list(range(0,1)), True)
NODE_LVL = {0:LVL_0_NODES, 1:LVL_1_NODES, 2:LVL_2_NODES, 3:LVL_3_NODES, 4:LVL_4_NODES, 5:LVL_5_NODES, 6:LVL_6_NODES}
LVL_BY_NODE = dict([(node, lvl) for lvl in NODE_LVL for node in NODE_LVL[lvl]])

# Define nodes belonging to each quadrant
QUAD1 = [3, 7, 8, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 37, 38, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
QUAD2 = [4, 9, 10, 19, 20, 21, 22, 39, 40, 41, 42, 43, 44, 45, 46, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]
QUAD3 = [5, 11, 12, 23, 24, 25, 26, 47, 48, 49, 50, 51, 52, 53, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
QUAD4 = [6, 13, 14, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 61, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]

# DEPRECATED: Parameters to transition out of
# PLEASE ONLY USE THE UPPERCASE ONES
HomeNode = HOME_NODE
RewardNode = WATERPORT_NODE
InvalidState = INVALID_STATE
WaterPortNode = RWD_STATE
UnrewNames = UNRWD_NAMES
RewNames = RWD_NAMES
AllNames = ALL_NAMES
UnrewNamesSub = UNRWD_NAMES_SUB
quad1 = QUAD1
quad2 = QUAD2
quad3 = QUAD3
quad4 = QUAD4
lvl6_nodes = LVL_6_NODES
lvl5_nodes = LVL_5_NODES
lvl4_nodes = LVL_4_NODES
lvl3_nodes = LVL_3_NODES
lvl2_nodes = LVL_2_NODES
lvl1_nodes = LVL_1_NODES
lvl0_nodes = LVL_0_NODES
