# Paths
OUTDATA_PATH = '../outdata/'

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
LVL_BY_NODE[HOME_NODE] = 1111

# Define nodes belonging to each quadrant
QUAD1 = [3, 7, 8, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 37, 38, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
QUAD2 = [4, 9, 10, 19, 20, 21, 22, 39, 40, 41, 42, 43, 44, 45, 46, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]
QUAD3 = [5, 11, 12, 23, 24, 25, 26, 47, 48, 49, 50, 51, 52, 53, 54, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
QUAD4 = [6, 13, 14, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]

assert len(QUAD1) == 31
assert len(QUAD2) == 31
assert len(QUAD3) == 31
assert len(QUAD4) == 31

HALF_UP = QUAD1 + QUAD2
HALF_DOWN = QUAD3 + QUAD4

HALF_LEFT = QUAD1 + QUAD3
HALF_RIGHT = QUAD2 + QUAD4

# Dict for each node: quadrant
node_quadrant_dict = {}
for q, quad in enumerate([QUAD1, QUAD2, QUAD3, QUAD4]):
    for n in quad:
        node_quadrant_dict[n] = q+3

# Dict for each node: sub-quadrant
node_subquadrant_dict = {}
SUBQUADRANT_dict = {}
for subq, level4_node in enumerate(LVL_4_NODES):
    p1, p2 = level4_node*2+1, level4_node*2+2
    n1, n2, n3, n4 = p1*2+1, p1*2+2, p2*2+1, p2*2+2
    # print(subq, "=>", n1, n2, n3, n4)
    this_subq = subq + 15
    node_subquadrant_dict[n1] = this_subq
    node_subquadrant_dict[n2] = this_subq
    node_subquadrant_dict[n3] = this_subq
    node_subquadrant_dict[n4] = this_subq
    node_subquadrant_dict[p1] = this_subq
    node_subquadrant_dict[p2] = this_subq
    node_subquadrant_dict[level4_node] = this_subq

    if this_subq not in SUBQUADRANT_dict:
        SUBQUADRANT_dict[this_subq] = []

    SUBQUADRANT_dict[this_subq].extend([level4_node, p1, p2, n1, n2, n3, n4])

ROW_1 = SUBQUADRANT_dict[15] + SUBQUADRANT_dict[16] + SUBQUADRANT_dict[19] + SUBQUADRANT_dict[20]
ROW_2 = SUBQUADRANT_dict[17] + SUBQUADRANT_dict[18] + SUBQUADRANT_dict[21] + SUBQUADRANT_dict[22]
ROW_3 = SUBQUADRANT_dict[23] + SUBQUADRANT_dict[24] + SUBQUADRANT_dict[27] + SUBQUADRANT_dict[28]
ROW_4 = SUBQUADRANT_dict[25] + SUBQUADRANT_dict[26] + SUBQUADRANT_dict[29] + SUBQUADRANT_dict[30]

COL_1 = SUBQUADRANT_dict[15] + SUBQUADRANT_dict[17] + SUBQUADRANT_dict[23] + SUBQUADRANT_dict[25]
COL_2 = SUBQUADRANT_dict[16] + SUBQUADRANT_dict[18] + SUBQUADRANT_dict[24] + SUBQUADRANT_dict[26]
COL_3 = SUBQUADRANT_dict[19] + SUBQUADRANT_dict[21] + SUBQUADRANT_dict[27] + SUBQUADRANT_dict[29]
COL_4 = SUBQUADRANT_dict[20] + SUBQUADRANT_dict[22] + SUBQUADRANT_dict[28] + SUBQUADRANT_dict[30]

# print(node_subquadrant_dict)

# Label each node in each subquadrant as viewed from corresponding level 2:
# Node order below: [straight(s), bent_straight (bs), opp_straight (o_s), opp_bent_straight (o_bs)]
OPP_PREFIX = 'o_'
BENT_PREFIX = 'b'
STRAIGHT = 's'
BENT_STRAIGHT = BENT_PREFIX + STRAIGHT
OPP_STRAIGHT = OPP_PREFIX + STRAIGHT
OPP_BENT_STRAIGHT = OPP_PREFIX + BENT_PREFIX + STRAIGHT

OPPOSITES = {
    STRAIGHT: OPP_STRAIGHT,
    OPP_STRAIGHT: STRAIGHT,
    BENT_STRAIGHT: OPP_BENT_STRAIGHT,
    OPP_BENT_STRAIGHT: BENT_STRAIGHT,
}

full_labels = {STRAIGHT: 'straight', BENT_STRAIGHT: 'bent straight', OPP_STRAIGHT: 'opposite straight', OPP_BENT_STRAIGHT: 'opposite bent straight'}
node_sets = [
    (63, 65, 64, 66),
    (68, 70, 67, 69),
    (73, 71, 74, 72),
    (78, 76, 77, 75),
    (95, 97, 96, 98),
    (100, 102, 99, 101),
    (105, 103, 106, 104),
    (110, 108, 109, 107),
    (79, 81, 80, 82),
    (84, 86, 83, 85),
    (89, 87, 90, 88),
    (94, 92, 93, 91),
    (111, 113, 112, 114),
    (116, 118, 115, 117),
    (121, 119, 122, 120),
    (126, 124, 125, 123),
]

# Dict for each quadrant: label in a quadrant
node_subquadrant_label_dict = {}
for ns in node_sets:
    n1, n2, n3, n4 = ns
    node_subquadrant_label_dict[n1] = STRAIGHT
    node_subquadrant_label_dict[n2] = BENT_STRAIGHT
    node_subquadrant_label_dict[n3] = OPP_STRAIGHT
    node_subquadrant_label_dict[n4] = OPP_BENT_STRAIGHT

# all 16 subquadrants of the maze
subquadrant_sets = {
    3: (15, 17, 16, 18),  # (s, bs, o_s, o_bs)
    4: (20, 22, 19, 21),
    5: (25, 23, 26, 24),
    6: (30, 28, 29, 27)
}

subquadrant_label_dict = {}
for subq in subquadrant_sets:
    n1, n2, n3, n4 = subquadrant_sets[subq]
    subquadrant_label_dict[n1] = STRAIGHT       # straight
    subquadrant_label_dict[n2] = BENT_STRAIGHT      # bent straight
    subquadrant_label_dict[n3] = OPP_STRAIGHT     # opposite straight
    subquadrant_label_dict[n4] = OPP_BENT_STRAIGHT    # opposite bent straight

quadrant_halves = {
    1: (3, 4),
    2: (5, 6),
}

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

los_node_mapping_lvl5_6 = {31: 63, 32: 65, 35: 71, 36: 73, 47: 95, 48: 97, 51: 103, 52: 105, 39: 79, 40: 81, 43: 87, 44: 89, 55: 111, 56: 113, 59: 119, 60: 121, 33: 68, 34: 70, 37: 76, 38: 78, 49: 100, 50: 102, 53: 108, 54: 110, 41: 84, 42: 86, 45: 92, 46: 94, 57: 116, 58: 118, 61: 124, 62: 126}
