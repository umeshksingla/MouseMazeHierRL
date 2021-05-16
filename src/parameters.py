## Constants

UnrewNames = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
RewNames = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
AllNames=RewNames+UnrewNames
UnrewNamesSub=['B5','B6','B7','D3','D4','D5','D7','D8','D9'] # excluding D6 which barely entered the maze

RWD_NODE = 116
HOME_NODE = 127
INVALID_STATE = -1
WATER_PORT_STATE = 128
FRAME_RATE = 30  # Hz


# Parameters to transition out of
HomeNode = HOME_NODE
RewardNode = RWD_NODE
InvalidState = INVALID_STATE
WaterPortNode = WATER_PORT_STATE

# Define cell numbers of end/leaf nodes
lvl6_nodes = list(range(63,127))
lvl5_nodes = list(range(31,63))
lvl4_nodes = list(range(15,31))
lvl3_nodes = list(range(7,15))
lvl2_nodes = list(range(3,7))
lvl1_nodes = list(range(1,3))
lvl0_nodes = list(range(0,1))
NODE_LVL = {0:lvl0_nodes, 1:lvl1_nodes, 2:lvl2_nodes, 3:lvl3_nodes, 4:lvl4_nodes, 5:lvl5_nodes, 6:lvl6_nodes}

# Define nodes belonging to each quadrant
quad1 = [3, 7, 8, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 37, 38, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
quad2 = [4, 9, 10, 19, 20, 21, 22, 39, 40, 41, 42, 43, 44, 45, 46, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]
quad3 = [5, 11, 12, 23, 24, 25, 26, 47, 48, 49, 50, 51, 52, 53, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
quad4 = [6, 13, 14, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 61, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]
