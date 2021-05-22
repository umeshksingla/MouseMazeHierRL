"""
Plot the node trajectories on maze
"""

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from numpy import ones

from MM_Maze_Utils import *
from parameters import HOME_NODE, RWD_NODE, FRAME_RATE
from utils import get_node_visit_times, get_all_night_nodes_and_times, get_wp_visit_times_and_rwd_times


def plot_trajectory(state_hist_all, episode_idx, save_file_name=None, figtitle=None, display=True):
    '''
    Plots specified simulated trajectories on the maze layout.
    
    state_hist_all: list of trajectories simulated by a model.
        Eg. [[0,1,3..], [28, 57, 116, ..], [0, 2, ..]]
    episode_idx: 'all', to plot all trajectories in state_hist_all
             int, to plot a specific bout/episode with index episode_idx
    
    Plots One maze figure with plotted trajectories and a color bar indicating nodes from entry to exit
    Returns: None
    '''

    def nodes2cell(state_hist_all):
        '''
        simulated trajectories, state_hist_all: {mouseID: [[TrajID x TrajSize]]}
        '''
        # print("state_hist_all", state_hist_all)
        state_hist_cell = []
        state_hist_xy = {}
        ma=NewMaze(6)
        for epID, epi in enumerate(state_hist_all):
            cells = []
            if not epi:
                continue
            for id,node in enumerate(epi):
                if id != 0 and node != HOME_NODE:
                    if node > epi[id-1]:
                        # if going to a deeper node
                        cells.extend(ma.ru[node])
                    elif node < epi[id-1]:
                        # if going to a shallower node
                        reverse_path = list(reversed(ma.ru[epi[id-1]]))
                        reverse_path = reverse_path + [ma.ru[node][-1]]
                        cells.extend(reverse_path[1:])
            if node==HOME_NODE:
                home_path = list(reversed(ma.ru[0]))
                cells.extend(home_path[1:])  # cells from node 0 to maze exit
            state_hist_cell.append(cells)
            state_hist_xy[epID] = np.zeros((len(cells),2))
            state_hist_xy[epID][:,0] = ma.xc[cells] + np.random.choice([-1,1],len(ma.xc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.xc[cells]))/2
            state_hist_xy[epID][:,1] = ma.yc[cells] + np.random.choice([-1,1],len(ma.yc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.yc[cells]))/2
        return state_hist_cell, state_hist_xy
    
    state_hist_cell, state_hist_xy = nodes2cell(state_hist_all)
    
    ma=NewMaze(6)
    # Draw the maze outline    
    fig,ax=plt.subplots(figsize=(9,9))
    plot(ma.wa[:,0],ma.wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
              xhide=True,yhide=True,axes=ax)
    re=[[-0.5,0.5,1,1],[-0.5,4.5,1,1],[-0.5,8.5,1,1],[-0.5,12.5,1,1],
       [2.5,13.5,1,1],[6.5,13.5,1,1],[10.5,13.5,1,1],
       [13.5,12.5,1,1],[13.5,8.5,1,1],[13.5,4.5,1,1],[13.5,0.5,1,1],
       [10.5,-0.5,1,1],[6.5,-0.5,1,1],[2.5,-0.5,1,1],
       [6.5,1.5,1,1],[6.5,11.5,1,1],[10.5,5.5,1,1],[10.5,7.5,1,1],
       [5.5,4.5,1,1],[5.5,8.5,1,1],[7.5,4.5,1,1],[7.5,8.5,1,1],[2.5,5.5,1,1],[2.5,7.5,1,1],
       [-0.5,2.5,3,1],[-0.5,10.5,3,1],[11.5,10.5,3,1],[11.5,2.5,3,1],[5.5,0.5,3,1],[5.5,12.5,3,1],
       [7.5,6.5,7,1]]
    for r in re:
        rect=patches.Rectangle((r[0],r[1]),r[2],r[3],linewidth=1,edgecolor='lightgray',facecolor='lightgray')
        ax.add_patch(rect)

    #plt.axis('off'); # turn off the axes

    # Converting cell positions to x,y positions in the maze
    # ma.ce contains x,y positions for each cell
    # print("state_hist_xy", state_hist_xy)
    if episode_idx == 'all':
        for id, epi in enumerate(state_hist_xy):
            x = state_hist_xy[epi][:,0]
            y = state_hist_xy[epi][:,1]
            t = np.linspace(0,1,x.shape[0]) # your "time" variable

            # set up a list of (x,y) points
            points = np.array([x,y]).transpose().reshape(-1,1,2)

            # set up a list of segments
            segs = np.concatenate([points[:-1],points[1:]],axis=1)

            # make the collection of segments
            lc = LineCollection(segs, cmap=plt.get_cmap('viridis'),linewidths=2) # jet, viridis hot
            lc.set_array(t) # color the segments by our parameter

            # plot the collection
            lines=ax.add_collection(lc); # add the collection to the plot
    else:
        x = state_hist_xy[episode_idx][:,0]
        y = state_hist_xy[episode_idx][:,1]
        t = np.linspace(0,1,x.shape[0]) # your "time" variable

        # set up a list of (x,y) points
        points = np.array([x,y]).transpose().reshape(-1,1,2)

        # set up a list of segments
        segs = np.concatenate([points[:-1],points[1:]],axis=1)

        # make the collection of segments
        lc = LineCollection(segs, cmap=plt.get_cmap('viridis'),linewidths=2) # jet, viridis hot
        lc.set_array(t) # color the segments by our parameter

        # plot the collection
        lines=ax.add_collection(lc); # add the collection to the plot

    cax=fig.add_axes([1.05, 0.05, 0.05, 0.9])
    cbar=fig.colorbar(lines,cax=cax)
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(['Entry','Exit'])
    cbar.ax.tick_params(labelsize=18)
    fig.suptitle(figtitle)
    fig = plt.gcf()
    if save_file_name:
        fig.savefig(save_file_name)
    if display:
        plt.show()
    return


def plot_maze_stats(data, datatype, save_file_name=None, display=True):
    """
    :param data: 1 x 127 array of values (a value for all 127 nodes in the maze
    from 0 to 126)
    :param datatype:
        'states': to just highlight nodes with same color, or
        'state_values': to see the gradient in color nodes based on the state values.
    """
    ma = NewMaze()
    if datatype == 'states':
        fr,_=np.histogram(data,bins=np.arange(2**(ma.le+1))-0.5)
    if datatype == 'state_values':
        fr = data
    col=np.array([[0,1,1,1],[1,.8,.8,1],[2,.6,.6,1],[3,.4,.4,1]])
    ax=PlotMazeFunction(fr,ma,mode='nodes',numcol=None,figsize=4,col=col);
    re=[[-0.5,0.5,1,1],[-0.5,4.5,1,1],[-0.5,8.5,1,1],[-0.5,12.5,1,1],
       [2.5,13.5,1,1],[6.5,13.5,1,1],[10.5,13.5,1,1],
       [13.5,12.5,1,1],[13.5,8.5,1,1],[13.5,4.5,1,1],[13.5,0.5,1,1],
       [10.5,-0.5,1,1],[6.5,-0.5,1,1],[2.5,-0.5,1,1],
       [6.5,1.5,1,1],[6.5,11.5,1,1],[10.5,5.5,1,1],[10.5,7.5,1,1],
       [5.5,4.5,1,1],[5.5,8.5,1,1],[7.5,4.5,1,1],[7.5,8.5,1,1],[2.5,5.5,1,1],[2.5,7.5,1,1],
       [-0.5,2.5,3,1],[-0.5,10.5,3,1],[11.5,10.5,3,1],[11.5,2.5,3,1],[5.5,0.5,3,1],[5.5,12.5,3,1],
       [7.5,6.5,7,1]]
    for r in re:
        rect=patches.Rectangle((r[0],r[1]),r[2],r[3],linewidth=1,edgecolor='lightgray',facecolor='lightgray')
        ax.add_patch(rect)
    plt.axis('off'); # turn off the axes
    fig = plt.gcf()
    if save_file_name:
        fig.savefig(save_file_name)
    if display:
        plt.show()
    return


def plot_nodes_vs_time(tf, colored_markers=False, init_time=None, time_window=None):
    """
    Plot traversed nodes (y-axis) over time (x-axis) for the selected time interval
    :param tf: trajectory file
    :param all_night_nodes_and_times: ndarray (n_nodes_traversed, 2) nodes and the time the animal was there
    :param times_to_rwd: times of reward delivery
    :returns: fig, axes
    """
    plt.figure(figsize=(15, 13))
    HOME_NODE_PLOTTING = -10
    all_night_nodes_and_times = get_all_night_nodes_and_times(tf)
    _, times_to_rwd = get_wp_visit_times_and_rwd_times(tf)
    all_night_nodes_and_times[all_night_nodes_and_times[:, 0] == HOME_NODE, 0] = HOME_NODE
    plt.plot(all_night_nodes_and_times[:, 1], all_night_nodes_and_times[:, 0], '.-')

    # Plot stars when the animal gets a reward
    plt.plot(times_to_rwd, RWD_NODE * ones(len(times_to_rwd)), linestyle='None', marker='^', label='rwd', markersize=6,
             color='#edc61c')

    if colored_markers:
        node_visit_times = list()
        for node in range(127):
            node_visit_times.append(get_node_visit_times(tf, node))
        for node in range(127):
            plt.plot(node_visit_times[node], node * ones(len(node_visit_times[node])), 'o')

    # Have visual separation for different node levels
    plt.axhline(0.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(2.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(6.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(14.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(30.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(62.5, color='brown', linestyle='--', linewidth='.8')
    plt.fill_betweenx([0.6, 2.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='gray')
    plt.fill_betweenx([2.6, 6.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='red')
    plt.fill_betweenx([6.6, 14.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='green')
    plt.fill_betweenx([14.6, 30.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='orange')
    plt.fill_betweenx([30.6, 62.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='yellow')
    plt.fill_betweenx([62.6, 126.4], 0, all_night_nodes_and_times[:, 1][-1], alpha=.1, color='blue')

    # plot times at home
    START_IDX = 0
    END_IDX = 1
    for bout in range(len(tf.no) - 1):
        plt.fill_betweenx([-13, -12], tf.fr[bout][END_IDX] / FRAME_RATE, tf.fr[bout + 1][START_IDX] / FRAME_RATE,
                          alpha=.2, color='black', label='at home' if bout == 0 else None)

    # TODO: use fill_betweenx to colored ribbon similar to the representing time at home, but to represent the quadrant the animal is at

    plt.grid()
    plt.title("All night trajectory")
    plt.ylabel("Node number")
    plt.xlabel("Time (s)")
    plt.legend()

    if init_time is not None and time_window is not None:
        plt.xlim(init_time, init_time + time_window)
    return plt.gcf(), plt.gca()