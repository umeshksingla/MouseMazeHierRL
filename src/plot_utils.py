"""
Plot the node trajectories on maze
"""

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from numpy import ones

from MM_Maze_Utils import *
from parameters import HOME_NODE, RWD_NODE, FRAME_RATE, NODE_LVL
from utils import get_node_visit_times, get_all_night_nodes_and_times, \
    get_wp_visit_times_and_rwd_times, nodes2cell, get_reward_times
import evaluation_metrics as em


def plot_trajectory(state_hist_all, episode_idx, save_file_name=None, figtitle=None, display=True, figsize=(9,9)):
    '''
    Plots specified simulated trajectories on the maze layout.
    
    state_hist_all: list of trajectories simulated by a model.
        Eg. [[0,1,3..], [28, 57, 116, ..], [0, 2, ..]]
    episode_idx: 'all', to plot all trajectories in state_hist_all
             int, to plot a specific bout/episode with index episode_idx
    
    Plots One maze figure with plotted trajectories and a color bar indicating nodes from entry to exit
    Returns: None
    '''
    state_hist_cell, state_hist_xy = nodes2cell(state_hist_all)
    
    ma=NewMaze(6)
    # Draw the maze outline    
    fig,ax=plt.subplots(figsize=figsize)
    plot(ma.wa[:,0],ma.wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
              xhide=True,yhide=True,axes=ax)
    re=[[-0.5,0.5,1,1],[-0.5,4.5,1,1],[-0.5,8.5,1,1],[-0.5,12.5,1,1],
       [2.5,13.5,1,1],[6.5,13.5,1,1],[10.5,13.5,1,1],
       [13.5,12.5,1,1],[13.5,8.5,1,1],[13.5,4.5,1,1],[13.5,0.5,1,1],
       [10.5,-0.5,1,1],[6.5,-0.5,1,1],[2.5,-0.5,1,1],
       [6.5,1.5,1,1],[6.5,11.5,1,1],[10.5,5.5,1,1],[10.5,7.5,1,1],
       [5.5,4.5,1,1],[5.5,8.5,1,1],[7.5,4.5,1,1],[7.5,8.5,1,1],[2.5,5.5,1,1],[2.5,7.5,1,1],
       [-0.5,2.5,3,1],[-0.5,10.5,3,1],[11.5,10.5,3,1],[11.5,2.5,3,1],[5.5,0.5,3,1],[5.5,12.5,3,1],
       [7.5,6.5,7,1]]  # coordinates of gray rectangles, the inaccessible regions of the maze
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

            # put a blue star in the beginning and a yellow star in the end of each trajectory
            plt.plot(points[ 0, 0, 0], points[ 0, 0, 1], "*", markersize=10, color="blue")
            plt.plot(points[-1, 0, 0], points[-1, 0, 1], "*", markersize=10, color="yellow")

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

        # put a blue star in the beginning and a yellow star in the end of each trajectory
        plt.plot(points[0, 0, 0], points[0, 0, 1], "*", markersize=10, color="blue")
        plt.plot(points[-1, 0, 0], points[-1, 0, 1], "*", markersize=10, color="yellow")

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
    return fig


def plot_nodes_vs_time(tf, colored_markers=False, init_time=None, time_window=None, include_grid=False,
                       separate_quadrants=True, custom_title=None):
    """
    Plot traversed nodes (y-axis) over time (x-axis) for the selected time interval
    :param tf: trajectory file
    :returns: fig, axes
    """
    plt.figure(figsize=(15, 13))
    HOME_NODE_PLOTTING = -10
    all_night_nodes_and_times = get_all_night_nodes_and_times(tf)
    _, times_to_rwd = get_wp_visit_times_and_rwd_times(tf)
    all_night_nodes_and_times[all_night_nodes_and_times[:, 0] == HOME_NODE, 0] = HOME_NODE_PLOTTING
    plt.plot(all_night_nodes_and_times[:, 1], all_night_nodes_and_times[:, 0], '.-')

    if colored_markers:
        node_visit_times = list()
        for node in range(127):
            node_visit_times.append(get_node_visit_times(tf, node))
        for node in range(127):
            plt.plot(node_visit_times[node], node * ones(len(node_visit_times[node])), 'o')

    # Have visual separation for different node levels and quadrants
    if separate_quadrants:
        label_shades="Color shades: levels 1 to 6\nColor intensity: quadrants"
    else:
        label_shades = "Color shades: levels 1 to 6"
    plt.plot([], [], ' ', label="Color shades: levels 1 to 6\nColor intensity: quadrants")
    plt.axhline(0.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(2.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(6.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(14.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(30.5, color='brown', linestyle='--', linewidth='.8')
    plt.axhline(62.5, color='brown', linestyle='--', linewidth='.8')
    end_time = all_night_nodes_and_times[:, 1][-1]
    plt.fill_betweenx([0.6, 2.4], 0, end_time, alpha=.1, color='gray')
    if separate_quadrants:
        plt.fill_betweenx([2.6, 3.4], 0, end_time, alpha=.05, color='red')
        plt.fill_betweenx([3.6, 4.4], 0, end_time, alpha=.1, color='red')
        plt.fill_betweenx([4.6, 5.4], 0, end_time, alpha=.15, color='red')
        plt.fill_betweenx([5.6, 6.4], 0, end_time, alpha=.2, color='red')

        plt.fill_betweenx([6.6, 8.4], 0, end_time, alpha=.05, color='green')
        plt.fill_betweenx([8.6, 10.4], 0, end_time, alpha=.1, color='green')
        plt.fill_betweenx([10.6, 12.4], 0, end_time, alpha=.15, color='green')
        plt.fill_betweenx([12.6, 14.4], 0, end_time, alpha=.2, color='green')

        plt.fill_betweenx([14.6, 18.4], 0, end_time, alpha=.05, color='orange')
        plt.fill_betweenx([18.6, 22.4], 0, end_time, alpha=.1, color='orange')
        plt.fill_betweenx([22.6, 26.4], 0, end_time, alpha=.15, color='orange')
        plt.fill_betweenx([26.6, 30.4], 0, end_time, alpha=.2, color='orange')

        plt.fill_betweenx([30.6, 38.4], 0, end_time, alpha=.05, color='yellow')
        plt.fill_betweenx([38.6, 46.4], 0, end_time, alpha=.15, color='yellow')
        plt.fill_betweenx([46.6, 54.4], 0, end_time, alpha=.25, color='yellow')
        plt.fill_betweenx([54.6, 62.4], 0, end_time, alpha=.35, color='yellow')

        plt.fill_betweenx([62.6, 78.4], 0, end_time, alpha=.05, color='blue')
        plt.fill_betweenx([78.6, 94.4], 0, end_time, alpha=.1, color='blue')
        plt.fill_betweenx([94.6, 110.4], 0, end_time, alpha=.15, color='blue')
        plt.fill_betweenx([110.6, 126.4], 0, end_time, alpha=.2, color='blue')
    else:
        plt.fill_betweenx([2.6, 6.4], 0, end_time, alpha=.1, color='red')
        plt.fill_betweenx([6.6, 14.4], 0, end_time, alpha=.1, color='green')
        plt.fill_betweenx([14.6, 30.4], 0, end_time, alpha=.1, color='orange')
        plt.fill_betweenx([30.6, 62.4], 0, end_time, alpha=.1, color='yellow')
        plt.fill_betweenx([62.6, 126.4], 0, end_time, alpha=.1, color='blue')

    # Plot stars when the animal gets a reward
    plt.plot(times_to_rwd, RWD_NODE * ones(len(times_to_rwd)) - .2, linestyle='None', marker='^', label='rwd',
             markersize=10, markerfacecolor='yellow', color='red')

    # plot times at home # NOTE: apparently there is a bug in fill_betweenx function which unpredictably gives a wrong visualization sometimes
    START_IDX = 0
    END_IDX = 1
    for bout in range(len(tf.no) - 1):
        plt.fill_betweenx([-13, -12], tf.fr[bout][END_IDX] / FRAME_RATE, tf.fr[bout + 1][START_IDX] / FRAME_RATE,
                          alpha=.2, color='black', label='at home' if bout == 0 else None)

    # TODO: use fill_betweenx to colored ribbon similar to the representing time at home, but to represent the quadrant the animal is at

    if include_grid: plt.grid()
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title("All night trajectory")
    plt.ylabel("Node number")
    plt.xlabel("Time (s)")
    plt.legend()

    if init_time is not None and time_window is not None:
        plt.xlim(init_time, init_time + time_window)
    return plt.gcf(), plt.gca()


def PlotMazeFunction_gradientcmap(fn, ma, interpolate_cell_values, colormap_name=None, numcol=None, figsize=4, axes=None,
                                  vmin=None, vmax=None):
    '''
    Plot the maze defined in `ma` with a function f overlaid in color

    See also MM_Traj_Utils.PlotMazeFunction
    :param fn: 1-by-128 array of state values for nodes on the maze
    :param ma: maze structure
    :param interpolate_cell_values: interpolate_cell_values (bool)
    :param colormap_name:
    :param numcol: color for the numbers. If numcol is None the numbers are omitted
    :param figsize: in inches
    :return: the axes of the plot with maze cells color-coded with state-values
    '''

    def nodes2cell_statevalues(V):
        '''
        Interpolate between node state-values to have a smoother transition of state-values across maze cells
        :param V: 1-by-128 array of state-values
        :return: 1-by-176 array of state-values on maze cells
        '''

        ma = NewMaze(6)
        cells = []

        for lvl in NODE_LVL:
            for j in NODE_LVL[lvl]:
                if j == 0:
                    root = HOME_NODE
                elif j % 2 == 0:
                    root = (j - 2) // 2
                elif j % 2 == 1:
                    root = (j - 1) // 2

                if j == 0:
                    values = np.linspace(V[root], V[j], len(ma.ru[j]))
                    cells.extend(values)
                else:
                    values = np.linspace(V[root], V[j], len(ma.ru[j]) + 1)
                    cells.extend(values[1:])
        return cells

    def generate_colors(f_cells, colormap_name, vmin=None, vmax=None):
        '''
        Convert numbers in array fn to a color array, colors where
        colors[0,:] = f and col[1:3, :] = RGB values corresponding to fn
        :param f_cells: array of state values for each maze cell
        :param colormap_name: name of matplotlib built-in colormap from matplotlib.pyplot.cm
        :return: colors (array of colors with f_cells on the first column and RGB on the next 3), cmappable
        '''
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        cmap = plt.cm.get_cmap(colormap_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmappable = ScalarMappable(norm=norm, cmap=cmap)
        cmap_norm = cmap(norm(f_cells))[:, :-1]  # Retrieving RGB array
        colors = np.concatenate((np.reshape(f_cells, (len(f_cells), 1)), cmap_norm), axis=1)

        return colors, cmappable

    if axes:
        ax=axes
        PlotMazeWall(ma, axes=ax, figsize=figsize)
    else:
        ax=PlotMazeWall(ma, axes=None, figsize=figsize)

    if interpolate_cell_values:
        f_cells = nodes2cell_statevalues(fn)
        colors, cmappable = generate_colors(f_cells, colormap_name, vmin=vmin, vmax=vmax)
        for j in range(len(ma.xc)):
            x = ma.xc[j];
            y = ma.yc[j]
            if not (fn is None):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, lw=0,
                                               color=colors[j, 1:]))  # draw with color f[]
            if numcol:
                plt.text(x - .35, y + .15, '{:d}'.format(j), color=numcol)  # number the cells
    else:
        fr, _ = np.histogram(fn, bins=np.arange(2 ** (ma.le + 1)) - 0.5)
        colors, cmappable = generate_colors(fr, colormap_name, vmin=vmin, vmax=vmax)

        for j,r in enumerate(ma.ru):
            x = ma.xc[r[-1]]; y=ma.yc[r[-1]]
            if not(fn is None):
                ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,lw=0,
                                           color=colors[j, 1:])) # draw with color f[]
            if numcol:
                plt.text(x-.35,y+.15,'{:d}'.format(j),color=numcol) # number the ends of a run
    return ax, cmappable


def plot_maze_stats(data, interpolate_cell_values=True, colormap_name=None, axes=None, save_file_name=None, display=True,
                    cbar=True, colorbar_label="", figtitle='', vmin=None, vmax=None):
    """
    :param data: list of maze nodes, cells or 1-by-128 array of state-values
    :param interpolate_cell_values (bool): True to interpolate, False to only color nodes and leave
        other cells colored in white
    :param colormap_name: name of matplotlib built-in colormap from matplotlib.pyplot.cm
    """
    # ma = NewMaze()
    # if datatype == 'states':
        # fr,_= np.histogram(data,bins=np.arange(2**(ma.le+1))-0.5)
        # col = np.array([[0, 1, 1, 1], [1, .8, .8, 1], [2, .6, .6, 1], [3, .4, .4, 1]])
        # ax = PlotMazeFunction(fr, ma, mode='nodes', numcol=None, col=col, axes=axes)

    # if datatype == 'state_values':
    #     ax, cmappable = PlotMazeFunction_gradientcmap(data, ma, datatype, colormap_name, axes=axes)
    #     plt.colorbar(cmappable, shrink=0.5)  # draw the colorbar

    ma = NewMaze()
    ax, cmappable = PlotMazeFunction_gradientcmap(data, ma, interpolate_cell_values, colormap_name, axes=axes,
                                                  vmin=vmin, vmax=vmax)

    re=[[-0.5,0.5,1,1],[-0.5,4.5,1,1],[-0.5,8.5,1,1],[-0.5,12.5,1,1],
       [2.5,13.5,1,1],[6.5,13.5,1,1],[10.5,13.5,1,1],
       [13.5,12.5,1,1],[13.5,8.5,1,1],[13.5,4.5,1,1],[13.5,0.5,1,1],
       [10.5,-0.5,1,1],[6.5,-0.5,1,1],[2.5,-0.5,1,1],
       [6.5,1.5,1,1],[6.5,11.5,1,1],[10.5,5.5,1,1],[10.5,7.5,1,1],
       [5.5,4.5,1,1],[5.5,8.5,1,1],[7.5,4.5,1,1],[7.5,8.5,1,1],[2.5,5.5,1,1],[2.5,7.5,1,1],
       [-0.5,2.5,3,1],[-0.5,10.5,3,1],[11.5,10.5,3,1],[11.5,2.5,3,1],[5.5,0.5,3,1],[5.5,12.5,3,1],
       [7.5,6.5,7,1]]  # coordinates of gray rectangles, the inaccessible regions of the maze
    for r in re:
        rect=patches.Rectangle((r[0],r[1]),r[2],r[3],linewidth=1,edgecolor='lightgray',facecolor='lightgray')
        ax.add_patch(rect)
    plt.axis('off'); # turn off the axes
    if cbar:
        plt.colorbar(cmappable, shrink=0.4, label=colorbar_label)
        # plt.colorbar(cmappable, ax=[ax], location='left', shrink=0.5)  # draw the colorbar

    fig = plt.gcf()
    fig.suptitle(figtitle)
    if save_file_name:
        fig.savefig(save_file_name)
    if display:
        plt.show()
    else:
        plt.clf()
        plt.close()
    return ax


def plot_trajs(episodes_mouse, save_file_path, title):
    """todo:
    episodes_mouse:

    Save every kth episodes when there are more than 100 episodes in total,
    otherwise save all.
    """
    k = 10
    for i, traj in enumerate(episodes_mouse):
        print("Saving traj", i, ":", traj[:5], '...', traj[-5:])
        if len(episodes_mouse) >= 100 and i >=10 and i%k != 0:
            continue
        plot_trajectory([traj], 'all',
                        save_file_name=os.path.join(save_file_path, f'traj_{i}.png'),
                        display=False,
                        figtitle=f'Traj {i} \n {title}')
        plt.clf()
        plt.close()
    return


def plot_episode_lengths(episodes_mouse, title, save_file_path=None, display=False):
    """todo
    A bar plot of all episode lengths
    episodes_mouse:

    """
    ax = plt.bar(range(len(episodes_mouse)), [len(e) for e in episodes_mouse],
            label='Episodes length')
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("number of steps")
    plt.legend()
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'epi_lengths_bars.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return ax


def plot_exploration_efficiency(episodes, title=None, save_file_path=None, display=False):
    """todo
    new_end_nodes_found:
        dict() of how many steps -> how many distinct end nodes
    """
    new_end_nodes_found = em.exploration_efficiency(episodes, re=True)
    f = plt.figure()
    ax = plt.plot(new_end_nodes_found.keys(), new_end_nodes_found.values(), 'o-')
    plt.xscale('log', base=10)

    if title: plt.title(title)
    plt.xlabel("end nodes visited")
    plt.ylabel("new end nodes found")
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'exp_efficiency.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return f


def plot_reward_path_lengths(episodes, title, save_file_path=None, dots=True, display=False):
    """todo
    time_reward_node:

    """
    visit_reward_node, time_reward_node = get_reward_times(episodes)
    if dots:
        plt.plot(time_reward_node, 'b.', label='Steps to reward')
    else:
        plt.plot(time_reward_node, 'b-', label='Steps to reward')
    plt.legend()
    plt.title(title)
    plt.xlabel("reward")
    plt.ylabel("number of steps")
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_dots.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_visit_freq(visit_frequency, title, save_file_path=None, display=False):
    """todo
    visit_frequency: 1x127 array

    """
    ax = plt.plot(visit_frequency)
    plt.title(title)
    plt.xlabel("node")
    plt.ylabel("no of visits")
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'visit_frequency_plot.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return ax
