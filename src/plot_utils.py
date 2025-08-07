"""
Several useful plotting functions
Plot the node trajectories on maze
"""
import pickle

from matplotlib.collections import LineCollection
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
import numpy as np
from matplotlib.pyplot import figure, subplot, title, suptitle
from numpy import ones
from sklearn.manifold import TSNE
from scipy.optimize import curve_fit

from MM_Maze_Utils import NewMaze, PlotMazeWall, PlotMazeFunction
from MM_Plot_Utils import plot, hist
from MM_Models import ModeMask
from MM_Traj_Utils import LoadTrajFromPath, PlotNodeBiasLocation, PlotNodeBias, TallyNodeStepTypes, SplitModeClips, NewNodes4
from parameters import OUTDATA_PATH, HOME_NODE, WATERPORT_NODE, FRAME_RATE, NODE_LVL, ALL_MAZE_NODES, UnrewNamesSub, \
    full_labels
import parameters as p
from parameters import LEAVE, DRINK, EXPLORE
from utils import get_node_visit_times, get_all_night_nodes_and_times, \
    get_wp_visit_times_and_rwd_times, nodes2cell, get_reward_times, \
    convert_traj_to_episodes, get_outward_pref_order, get_revisits, get_end_nodes_revisits, get_unique_node_revisits, \
    calculate_normalized_visit_frequency_by_level, calculate_normalized_visit_frequency
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
            x = epi[:,0]
            y = epi[:,1]
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
        for node in ALL_MAZE_NODES:
            node_visit_times.append(get_node_visit_times(tf, node))
        for node in ALL_MAZE_NODES:
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
    plt.plot(times_to_rwd, WATERPORT_NODE * ones(len(times_to_rwd)) - .2, linestyle='None', marker='^', label='rwd',
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
    :param interpolate_cell_values: interpolate_cell_values (bool)  # TODO: change this name since it is not only doing the
                                                                        interpolation, but also change the way the function
                                                                        is calculated
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
        other cells colored in white # TODO: change this name since it is not only doing the interpolation, but also change the
                                        way the function is calculated
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

    Save every kth episode when there are more than 20 episodes in total,
    otherwise save all.
    """
    print("# Trajectories", len(episodes_mouse))
    k = 5
    for i, traj in enumerate(episodes_mouse):
        if i >= 50:
            break
        if len(episodes_mouse) >= 20 and i >= 10 and i%k != 0:
            continue
        print("Saving traj", i, ":", traj[:5], '...', traj[-5:])
        plot_trajectory([traj], 'all',
                        save_file_name=os.path.join(save_file_path, f'traj_{i}.png'),
                        display=False,
                        figtitle=f'Traj {i} \n {title}')
        plt.clf()
        plt.close()
    return


def plot_episode_lengths(tfs_labels, re=False, title='', save_file_path=None, display=False):
    """
    A bar plot of all episode lengths
    """
    plt.figure(figsize=(10,5))

    mouse = 'B1' if re else 'B5'
    animal_tf_label = [(LoadTrajFromPath(OUTDATA_PATH + f'{mouse}-tf'), f'mouse {mouse}')]

    all_lengths = []
    colors = []
    labels = []
    for i, (tf, label) in enumerate(animal_tf_label + tfs_labels):
        color = p.ANIMAL_COLOR if i == 0 else p.COLORS[i - 1]
        lengths = [len(e) for e in tf.no]
        all_lengths.append(lengths)
        colors.append(color)
        labels.append(label)

    plt.hist(all_lengths, bins=100, color=colors, label=labels, density=True, stacked=True)

    plttitle = 'Bout lengths'
    if title:
        plttitle = plttitle + '\n' + title
    plt.title(plttitle)
    plt.xlabel("time")
    plt.ylabel("episode length in number of nodes")
    plt.legend()
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'epi_lengths_bars.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_mode_fraction():
    ma = NewMaze(6)

    # Mode analysis for all animals
    Names = p.RewNames + p.UnrewNamesSub
    k = len(p.RewNames)

    for i, nickname in enumerate(Names):
        tf = LoadTrajFromPath(OUTDATA_PATH + nickname + '-tf')
        cl0 = SplitModeClips(tf, ma, re=i < k)  # find the clips; no drink mode for unrewarded animals

        #### ONLY FIRST HALF
        cn = np.cumsum(cl0[:, 2])  # cumulative number of nodes
        ha = np.where(cn > cn[-1] / 2)[0][0]  # index for half the number of nodes
        cl1 = cl0[:ha];
        cl2 = cl0[ha:]  # two clip arrays for first and second half of nodes
        cl = cl1
        ####

        ti = np.array(
            [tf.no[c[0]][c[1] + c[2], 1] - tf.no[c[0]][c[1], 1] for c in cl])  # duration in frames of each clip
        nn = np.array([np.sum(cl[np.where(cl[:, 3] == p.LEAVE)][:, 2]),
                       np.sum(cl[np.where(cl[:, 3] == p.DRINK)][:, 2]),
                       np.sum(cl[np.where(cl[:, 3] == p.EXPLORE)][:, 2])])  # number of node steps in each mode
        nf = np.array([np.sum(ti[np.where(cl[:, 3] == p.LEAVE)]),
                       np.sum(ti[np.where(cl[:, 3] == p.DRINK)]),
                       np.sum(ti[np.where(cl[:, 3] == p.EXPLORE)])])  # number of frames in each mode
        tr = np.zeros((3, 3))  # number of transitions between the 3 modes
        for i in range(1, len(cl)):
            tr[cl[i - 1, 3], cl[i, 3]] += 1
        ce = cl[np.where(cl[:, 3] == p.EXPLORE)]  # clips of exploration
        ne = np.concatenate(
            [tf.no[c[0]][c[1]:c[1] + c[2], 0] for c in ce])  # nodes excluding the last state in each clip
        le = 6  # end nodes only
        ln = list(range(2 ** le - 1, 2 ** (le + 1) - 1))  # list of node numbers in level le
        ns = ne[np.isin(ne, ln)]  # restricted to desired nodes
        wcn = NewNodes4(ns, nf[2] / len(ns))  # compute new nodes vs all nodes for exploration mode only
        with open(OUTDATA_PATH + nickname + '-Modes1', 'wb') as f:  # save in one file per animal
            pickle.dump((nn, nf, tr, wcn), f)

    a = len(Names)
    # nn = np.zeros((a, 3))  # number of nodes in each mode
    nf = np.zeros((a, 3))  # number of frames in each mode
    # tr = np.zeros((a, 3, 3))  # number of transitions between modes
    ff = np.zeros((a, 3))  # fraction of frames in each mode
    # wcn = []  # these arrays have different shape
    for i, nickname in enumerate(Names):
        with open(OUTDATA_PATH + nickname + '-Modes1', 'rb') as f:
            _, nf1, _, _ = pickle.load(f)
        # nn[i] = nn1
        nf[i] = nf1
        # tr[i] = tr1
        # wcn += [wcn1]
        s = sum(nf[i])
        ff[i] = nf[i] / s

    c, n, nf = em.exploration_efficiency(tf, re=True, le=6)

    labels = 'Leave', 'Drink', 'Explore'
    av = np.mean(ff[:k], axis=0)
    sizes = [av[0], av[1], av[2]]
    explode = (0, 0, 0.1)  # only "explode" the 2nd slice
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
            shadow=False, startangle=90, colors=['red', 'cyan', 'green'])
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
    #         shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.savefig('figs/RewPie.pdf')
    plt.show()


def plot_exploration_efficiency(tfs_labels, re, half=0, le=6, title='', save_file_path=None, display=False):
    """
    :param episodes: [[], [], ...] (list of list of nodes for each trajectory)
    :param re: True for rewarded animals, False for unrewarded (i.e. if you want
    to treat a visit to 116 as rewarded then True. If there is no reward and it's
    only exploration, keep it False).
    :param half: 'separate' if plot two halves separately on one plot, 0 if all night,
    1 if only first half, 2 if only second half
    """

    if re:
        animal_set = p.RewNames
    else:
        animal_set = p.UnrewNamesSub

    def get_animal_dict(h):
        if re:
            ee_dict = em.get_rewarded_ee(h, le)
            animal_lbl = p.REW_ANIMALS_PLOT_LABEL
            color = {-1: 'g', 0: p.ANIMAL_COLOR, 1: 'c', 2: 'y'}[h]
        else:
            ee_dict = em.get_unrewarded_ee(h, le)
            animal_lbl = p.UNREW_ANIMALS_PLOT_LABEL
            color = p.ANIMAL_COLOR
        if h != 0:
            animal_lbl = animal_lbl + f'-h={h}'
        return ee_dict, animal_lbl, color

    def plot_animal_ee(ee_dict, lbl, color, linestyle, alpha):
        c, n, _ = ee_dict[animal_set[0]]  # plot one animal (to get the legend right)
        plt.plot(c, n, color=color, linestyle=linestyle, alpha=alpha, linewidth=1, label=lbl)
        for nickname in animal_set[1:]:  # plot rest of the animals
            c, n, _ = animals_ee_dict[nickname]
            plt.plot(c, n, color=color, linestyle=linestyle, alpha=alpha, linewidth=1)
        return

    plt.figure()

    # animal data
    if half == 'separate':

        if re:
            animals_ee_dict, animal_label, color = get_animal_dict(-1)
            plot_animal_ee(animals_ee_dict, animal_label, color, linestyle='--', alpha=0.8)

        animals_ee_dict, animal_label, color = get_animal_dict(1)
        plot_animal_ee(animals_ee_dict, animal_label, color, linestyle='--', alpha=0.4)

        animals_ee_dict, animal_label, color = get_animal_dict(2)
        plot_animal_ee(animals_ee_dict, animal_label, color, linestyle='--', alpha=0.5)

    else:
        assert half in [0, 1, 2]
        animals_ee_dict, animal_label, color = get_animal_dict(half)
        plot_animal_ee(animals_ee_dict, animal_label, color, linestyle='-.', alpha=0.4)

    # random
    c, n = em.get_random_ee(le)
    plt.plot(c, n, 'black', linestyle='-.', label='random')

    # DFS
    c, n = em.get_dfs_ee(le)
    plt.plot(c, n, 'green', label='optimal')

    # Get this agent's ee
    for i, (tf, label) in enumerate(tfs_labels):
        if half == 'separate':
            c, n, _ = em.exploration_efficiency(tf, re=re, le=le, half=1)
            plt.plot(c, n, f'{p.COLORS[i]}o-', label=(label if label else f'agent {i}') + '-h1')
            c, n, _ = em.exploration_efficiency(tf, re=re, le=le, half=2)
            plt.plot(c, n, f'{p.COLORS[i]}o-', alpha=0.5, label=(label if label else f'agent {i}') + '-h2')
        else:
            c, n, _ = em.exploration_efficiency(tf, re=re, le=le, half=half)
            agent_label = label if label else f'agent {i}'
            if half != 0:
                agent_label = agent_label + f'-h{half}'
            plt.plot(c, n, f'{p.COLORS[i]}o-', label=agent_label)

    plt.xscale('log', base=10)
    plttitle = f'Exploration Efficiency (Level {le})'
    if title:
        plttitle = plttitle + '\n' + title
    plt.title(plttitle)
    plt.xlabel(f"Nodes visited (level={le})")
    plt.ylabel(f"New nodes found (level={le})")
    plt.legend()
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'exp_efficiency_le{le}_half={half}.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def nodebias(tr, ma):
    tu = TallyNodeStepTypes(tr,ma)
    n = 2**ma.le-1 # number of nodes below end node level
    # bo = (tu[:n,2]+tu[:n,3])/np.sum(tu[:n,:],axis=1) # (outleft+outright)/(outleft+outright+inleft+inright)
    # so = np.sqrt((tu[:n,2]+tu[:n,3])*(tu[:n,0]+tu[:n,1])/np.sum(tu[:n,:],axis=1)**3) # std dev
    # plot(bo,fmts=['g-'],legend=['back'],linewidth=2,
    #      xlabel='Node',ylabel='back(back+left+right)',figsize=(10,3),grid=True);
    # plt.errorbar(range(len(bo)),bo,yerr=so,fmt='none');
    bl = tu[:n,0]/(tu[:n,0]+tu[:n,1]) # inleft/(inleft+inright)
    sl = np.sqrt(tu[:n,0]*tu[:n,1]/(tu[:n,0]+tu[:n,1])**3) # std dev
    # plot(bl,fmts=['r-'],legend=['left'],linewidth=2,
    #      xlabel='Node',ylabel='left/(left+right)',figsize=(10,3),grid=True)
    # plt.errorbar(range(len(bl)),bl,yerr=sl,fmt='none')
    return bl, sl


def plot_percent_turns(tfs_labels, re=False, title='', save_file_path=None, display=False):

    mouse = 'B1' if re else 'B5'
    animal_tf_label = [(LoadTrajFromPath(OUTDATA_PATH + f'{mouse}-tf'), f'mouse {mouse}')]
    ma = NewMaze(6)

    labels = []
    bls = []
    sls = []
    fmts = []
    for i, (tf, label) in enumerate(animal_tf_label + tfs_labels):
        labels.append(label if label else f'agent {i}')
        bl, sl = nodebias(tf, ma)
        bls.append(bl)
        sls.append(sl)
        fmts.append((p.ANIMAL_COLOR if i == 0 else p.COLORS[i-1]) + '-')

    plot([[0.5] * 64] + bls,
         fmts=['k:'] + fmts,
         legend=['random'] + labels,
         figsize=(10, 3),
         linewidth=1.2, xlabel='Node', ylabel='Left bias', grid=True, loc='lower left')

    for bl, sl in zip(bls, sls):
        plt.errorbar(range(len(bl)), bl, yerr=sl, fmt='none')

    plttitle = 'Node Biases'
    if title:
        plttitle = plttitle + '\n' + title
    plt.title(plttitle)
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'node_bias.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()

    for i, (tf, label) in enumerate(animal_tf_label + tfs_labels):
        plt.figure()
        PlotNodeBiasLocation(tf, ma)
        plttitle = 'Spatial distribution of left-right Bias'
        if title:
            plttitle = plttitle + '\n' + title
        plt.title(plttitle)
        if save_file_path:
            os.makedirs(save_file_path, exist_ok=True)
            plt.savefig(os.path.join(save_file_path, f'node_bias_maze_{labels[i]}_{i}.png'), bbox_inches='tight', dpi='figure')
        if display:
            plt.show()
        plt.clf()
        plt.close()
    return


def plot_percent_turns_DEPRECATED(tf, title=None, save_file_path=None, display=False):
    raise NotImplementedError
    seqs_level2 = [[0, 1, 4],
                   [0, 1, 3],
                   [0, 2, 6],
                   [0, 2, 5]]

    seqs_level3 = []
    for s in seqs_level2:
        l, r = 2*s[-1]+1, 2*s[-1] + 2
        seqs_level3.append(s + [l])
        seqs_level3.append(s + [r])

    seqs_level4 = []
    for s in seqs_level3:
        l, r = 2*s[-1]+1, 2*s[-1] + 2
        seqs_level4.append(s + [l])
        seqs_level4.append(s + [r])

    seqs_level5 = []
    for s in seqs_level4:
        l, r = 2*s[-1]+1, 2*s[-1] + 2
        seqs_level5.append(s + [l])
        seqs_level5.append(s + [r])

    # data
    with open(p.OUTDATA_PATH + 'outward_turns_unrewarded.pkl', 'rb') as f:
        pref_unrewarded_le = pickle.load(f)

    plt.figure()
    for le, seqs_level in zip([2, 3, 4, 5], [seqs_level2, seqs_level3, seqs_level4, seqs_level5]):
        outward_prefs = [0]
        total = 0
        for node_seq in seqs_level:
            turn_node = node_seq[-1]
            pref_order = get_outward_pref_order(turn_node)
            assert len(pref_order) == 2
            counts, seq_samples = em.percent_turns(tf, node_seq, pref_order[0], pref_order[1])
            # print(node_seq, turn_node, pref_order, counts, seq_samples)
            if seq_samples < 2:
                continue
            outward_prefs.append(counts[pref_order[0]])
            total += seq_samples
        print(f"le={le}:", outward_prefs)

        plt.plot([le]*len(pref_unrewarded_le[le]), pref_unrewarded_le[le],  'b.', label=f'unrewarded' if le == 2 else '')
        plt.plot(le, sum(outward_prefs)*100/total, 'ro', label=f'agent' if le == 2 else '')
        plt.plot(le, 50.0, 'ko', label=f'random' if le == 2 else '')

    plt.xlabel('Level')
    plt.ylabel('Percentage outward turns')
    plt.ylim([0, 100])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(loc='lower right')
    if title:
        plt.title(title)
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'percent_outward_turns.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_first_endnode_labels(tfs_labels, re=False, title='', save_file_path=None, display=False):

    animal_lbl = p.REW_ANIMALS_PLOT_LABEL if re else p.UNREW_ANIMALS_PLOT_LABEL
    precomputed_file = 'first_endnode_label_rewarded.pkl' if re else 'first_endnode_label_unrewarded.pkl'

    plt.figure()

    # animal data
    with open(p.OUTDATA_PATH + precomputed_file, 'rb') as f:
        first_endnode_label_animal = pickle.load(f)

    for i, k in enumerate(first_endnode_label_animal):
        plt.plot([k]*len(first_endnode_label_animal[k]), first_endnode_label_animal[k], f'{p.ANIMAL_COLOR}.', label=animal_lbl if i == 0 else '')

    # random
    plt.plot(list(first_endnode_label_animal.keys()), [25.0] * len(first_endnode_label_animal), 'ko', label='random')

    # this agent
    for i, (tf, label) in enumerate(tfs_labels):
        first_visit_label_fracs = em.first_endnode_label(tf)
        print(f"first_visit_label_fracs {i}", first_visit_label_fracs)
        plt.plot(list(first_visit_label_fracs.keys()), list(first_visit_label_fracs.values()), f'{p.COLORS[i]}o', label=label if label else f'agent {i}')

    plt.xticks(rotation=5)
    plt.ylim([0, 100])
    plt.ylabel('Preference node type in the corner')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(loc='upper right')
    plt.title(f"Type of endnode first hit on entering a subQ\n{title}")
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'first_endnode_hits.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return

    # # Plotting on the maze
    # plt.figure()
    # first_visit_plot = np.array([0]*len(ALL_MAZE_NODES))
    # first_visit_plot[63] = first_visit_label_fracs.get(p.full_labels[p.STRAIGHT], 0)  # s
    # first_visit_plot[64] = first_visit_label_fracs.get(p.full_labels[p.OPP_STRAIGHT], 0)  # o_s
    # first_visit_plot[65] = first_visit_label_fracs.get(p.full_labels[p.BENT_STRAIGHT], 0)  # bs
    # first_visit_plot[66] = first_visit_label_fracs.get(p.full_labels[p.OPP_BENT_STRAIGHT], 0)  # o_bs
    # PlotMazeFunction(first_visit_plot/100, NewMaze(6), mode='nodes', numcol='g', figsize=6)
    # if title:
    #     plt.title(title)
    # if save_file_path:
    #     plt.savefig(os.path.join(save_file_path, f'first_endnode_hits_maze.png'), bbox_inches='tight', dpi='figure')
    # if display:
    #     plt.show()
    # plt.clf()
    # plt.close()


def plot_opposite_node_preference(tfs_labels, re=False, title='', save_file_path=None, display=False):

    plt.figure()

    mouse = 'B1' if re else 'B5'
    animal_tf_label = [(LoadTrajFromPath(OUTDATA_PATH+f'{mouse}-tf'), f'mouse {mouse}')]

    for i, (tf, label) in enumerate(animal_tf_label + tfs_labels):
        label_transition_counts = em.second_endnode_label(tf)

        total_transitions = 0
        for l1, l2 in label_transition_counts.items():
            total_transitions += sum(l2.values())
            # factor = 100.0 / sum(c.values())
            # label_transition_counts[i] = {k: round(v * factor, 2) for k, v in c.items()}  # normalize
        opposite_transitions = sum([
            label_transition_counts[p.STRAIGHT].get(p.OPP_STRAIGHT, 0),
            label_transition_counts[p.OPP_STRAIGHT].get(p.STRAIGHT, 0),
            label_transition_counts[p.BENT_STRAIGHT].get(p.OPP_BENT_STRAIGHT, 0),
            label_transition_counts[p.OPP_BENT_STRAIGHT].get(p.BENT_STRAIGHT, 0)
        ])
        opp_transition_percent = (opposite_transitions * 100) / total_transitions
        print("i", label, opp_transition_percent)
        # print("same", percents)
        # percents = np.array(percents)
        # percents = percents[percents != 0]
        # print("same", percents)
        color = (p.ANIMAL_COLOR if i == 0 else p.COLORS[i - 1])
        # plt.plot([1]*len(percents), percents, f'{color}o', label=label if label else f'agent {i}')
        plt.plot(1, opp_transition_percent, f'{color}o', label=label if label else f'agent {i}')

    plt.xticks([1], ['same subquad'])
    # plt.ylabel('opposite node preference at endnodes')
    plt.legend(loc='upper right')
    plt.ylim([0, 100])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f"Preference for the opposite node in the same subquadrant\n{title}")
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'opposite_endnode_preference.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_revisits():
    raise NotImplementedError
    for sub in p.UnrewNamesSub[:2]:
        revisit = get_revisits(sub)
        for node in [0]:
            total = len(revisit[node])
            plt.figure(figsize=(15, 4))
            plt.plot(range(total), revisit[node])
            plt.title(f"animal {sub}, node {node}")
            plt.xlabel("as time goes by")
            plt.ylabel("number of nodes in between revisits")
            plt.show()
    return


def plot_end_node_revisits(tf, title='', save_file_path=None, display=False):
    raise NotImplementedError
    N = 63
    end_nodes_revisit = get_end_nodes_revisits(tf)
    for node in [N]:
        revisits = end_nodes_revisit[node]
        total = len(revisits)
        plt.figure(figsize=(15, 4))
        plt.plot(range(total), revisits, 'o')
        plt.title(f"{title} node {node}")
        plt.xlabel("as time goes by")
        #         plt.ylim([-1, 10])
        plt.gca().yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.ylabel("Count of end-nodes in between revisits")
        if save_file_path:
            plt.savefig(os.path.join(save_file_path, f'fraction_endnode_revisits_node{node}_alltime.png'),
                        bbox_inches='tight', dpi='figure')
        if display:
            plt.show()
        plt.clf()
        plt.close()
    return


def plot_node_revisits_level_halves(tfs_labels, level_to_plot,  re=False, title='', save_file_path=None, display=False):

    plt.figure(figsize=(9, 4))
    mouse = 'B1' if re else 'B5'
    animal_tf_label = [(LoadTrajFromPath(OUTDATA_PATH+f'{mouse}-tf'), f'mouse {mouse}')]
    for i, (tf, label) in enumerate(animal_tf_label + tfs_labels):
        print(label)
        revisit_phase = [None, None]
        revisit_phase[0] = get_revisits(tf, level_to_plot, 'first_half')
        revisit_phase[1] = get_revisits(tf, level_to_plot, 'second_half')
        for half in [0, 1]:
            revisit = revisit_phase[half]
            to_plot = []
            for node in p.NODE_LVL[level_to_plot]:
                to_plot += revisit[node]
            total = len(to_plot)
            d = {x: to_plot.count(x) / total for x in to_plot if x <= 20}
            d = sorted(d.items(), key=lambda x: x[0])
            color = (p.ANIMAL_COLOR if i == 0 else p.COLORS[i-1]) + ('.:' if half else '.-')
            plt.plot([k[0] for k in d], [k[1] for k in d], color, label=(label if label else f'agent {i}') + f': Half {half+1}')
    plt.legend()
    plt.title(f"Revisits{title}\nLevel {level_to_plot} (excludes Reward subquadrant)")
    plt.xlabel("number of nodes before revisiting the same node")
    plt.ylabel("fraction of total revisits")
    # plt.xlim([0.1, 10])
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'fraction_node_revisits_level{level_to_plot}_{title}_halves.png'),
                    bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()

    return


def plot_end_node_revisits_level_all_time(tf, levels=p.NODE_LVL, title='', save_file_path=None, display=False):
    revisits = get_end_nodes_revisits(tf, 'all')
    for level in levels:
        plt.figure(figsize=(15, 3))
        to_plot = []
        for node in p.NODE_LVL[level]:
            to_plot += revisits[node]
        total = len(to_plot)
        d = {x: to_plot.count(x) / total for x in to_plot}
        #     d = dict(filter(lambda x: x[0] <= 50, d.items()))
        plt.bar(d.keys(), d.values(), width=0.4)
        plt.title(f"{title}\nLevel {level}:   Sample size={len(to_plot)}")
        plt.xlabel("number of endnodes before revisiting the same node")
        plt.ylabel("fraction of total revisits")
        plt.xlim([-1, 50])
        plt.ylim([0.0, max(d.values()) + 0.05])
        if save_file_path:
            os.makedirs(save_file_path, exist_ok=True)
            plt.savefig(os.path.join(save_file_path, f'fraction_endnode_revisits_level{level}_{title}_alltime.png'),
                        bbox_inches='tight', dpi='figure')
        if display:
            plt.show()
        plt.clf()
        plt.close()
    return


def plot_end_node_revisits_level_halves(tf, levels=p.NODE_LVL, title='', save_file_path=None, display=False):
    revisit_phase = [None, None]
    revisit_phase[0] = get_end_nodes_revisits(tf, 'first_half')
    revisit_phase[1] = get_end_nodes_revisits(tf, 'second_half')
    for level in levels:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        for half in [0, 1]:
            revisit = revisit_phase[half]
            to_plot = []
            for node in p.NODE_LVL[level]:
                to_plot += revisit[node]
            total = len(to_plot)
            d = {x: to_plot.count(x) / total for x in to_plot}
            ax[half].bar(d.keys(), d.values(), width=0.45)
            ax[half].set_title(f"{title}\nLevel {level}:   Sample size={len(to_plot)}  Half={half + 1}")
            ax[half].set_xlabel("number of endnodes before revisiting the same node")
            ax[half].set_ylabel("fraction of total revisits")
            ax[half].set_xlim([-1, 50])
            ax[half].set_ylim([0.0, max(d.values()) + 0.05])
        if save_file_path:
            os.makedirs(save_file_path, exist_ok=True)
            plt.savefig(os.path.join(save_file_path, f'fraction_endnode_revisits_level{level}_{title}_halves.png'),
                        bbox_inches='tight', dpi='figure')
        if display:
            plt.show()
        plt.clf()
        plt.close()
    return


def plot_unique_node_revisits_level_halves(tf, levels=p.NODE_LVL, title='', save_file_path=None, display=False):
    revisit_phase = [None, None]
    revisit_phase[0] = get_unique_node_revisits(tf, 'first_half')
    revisit_phase[1] = get_unique_node_revisits(tf, 'second_half')
    for level in levels:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        for half in [0, 1]:
            revisit = revisit_phase[half]
            to_plot = []
            for node in p.NODE_LVL[level]:
                if node not in revisit: continue
                to_plot += revisit[node]
            total = len(to_plot)
            d = {x: to_plot.count(x) / total for x in to_plot}
            ax[half].bar(d.keys(), d.values(), width=0.45)
            ax[half].set_title(f"{title}\nLevel {level}:   Sample size={len(to_plot)}  Half={half + 1}")
            ax[half].set_xlabel("number of unique nodes before revisiting the same node")
            ax[half].set_ylabel("fraction of total revisits")
            ax[half].set_xlim([-1, 50])
            ax[half].set_ylim([0.0, max(d.values()) + 0.05])
        if save_file_path:
            os.makedirs(save_file_path, exist_ok=True)
            plt.savefig(os.path.join(save_file_path, f'fraction_uniquenode_revisits_level{level}_{title}_halves.png'),
                        bbox_inches='tight', dpi='figure')
        if display:
            plt.show()
        plt.clf()
        plt.close()
    return


def plot_decision_biases(tfs_labels, re, title='', save_file_path=None, display=False):
    """
    Plots the decision biases and prints their standard deviation for each agent
    """

    animal_label = p.REW_ANIMALS_PLOT_LABEL if re else p.UNREW_ANIMALS_PLOT_LABEL

    figure()

    # load precomputed animal biases
    bi_data = em.get_decision_biases_animal(re)

    # get this agent's biases
    labels = []
    tfs = []
    for i, (tf, label) in enumerate(tfs_labels):
        labels.append(label if label else f'agent {i}')
        tfs.append(tf)

    bi_agents, _ = em.get_decision_biases(tfs, re)

    # fmts = [f'{p.ANIMAL_COLOR}.', 'k+'] + [(p.COLORS[i] + '.') for i, _ in enumerate(labels)]
    fmts = [f'{p.ANIMAL_COLOR}.', 'k+'] + ['c.']*10 + ['r.']*10 + ['g.']*10

    # plot biases BF vs SF
    ax = subplot(121)
    x = [bi_data[:, 0], [2 / 3]] + [[bi_agents[i, 0]] for i in range(len(bi_agents))]
    y = [bi_data[:, 2], [2 / 3]] + [[bi_agents[i, 2]] for i in range(len(bi_agents))]
    plot(x, y, fmts=fmts, markersize=7,
         xlim=[0, 1], ylim=[0, 1], equal=True, axes=ax,
         # legend=[animal_label, 'random']+labels,
         xlabel='$P_{\mathrm{SF}}$', ylabel='$P_{\mathrm{BF}}$', loc='lower left')

    # plot biases BS vs SA
    ax = subplot(122)
    x = [bi_data[:, 1], [1 / 2]] + [[bi_agents[i, 1]] for i in range(len(bi_agents))]
    y = [bi_data[:, 3], [1 / 2]] + [[bi_agents[i, 3]] for i in range(len(bi_agents))]
    plot(x, y, fmts=fmts, markersize=7,
         xlim=[0, 1], ylim=[0, 1], equal=True, axes=ax,
         # legend=[animal_label, 'random']+labels,
         xlabel='$P_{\mathrm{SA}}$', ylabel='$P_{\mathrm{BS}}$', loc='lower left')

    plttitle = 'Decision Biases'
    if title:
        plttitle = plttitle + '\n' + title
    suptitle(plttitle)
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'decision_biases.png'), bbox_inches='tight', dpi='figure')
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_markov_fit_non_pooling(tf, re, title='', save_file_path=None, display=False):
    """
    """
    if len(tf.no) <= 5:
        print("Not plotting markov_fit_pooling coz of insufficient number of episodes.")
        return
    start = time.time()
    print("plotting markov_fit_non_pooling")
    [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr] = em.markov_fit_non_pooling(tf, re)
    print("plot_markov_fit_non_pooling", time.time()-start, "seconds")
    plot([hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr],
             fmts=['r.-', 'g.-', 'b.-', 'y.-'], markersize=8, linewidth=1,
             xlabel='Average depth of history', ylabel='Cross-entropy',
             legend=['fix test', 'var test', 'fix train', 'var train'],
             loc='upper center', figsize=(5, 4), title=f'Markov Fit (non pooling)\n{title}')

    print("avg depth for min cross entropy")
    ef, hf = sorted(zip(cf5, hf5))[0]
    ev, hv = sorted(zip(cv5, hv5))[0]

    # mark the minimum ones
    plt.plot(hf, ef, fillstyle='none', markersize=15, color='tab:red', marker='o')
    plt.plot(hv, ev, fillstyle='none', markersize=15, color='tab:green', marker='o')

    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'markov_fit_non_pooling.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_markov_fit_pooling(tf, label, re, title='', save_file_path=None, display=False):
    """
    """
    if len(tf.no) <= 6:
        print("Not plotting markov_fit_pooling coz of insufficient number of episodes.")
        return
    start = time.time()
    print("plotting markov_fit_pooling")
    [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr] = em.markov_fit_pooling(tf, re)
    print("plot_markov_fit_pooling end", time.time()-start, "seconds")
    plttitle = 'Markov Fit (pooling)'
    if title:
        plttitle = plttitle + '\n' + title
    plot([hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr],
         fmts=['r.-', 'g.-', 'b.-', 'y.-'], markersize=8, linewidth=1,
         xlabel='Average depth of history', ylabel='Cross-entropy',
         # ylim=[1.15, 1.75],
         legend=['fix test', 'var test', 'fix train', 'var train'],
         loc='lower left', figsize=(5, 4), title=plttitle)

    print("avg depth for min cross entropy")
    ef, hf = sorted(zip(cf5, hf5))[0]
    ev, hv = sorted(zip(cv5, hv5))[0]
    print("fix", hf, ef)
    print("var", hv, ev)

    # mark the minimum ones
    plt.plot(hf, ef, fillstyle='none', markersize=15, color='tab:red', marker='o')
    plt.plot(hv, ev, fillstyle='none', markersize=15, color='tab:green', marker='o')
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'markov_fit_pooling_{label}.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_trajectory_features(episodes, title=None, save_file_path=None, display=False):

    simulated_features = em.get_feature_vectors(episodes)
    print("simulated_features", simulated_features.shape)

    # animal data
    mouse_episodes = convert_traj_to_episodes(LoadTrajFromPath(p.OUTDATA_PATH + 'B5-tf'))
    mouse_features = em.get_feature_vectors(mouse_episodes)
    print("mouse_features", mouse_features.shape)

    # tsne = TSNE()
    # tsne_results = tsne.fit_transform(simulated_features)
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label='agent')
    #
    # tsne_1 = TSNE()
    # tsne_results_1 = tsne_1.fit_transform(mouse_features)
    # plt.scatter(tsne_results_1[:, 0], tsne_results_1[:, 1], label='mouse B5')
    # plt.xlabel('t-sne1')
    # plt.ylabel('t-sne2')
    #
    # plt.legend()
    # plt.show()
    # plt.close()

    plt.figure()
    plt.scatter(mouse_features[:, 0], mouse_features[:, 1], s=10, alpha=0.6, label='mouse')
    plt.scatter(simulated_features[:, 0], simulated_features[:, 1], s=10, label='agent')
    plt.xlabel('rotational velocity')
    plt.ylabel('mean diffusivity')
    plt.legend()
    if save_file_path: plt.savefig(os.path.join(save_file_path, f'rot_vs_diff.png'))
    if display: plt.show()
    plt.clf(); plt.close()

    plt.figure()
    plt.scatter(mouse_features[:, 1], mouse_features[:, 2], s=10, alpha=0.6, label='mouse')
    plt.scatter(simulated_features[:, 1], simulated_features[:, 2], s=10, label='agent')
    plt.xlabel('mean diffusivity')
    plt.ylabel('tortuosity')
    plt.legend()
    if save_file_path: plt.savefig(os.path.join(save_file_path, f'diff_vs_tort.png'))
    if display: plt.show()
    plt.clf(); plt.close()

    plt.figure()
    plt.scatter(mouse_features[:, 0], mouse_features[:, 2], s=10, alpha=0.6, label='mouse')
    plt.scatter(simulated_features[:, 0], simulated_features[:, 2], s=10, label='agent')
    plt.xlabel('rotational velocity')
    plt.ylabel('tortuosity')
    plt.legend()
    if save_file_path: plt.savefig(os.path.join(save_file_path, f'rot_vs_tort.png'))
    if display: plt.show()
    plt.clf(); plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        mouse_features[:, 0],
        mouse_features[:, 1],
        mouse_features[:, 2],
        color='b', s=10, alpha=0.6, label='mouse B5')
    ax.scatter(
        simulated_features[:, 0],
        simulated_features[:, 1],
        simulated_features[:, 2],
        color='r', s=10, label='agent')
    ax.set_xlabel('mean rotational_velocity')
    ax.set_ylabel('mean diffusivity')
    ax.set_zlabel('tortuosity')
    ax.set_title(title)
    plt.legend()
    if save_file_path: plt.savefig(os.path.join(save_file_path, f'trajectory_features.png'))
    if display: plt.show()
    plt.clf(); plt.close()
    return


def reward_lengths():
    u=np.concatenate(w, axis=-1)
    ar = u[2]  # number of rewards prior to this path
    al = u[3]  # length of this path
    ar1 = ar[al < 1000];
    al1 = al[al < 1000]  # eliminate single outlier

    rl = sorted(list(set(ar)))  # sorted list of all reward numbers
    re = [];
    me = [];
    sd = [];
    se = []
    for r in rl:
        x = al1[ar1 == r]
        re += [r]
        me += [np.median(x)]  # median of all the durations at that reward number

    popt, pcov = curve_fit(func, re[:101], me[:101], p0=[50, .1, 10])  # fit the exponential

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [1, 3.5]})

    plot(ar, al, fmts=['co'], markersize=2,
         xlabel='Number of rewards experienced', ylabel='Length of water run (steps)', axes=axes[1, 0]);
    plot(re, me, fmts=['bo'], markersize=4, axes=axes[1, 0]);
    x = np.arange(re[-1] + 1)  # reward number
    plot(x, func(x, *popt), fmts=['b-'], linewidth=2, ylim=[0, 160], xlim=[-0.5, 100.5],
         legend=['All water runs', 'Median', 'Exp fit, decay = {:.1f}'.format(1 / popt[1])],  # ,'Mean ± SEM'
         loc='upper right', axes=axes[1, 0]);

    hist([al[al == 6], al[al > 6]], bins=np.arange(-1, 161, 2), xlabel='Number of runs',
         orientation='horizontal', xhide=False, yhide=False, ylim=[0, 160], xscale='log', xticks=[1e0, 1e1, 1e2, 1e3],
         axes=axes[1, 1]);  # ,yticks=[0,0.5,1],
    axes[1, 1].tick_params(labelleft=False)

    bin = 5  # reward binning
    g0 = np.where(al == 6)[0]  # perfect runs
    g1 = np.where(al > 6)[0]  # longer runs
    r0 = sorted(ar[g0])  # sorted reward numbers for perfect paths
    r1 = sorted(ar[g1])  # sorted reward numbers for longer paths
    # _,n,bins,_=hist([r0,r1],bins=np.arange(140//bin)*bin);
    n0, bins = np.histogram(r0, bins=np.arange(140 // bin) * bin);  # histo of reward numbers for perfect paths
    n1, _ = np.histogram(r1, bins=np.arange(140 // bin) * bin);  # histo of reward numbers for longer paths
    y0 = n0 / (n0 + n1)  # fraction of perfect paths in each reward number bin
    y1 = n1 / (n0 + n1)  # fraction of longer paths in each reward number bin
    x = bins[:-1]  # reward number bins
    du = [np.median(u[0, np.logical_and(u[3] == 6, np.logical_and(u[2] >= i, u[2] < i + bin))])
          # median duration of perfect paths
          for i in x]  # in each bin of reward numbers

    ax1 = axes[0, 0]  # top left graph
    color = 'tab:orange'
    ax1.set_ylabel('Fraction perfect', color=color, fontsize=12)
    ax1.plot(x, y0, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(labelbottom=False)
    ax1.set_xlim([-0.5, 100.5])
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:purple'
    ax2.set_ylabel('Duration (s)', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(x, du, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 10])

    # plot(x,y0,fmts=['r-','g-'],ylim=[0,1],ylabel='Fraction perfect',grid=True,
    #     loc='center right',xlim=[-0.5,100.5],axes=axes[0,0]); # ,title='Runs from entrance to water port: length distribution vs rewards'

    axes[0, 1].axis('off')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(wspace=0.04, hspace=0.1)
    plt.savefig('figs/WaterRunLengthMedianHistStack.pdf')
    return


def plot_reward_path_lengths(tfs_labels, title, save_file_path=None, dots=True, display=False):
    """todo
    time_reward_node:

    """
    tf, label = tfs_labels[0]
    episodes = convert_traj_to_episodes(tf)
    visit_reward_node, time_reward_node = get_reward_times(episodes)
    if dots:
        plt.plot(time_reward_node, 'b.', label='Steps to reward')
    else:
        plt.plot(time_reward_node, 'b-', label='Steps to reward')
    # plt.hist(time_reward_node)
    plt.legend()
    plt.title(title)
    plt.xlabel("reward")
    plt.ylabel("number of steps")
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_dots={dots}.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_reward_runs(tfs_labels, title, save_file_path=None, display=False):
    # Exponential fit to the median run for each reward value; plot all data in background
    def func(x, a, b, c):  # exponential fit
        return a * np.exp(-b * x) + c

    w = []
    for i, nickname in enumerate(p.RewNames):
        tf = LoadTrajFromPath(p.OUTDATA_PATH + nickname + '-tf')
        re = np.array([y[0] + tf.fr[i, 0] for i, r in enumerate(tf.re) for y in r])  # reward times in frames
        wd = []
        wt = []
        wr = []
        wl = []
        for j, b in enumerate(tf.no):
            wk = np.where(b[:, 0] == 116)[0]  # node index of visits to water port
            if len(wk) > 0:  # if there was a visit to water port
                d = b[wk[0], 1]  # frames from start of bout to water port
                t = tf.fr[j, 0] + d  # absolute time of that visit in frames
                rk = np.where(re > t)[0]  # rewards after time t
                if len(rk) == 0:  # if no reward after t
                    r = len(re)  # total number of rewards
                else:
                    r = rk[0]  # number of rewards received before t
                wd += [d / 30]  # seconds from start of bout to water port
                wt += [t / 30]  # absolute time of that visit in seconds
                wr += [r]  # number of rewards prior to that visit
                wl += [wk[0]]  # number of steps to water port
        w += [np.array([wd, wt, wr, wl])]

    # with open('outdata/FirstWaterRuns', 'rb') as f:
    #     RewNames, w = pickle.load(f)

    u = np.concatenate(w, axis=-1)
    ar = u[2]  # number of rewards prior to this path
    al = u[3]  # length of this path
    ar1 = ar[al < 1000];
    al1 = al[al < 1000]  # eliminate single outlier

    rl = sorted(list(set(ar)))  # sorted list of all reward numbers
    re = [];
    me = [];
    sd = [];
    se = []
    for r in rl:
        x = al1[ar1 == r]
        re += [r]
        me += [np.median(x)]  # median of all the durations at that reward number

    popt, pcov = curve_fit(func, re[:101], me[:101], p0=[50, .1, 10])  # fit the exponential

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [1, 3.5]})

    plot(ar, al, fmts=['co'], markersize=2,
         xlabel='Number of rewards experienced', ylabel='Length of water run (steps)', axes=axes[1, 0]);
    plot(re, me, fmts=['bo'], markersize=4, axes=axes[1, 0]);
    x = np.arange(re[-1] + 1)  # reward number
    plot(x, func(x, *popt), fmts=['b-'], linewidth=2, ylim=[0, 160], xlim=[-0.5, 100.5],
         legend=['All water runs', 'Median', 'Exp fit, decay = {:.1f}'.format(1 / popt[1])],  # ,'Mean ± SEM'
         loc='upper right', axes=axes[1, 0]);

    hist([al[al == 6], al[al > 6]], bins=np.arange(-1, 161, 2), xlabel='Number of runs',
         orientation='horizontal', xhide=False, yhide=False, ylim=[0, 160], xscale='log', xticks=[1e0, 1e1, 1e2, 1e3],
         axes=axes[1, 1]);  # ,yticks=[0,0.5,1],
    axes[1, 1].tick_params(labelleft=False)

    bin = 5  # reward binning
    g0 = np.where(al == 6)[0]  # perfect runs
    g1 = np.where(al > 6)[0]  # longer runs
    r0 = sorted(ar[g0])  # sorted reward numbers for perfect paths
    r1 = sorted(ar[g1])  # sorted reward numbers for longer paths
    # _,n,bins,_=hist([r0,r1],bins=np.arange(140//bin)*bin);
    n0, bins = np.histogram(r0, bins=np.arange(140 // bin) * bin);  # histo of reward numbers for perfect paths
    n1, _ = np.histogram(r1, bins=np.arange(140 // bin) * bin);  # histo of reward numbers for longer paths
    y0 = n0 / (n0 + n1)  # fraction of perfect paths in each reward number bin
    y1 = n1 / (n0 + n1)  # fraction of longer paths in each reward number bin
    x = bins[:-1]  # reward number bins
    du = [np.median(u[0, np.logical_and(u[3] == 6, np.logical_and(u[2] >= i, u[2] < i + bin))])
          # median duration of perfect paths
          for i in x]  # in each bin of reward numbers

    ax1 = axes[0, 0]  # top left graph
    color = 'tab:orange'
    ax1.set_ylabel('Fraction perfect', color=color, fontsize=12)
    ax1.plot(x, y0, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(labelbottom=False)
    ax1.set_xlim([-0.5, 100.5])
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:purple'
    ax2.set_ylabel('Duration (s)', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(x, du, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 10])

    # plot(x,y0,fmts=['r-','g-'],ylim=[0,1],ylabel='Fraction perfect',grid=True,
    #     loc='center right',xlim=[-0.5,100.5],axes=axes[0,0]); # ,title='Runs from entrance to water port: length distribution vs rewards'

    axes[0, 1].axis('off')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(wspace=0.04, hspace=0.1)
    plt.title(f'Runs from entrance to water port\n{title}')
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'WaterRunLengthMedianHistStack.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()


    return


def plot_visit_freq_by_level(tfs_labels, re=False, title='', save_file_path=None, display=False):

    ma = NewMaze()

    def get_explore_mode_nodes(tf, re):
        cl = SplitModeClips(tf, ma, re=re)
        ce = cl[np.where(cl[:, 3] == EXPLORE)]  # clips of exploration
        ne = np.concatenate([tf.no[c[0]][c[1]:c[1] + c[2], 0] for c in ce])  # nodes excluding the last state in each clip
        return ne

    animal_set = p.RewNames if re else p.UnrewNamesSub
    animal_lbl = p.REW_ANIMALS_PLOT_LABEL if re else p.UNREW_ANIMALS_PLOT_LABEL

    plt.figure()
    plot_level = "Level"

    # animal data
    tf = LoadTrajFromPath(OUTDATA_PATH + f'{animal_set[0]}-tf')  # plot one animal (to get the legend right)
    ne = get_explore_mode_nodes(tf, re)
    visit_frequency = calculate_normalized_visit_frequency_by_level([ne])
    print(visit_frequency)
    fmt = f'{p.ANIMAL_COLOR}-.'
    plt.plot(visit_frequency, fmt, alpha=0.4, linewidth=1, label=animal_lbl)
    for nickname in animal_set[1:]:  # plot rest of the animals
        tf = LoadTrajFromPath(OUTDATA_PATH + f'{nickname}-tf')
        ne = get_explore_mode_nodes(tf, re)
        visit_frequency = calculate_normalized_visit_frequency_by_level([ne])
        plt.plot(visit_frequency, fmt, alpha=0.4, linewidth=1)

    # random
    plt.plot([0.01344607, 0.02275074, 0.04698593, 0.09454197, 0.1892453, 0.37661891, 0.25032271], 'k-.', label='random')

    # agent data
    for i, (tf, label) in enumerate(tfs_labels):
        ne = get_explore_mode_nodes(tf, re)
        visit_frequency = calculate_normalized_visit_frequency_by_level([ne])
        print("visit_frequency", visit_frequency)
        color = p.COLORS[i]
        fmt = f'{color}o-'
        plt.plot(visit_frequency, fmt, label=label if label else f'agent {i}')

    plt.title(f'Normalized visit frequency by level\n{title}')
    plt.xlabel(plot_level)
    plt.ylabel(f"Fraction of visits to each level")
    plt.legend(loc='upper left')
    plt.ylim([0.0, 0.4])
    if save_file_path:
        os.makedirs(save_file_path, exist_ok=True)
        plt.savefig(os.path.join(save_file_path, f'norm_visit_frequency_by_level_plot.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_outside_inside_ratio(tfs, re, title='', save_file_path=None, display=False):
    print(f"calculating outside inside ratio...")

    if re:
        precomputed_oi_file = 'oiratio_rewarded.pkl'
        animal_label = p.REW_ANIMALS_PLOT_LABEL
    else:
        precomputed_oi_file = 'oiratio_unrewarded.pkl'
        animal_label = p.UNREW_ANIMALS_PLOT_LABEL

    # animal data
    with open(OUTDATA_PATH + precomputed_oi_file, 'rb') as f:
        animal_ratios = pickle.load(f)

    # get this agent's oi ratio
    ratios = []
    for tf, label in tfs:
        ratio = em.outside_inside_ratio(tf, re)
        ratios.append((ratio, label))

    plt.figure()

    plt.plot([1], [1.0], 'ko', label='random')
    plt.plot([1]*len(animal_ratios), list(animal_ratios.values()), 'b.', label=f'{animal_label} (mean 2.2)')
    for i, (ratio, label) in enumerate(ratios):
        plt.plot([1], [ratio], f'{p.COLORS[i]}o', label=f'{label if label else f"agent {i}"} - {round(ratio, 2)}')

    plt.ylim([0, 4])
    plttitle = 'Ratio of visits to outer vs inner leaf nodes'
    if title:
        plttitle = plttitle + '\n' + title
    plt.title(plttitle)
    plt.ylabel(f"ratio")
    plt.tick_params(axis='x', bottom=False, labelbottom=False)
    plt.legend()
    if save_file_path:
        plt.savefig(os.path.join(save_file_path, f'outside_inside_ratio.png'))
    if display:
        plt.show()
    plt.clf()
    plt.close()
    return


if __name__ == '__main__':
    from MM_Traj_Utils import LoadTrajFromPath
    from collections import defaultdict

    # for sub in p.UnrewNamesSub:
    #     print(sub)
    #     animal_tf_label = [(LoadTrajFromPath(p.OUTDATA_PATH + f'{sub}-tf'), f'mouse {sub}')]
    #     plot_opposite_node_preference(animal_tf_label, title=sub, display=True)

    # for sub in p.UnrewNamesSub[:1]:
    #     print(sub)
    #     tf = LoadTrajFromPath(p.OUTDATA_PATH + f'{sub}-tf')
    #     print(tf)
        # plot_node_revisits_level_halves(tf, [6], title=sub, save_file_path=f'../../figs/samples_from_data/{sub}', display=False)
        # plot_end_node_revisits_level_halves(tf, [6], title=sub, save_file_path=f'../../figs/samples_from_data/{sub}', display=False)
        # plot_unique_node_revisits_level_halves(tf, [6], title=sub, save_file_path=f'../../figs/samples_from_data/{sub}', display=False)
        # print("\n")
        # episodes = convert_traj_to_episodes(tf)
        # plot_trajs(episodes, title=sub, save_file_path=f'../../figs/samples_from_data/{sub}_trajs')

    # ee_dict = {}
    # for sub in p.UnrewNamesSub[:1]:
    #     a = plot_markov_fit_pooling(LoadTrajFromPath(OUTDATA_PATH + f'{sub}-tf'), sub, re=False, display=True)
    #     ee_dict[sub] = a
    #
    # print(ee_dict)