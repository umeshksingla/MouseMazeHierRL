import csv
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import glob
from matplotlib.collections import LineCollection
import numpy as np

from MM_Traj_Utils import NewMaze
from MM_Plot_Utils import plot


class NagyTraj:
    def __init__(self, traj_rows):
        self.xy = np.array([[t.x, t.y] for t in traj_rows])


class NagyTrajRow:
    def __init__(self, frame_row):
        self.frame_num = int(frame_row[0])
        self.num_idv = int(frame_row[1])
        self.rat_id = frame_row[2]
        self.x = np.float(frame_row[3])
        if not np.isnan(self.x):
            self.x = np.int(self.x)
        self.y = np.float(frame_row[4])
        if not np.isnan(self.y):
            self.y = np.int(self.y)
        self.orientation = float(frame_row[5])
        self.test()

    def test(self):
        assert self.num_idv == 1

    def __str__(self):
        return f'TrajRow{self.rat_id, self.frame_num, self.x, self.y}'


def loadNagyTraj(filepath):
    with open(filepath, newline='') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        traj_rows = []
        for row in r:
            print(row)
            traj_rows.append(NagyTrajRow(row))
    return NagyTraj(traj_rows)


def plot_trajectory(plot_points, title='trajectory', interval=30):
    plot_animation = True
    k = 5  # every kth point

    def update_line(num, points, line):
        line.set_data(points[..., :num])
        return line,

    x_min_axis, x_max_axis = np.nanmin(plot_points[0, :]) - 10, np.nanmax(plot_points[0, :]) + 10
    y_min_axis, y_max_axis = np.nanmin(plot_points[1, :]) - 10, np.nanmax(plot_points[1, :]) + 10

    xlim = (x_min_axis, x_max_axis)
    ylim = (y_min_axis, y_max_axis)
    print(xlim, ylim)

    # add a point that goes to left bottom to mark the end of animation
    if plot_animation:
        plot_points = np.array([
            np.append(plot_points[0, :][::k], xlim[0]),  # plot every kth point
            np.append(plot_points[1, :][::k], ylim[0])
        ])

    fig1 = plt.figure()
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel('x')
    plt.title(title)

    if plot_animation:
        l, = plt.plot([], [],  'r-', linewidth=0.8)
        line_ani = animation.FuncAnimation(fig1, update_line,
                                           fargs=(plot_points, l),
                                           interval=interval, blit=True, repeat=False)
        plt.plot(plot_points[0, -1], plot_points[1, -1], '*')
    else:
        plt.plot(plot_points[0, :], plot_points[1, :], 'r-', linewidth=0.8)
    plt.show()
    return


if __name__ == '__main__':
    files = glob.glob('../matenagy/single1-7/single1_*')
    for tfile in files[:3]:
        tf = loadNagyTraj(tfile)
        translated_xy = np.array([tf.xy[:, 0] - 900, 400-tf.xy[:, 1]]).T
        print(translated_xy)
        x = translated_xy[:, 0]
        y = translated_xy[:, 1]
        # plt.plot(translated_xy[:, 0], translated_xy[:, 1], '-.', markersize=0.1, cmap=mpl.colormaps['viridis'])

        plot_trajectory(translated_xy.T)

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        # axes.set_xlim(-700, 700)
        # axes.set_ylim(-700, 700)
        # plt.style.use("ggplot")
        #
        # def animate(i):
        #     # axes.cla()  # clear the previous image
        #     axes.plot(x[:i], y[:i], color="blue")
        #
        # anim = FuncAnimation(fig, animate, frames=len(x) + 1, interval=100, blit=False)


        # ma = NewMaze(6)
        # # Draw the maze outline
        # fig, ax = plt.subplots(figsize=(9, 9))
        # # plot([-400, -400], [400, 400], fmts=['k.'], equal=True, linewidth=2, yflip=True,
        # #      xhide=True, yhide=True, axes=ax)
        # ax.plot([-400, -400], [400, 400])
        #
        # x = translated_xy[:, 0]
        # y = translated_xy[:, 1]
        # print(translated_xy)
        # t = np.linspace(0, 1, x.shape[0])  # your "time" variable
        # plt.plot(translated_xy[:, 0], translated_xy[:, 1])
        # plt.set_cmap(plt.get_cmap('viridis'))
        # # set up a list of (x,y) points
        # # points = translated_xy # np.array([x, y]).transpose().reshape(-1, 1, 2)
        # # print(points)
        # # set up a list of segments
        # # segs = np.concatenate([points[:-1], points[1:]], axis=1)
        # # make the collection of segments
        # # lc = LineCollection([translated_xy], cmap=plt.get_cmap('viridis'), linewidths=0.1)  # jet, viridis hot
        # # lc.set_array(t)  # color the segments by our parameter
        # # print(lc)
        # # lines = ax.add_collection(lc);  # add the collection to the plot
        # # cax = fig.add_axes([200, 200, 200, 200])
        # # cbar = fig.colorbar(lines, cax=cax)
        # # cbar.set_ticks([0, 1])
        # # cbar.set_ticklabels(['Entry', 'Exit'])
        # # cbar.ax.tick_params(labelsize=18)
        # plt.show()
