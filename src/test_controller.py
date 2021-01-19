import pickle as pkl
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from utility_func import *
from controllers import *
from drone_models import Crazyflie, Phantom3, Phantom3_vel

filename_target = '/home/mostertag/Workbench/Drone/opt-trajectory/results/20200422/Nq80/20200422_023616_Nq80_vmax15.0_amax10.0_B10.0.pkl'
ind_error = [21, 23] #list(range(24)) #[2, 4, 7, 9, 12, 14, 20, 21, 23] # smaller subset where Drone issues on [4, 9, 14, 21, 23]

def reset_drone(drone):
    drone.t_prev = 0
    drone.s_p, drone.s_v, drone.s_a, drone.s_j = drone.calc_traj(0)
    drone.s_state = init_state(drone.s_p, drone.s_v, drone.s_a, drone.s_j, yaw=0)
    drone.drone_model = Phantom3()


def quadplot_create(fig_num=1):
    """
    Creates a 3d plot for tracking progress of quadcopter
    """
    if not(plt.isinteractive()):
        plt.ion()

    plt.figure(fig_num)
    plt.clf()
    h_ax = plt.axes(projection='3d')
    h_ax.set_xlabel('x (m)')
    h_ax.set_ylabel('y (m)')
    h_ax.set_zlabel('z (m)')

    return h_ax


def quadplot_update(h_ax, s_traj, s_plan, t_curr=None):
    """
    Updates plot designated by an axis handle

    Note: s_traj will have np.nan values for any points not yet collected
    """
    # Find min/max position values for each axis to normalize
    s_min = np.nanmin(s_traj[0:3, :], axis=1)
    s_max = np.nanmax(s_traj[0:3, :], axis=1)
    s_maxrange = np.max(s_max - s_min) * 1.1 # Add a 10% buffer to edges
    if s_maxrange < 2:
        s_maxrange = 2
    s_avg = (s_max + s_min) / 2

    # Plot valid points
    h_lines = h_ax.get_lines()
    if len(h_lines) < 2:
        h_ax.plot3D(s_traj[0, :], s_traj[1, :], s_traj[2, :])
        h_ax.plot3D(s_plan[0, :], s_plan[1, :], s_plan[2, :], '--')
    else:
        h_lines[0].set_data_3d(s_traj[0, :], s_traj[1, :], s_traj[2, :])
        h_lines[1].set_data_3d(s_plan[0, :], s_plan[1, :], s_plan[2, :])

    # Set equalized axis limits
    h_ax.set_xlim(s_avg[0] - s_maxrange / 2, s_avg[0] + s_maxrange / 2)
    h_ax.set_ylim(s_avg[1] - s_maxrange / 2, s_avg[1] + s_maxrange / 2)
    h_ax.set_zlim(s_avg[2] - s_maxrange / 2, s_avg[2] + s_maxrange / 2)

    if t_curr:
        h_ax.set_title('Simulation t = {0:2.3f}'.format(t_curr))

    plt.draw()

if __name__ == '__main__':
    dt = 0.1
    t_max = 500

    num_step = np.ceil(t_max / dt).astype(np.int64)

    with open(filename_target, 'rb') as fid:
        data_in = pkl.load(fid)

    drone_test = data_in['drone']['drones'][20]
    reset_drone(drone_test)

    arr_s = np.zeros((13, 0))
    arr_s_plan = np.zeros((13, 0))

    plt.ion()
    h_ax = quadplot_create()

    for ind in range(num_step):
        drone_test.update(dt, b_sample=False)

        temp_s_plan = np.zeros((13, 1))
        temp_s = np.zeros((13, 1))

        temp_s_plan[0:3, 0] = drone_test.s_p
        temp_s_plan[3:6, 0] = drone_test.s_v
        temp_s_plan[6:9, 0] = drone_test.s_a
        temp_s_plan[9:12, 0] = drone_test.s_j
        temp_s[:, 0] = drone_test.s_state

        arr_s = np.append(arr_s, temp_s, axis=1)
        arr_s_plan = np.append(arr_s_plan, temp_s_plan, axis=1)

        if ind % 10 == 0:
            quadplot_update(h_ax, s_traj=arr_s, s_plan=arr_s_plan)
            plt.pause(0.01)




