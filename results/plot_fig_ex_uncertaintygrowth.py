import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib

from env_models import Model_Fig2
from drones import Drone_Ostertag2020
from drone_models import Phantom3

def kalman_update(Sig, H, W, V, fs):
    Sig_temp = Sig + W * (1 / fs)
    K = Sig_temp @ H.T @ np.linalg.inv(H @ Sig_temp @ H.T + V)
    Sig_out = (np.eye(Sig.shape[0]) - K @ H) @ Sig_temp

    return Sig_out


if __name__ == '__main__':
    t_length = 600
    Sig0 = np.array([[30, 0, 0], [0, 30, 0], [0, 0, 30]])
    V = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    W = np.array([[0.165, 0, 0], [0, 0.15, 0], [0, 0, 0.07]])
    fs = 1

    env = Model_Fig2()
    try:
        with open('fig2_drone.pkl', 'rb') as fid:
            drone = pkl.load(fid)
    except:
        drone = Drone_Ostertag2020(drone_model=Phantom3, drone_id=1, b_verbose=True, b_logging=False, env_model=env,
                                   vmax=16, amax=16.3, jmax=1000.0, fs=1, obs_rad=10, covar_obs=5)
        with open('fig2_drone.pkl', 'wb') as fid:
            pkl.dump(drone, fid)

    drone.covar_s = 30 * np.eye(3)

    print(' ')

    # Run short time simulation
    N_t = 1000
    t_traj = np.linspace(0, 100, N_t)
    arr_s_p = np.zeros((N_t, 3))
    arr_s_v = np.zeros((N_t, 3))
    arr_s_a = np.zeros((N_t, 3))
    arr_s_j = np.zeros((N_t, 3))

    for ind, t in enumerate(t_traj):
        s_p, s_v, s_a, s_j = drone.calc_traj(t)

        arr_s_p[ind, :] = s_p
        arr_s_v[ind, :] = s_v
        arr_s_a[ind, :] = s_a
        arr_s_j[ind, :] = s_j

    arr_s_v_mag = np.sqrt(np.sum(arr_s_v ** 2, axis=1))
    arr_s_a_mag = np.sqrt(np.sum(arr_s_a ** 2, axis=1))

    arr_q = env.get_pois()

    fig1 = plt.figure(1)
    plt.scatter(arr_q[:, 0], arr_q[:, 1])
    plt.plot(arr_s_p[:, 0], arr_s_p[:, 1])

    fig2, axs = plt.subplots(2, 1)
    axs[0].plot(t_traj, arr_s_v_mag)
    axs[1].plot(t_traj, arr_s_a_mag)
    plt.show(block=False)

    T1_2 = 30
    T2_3 = 11
    T3_1 = 26
    T_total = T1_2 + T2_3 + T3_1

    T1 = 0
    T2 = T1 + T1_2
    T3 = T2 + T2_3

    t_run = np.arange(t_length, dtype=np.int)

    list_Sig = [Sig0]
    list_t = [0]
    for t in t_run:
        t_temp = t % T_total

        b_obs = False
        if t_temp == T1 or t_temp == T1 + 1 or t_temp == T1 + 2:
            H_t = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            b_obs = True
        elif t_temp == T2 or t_temp == T2 + 1:
            H_t = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            b_obs = True
        elif t_temp == T3:
            H_t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            b_obs = True
        else:
            H_t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        if b_obs == True:
            list_Sig.append(list_Sig[-1] + W / fs)
            list_t.append(t)
            list_Sig.append(kalman_update(list_Sig[-2], H_t, W, V, fs))
        else:
            list_Sig.append(kalman_update(list_Sig[-1], H_t, W, V, fs))
        list_t.append(t)

    arr_Sig = np.array(list_Sig)
    arr_t = np.array(list_t)
    max_Sig = np.max(arr_Sig, axis=1)

    upper_Sig = np.max(max_Sig, axis=1)

    fig, axs = plt.subplots(2,1, figsize=(6, 6), dpi=200)
    axs[1].plot(arr_t[:602], upper_Sig[:602], color='0.8', linewidth=8, label='$\lambda_{max}$', zorder=-1)
    axs[1].plot(arr_t[:602], max_Sig[:602, 0], linewidth=3, label='$\lambda_1$', zorder=2)
    axs[1].plot(arr_t[:602], max_Sig[:602, 1], linewidth=1.5, label='$\lambda_2$', zorder=0)
    axs[1].plot(arr_t[:602], max_Sig[:602, 2], linewidth=1.5, label='$\lambda_3$', zorder=1)
    axs[1].set_xlim([455, 580])
    axs[1].set_ylim([1, 16])
    axs[1].legend(loc=1, facecolor='white', framealpha=1)
    axs[1].set_xticklabels('')
    axs[1].set_yticklabels('')
    axs[1].grid(b=True, which='major', axis='both')
    plt.show(block=False)

    # Place locations of interest that match above parameters
    q1 = np.array([100, 100])
    q2 = q1 + [30, 0]
    q3 = q1 + [24.25, -9.377]
    b_rad = 4

    arr_q = np.vstack((q1, q2, q3))
    axs[0].scatter(q1[0], q1[1], s=[60], zorder=3, label='$q_1$')
    axs[0].scatter(q2[0], q2[1], s=[60], zorder=3, label='$q_2$')
    axs[0].scatter(q3[0], q3[1], s=[60], zorder=3, label='$q_3$')
    for q in arr_q:
        axs[0].add_patch(matplotlib.patches.Circle(q, radius=b_rad, color='0.8', zorder=-1))
    axs[0].plot(arr_s_p[:152, 0] / 2.5, arr_s_p[:152, 1] / 2.5, 'k-', linewidth=2, zorder=10)

    arr_q = np.vstack((q1, q2, q3, q1))
    #axs[0].plot(arr_q[:, 0], arr_q[:, 1], 'k', linewidth=2, zorder=-2)
    axs[0].set_xticks([])
    axs[0].set_xticklabels('')
    axs[0].set_yticks([])
    axs[0].set_yticklabels('')
    axs[0].set_xlim([95.5, 141])
    axs[0].set_ylim([85, 107])
    axs[0].legend(loc=1, facecolor='white', framealpha=1)
    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig('fig_ex_uncertainty.png', transparent=True)

    print(' ')