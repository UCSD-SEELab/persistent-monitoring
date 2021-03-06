#!/usr/bin/env python3
"""
main_sim.py
Michael Ostertag

A simulation environment to test persistent monitoring algorithms using robotic platforms.

The simulation keeps track of an environmental model (env_model) and drone states (swarm_controller).
"""

import time
import argparse

import os
import sys
import math
import logging
import numpy as np
import pickle as pkl
from env_models import Model_Base, Model_Randomized, Model_Fig1
from drones import Drone_Base, Drone_Ostertag2020, Drone_Constant, Drone_Ostertag2019, Drone_Smith2012
from drones import Drone_Smith2012_Regions, Drone_Ostertag2019_Regions
from drone_models import Crazyflie, Phantom3, Phantom3_vel

import matplotlib.cm as colors
import matplotlib.pyplot as plt

from datetime import datetime

####################
class Sim_Environment():
    """
    Simulation environment for single robotic platforms measuring N random points of interest
    """

    def __init__(self, swarm_controller=None, env_model=None, step_time=1, steps_per_sample=1,
                 b_verbose=True, b_logging=True):
        """
        """

        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.step_time = step_time
        self.steps_per_sample = steps_per_sample

        self.step_count = 0

        self.swarm_controller = swarm_controller
        if (swarm_controller == None):
            self.b_swarm_controller = False
        else:
            self.b_swarm_controller = True

        self.env_model = env_model
        if (self.env_model == None):
            self.b_env_model = False
        else:
            self.b_env_model = True

        self.filename_results = time.strftime('Result_%Y%m%d_%H%M%S', time.localtime())

        plt.ion()  # Enable interactive plotting

    def set_swarm_controller(self, swarm_controller):
        """
        Links a swarm controller to the simulation environment after initialization
        """
        self.swarm_controller = swarm_controller
        self.b_swarm_controller = True

        if (self.b_env_model):
            self.swarm_controller.set_env_model( self.env_model )

    def set_env_model(self, env_model):
        """
        Links an environmental model to the simulation
        """
        self.env_model = env_model
        self.b_env_model = True

        if (self.b_swarm_controller):
            self.swarm_controller.set_env_model( self.env_model )

    def init_sim(self, num_steps, steps_per_sample):
        """
        Initializes the simulation
        """
        if (self.env_model):
            self.env_model.init_sim()

        if (self.swarm_controller):
            self.swarm_controller.init_sim(num_steps, steps_per_sample)

        self.step_count = 0

    def update(self):
        """
        Updates the environmental model and then the plan for each robotic platform
        """
        b_sample = (self.step_count % self.steps_per_sample) == 0

        if (self.env_model):
            self.env_model.update( dt=self.step_time )

        if (self.swarm_controller):
            self.swarm_controller.update( dt=self.step_time, b_sample=b_sample)

        self.step_count += 1

    def save_results(self, param_vmax, param_B):
        """
        Save the configuration results from the environmental model and the
        covariance results from the individual drones to a csv file
        """
        list_env_config = {'N_q': self.env_model.N_q, 'env_size': self.env_model.env_size, 'list_q':self.env_model.list_q,
                           'covar_env': self.env_model.covar_env}
        list_drone_config = {'drones':self.swarm_controller.list_drones, 'B':param_B}
        list_results = self.swarm_controller.get_results()

        np.set_printoptions(linewidth=1024, suppress=True)

        filename = datetime.now().strftime('%Y%m%d_%H%M%S_Nq{0:02d}_vmax{1}_amax{2}_B{3}.pkl'.format(self.env_model.N_q,
                                                                                         param_vmax, param_amax, param_B))
        directory_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      datetime.now().strftime('../results/%Y%m%d'))
        if not os.path.exists(directory_save):
            os.makedirs(directory_save)
        with open(os.path.join(directory_save, filename), 'wb') as fid:
            pkl.dump({'env':list_env_config, 'drone':list_drone_config, 'results':list_results,
                      'time':{'t_step':step_time, 'step_per_sample':steps_per_sample}}, fid)

    def visualize(self):
        plt.clf()
        plt.figure(1)
        plt.subplot(111)

        if (self.b_env_model):
            self.env_model.visualize()

        if (self.b_swarm_controller):
            self.swarm_controller.visualize()

        plt.draw()
        plt.pause(0.01)


####################
class SwarmController():
    """
    SwarmController controls the communication between all drones and manages
    their movement calculation, updates, and visualizations
    """

    def __init__(self, env_model=None,
                 b_verbose=True, b_logging=True):
        """
        gamma, poi, covar_e, map_terrain=np.matrix([]), map_data=np.matrix([]),
                 N_drones=1, pos_start=0, list_pos_start=[], step_size=1, step_time=1,
                 param_covar_0_scale=100, param_v_max=10, param_n_obs=5, param_ros=1,
                 b_verbose=True, b_logging=True):
        """
        """
        Initializes the swarm controller
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging

        self.env_model = env_model
        if (self.env_model == None):
            self.b_env_model = False
        else:
            self.b_env_model = True

        self.list_drones = []

    def add_drone(self, drone_model, planner, drone_id, b_verbose=True, b_logging=True, cfg=None):
        """
        Adds a drone specified by drone_type, which should be a drone model that conforms to the standard in drones.py.
        cfg is a dictionary that holds the configuration of the drone for all required parameters by the class.
        """
        #try:
        temp_drone = planner(drone_model=drone_model(), drone_id=drone_id, b_verbose=b_verbose, b_logging=b_logging, **cfg)
        self.list_drones.append(temp_drone)
        print('Drone ({0}) added to swarm_controller'.format(planner.__name__))
        #except:
        #    print(sys.exc_info())
        #    print('Drone ({0}) failed to be created with parameters {1}'.format(planner.__name__, cfg))

    def init_sim(self, num_steps, steps_per_sample):
        """
        Initialize each drone for simulation that is about to start
        """
        for drone in self.list_drones:
            drone.init_sim()

        self.arr_covar = np.zeros((len(self.list_drones), num_steps * (steps_per_sample + 1), self.env_model.N_q + 1))
        self.arr_s = np.zeros((len(self.list_drones), 13, num_steps* steps_per_sample))
        self.arr_s_plan = np.zeros((len(self.list_drones), 4, 3, num_steps * steps_per_sample))
        self.arr_b_sample = np.zeros(num_steps * steps_per_sample).astype(bool)

        self.ind_arr = 0
        self.ind_arr_covar = 0

    def update(self, dt, b_sample=True):
        """
        description
        """
        if b_sample:
            temp_covar = np.zeros((len(self.list_drones), 2, self.env_model.N_q + 1))
            self.arr_covar[:, self.ind_arr_covar:self.ind_arr_covar + 2, :] = temp_covar
            self.ind_arr_covar += 2
        else:
            temp_covar = np.zeros((len(self.list_drones), 1, self.env_model.N_q + 1))
            self.arr_covar[:, self.ind_arr_covar:self.ind_arr_covar + 1, :] = temp_covar
            self.ind_arr_covar += 1

        temp_s_plan = np.zeros((len(self.list_drones), 4, 3, 1))
        temp_s = np.zeros((len(self.list_drones), 13, 1))

        for ind_drone, drone in enumerate(self.list_drones):
            temp_covar[ind_drone, :, :] = drone.update(dt, b_sample=b_sample)
            temp_s_plan[ind_drone, 0, :, 0] = drone.s_p
            temp_s_plan[ind_drone, 1, :, 0] = drone.s_v
            temp_s_plan[ind_drone, 2, :, 0] = drone.s_a
            temp_s_plan[ind_drone, 3, :, 0] = drone.s_j
            temp_s[ind_drone, :, 0] = drone.s_state

        self.arr_s[:, :, self.ind_arr:self.ind_arr + 1] = temp_s
        self.arr_s_plan[:, :, :, self.ind_arr:self.ind_arr + 1] = temp_s_plan
        self.arr_b_sample[self.ind_arr] = b_sample
        self.ind_arr += 1

    def reset_covar(self):
        """
        Reset self.list_covar_max to initial value
        """
        self.arr_covar = np.zeros((len(self.list_drones), 0, self.env_model.N_q + 1))

    def set_env_model(self, env_model):
        """
        Sets the environmental model and resets all drones
        """
        self.env_model = env_model
        # for drone in self.list_drones:
        #     drone.set_terrain_map(self.map_terrain) # Drones not currently using map

    def get_results(self):
        """
        Return results of the drone IDs and their covariance values
        """
        return {'covar':self.arr_covar, 's_plan':self.arr_s_plan, 's_real':self.arr_s, 'b_samp':self.arr_b_sample}

    def get_simplified_drones(self):
        """
        Return list of parameters extracted from drones. Parameters are:
            drone.__name__
            drone.vmax
            drone.amax
            drone.jmax
            drone.covar_bound
            drone.drone_model
            drone.fs
            drone.fu
            drone.covar_obs
            drone.obs_rad
        """
        list_drone_simp = []
        for drone in self.list_drones:
            # Parameters common to all drones
            temp_dict = {'class': drone, #type(drone).__name__,
                         'vmax': drone.vmax,
                         'amax': drone.amax,
                         'jmax': drone.jmax,
                         'covar_bound': drone.get_optimalbound(),
                         'controller': drone.controller.__name__,
                         'drone_model': drone.drone_model,
                         'fs': drone.fs,
                         'covar_obs': drone.covar_obs,
                         'obs_rad': drone.obs_rad}

            # Parameter only for B-spline-based planners
            if hasattr(drone, 'fu'):
                temp_dict['fu'] = drone.fu
            # Parameter only for segmenting path controllers
            if hasattr(drone, 'J'):
                temp_dict['J'] = drone.J

            list_drone_simp.append(temp_dict)

        return list_drone_simp

    def reset_drones(self, theta0, covar_0_scale=100):
        """
        Resets the drone to a provided theta0 with some initial
        covariance scaled by covar_0_scale while maintaining any generated
        models
        """
        self.reset_covar()
        for ind, drone in enumerate(self.list_drones):
            drone.init_covar(covar_0_scale)
            drone.set_theta(theta0)

    def visualize(self):
        for ind, drone in enumerate(self.list_drones):
            drone.visualize(colors.get_cmap('tab10')(ind % 10))

####################
"""
Main control loop
"""
if __name__ == '__main__':
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--vmax', type=float, default=15.0,
                        help='maximum velocity of drone')
    parser.add_argument('--amax', type=float, default=10.0,
                        help='maximum acceleration of drone')
    parser.add_argument('--jmax', type=float, default=1000.0,
                        help='maximum jerk of drone')
    parser.add_argument('--B', type=float, default=20.0,
                        help='radius of observation window')
    parser.add_argument('--V', type=float, default=20.0,
                        help='noise in observation model')
    parser.add_argument('--Nq', type=int, default=20,
                        help='number of points of interest (q)')
    parser.add_argument('--Nsteps', type=int, default=1200,
                        help='number of simulation steps')
    parser.add_argument('--Ntests', type=int, default=10,
                        help='number of independent tests')
    parser.add_argument('--verbose', action='store_true',
                        help='enables verbose output')
    parser.add_argument('--visual', action='store_true', 
                        help='enables visual output')
    parser.add_argument('--logging', action='store_true',
                        help='enables debug logging')
    parser.add_argument('--overlap', action='store_true',
                        help='forces at least two points of interest to have overlapping sensing region')
    args = parser.parse_args()

    N_steps = args.Nsteps
    N_drones = 1
    env_size = np.array([450, 450])
    step_size = 1
    param_b_visualize = args.visual

    if param_b_visualize:
        step_time = 0.1
        steps_per_sample = 10
        N_steps = N_steps * steps_per_sample
    else:
        step_time = 0.1
        steps_per_sample = 10
        N_steps = N_steps * steps_per_sample

    N_tests = args.Ntests
    param_ros = 1
    param_covar_0_scale = 100
    param_vmax = args.vmax
    param_amax = args.amax
    param_jmax = args.jmax
    param_obs_rad = args.B
    param_N_q = args.Nq

    b_verbose = args.verbose
    b_logging = args.logging
    b_overlap = args.overlap

    if (b_logging):
        logger_filename = time.strftime('Log_%Y%m%d_%H%M%S.log', time.localtime())
        # TODO If Logs doesn't exist, create folder.
        logger_format = "[%(asctime)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
        logging.basicConfig(filename='Logs/' + logger_filename, level=logging.DEBUG,
                            format=logger_format, datefmt='%Y%m%d %H%M%S')
        logger = logging.getLogger('root')

    # Configure simulation environment and initialize
    sim_env = Sim_Environment(swarm_controller=None, env_model=None, steps_per_sample=steps_per_sample,
                              step_time=step_time, b_verbose=True, b_logging=True)

    for ind_test in range(N_tests):
        print('Test {0:4d}/{1}'.format(ind_test + 1, N_tests))

        # Setup environmental model
        # env_model = Model_Base(env_size=env_size, step_size=step_size, N_q=param_N_q,
        #                        b_verbose=b_verbose, b_logging=b_logging)
        env_model = Model_Randomized(env_size=env_size, step_size=step_size, N_q=param_N_q, B=param_obs_rad,
                                     b_verbose=b_verbose, b_logging=b_logging)
        # env_model = Model_Fig1()
        sim_env.set_env_model(env_model)

        fs = 1 / (steps_per_sample * step_time)

        # Set up swarm controller and drones to test
        swarm_controller = SwarmController(env_model=env_model, b_verbose=b_verbose, b_logging=b_logging)
        sim_env.set_swarm_controller(swarm_controller)
        # Constant
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_9',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':9, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_10',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':10, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_12',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_14',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_16',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Constant, drone_id='Drone_Constant_16',
                                   b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        # Linear
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_9', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 9, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_10', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 10, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_12', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 12, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_14', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 14, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_16', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 16, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Smith2012_Regions,
                                   drone_id='Drone_Smith2012_Regions_18', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 18, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        # ACC Controller
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_9', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':9, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_10', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':10, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_12', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_14', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_16', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3_vel, planner=Drone_Ostertag2019_Regions,
                                   drone_id='Drone_Ostertag2019_Regions_18', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':param_amax, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        # New Controllers
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_12_85pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':16.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_12_90pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':17.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_12_95pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':18.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_12_100pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':12, 'amax':19.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_14_85pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':16.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_14_90pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':17.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_14_95pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':18.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_14_100pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':14, 'amax':19.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_16_85pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':16.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_16_90pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':17.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_16_95pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':18.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_16_100pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':16, 'amax':19.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_18_85pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':16.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_18_90pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':17.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_18_95pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':18.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_18_100pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':18, 'amax':19.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_20_85pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':20, 'amax':16.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_20_90pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':20, 'amax':17.3, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_20_95pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':20, 'amax':18.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})
        swarm_controller.add_drone(drone_model=Phantom3, planner=Drone_Ostertag2020,
                                   drone_id='Drone_Ostertag2020_20_100pct', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':20, 'amax':19.2, 'jmax':param_jmax,
                                        'fs':fs, 'obs_rad':param_obs_rad})

        sim_env.init_sim(num_steps=N_steps, steps_per_sample=steps_per_sample)

        arr_timechecks = np.linspace(0, N_steps, 10 + 1).astype(int)

        for ind in range(N_steps):
            sim_env.update()

            if ind in arr_timechecks:
                print('{0}/{1}'.format(ind, N_steps))

            # Update visualization
            if param_b_visualize:
                sim_env.visualize()
                time.sleep(0.001)

        print('{0}/{1}'.format(N_steps, N_steps))
        sim_env.save_results(param_vmax=param_vmax, param_B=param_obs_rad)

    if (b_logging):
        logging.shutdown()
