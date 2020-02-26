"""
drones.py
Michael Ostertag

Classes for drones under test
    1. Drone_Constant
    2. Drone_Smith
    3. Drone_Ostertag2019
    4. Drone_Ostertag2020
"""

import time
import sys

import math
import logging
import numpy as np
from scipy import linalg
from shapely.geometry import Polygon, LineString, LinearRing, MultiPoint, Point, MultiLineString
from shapely.ops import unary_union, split, cascaded_union, snap, linemerge
from shapely.affinity import translate
import shapely

# Saving out errors
from datetime import datetime
import pickle as pkl

from functools import partial

# SymPy used for generating functions to solve for Ostertag Greedy KF 
# implementation
import sympy

import matplotlib.cm as colors
import matplotlib.pyplot as plt

import pulp     # PuLP is a linear programming optimization library
import cvxpy    # cvxpy is a convex optimization wrapper for different optimization tools
from concorde.tsp import TSPSolver

from utility_func import *


####################
def generate_func_kfsteadystate(depth=10):
    """
    Creates the functions to solve the steady state Kalman Filter equations 
    around a loop with multiple consecutive observations of a point
    """
    V = sympy.symbols('V')
    Twc = sympy.symbols('Twc')
    W = sympy.symbols('W')
    list_lambda = sympy.symbols(['lambda_T{0}'.format(i) for i in range(depth)])
    list_func = []
    
    for num_obs in range(1, depth+1):
        # time_start = time.time()
        # print('Depth {0}'.format(num_obs))
        if (num_obs > 1):
            expr = sympy.Eq(list_lambda[0], list_lambda[num_obs-1] + (Twc - num_obs)*W - \
                            list_lambda[num_obs-1]**2 / (list_lambda[num_obs-1] + V))
            for k in range(num_obs-1, 0, -1):
                expr = expr.subs(list_lambda[k], 
                                 list_lambda[k-1] + W - list_lambda[k-1]**2 / \
                                 (list_lambda[k-1] + V))
        else:
           expr = sympy.Eq(list_lambda[0], list_lambda[0] + (Twc - num_obs)*W - \
                           list_lambda[0]**2 / (list_lambda[0] + V))

        list_func.append(expr)
        # print('{0:0.3f} s'.format(time.time() - time_start))
    
    return list_func

####################
# Trajectory functions
def tfunc_linear(t, p1, p2, t1, t2):
    """

    """
    alpha = (t2 - t) / (t2 - t1)
    assert ((alpha <= 1) and (alpha >= 0))

    s_out = alpha * p1 + (1 - alpha) * p2

    return s_out

def tfunc_bspline(t, bspl):
    """

    """
    s_out = bspl(t)

    return s_out

def tfunc_const(t, val):
    """

    """
    return val

####################
# Observation window functions
def obs_window_circular(p, q, rad=10):
    """
    A circular observation window where p is the current location of the robotic platform, q is either a single location
    of a POI or a 2d array of locations, and rad is the radius
    """
    if q.ndim == 1:
        dist_p_q = np.sqrt(np.sum(np.power(q - p, 2)))
    elif q.ndim == 2:
        dist_p_q = np.sqrt(np.sum(np.power(q - p, 2), axis=1))
    else:
        # ERROR
        dist_p_q = 0

    return dist_p_q <= rad

####################
class Drone_Base():
    """
    Base class for drones that holds all basic functionality
    """
    DEFAULT_VMAX = 10
    DEFAULT_AMAX = 100
    DEFAULT_JMAX = 400
    DEFAULT_COVAR_OBS = 40
    DEFAULT_FS = 1

    THETA = np.linspace(0, 2*np.pi, 20)

    def __init__(self, drone_id, env_model=None, b_verbose=True, b_logging=True, **cfg):
        """
        OLD PARAMS. TO BE UPDATED
        Initializing the drone requires at minimum:
            loop_path: a path to follow 
            poi: a list of points of interest 
            covar_e: covariance matrix for environment noise
            obs_window: an observation window
            covar_obs: covariance matrix for observation noise
        
        and optionally:
            drone_id: unique number representing a drone
            theta0: initial position around loop_path
            fs: sampling frequency
            covar_s_0_scale: initial covariance of each poi
            v_max: maximum velocity
            v_min: minimum velocity
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging

        self.drone_id = drone_id        # Unique drone ID

        self.drone_desc = 'Drone_Base class'

        if ('vmax' in cfg.keys()):
            self.vmax = cfg['vmax']
        else:
            self.vmax = Drone_Base.DEFAULT_VMAX

        if ('amax' in cfg.keys()):
            self.amax = cfg['amax']
        else:
            self.amax = Drone_Base.DEFAULT_AMAX

        if ('jmax' in cfg.keys()):
            self.jmax = cfg['jmax']
        else:
            self.jmax = Drone_Base.DEFAULT_JMAX

        if ('fs' in cfg.keys()):
            self.fs = cfg['fs']
        else:
            self.fs = Drone_Base.DEFAULT_FS

        if ('obs_rad' in cfg.keys()):
            self.obs_rad = cfg['obs_rad']
        else:
            self.obs_rad = 10
        self.obs_window = partial(obs_window_circular, rad=self.obs_rad)

        if ('covar_obs' in cfg.keys()):
            self.covar_obs = cfg['covar_obs']
        else:
            self.covar_obs = Drone_Base.DEFAULT_COVAR_OBS

        self.env_model = env_model
        if (self.env_model):
            self.list_q = self.env_model.get_pois()
            self.init_sensing()
            self.s_p, self.s_v, self.s_a, self.s_j = self.form_traj()
            self.b_traj_ready = True
            self.t_prev = 0
        else:
            self.s_p = None
            self.s_v = None
            self.s_a = None
            self.s_j = None
            self.b_traj_ready = False

    def form_traj(self):
        """
        Calculates a basic trajectory using TSP and then a velocity set to max velocity
        """
        if not(self.env_model):
            self.s_p = None
            self.s_v = None
            self.s_a = None
            self.s_j = None
            self.b_traj_ready = False
            return

        self.list_q = self.env_model.get_pois()
        self.N_q = len(self.list_q)

        self.create_s_function_map()

        return self.calc_traj(0)

    def greedy_knockdown_algorithm(self, arr_T_min, T_notobs):
        """
        Implementation of the Greedy Knockdown Algorithm from Ostertag (2020)
        """

        print('Greedy Knockdown Algorithm')
        print(arr_T_min)

        # Generate Kalman filter steady state equations that need to be solved
        # for each iteration of the loop
        kf_depth = 6

        if (self.list_kf_eqs is None):
            self.list_kf_eqs = generate_func_kfsteadystate(depth=kf_depth)

        V = sympy.symbols('V')
        Twc = sympy.symbols('Twc')
        W = sympy.symbols('W')
        lambda_T0 = sympy.symbols('lambda_T0')

        N_loops = 100
        N_kf = np.zeros((N_loops, self.N_q)).astype(int)  # Number of observations at each point of interest
        N_kf[0, :] = 1
        sig_max = np.zeros((N_loops - 1, self.N_q))
        for ind_loop in range(N_loops - 1):
            # Observed path should move at either max speed or minimum required time to take one sample
            T_obs = np.max(np.append(arr_T_min.reshape(-1, 1), N_kf[ind_loop, :].reshape(-1, 1) / self.fs, axis=1), axis=1)

            # Predicted worst case time for single loop with d_i observations per location i
            T_loop = np.ceil((np.sum(T_obs) + T_notobs) * self.fs) / self.fs

            # Calculate steady state value for each point
            for ind_i in range(self.N_q):
                eq_temp = self.list_kf_eqs[N_kf[ind_loop, ind_i] - 1].subs(
                    {V: self.covar_obs, W: self.covar_e[ind_i, ind_i], Twc: T_loop})
                # Solve for steady-state covariance using sympy.nsolve by providing an initial solution vector from the
                # previously solved step
                if (ind_loop == 0):
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, self.covar_s[ind_i, ind_i])
                else:
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, sig_max[ind_loop - 1, ind_i])

            print(N_kf[ind_loop, :])
            print(sig_max[ind_loop, :])
            print('\n')

            N_kf[ind_loop + 1, :] = N_kf[ind_loop, :]
            # Add observation the poi with the highest uncertainty
            ind_max = np.argmax(sig_max[ind_loop, :])
            N_kf[ind_loop + 1, ind_max] += 1

            # TODO Update end condition
            if (N_kf[ind_loop + 1, ind_max] >= kf_depth):
                break

        sig_max_temp = np.max(sig_max, axis=1)
        ind_row_max = np.argmin(sig_max_temp[(sig_max_temp > 0).flatten()])
        self.covar_bound = sig_max_temp[ind_row_max]
        ind_row_min = np.argmin(sig_max_temp[(sig_max_temp > 0).flatten()])

        print('Optimal after {0} obs'.format(ind_row_min))
        print(N_kf[ind_row_min, :])
        print(sig_max[ind_row_min, :])
        print('\n')

        return N_kf[ind_row_min, :]

    def calc_tsp_order(self, list_q):
        """
        Calculates the TSP order given a set of points of interest list_q that contain position information using the
        Concorde TSP solver. Returns a transform to change list_q into an ordered list
        """
        x = list_q[:, 0]
        y = list_q[:, 1]

        solver = TSPSolver.from_data(x, y, norm='EUC_2D')
        q_order = solver.solve()

        q_trans = np.zeros((self.N_q, self.N_q))
        for ind_row, ind_q in enumerate(q_order.tour):
            q_trans[ind_row, ind_q] = 1

        return q_trans

    def init_sensing(self):
        """
        Initialize sensing for the drone
        """
        self.N_q = len(self.list_q)
        self.list_s = np.zeros((self.N_q, 1))
        self.covar_e = self.env_model.get_covar_env()
        self.init_covar_s(covar_scale=100)

    def init_covar_s(self, covar_scale=100):
        """
        Initialize covariance matrix for s, the estimate of x
        """
        self.covar_s = self.covar_e * covar_scale
    
    def log_iter(self):
        """
        Log relevant information for current iteration:
            dtheta
            theta
            pos
            covar_s
        """
        if (self.b_logging):
            self.logger.debug('Drone {0}'.format(self.drone_id))
            self.logger.debug('dtheta = {0:0.2f}'.format(self.dtheta))
            self.logger.debug('theta = {0:0.2f}'.format(self.theta))
            self.logger.debug('pos = ({0:0.2f}, {1:0.2f})'.format(self.pos[0], self.pos[1]))
            # Create long single line of diagonal elements of covariance matrix
            self.logger.debug('covar_s  = ' + np.array2string(np.diag(self.covar_s), separator=', ', formatter={'float_kind':lambda x: '{0:6.3f}'.format(x)}, max_line_width=2148))
            if (self.H.shape[0] > 0):
                self.logger.debug('H = ' + np.array2string(self.H.flatten(), separator=', ', formatter={'int_kind':lambda x: '{0:1d}'.format(x)}, max_line_width=2148))
            else:
                self.logger.debug('H = ' + np.array2string(np.zeros((self.N_q,1)).flatten(), separator=', ', formatter={'int_kind':lambda x: '{0:1d}'.format(x)}, max_line_width=2148))
        
        if (self.b_verbose):
            print('Drone {0}'.format(self.drone_id))
            print('  t = {0:0.2f}'.format(self.t_prev))
            print('  pos = ({0:0.2f}, {1:0.2f})'.format(self.s_p[0], self.s_p[1]))
            print('  vel = ({0:0.2f}, {1:0.2f})'.format(self.s_v[0], self.s_v[1]))

    def calc_path_length(self, loop_path):
        """
        Calculate path length assuming linear interpolation between points and
        each row is a new point
        """
        L = np.sum(np.sqrt(np.sum(np.power(loop_path[1:,:] - loop_path[:-1,:], 2), axis=1)))
        return L

    def calc_traj(self, t):
        """
        Calculates the trajectory (position, velocity, acceleration, and jerk) at a given time point
        """
        # If look up table hasn't been generated yet, then generate
        if (self.s_fmap is None):
            self.create_s_function_map()

        # Using look up table, find the points of interest on either side of current position
        t_mod = t % self.s_fmap[-1][1]
        s_funcs = next(entry for entry in self.s_fmap if (entry[0] <= t_mod < entry[1]))
        s_p = s_funcs[2](t_mod)
        s_v = s_funcs[3](t_mod)
        s_a = s_funcs[4](t_mod)
        s_j = s_funcs[5](t_mod)

        if any(s_p != s_p):
            filename = datetime.now().strftime('traj_error_%Y%m%d_%H%M%S.pkl')
            print(' traj_error. Saving pickle file ({0})'.format(filename))
            pkl.dump({'s_funcs':s_funcs, 't':t, 't_mod':t_mod, 's_p':s_p, 's_v':s_v, 's_a':s_a, 's_j':s_j},
                     open(filename, 'wb'))

        return s_p, s_v, s_a, s_j

    def create_s_function_map(self):
        """
        Creates a look-up table that links current time around the loop to a segment in a piecewise trajectory function
        """
        self.s_fmap = []

        temp_s_p1 = np.zeros((5, 2))
        s_t2 = np.array([1, 2, 3, 4, 5])
        s_t1 = np.roll(s_t2, 1, axis=0)
        s_t1[0] = 0

        for t1, t2, p1 in zip(s_t1, s_t2, temp_s_p1):
            self.s_fmap.append([t1, t2,
                                partial(tfunc_const, val=p1),
                                partial(tfunc_const, val=np.zeros(p1.shape)),
                                partial(tfunc_const, val=np.zeros(p1.shape)),
                                partial(tfunc_const, val=np.zeros(p1.shape)),
                                {'type': 'constant', 't1': t1, 't2': t2, 'p1': p1, 'T': (t2 - t1)}])

    def calc_pos(self, t):
        """
        Calculates position along closed path based on time t
        """
        # If look up table hasn't been generated yet, then generate
        if (self.s_fmap is None):
            self.create_s_function_map()

        # Using look up table, find the points of interest on either side of current position
        t_mod = t % self.s_fmap[-1][1]
        s_func = next(entry[2] for entry in self.s_fmap if (entry[0] <= t_mod < entry[1]))
        
        return s_func(t_mod)

    def calc_vel(self, t):
        """
        Calculates velocity along closed path
        """
        # If look up table hasn't been generated yet, then generate
        if (self.s_fmap is None):
            self.create_s_function_map()

        # Using look up table, find the points of interest on either side of current position
        t_mod = t % self.s_fmap[-1][1]
        s_func = next(entry[3] for entry in self.s_fmap if (entry[0] <= t_mod < entry[1]))

        return s_func(t_mod)

    def calc_acc(self, t):
        """
        Calculates acceleration along closed path
        """
        # If look up table hasn't been generated yet, then generate
        if (self.s_fmap is None):
            self.create_s_function_map()

        # Using look up table, find the points of interest on either side of current position
        t_mod = t % self.s_fmap[-1][1]
        s_func = next(entry[4] for entry in self.s_fmap if (entry[0] <= t_mod < entry[1]))

        return s_func(t_mod)

    def calc_jrk(self, t):
        """
        Calculates jerk along closed path
        """
        # If look up table hasn't been generated yet, then generate
        if (self.s_fmap is None):
            self.create_s_function_map()

        # Using look up table, find the points of interest on either side of current position
        t_mod = t % self.s_fmap[-1][1]
        s_func = next(entry[5] for entry in self.s_fmap if (entry[0] <= t_mod < entry[1]))

        return s_func(t_mod)

    def observed_pois(self):
        """
        Returns a list of observed points of interest based on the defined observation window
        """
        if (self.obs_window):
            b_ind_obs = self.obs_window(self.s_p, self.list_q)
        else:
            b_ind_obs = []

        ind_obs = [ind for ind, val in enumerate(b_ind_obs) if val == 1]

        test = self.list_q[b_ind_obs, :]

        return self.list_q[b_ind_obs, :], ind_obs

    def init_sim(self):
        """
        Initialize drone for simulation
        """
        pass

    def update(self, dt):
        """
        Moves the drone based on the velocity calculated by calc_movement(). 
        Then, captures an observation and updates the Kalman filter.
        """
        t = self.t_prev + dt
        # Update current position
        self.s_p, self.s_v, self.s_a, self.s_j = self.calc_traj(t)

        # Update current prediction of estimated states, capture an observation, and update Kalman filter
        self.update_predict(dt)
        self.capture_data()
        self.update_kalman()
        
        # Log information
        self.log_iter()

        self.t_prev = t
    
    def capture_data(self):
        """
        Captures data for any points within observation window B(theta) where 
        B(theta) = self.pos + self.B
        
        The observation matrix H is saved where y = Hx + noise
        """
        q_obs, ind_detected = self.observed_pois()

        self.M_obs = len(q_obs)
        
        # Reset observation matrix to correct size
        self.H = np.zeros((self.M_obs, self.N_q))
        
        for ind_row, ind_detected in enumerate(ind_detected):
            self.H[ind_row, ind_detected] = 1
        
        # Create observations
        # To be updated with appropriate noise models
        self.y = np.zeros((self.M_obs, 1))
        self.covar_V = self.covar_obs*np.identity(self.M_obs)

    def update_predict(self, dt):
        """
        Updates prediction of estimate using knowledge of environmental model
        """
        # list_s is the current approximation of the sensed state. The model update are expected values from a provided
        # model
        self.list_s = self.list_s + self.calc_model_update(dt)

        # covar_s is the uncertainty of the phenomenon, which increases by the model/environmental covariance in a
        # linear fashion due to the Wiener process model
        self.covar_s = self.covar_s + self.covar_e * dt
        
    def calc_model_update(self, dt):
        """
        The model update provides the expected evolution of values of the phenomenon being measured based on the
        current knowledge of the phenomenon
        """
        model_update = np.zeros(self.list_s.shape)

        return model_update

    def update_kalman(self):
        """
        Updates covariance of s using noise levels
        """
        # Calculate Kalman gain
        K = linalg.solve(np.matmul(np.matmul(self.H,self.covar_s), self.H.T) + self.covar_V, 
                         np.matmul(self.H, self.covar_s)).T
        
        # Update estimates and covariances
        self.list_s = self.list_s + np.matmul(K, self.y - np.matmul(self.H, self.list_s))
        self.covar_s = np.matmul(np.identity(self.covar_s.shape[0]) - np.matmul(K, self.H), self.covar_s)

    def get_covar_s(self):
        """
        Returns the current covariance matrix for the estimate of s immediately
        preceeding the next point
        """
        return self.covar_s
            
    def visualize(self, plot_color):
        """
        Plots the drone position and its viewing window on previously selected
        figure
        """
        # Plot position
        plt.scatter(self.s_p[0], self.s_p[1], color=plot_color, marker='x')
        
        # Plot sensing region
        B_x = np.cos(Drone_Base.THETA) * self.obs_rad + self.s_p[0]
        B_y = np.sin(Drone_Base.THETA) * self.obs_rad + self.s_p[1]
        plt.plot(B_x, B_y, color=plot_color)
    
    
    def get_optimalbound(self):
        """ 
        Returns optimal bound for the chosen velocity controller or -1 if no
        bound exists. To be updated and implemented by any controller with
        an optimal bound
        """
        return -1


####################
class Drone_Constant(Drone_Base):
    """
    Drone that flies at a constant speed.
    """
    
    def __init__(self, drone_id, env_model=None, b_verbose=True, b_logging=True, **cfg):
        """
        Initialze a drone that flies at a constant velocity around the path
        
        New variables:
            v_const: constant velocity that the drone flies at
        """
        self.drone_desc = 'Constant velocity drone'

        super(Drone_Constant, self).__init__(drone_id=drone_id, env_model=env_model,
             b_logging=b_logging, b_verbose=b_verbose, **cfg)

    def form_traj(self):
        """
        Calculates a basic trajectory using TSP and then a velocity set to max velocity
        """
        if not(self.env_model):
            self.s_p = None
            self.s_v = None
            self.s_a = None
            self.s_j = None
            self.b_traj_ready = False
            return

        list_q = self.env_model.get_pois()
        q_trans = self.calc_tsp_order(list_q)

        self.list_q = q_trans @ list_q
        self.covar_e = q_trans @ self.covar_e @ q_trans.T

        self.create_s_function_map()

        return self.calc_traj(0)

    def create_s_function_map(self):
        """
        Creates a look-up table that links current time around the loop to a segment in a piecewise trajectory function
        """
        self.s_fmap = []

        temp_s_p1 = np.array(self.list_q)
        temp_s_p2 = np.roll(temp_s_p1, -1, axis=0)
        s_p_dist = np.sqrt(
            np.sum(np.power(temp_s_p1 - np.append([temp_s_p1[-1, :]], temp_s_p1[:-1, :], axis=0), 2), axis=1))
        s_t2 = np.cumsum(s_p_dist / self.vmax)
        s_t1 = np.roll(s_t2, 1, axis=0)
        s_t1[0] = 0

        for t1, t2, p1, p2 in zip(s_t1, s_t2, temp_s_p1, temp_s_p2):
            self.s_fmap.append([t1, t2,
                                partial(tfunc_linear, t1=t1, t2=t2, p1=p1, p2=p2),
                                partial(tfunc_const, val=(p2 - p1) / np.sqrt(np.sum(np.power(p2 - p1, 2))) * self.vmax),
                                partial(tfunc_const, val=np.zeros(p1.shape)),
                                partial(tfunc_const, val=np.zeros(p1.shape)),
                                {'type': 'linear', 't1': t1, 't2': t2, 'p1': p1, 'p2': p2, 'T': (t2 - t1)}])

####################
class Drone_Smith(Drone_Base):
    """
    Drone that flies with a velocity controller as proposed in Smith (2012)
        Smith SL, Schwager M, Rus D. Persistent robotic tasks: Monitoring and 
        sweeping in changing environments. IEEE Transactions on Robotics. 2012 
        Apr;28(2):410-26.
    
    New variables:
        J: number of rectangular segments to use as basis functions
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, v_max=20, v_min=0, 
                 covar_s_0_scale=100, J=100, b_verbose=True, b_logging=True):
        """
        Initialze a drone that flies at a constant velocity around the path
        """
        super(Drone_Smith, self).__init__(loop_path, poi, covar_e, 
             obs_window, covar_obs, drone_id=drone_id, theta0=theta0, fs=fs, 
             v_max=v_max, v_min=v_min, covar_s_0_scale=covar_s_0_scale,
             b_logging=b_logging, b_verbose=b_verbose)
        
        self.J = J
        self.create_v_controller() # Creates an optimal velocity controller
        
        
    def create_v_controller(self):
        """
        Creates an optimal position-dependent velocity controller with J steps
        based on the methodology outlined in Smith (2012)
        """
        # Growth in uncertainty is due to environmental noise
        p = np.diag(self.covar_e)
        
        # Decrease in uncertainty is from Kalman filter, which can only measure
        # when q is within sensing range. The decrease at steady state can be
        # approximated as the product of amount of time for 1 cycle of the loop
        # and the growth of the point
        
        # Total amount of time that each point is observed
        dtheta_obs = np.zeros((self.N_q, 1))
        list_gamma_obs = []     # list of observation segments
        list_theta_obs = []     # list of theta for beginning and ending of observation segments
        
        for ind_q, q in enumerate(self.list_q):            
            # Create region around q in which q can be sensed
            B_q = translate(self.B, xoff=q[0], yoff=q[1])
            # Split gamma and determine which section can sense point q
            gamma_temp = LineString(self.gamma)
            gamma_split = split(gamma_temp, B_q)
            
            list_dtheta = np.array([gamma_seg.length for gamma_seg in gamma_split])
            list_theta = np.cumsum(list_dtheta)
            # list_valid method fails around start/stop of loop 
            # list_valid = np.array([B_q.contains(Point(np.array(gamma_seg)[1,:])) for gamma_seg in gamma_split])
            # Valid/invalid segments will alternate. Determine if second
            # segment is valid or not. If too large, likely invalid.
            list_valid = np.zeros(list_theta.shape).astype(bool).flatten()
            if (list_dtheta[1] > self.L/2):
                list_valid[0::2] = True
            else:
                list_valid[1::2] = True
                
            N_segs = list_valid.shape[0]
            dtheta_obs[ind_q] = np.sum(list_dtheta[list_valid])
            theta_obs = []
            gamma_obs = []
            for ind_seg, valid, theta, gamma_seg in zip(range(N_segs), list_valid, list_theta, gamma_split):
                if (valid):
                    gamma_obs.append(gamma_seg)
                    if (ind_seg == 0):
                        theta_obs.append([0, theta])
                    elif (ind_seg == (N_segs-1)):
                        theta_obs.append([list_theta[ind_seg-1], self.L])
                    else:
                        theta_obs.append([list_theta[ind_seg-1], theta])
            
            list_theta_obs.append(theta_obs)
            list_gamma_obs.append(gamma_obs)
        
        list_gamma_obs_flat = [gamma_seg for sublist in list_gamma_obs for gamma_seg in sublist]
        gamma_cumobs = MultiLineString(list_gamma_obs_flat)
        gamma_cumobs = linemerge(gamma_cumobs)
        
        # Observed path should move at either max speed or minimum required 
        # time to take one sample
        T_obs = np.max(np.append(dtheta_obs/self.v_max, np.ones((self.N_q, 1))/self.fs, axis=1), axis=1)
        
        # Unobserved path should move at top speed
        if (isinstance(gamma_cumobs, shapely.geometry.LineString)):
            L_notobs = self.L - gamma_cumobs.length
        elif (isinstance(gamma_cumobs, shapely.geometry.MultiLineString)):
            L_notobs = self.L
            for gamma_cumobs_ in gamma_cumobs:
                    L_notobs -= gamma_cumobs_.length
                    
        T_notobs = L_notobs / self.v_max
        
        # Predicted time for single loop with 1 obs per location
        T0 = np.sum(T_obs) + T_notobs
        # First-order approximation of Kalman filter decrease
        c = T0 * p
        
        # Create basis functions
        list_beta = np.linspace(0, self.L, num=(self.J+1))
        list_beta = list_beta[0:-1].reshape((-1,1))

        list_dtheta_beta = self.calc_int_beta(list_theta_obs, list_beta)
        N_segs_total = sum([len(dtheta_beta_seg)**2 for dtheta_beta_seg in list_dtheta_beta[:,0]])
        
        # Calculate K as defined in Smith (2012) Eq. (8)
        K = np.zeros((self.N_q, self.J))
        for ind_i in range(self.N_q):
            for ind_j in range(self.J):
                K[ind_i, ind_j] = np.sum(list_dtheta_beta[ind_i,ind_j]) - p[ind_i]/c[ind_i]*self.L/self.J
        
        # Calculate X as defined in Smith (2012) Eq. (19)
        list_dtheta_beta_edge2edge = self.calc_int_beta_edge2edge(list_theta_obs, list_beta)
        X = np.zeros((N_segs_total, self.J))
        ind_seg = 0
        for ind_i in range(self.N_q):
            N_segs = len(list_dtheta_beta[ind_i, ind_j])
            for k in range(N_segs):
                for b in range(N_segs):
                    for ind_j in range(self.J):
                        X_growth = p[ind_i]*list_dtheta_beta_edge2edge[ind_seg, ind_j]
                        ind_decay = (k - np.arange(b + 1)) % N_segs
                        X_decay = c[ind_i]*sum([list_dtheta_beta[ind_i, ind_j][ind] for ind in ind_decay])
                        X[ind_seg, ind_j] = X_growth - X_decay
                    ind_seg += 1
        
        # Set up optimization problem
        prob_statement = pulp.LpProblem('Smith 2012', pulp.LpMinimize)
        
        # Create variables
        list_alphavar = [pulp.LpVariable('a{0:03d}'.format(i), lowBound=1/self.v_max,
                         cat=pulp.LpContinuous) for i in range(self.J)]
        marginvar = pulp.LpVariable('B', lowBound=0, cat=pulp.LpContinuous)
        
        # Add objective statement
        prob_statement += (marginvar)
        
        # Add constraints with X
        for ind_X, X_row in enumerate(X):
            prob_statement += pulp.LpConstraint((pulp.lpDot(X_row, list_alphavar) - marginvar), 
                                        sense=pulp.constants.LpConstraintLE, 
                                        name='X const{0}'.format(ind_X), rhs=0)
        # Add constraints with K
        for ind_K, K_row in enumerate(K):
            prob_statement += pulp.LpConstraint((pulp.lpDot(K_row, list_alphavar)), 
                                        sense=pulp.constants.LpConstraintGE, 
                                        name='K const{0}'.format(ind_K), rhs=0)
        
        # prob_statement.writeLP('SmithModel.lp')
        prob_statement.solve()
        
        list_alpha = np.array([v.varValue for v in prob_statement.variables()])[1:].reshape((-1,1))
        
        self.v_controller = np.append(list_beta, list_alpha, axis=1)
        print('Smith v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
        if (self.b_logging):
            self.logger.debug('v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
            self.logger.debug('alpha  = ' + np.array2string(list_alpha.flatten(), separator=', ', formatter={'float_kind':lambda x: '{0:2.5f}'.format(x)}, max_line_width=2148))

    
    def calc_int_beta(self, list_theta_obs, list_beta):
        """
        Calculates the length of path along each gamma segment, denoted by 
        theta_start and theta_stop, where q_i is observable that is in 
        list_beta[j:j+1]
        """
        list_dtheta_beta = []
        
        # Iterate through all points of interest
        for ind_i, theta_obs in enumerate(list_theta_obs):
            dtheta_beta = []
            # Iterate through all betas
            beta_start = list_beta.item(0)
            for ind_j in range(self.J):                
                if (ind_j == (len(list_beta) - 1)):
                    beta_stop = self.L
                else:
                    beta_stop = list_beta.item(ind_j+1)

                # Iterate through all different observable segments of gamma
                dtheta_beta_seg = []
                for theta_obs_seg in theta_obs:
                    temp_dtheta = 0
                    
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs_seg[0]
                    theta_stop = theta_obs_seg[1]
                    
                    # Covers all cases
                    if (theta_stop < theta_start):
                        if (theta_start < beta_stop):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (beta_stop - theta_start)])
                        elif (theta_stop > beta_start):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (theta_stop - beta_start)])
                    else:
                        if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                            temp_dtheta += min([(theta_stop - beta_start),
                                                (beta_stop - beta_start), 
                                                (theta_stop - theta_start),
                                                (beta_stop - theta_start)])
                    dtheta_beta_seg.append(temp_dtheta)
                dtheta_beta.append(dtheta_beta_seg)
                beta_start = beta_stop
            list_dtheta_beta.append(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    
    
    def calc_int_beta_edge2edge(self, list_theta_obs, list_beta):
        """
        Calculates the length of path overlapping with the basis functions 
        from the end of observable segment to the end of the next segment.
        theta_start and theta_stop represent the end of the first observable
        segment and end of the target observable segment, respectively, where 
        q_i is observable that is in list_beta[j:j+1]
        """
        list_dtheta_beta = []

        # Iterate through all points of interest and all combinations of segments
        for ind_i, theta_obs in enumerate(list_theta_obs):
            N_segs = len(theta_obs)
            # Iterate through all combinations
            dtheta_beta = []
            for k in range(N_segs):
                for b in range(N_segs):
                    ind_start = (k - b - 1) % N_segs
                    ind_stop = k
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs[ind_start][1]
                    theta_stop = theta_obs[ind_stop][1]
                    
                    beta_start = list_beta.item(0)
                    # Iterate through all betas
                    dtheta_beta_seg = []
                    for ind_j in range(self.J):                
                        if (ind_j == (len(list_beta) - 1)):
                            beta_stop = self.L
                        else:
                            beta_stop = list_beta.item(ind_j+1)
                        
                        temp_dtheta = 0
                        
                        # Covers all cases
                        if (theta_stop < theta_start):
                            if (theta_start < beta_stop):
                                temp_dtheta += min([(beta_stop - beta_start),
                                                    (beta_stop - theta_start)])
                            elif (theta_stop > beta_start):
                                temp_dtheta += min([(beta_stop - beta_start),
                                                    (theta_stop - beta_start)])
                        elif (theta_stop > theta_start):
                            if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                                temp_dtheta += min([(theta_stop - beta_start),
                                                    (beta_stop - beta_start), 
                                                    (theta_stop - theta_start),
                                                    (beta_stop - theta_start)])
                        else:
                            # Condition means checking beta over entire loop
                            temp_dtheta += (beta_stop - beta_start)
                            
                        dtheta_beta_seg.append(temp_dtheta)
                        beta_start = beta_stop
                    dtheta_beta.append(dtheta_beta_seg)
            list_dtheta_beta.extend(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    
    
    def calc_movement(self):
        """
        Velocity depends on current position and movement that can be taken 
        until next step
        """
        self.dtheta = 0
        time_plan = 1/self.fs
        ind_1 = np.argmin(self.v_controller[:,0] < self.theta)
        ind_1 = (ind_1 - 1) % self.J
        while (time_plan > 0):
            ind_2 = (ind_1 + 1) % self.J
            v_seg = 1/self.v_controller[ind_1, 1]
            dist_seg = (self.v_controller[ind_2, 0] - (self.theta + self.dtheta)) % self.L
            t_seg = (dist_seg / v_seg) # time to reach the next beta segment
            if (t_seg >= time_plan):
                self.dtheta += time_plan*v_seg
                time_plan = 0
            else:
                self.dtheta += dist_seg
                time_plan -= t_seg
                ind_1 = ind_2


####################       
class Drone_Ostertag2019(Drone_Base):
    """
    Drone that flies with a velocity controller as proposed in Ostertag (2018)

    Utilizes a greedy algorithm to find the optimal velocity controller that
    meets a minimium bound of the maximum steady state Kalman filter 
    uncertainty
    
    New variables:
        J: number of rectangular segments to use as basis functions
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, v_max=20, v_min=0, 
                 covar_s_0_scale=100, J=100, b_logging=True, b_verbose=True):
        """
        Initialze a drone that 
        """
        super(Drone_Ostertag2019, self).__init__(loop_path, poi, covar_e,
             obs_window, covar_obs, drone_id=drone_id, theta0=theta0, fs=fs, 
             v_max=v_max, v_min=v_min, covar_s_0_scale=covar_s_0_scale,
             b_logging=b_logging, b_verbose=b_verbose)
        
        self.J = J
        self.create_v_controller() # Creates an optimal velocity controller
        
        
    def create_v_controller(self):
        """
        Creates an optimal position-dependent velocity controller with J steps
        based on the methodology outlined in Ostertag (2018)
        """
        # Total amount of time that each point is observed
        dtheta_obs = np.zeros((self.N_q, 1))
        list_gamma_obs = []     # list of observation segments
        list_theta_obs = []     # list of theta for beginning and ending of observation segments
        
        for ind_q, q in enumerate(self.list_q):            
            # Create region around q in which q can be sensed
            B_q = translate(self.B, xoff=q[0], yoff=q[1])
            # Split gamma and determine which section can sense point q
            gamma_temp = LineString(self.gamma)
            gamma_split = split(gamma_temp, B_q)
            
            list_dtheta = np.array([gamma_seg.length for gamma_seg in gamma_split])
            list_theta = np.cumsum(list_dtheta)
            # Valid/invalid segments will alternate. Determine if second
            # segment is valid or not. If too large, likely invalid.
            list_valid = np.zeros(list_theta.shape).astype(bool).flatten()
            if (list_dtheta[1] > self.L/2):
                list_valid[0::2] = True
            else:
                list_valid[1::2] = True
            
            N_segs = list_valid.shape[0]
            dtheta_obs[ind_q] = np.sum(list_dtheta[list_valid])
            theta_obs = []
            gamma_obs = []
            for ind_seg, valid, theta, gamma_seg in zip(range(N_segs), list_valid, list_theta, gamma_split):
                if (valid):
                    gamma_obs.append(gamma_seg)
                    if (ind_seg == 0):
                        theta_obs.append([0, theta])
                    elif (ind_seg == (N_segs-1)):
                        theta_obs.append([list_theta[ind_seg-1], self.L])
                    else:
                        theta_obs.append([list_theta[ind_seg-1], theta])
            
            list_theta_obs.append(theta_obs)
            list_gamma_obs.append(gamma_obs)
        
        list_gamma_obs_flat = [gamma_seg for sublist in list_gamma_obs for gamma_seg in sublist]
        gamma_cumobs = MultiLineString(list_gamma_obs_flat)
        gamma_cumobs = linemerge(gamma_cumobs)
        
        # Unobserved path should move at top speed
        if (isinstance(gamma_cumobs, shapely.geometry.LineString)):
            L_notobs = self.L - gamma_cumobs.length
        elif (isinstance(gamma_cumobs, shapely.geometry.MultiLineString)):
            L_notobs = self.L
            for gamma_cumobs_ in gamma_cumobs:
                    L_notobs -= gamma_cumobs_.length
        
        T_notobs = L_notobs / self.v_max
        
        # Generate Kalman filter steady state equations that need to be solved
        # for each iteration of the loop
        kf_depth = 6
        self.list_kf_eqs = generate_func_kfsteadystate(depth=kf_depth)

        V = sympy.symbols('V')
        Twc = sympy.symbols('Twc')
        W = sympy.symbols('W')
        lambda_T0 = sympy.symbols('lambda_T0')
        
        N_loops = 100
        N_kf = np.zeros((N_loops, self.N_q)).astype(int) # Number of observations at each point of interest
        N_kf[0,:] = 1
        sig_max = np.zeros((N_loops-1, self.N_q))
        for ind_loop in range(N_loops-1):
            # Observed path should move at either max speed or minimum required 
            # time to take one sample
            T_obs = np.max(np.append(dtheta_obs/self.v_max, N_kf[ind_loop,:].reshape(-1,1)/self.fs, axis=1), axis=1)
            
            # Predicted worst case time for single loop with d_i observations
            # per location i
            T_loop = np.ceil((np.sum(T_obs) + T_notobs)/self.fs)*self.fs
            
            # Calculate steady state value for each point
            for ind_i in range(self.N_q):
                eq_temp = self.list_kf_eqs[N_kf[ind_loop,ind_i] - 1].subs({V:self.covar_obs, W:self.covar_e[ind_i, ind_i], Twc:T_loop})
                if (ind_loop == 0):
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, self.covar_s[ind_i, ind_i])
                else:
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, sig_max[ind_loop-1, ind_i])
            
            N_kf[ind_loop+1, :] = N_kf[ind_loop, :]
            # Add observation the poi with the highest uncertainty
            ind_max = np.argmax(sig_max[ind_loop, :])
            N_kf[ind_loop+1, ind_max] += 1
            
            if (N_kf[ind_loop+1, ind_max] >= kf_depth):
                break
        
        sig_max_temp = sig_max.max(axis=1)
        ind_row_max = np.argmin(sig_max_temp[(sig_max_temp > 0).flatten()])
        
        if (self.b_verbose):
            print(N_kf[ind_row_max,:])
            print(sig_max[ind_row_max,:])
        if (self.b_logging):
            self.logger.debug(N_kf[ind_row_max,:].flatten())
            self.logger.debug(sig_max[ind_row_max,:].flatten())
        
        # Create basis functions
        list_beta = np.linspace(0, self.L, num=(self.J+1))
        list_beta = list_beta[0:-1].reshape((-1,1))
        
        # Calculate the betas and portion of betas from which POI can be
        # observed
        list_dtheta_beta = self.calc_int_beta(list_theta_obs, list_beta)
        
        # Create list of alpha coefficients for optimization
        list_coeff = np.zeros((self.N_q, self.J))
        for ind_i in range(self.N_q):
            for ind_j in range(self.J):
                list_coeff[ind_i, ind_j] = np.sum(list_dtheta_beta[ind_i, ind_j])

        # Set up optimization problem
        prob_statement = pulp.LpProblem('Ostertag 2018', pulp.LpMinimize)
        
        # Create variables
        list_alphavar = [pulp.LpVariable('a{0:03d}'.format(i), lowBound=1/self.v_max,
                         cat=pulp.LpContinuous) for i in range(self.J)]
        
        # Add objective statement
        prob_statement += pulp.lpSum(list_alphavar)
        
        # Add constraints from greedy Kalman Filter alg
        for ind_coeff, coeff_row in enumerate(list_coeff):
            prob_statement += pulp.LpConstraint((pulp.lpDot(coeff_row, list_alphavar)), 
                                        sense=pulp.constants.LpConstraintGE, 
                                        name='KF const{0}'.format(ind_coeff), rhs=N_kf[ind_row_max, ind_coeff]/self.fs)
        
        # prob_statement.writeLP('OstertagModel.lp')
        prob_statement.solve()
        
        list_alpha = np.array([v.varValue for v in prob_statement.variables()]).reshape((-1,1))
        
        try:
            self.v_controller = np.append(list_beta, list_alpha, axis=1)
        except:
            print('list_beta: %d'.format(len(list_beta)))
            print('list_alpha %d'.format(len(list_alpha)))
        
        print('Ostertag v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
        self.covar_bound = sig_max_temp[ind_row_max]
        # print(list_alpha)
        if (self.b_logging):
            self.logger.debug('v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
            self.logger.debug('alpha  = ' + np.array2string(list_alpha.flatten(), separator=', ', formatter={'float_kind':lambda x: '{0:2.5f}'.format(x)}, max_line_width=2148))


    def calc_int_beta(self, list_theta_obs, list_beta):
        """
        Calculates the length of path along each gamma segment, denoted by 
        theta_start and theta_stop, where q_i is observable that is in 
        list_beta[j:j+1]
        """
        list_dtheta_beta = []
        
        # Iterate through all points of interest
        for ind_i, theta_obs in enumerate(list_theta_obs):
            dtheta_beta = []
            # Iterate through all betas
            beta_start = list_beta.item(0)
            for ind_j in range(self.J):                
                if (ind_j == (len(list_beta) - 1)):
                    beta_stop = self.L
                else:
                    beta_stop = list_beta.item(ind_j+1)

                # Iterate through all different observable segments of gamma
                dtheta_beta_seg = []
                for theta_obs_seg in theta_obs:
                    temp_dtheta = 0
                    
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs_seg[0]
                    theta_stop = theta_obs_seg[1]
                    
                    # Covers all cases
                    if (theta_stop < theta_start):
                        if (theta_start < beta_stop):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (beta_stop - theta_start)])
                        elif (theta_stop > beta_start):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (theta_stop - beta_start)])
                    else:
                        if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                            temp_dtheta += min([(theta_stop - beta_start),
                                                (beta_stop - beta_start), 
                                                (theta_stop - theta_start),
                                                (beta_stop - theta_start)])
                    dtheta_beta_seg.append(temp_dtheta)
                dtheta_beta.append(dtheta_beta_seg)
                beta_start = beta_stop
            list_dtheta_beta.append(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    

    def calc_movement(self):
        """
        Velocity depends on current position and movement that can be taken 
        until next step
        """
        self.dtheta = 0
        time_plan = 1/self.fs
        ind_1 = np.argmin(self.v_controller[:,0] < self.theta)
        ind_1 = (ind_1 - 1) % self.J
        while (time_plan > 0):
            ind_2 = (ind_1 + 1) % self.J
            v_seg = 1/self.v_controller[ind_1, 1]
            dist_seg = (self.v_controller[ind_2, 0] - (self.theta + self.dtheta)) % self.L
            t_seg = (dist_seg / v_seg) # time to reach the next beta segment
            if (t_seg >= time_plan):
                self.dtheta += time_plan*v_seg
                time_plan = 0
            else:
                self.dtheta += dist_seg
                time_plan -= t_seg
                ind_1 = ind_2
 
  
    def get_optimalbound(self):
        """
        Return the theoretical optimal bound for the algorithm
        """
        return self.covar_bound


####################
class Drone_Ostertag2020(Drone_Base):
    """
    Drone that flies with a feasible trajectory controller as proposed in Ostertag (2020)

    Utilizes an iterative algorithm that alternates between the Greedy Knockdown Algorithm and a time-minimizing
     feasible trajectory calculated using B-splines to find the optimal trajectory that minimizes the maximum steady
     state Kalman filter uncertainty

    New variables:
        ??
    """
    DEFAULT_FU = 20

    def __init__(self, drone_id, env_model=None, b_verbose=True, b_logging=True, **cfg):
        """
        Initialze a drone that
        """
        self.list_kf_eqs = None

        if 'fu' in cfg.keys():
            self.fu = cfg['fu']
        else:
            self.fu = Drone_Ostertag2020.DEFAULT_FU

        super(Drone_Ostertag2020, self).__init__(drone_id=drone_id, env_model=env_model,
                                                 b_logging=b_logging, b_verbose=b_verbose,
                                                 **cfg)

        self.drone_desc = 'Drone with minimum-time feasible B-spline trajectory'

    def form_problem(self, m, d, b_slack=False):
        """
        Forms and returns the formatted convex optimization problem and associated parameters
        """
        m_valid = m + K_CONST + 1
        dim_valid = d

        # Form objective statement using B-spline with k = 3
        # NOTE: c contains all of the control point locations for all dimensions. Each dimension is stacked so that c is
        # a vector
        c = cvxpy.Variable((m_valid, dim_valid))
        gamma_v = cvxpy.Variable()
        gamma_a = cvxpy.Variable()
        gamma_j = cvxpy.Variable()
        gamma_gamma = cvxpy.Variable()

        col = np.zeros(m_valid)
        col[0], col[1], col[2], col[3] = -1, 3, -3, 1
        row = np.zeros(m + 1)
        row[0] = 1
        A_ = linalg.toeplitz(col, row)
        A = A_ @ A_.T

        ## Form objective statement
        J = 0
        if b_slack:
            # Minimize slack variables to check constraints
            """
            J += gamma_v
            J += 0.01 * gamma_j
            """
            J += gamma_gamma

            J += 0.001 * gamma_a + 0.001 * gamma_j

            # for ind_dim in range(dim_valid):
            #     # J += (1 / param_dtau)**(2*K_CONST - 1) * cvxpy.quad_form(c[:, ind_dim], A) # Equivalent to c_dim.T @ A @ c_dim
            #     J += 1 * cvxpy.quad_form(c[:, ind_dim], A)

        else:
            for ind_dim in range(dim_valid):
                # J += (1 / param_dtau)**(2*K_CONST - 1) * cvxpy.quad_form(c[:, ind_dim], A) # Equivalent to c_dim.T @ A @ c_dim
                J += cvxpy.quad_form(c[:, ind_dim], A)

        # Configure constraints
        #   0: s(0)     = p_in
        #   1: s(T)     = p_out
        #   2: s^(1)(0) = v_in
        #   3: s^(1)(T) = v_out
        #   4: s^(2)(0) = 0
        #   5: s^(2)(T) = 0
        #   6: s^(3)(0) = 0
        #   7: s^(3)(T) = 0
        D = np.zeros((8, m_valid))

        # Define parameters for disciplined convex programming
        param_b = cvxpy.Parameter((8, dim_valid))
        param_B = cvxpy.Parameter(nonneg=True)
        param_q = cvxpy.Parameter((dim_valid,))
        param_dtau = cvxpy.Parameter(nonneg=True)
        param_vmax = cvxpy.Parameter(nonneg=True)
        param_amax = cvxpy.Parameter(nonneg=True)
        param_jmax = cvxpy.Parameter(nonneg=True)

        # Condition on slack variable
        # st = [gamma >= 0]
        st = []

        # Equality Constraints for enter/exit conditions
        # Position: s(0) and s(T)
        D[0, :3] = [1 / 6, 2 / 3, 1 / 6]
        D[1, -4:-1] = [1 / 6, 2 / 3, 1 / 6]

        # Velocity: s^(1)(0) and s^(1)(T)
        D[2, :3] = np.array([-1 / 2, 0, 1 / 2])
        D[3, -4:-1] = np.array([-1 / 2, 0, 1 / 2])

        # Acceleration: s^(2)(0) and s^(2)(T)
        D[4, :3] = np.array([1, -2, 1])
        D[5, -4:-1] = np.array([1, -2, 1])

        # Jerk: s^(3)(0) and s^(3)(T)
        D[6, :4] = np.array([-1, 3, -3, 1])
        D[7, -4:] = np.array([-1, 3, -3, 1])

        st_eq = D @ c == param_b
        st += [st_eq]

        # Inequality Constraints for remaining in S_free
        # Constraints on location, velocity, and acceleration
        c_imp_p = c[(K_CONST - 1):(m + K_CONST - 2), :dim_valid]
        for _c in c_imp_p:
            st += [cvxpy.sum((_c - param_q) ** 2) <= param_B ** 2]

        # Velocity
        # c_v_{i} = c_{i} - c_{i-1}
        for ind_i in range(K_CONST - 1, m + K_CONST - 2):
            _c_v = (c[ind_i, :dim_valid] - c[ind_i - 1, :dim_valid]) / param_dtau
            st += [cvxpy.sum(_c_v ** 2) - gamma_v <= param_vmax ** 2]

        # Accleration
        # c_v_{i} = c_{i} - 2*c_{i-1} + c_{i-2}
        for ind_i in range(K_CONST - 1, m + K_CONST - 1):
            _c_a = (c[ind_i, :dim_valid] - 2 * c[ind_i - 1, :dim_valid] + c[ind_i - 2, :dim_valid]) / param_dtau ** 2
            st += [cvxpy.sum(_c_a ** 2) - gamma_a <= param_amax ** 2]

        # Jerk
        # c_v_{i} = c_{i} - 3*c_{i-1} + 3*c_{i-2} - c_{i-3}
        for ind_i in range(K_CONST, m + K_CONST):
            _c_j = (c[ind_i, :dim_valid] - 3 * c[ind_i - 1, :dim_valid] + 3 * c[ind_i - 2, :dim_valid] - c[ind_i - 3,
                                                                                                         :dim_valid]) / param_dtau ** 3
            st += [cvxpy.sum(_c_j ** 2) - gamma_j <= param_jmax ** 2]

        if b_slack:
            # Max norm of slack
            st += [gamma_v / param_vmax ** 2 <= gamma_gamma,
                   gamma_a / param_amax ** 2 <= gamma_gamma,
                   gamma_j / param_jmax ** 2 <= gamma_gamma]
        else:
            gamma_v.value = 0
            gamma_a.value = 0
            gamma_j.value = 0

        opt_prob = cvxpy.Problem(cvxpy.Minimize(J), st)
        prob_vars = {'c': c,
                'gamma_v': gamma_v,
                'gamma_a': gamma_a,
                'gamma_j': gamma_j,
                'gamma': gamma_gamma
                }
        prob_params = {'b': param_b,
                  'B': param_B,
                  'q': param_q,
                  'dtau': param_dtau,
                  'vmax': param_vmax,
                  'amax': param_amax,
                  'jmax': param_jmax
                  }

        return opt_prob, prob_params, prob_vars

    def opt_bst(self, q_pos, p_in, p_out, v_in, v_out, T_min):
        """
        Optimization to find minimum time, feasible B-spline path for a single segment
        """
        print('q_pos = {0}'.format(q_pos))
        print('p_in = {0}'.format(p_in))
        print('p_out = {0}'.format(p_out))
        print('v_in = {0}'.format(v_in))
        print('v_out = {0}'.format(v_out))

        dim_valid = min([q_pos.shape[-1], p_in.shape[-1], p_out.shape[-1], v_in.shape[-1], v_out.shape[-1]])
        m = np.floor(T_min * self.fu).astype(int)

        opt_prob, prob_params, prob_vars = self.form_problem(m, dim_valid, b_slack=True)

        # Input values for parameters
        prob_params['q'].value = q_pos
        prob_params['B'].value = self.obs_rad
        prob_params['vmax'].value = self.vmax
        prob_params['amax'].value = self.amax
        prob_params['jmax'].value = self.jmax

        # TODO Implement more intelligent search method
        stop_gammatol = 0.01
        stop_ttol = 0.05
        maxiter = 20
        smallstep = 1.05
        backstep = 1 / (smallstep)**(1/5)
        barr_valid = np.zeros(maxiter)
        barr_smallstep = np.ones(maxiter)
        barr_solverfailed = np.zeros(maxiter)
        arr_t = np.zeros(maxiter)
        arr_gamma = np.zeros(maxiter)
        arr_gamma_v = np.zeros(maxiter)
        arr_gamma_a = np.zeros(maxiter)
        arr_gamma_j = np.zeros(maxiter)
        b_forward = True
        arr_c_best = np.zeros((2,2))
        m_best = m
        dt_best = T_min / m
        ind_t_best = -1

        # Initial
        arr_t[0] = T_min
        barr_smallstep[0] = 0

        for ind in range(0, maxiter-1):
            t = arr_t[ind]

            # Set knot spacing parameters
            dt = t / m
            prob_params['dtau'].value = dt

            b = np.zeros((8, dim_valid))
            b[0, :] = p_in[:dim_valid]
            b[1, :] = p_out[:dim_valid]
            b[2, :] = v_in[:dim_valid] * dt
            b[3, :] = v_out[:dim_valid] * dt
            prob_params['b'].value = b

            if self.b_verbose:
                print('--SLACK--')
                print('  t:  {0:0.4f}'.format(t))
                print('  m:   {0:d}'.format(m))
                print('  dt:  {0:0.4f}'.format(dt))
            try:
                trun_start = time.process_time()
                opt_prob.solve(solver=cvxpy.MOSEK) #, verbose=True) # if issue
                trun_end = time.process_time()
                if self.b_verbose:
                    print('  proc: {0:0.2f} ms'.format(1000*(trun_end - trun_start)))
            except:
                barr_solverfailed[ind] = 1
                print('  Solver failed')

            if not(barr_solverfailed[ind]):
                slack_max = np.sign(prob_vars['gamma'].value)*np.sqrt(np.abs(prob_vars['gamma'].value))
                slack_v = np.sign(prob_vars['gamma_v'].value)*np.sqrt(np.abs(prob_vars['gamma_v'].value))
                slack_a = np.sign(prob_vars['gamma_a'].value)*np.sqrt(np.abs(prob_vars['gamma_a'].value))
                slack_j = np.sign(prob_vars['gamma_j'].value)*np.sqrt(np.abs(prob_vars['gamma_j'].value))
                arr_gamma[ind] = slack_max
                arr_gamma_v[ind] = slack_v
                arr_gamma_a[ind] = slack_a
                arr_gamma_j[ind] = slack_j

                if (opt_prob.status == 'optimal'):
                    print('  ovr: {0:0.4f}'.format(slack_max))
                    print('  vel: {0:0.4f}'.format(slack_v))
                    print('  acc: {0:0.4f}'.format(slack_a))
                    print('  jer: {0:0.4f}'.format(slack_j))
                else:
                    barr_solverfailed[ind] = 1
                    print('  Optimization did not converge')
                    debug_data, debug_chain, debug_invdata = opt_prob.get_problem_data(cvxpy.MOSEK)
                    print('  Status: ', opt_prob.status)

            # Adjust t based on little-big jumps
            # Little jumps are used to approximate derivative, big jump uses the derivative to explore larger space

            """
            if big jump
                if invalid
                    set up small jump
                if valid
                    set up back search
            if small jump
                if invalid
                    set up big jump
                if valid
                    set up back search
                    
            if invalid
                if big jump,
                    set up small jump
                else
                    set up big jump
            else
                set up back search
                
            if solver failed
                try again with small jump
            """

            if barr_solverfailed[ind]:
                if b_forward:
                    arr_t[ind+1] = t * smallstep
                    barr_smallstep[ind+1] = 1
                    print('    solver failed. small step to t = {0:0.3f}'.format(arr_t[ind+1]))
                elif not(b_forward):
                    print('    solver failed. use previous best of t = {0:0.3f}'.format(arr_t[ind_t_best]))
                    break
            else:
                if slack_max <= stop_gammatol:
                    # Begin back search
                    # TODO Make smarter back search
                    if (ind_t_best == -1) or (arr_t[ind_t_best] > t):
                        ind_t_best = ind
                        arr_c_best = prob_vars['c'].value
                        m_best = m
                        dt_best = t / m

                    barr_valid[ind] = 1
                    b_forward = False

                    if t == T_min:
                        print('    result valid. t = T_min = {0:0.3f}'.format(arr_t[ind]))
                        break
                    else:
                        arr_t[ind+1] = t * backstep
                        print('    result valid. back step to t = {0:0.3f}'.format(arr_t[ind+1]))
                else:
                    if not(b_forward):
                        print('    result invalid. use previous best of t = {0:0.3f}'.format(arr_t[ind_t_best]))
                        break
                    if barr_smallstep[ind]:
                        # Do big step
                        ind_prev = np.max(np.where(np.logical_and(np.logical_not(barr_smallstep), np.logical_not(barr_solverfailed))))
                        t_prev = arr_t[ind_prev]
                        t_curr = arr_t[ind]
                        val_prev = arr_gamma[ind_prev]
                        val_curr = arr_gamma[ind]
                        val_slope = (val_curr - val_prev) / (t_curr - t_prev)
                        t_int = t_curr - val_curr / val_slope
                        arr_t[ind+1] = t_int
                        barr_smallstep[ind+1] = 0
                        print('    result invalid. big step to t = {0:0.3f}'.format(arr_t[ind+1]))

                        # TODO Adjust problem formulation by increasing m?
                    else:
                        # Do small step
                        arr_t[ind+1] = t * smallstep
                        barr_smallstep[ind+1] = 1
                        print('    result invalid. small step to t = {0:0.3f}'.format(arr_t[ind+1]))

        return arr_c_best, m_best, dt_best

    def split_T_obs(self, list_traj):
        """
        Extract the total time of non-observed regions and the amount of time observed for each
        specific POI
        """
        T_notobs = 0
        arr_T = np.zeros(self.N_q)
        ind_q = 0
        for traj_segment in list_traj:
            if traj_segment['type'] == 'bspline':
                arr_T[ind_q] = traj_segment['T']
                ind_q += 1
            elif traj_segment['type'] =='linear':
                T_notobs += traj_segment['T']

        return arr_T, T_notobs

    def calc_control_points(self, arr_T_min, b_calc=None):
        """
        Calculates control points and knots for a B-spline representation of a trajectory
        """

        print('Calculating B-spline trajectories with minimum times of:')
        print(arr_T_min)

        list_traj = []

        # Iterate through POI to determine path in B and to the next POI
        for ind_curr in range(len(self.list_q)):
            T_min = arr_T_min[ind_curr]

            ind_prev = ind_curr - 1
            ind_next = (ind_curr + 1) % len(self.list_q)

            pos_prev = self.list_q[ind_prev]
            pos_curr = self.list_q[ind_curr]
            pos_next = self.list_q[ind_next]

            print('\nPOI {0:03d} at ({1:3.1f}, {2:3.1f})'.format(ind_curr, pos_curr[0], pos_curr[1]))
            print('  T_min: {0}'.format(T_min))

            if not(b_calc is None) and not(b_calc[ind_curr]):
                # If control points are valid from previous calculation, then do not recalculate
                list_traj.append(self.list_traj[2*ind_curr])     # bspline
                list_traj.append(self.list_traj[2*ind_curr + 1]) # linear
                print('  No change from previous iteration. Reusing trajectory.')
            else:
                print('  Change from previous iteration. Recalculating.')
                # Calculate B-spline control points (pos, vel, acc, jer) for trajectory in region B
                pos_din = pos_curr - pos_prev
                pos_dout = pos_next - pos_curr

                theta_in = np.arctan2(-pos_din[1], -pos_din[0])
                theta_out = np.arctan2(pos_dout[1], pos_dout[0])
                pos_in = pos_curr + np.array([np.cos(theta_in), np.sin(theta_in)]) * self.obs_rad
                pos_out = pos_curr + np.array([np.cos(theta_out), np.sin(theta_out)]) * self.obs_rad

                v_in = self.vmax * pos_din / np.sum(pos_din ** 2) ** 0.5
                v_out = self.vmax * pos_dout / np.sum(pos_dout ** 2) ** 0.5
                c_p, m, dt = self.opt_bst(q_pos=pos_curr, p_in=pos_in, p_out=pos_out, v_in=v_in, v_out=v_out, T_min=T_min)
                c_v, c_a, c_j = calc_ctrl_deriv(c_p, dt)
                list_traj.append({'type':'bspline', 'c_p':c_p, 'c_v':c_v, 'c_a':c_a, 'c_j':c_j, 'm':m, 'dt':dt,
                                  'T_min':T_min, 'T':m*dt})

                pos_in_next = pos_next - np.array([np.cos(theta_out), np.sin(theta_out)]) * self.obs_rad
                t_to_next = np.sum((pos_in_next - pos_out) ** 2) ** 0.5 / self.vmax
                list_traj.append({'type':'linear', 'p1': pos_out, 'p2': pos_in_next, 'v':self.vmax, 'T':t_to_next})

        return list_traj

    def form_traj(self):
        """
        Calculates a minimum-time, feasible trajectory using the methods outlined in Ostertag (2020)
        1. Calculate point order using TSP
        2. Iterate:
          2a. Solve for minimum time path using B-spline formulation that meets observation time constraints
          2b. Solve for optimal number of observations using Greedy Knockdown Algorithm
          2c. If optimal number of observations is the same, exit
        """
        if not(self.env_model):
            self.s_p = None
            self.s_v = None
            self.s_a = None
            self.s_j = None
            self.b_traj_ready = False
            return

        list_q = self.env_model.get_pois()
        q_trans = self.calc_tsp_order(list_q)

        self.list_q = q_trans @ list_q
        self.covar_e = q_trans @ self.covar_e @ q_trans.T

        # Initial trajectory calculation for d_i = 1 for all i to seed the Greedy Knockdown Algorithm
        self.list_traj = self.calc_control_points(np.ones(self.N_q) / self.fs)
        arr_T_obs, T_notobs = self.split_T_obs(self.list_traj)

        # Greedy Knockdown to calculate optimal number of observations
        self.list_d = self.greedy_knockdown_algorithm(arr_T_obs, T_notobs)
        list_d_prev = np.ones(self.list_d.shape)
        while not(all(self.list_d == list_d_prev)):
            # Determine which points have had a changed number of observations
            arr_b_recalc = np.logical_not(self.list_d == list_d_prev)
            list_d_prev = self.list_d[:]
            self.list_traj = self.calc_control_points(list_d_prev / self.fs, b_calc=arr_b_recalc)
            arr_T_obs, T_notobs = self.split_T_obs(self.list_traj)
            self.list_d = self.greedy_knockdown_algorithm(arr_T_obs, T_notobs)

        self.create_s_function_map()

        return self.calc_traj(0)


    def create_s_function_map(self):
        """
        Creates a look-up table that links current time around the loop to a segment in a piecewise trajectory function
        """
        self.s_fmap = []

        s_t2 = np.cumsum(np.array([traj_seg['T'] for traj_seg in self.list_traj]))
        s_t1 = np.roll(s_t2, 1, axis=0)
        s_t1[0] = 0

        for t1, t2, traj_seg in zip(s_t1, s_t2, self.list_traj):
            if traj_seg['type'] == 'linear':
                p1 = traj_seg['p1']
                p2 = traj_seg['p2']
                v = traj_seg['v']
                self.s_fmap.append([t1, t2,
                                    partial(tfunc_linear, t1=t1, t2=t2, p1=p1, p2=p2),
                                    partial(tfunc_const,
                                            val=(p2 - p1) / np.sqrt(np.sum(np.power(p2 - p1, 2))) * self.vmax),
                                    partial(tfunc_const, val=np.zeros(p1.shape)),
                                    partial(tfunc_const, val=np.zeros(p1.shape)),
                                    {'type': 'linear', 't1': t1, 't2': t2, 'p1': p1, 'p2': p2, 'T': (t2 - t1)}])
            elif traj_seg['type'] == 'bspline':
                k = 3
                m = traj_seg['m']
                c_p = traj_seg['c_p']
                c_v = traj_seg['c_v']
                c_a = traj_seg['c_a']
                c_j = traj_seg['c_j']

                dt = traj_seg['dt']

                tau = np.arange(-k * dt, (m + k + 1 + 1) * dt, dt)[0:m + 2 * k + 1 + 1] + t1
                # NOTE: Additional "+ 1" at stopping point is required to generate knot exactly at (M + k_test + 1)
                tau_p = tau
                tau_v = tau[:-1]
                tau_a = tau[:-2]
                tau_j = tau[:-3]

                self.s_fmap.append([t1, t2,
                                    form_bspline(c_p, tau_p, k),
                                    form_bspline(c_v, tau_v, k-1),
                                    form_bspline(c_a, tau_a, k-2),
                                    form_bspline(c_j, tau_j, k-3),
                                    traj_seg])
        print('New trajectory function map generated.')