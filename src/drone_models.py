import numpy as np


class Crazyflie:
    def __init__(self):
        """
        Parameters for the Crazyflie 2.0 based on code from MEAM 620 and Bernd Pfrommer

        Model assumptions based on physical measurements:

        motor + mount + vicon marker = mass point of 3g
        arm length of mass point: 0.046m from center
        battery pack + main board are combined into cuboid (mass 18g) of dimensions:

            width  = 0.03m
            depth  = 0.03m
            height = 0.012m
        """

        m = 0.030  # weight (in kg) with 5 vicon markers (each is about 0.25g)
        g = 9.81  # gravitational constant
        I = np.array([[1.43e-5, 0, 0],
                      [0, 1.43e-5, 0],
                      [0, 0, 2.89e-5]])  # inertial tensor in m^2 kg
        L = 0.046  # arm length in m

        self.mass = m
        self.I = I
        self.invI = np.linalg.inv(I)
        self.grav = g
        self.arm_length = L

        self.maxangle = 40 * np.pi / 180  # you can specify the maximum commanded angle here
        self.maxF = 2.5 * m * g  # left these untouched from the nano plus
        self.minF = 0.05 * m * g  # left these untouched from the nano plus

        self.k_pi = np.array([5, 5, 50])
        self.k_di = np.array([2.5, 2.5, 10])

        self.k_p = np.array([250, 250, 250])
        self.k_d = np.array([20, 20, 20])


class Phantom3:
    def __init__(self):
        """
        Parameters for the Phantom3

        Model assumptions from Phantom3 description
        """

        m = 1.216
        g = 9.81
        I = np.array([[0.03356,       0,       0],
                      [      0, 0.03356,       0],
                      [      0,       0, 0.06782]]) # Scaled version of Crazyflie inertial tensor
        L = 0.350  # arm length in m

        self.mass = m
        self.I = I
        self.invI = np.linalg.inv(I)
        self.grav = g
        self.arm_length = L

        self.maxangle = 60 * np.pi / 180
        self.maxF = 2.20 * m * g
        self.minF = 0.05 * m * g

        self.k_pi = np.array([1, 1, 3])
        self.k_di = np.array([2, 2, 3])

        self.k_p = np.array([250, 250, 250])
        self.k_d = np.array([20, 20, 20])

        self.maxXYaccel = 2 * self.maxF / self.mass * np.sin(self.maxangle)

class Phantom3_vel:
    def __init__(self):
        """
        Parameters for the Phantom3 and a velocity-based plan

        Model assumptions from Phantom3 description
        """

        m = 1.216
        g = 9.81
        I = np.array([[0.03356,       0,       0],
                      [      0, 0.03356,       0],
                      [      0,       0, 0.06782]]) # Scaled version of Crazyflie inertial tensor
        L = 0.350  # arm length in m

        self.mass = m
        self.I = I
        self.invI = np.linalg.inv(I)
        self.grav = g
        self.arm_length = L

        self.maxangle = 60 * np.pi / 180
        self.maxF = 2.20 * m * g
        self.minF = 0.05 * m * g

        self.k_pi = np.array([1.1, 1.1, 3.3])
        self.k_di = np.array([3, 3, 4.5])

        self.k_p = np.array([250, 250, 250])
        self.k_d = np.array([20, 20, 20])

        self.maxXYaccel = 1.0 * self.maxF / self.mass * np.sin(self.maxangle)
