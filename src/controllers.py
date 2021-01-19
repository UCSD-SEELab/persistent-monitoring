import numpy as np

from utils_sim import *

def quadEOM_readonly(t, s, F, M, model_drone):
    """
    QUADEOM_READONLY Solve quadrotor equation of motion
        quadEOM_readonly calculates the derivative of the state vector

    INPUTS:
    t      - 1 x 1, time
    s      - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
    F      - 1 x 1, thrust output from controller (only used in simulation)
    M      - 3 x 1, moments output from controller (only used in simulation)
    params - struct, output from crazyflie() and whatever parameters you want to pass in

    OUTPUTS:
    sdot   - 13 x 1, derivative of state vector s

    NOTE: You should not modify this function
    See Also: quadEOM_readonly, crazyflie
    """
    #************ EQUATIONS OF MOTION ************************
    # Limit the force and moments due to actuator limits
    A = np.array([[0.25,                           0, -0.5/model_drone.arm_length],
                  [0.25,  0.5/model_drone.arm_length,                           0],
                  [0.25,                           0,  0.5/model_drone.arm_length],
                  [0.25, -0.5/model_drone.arm_length,                           0]])

    prop_thrusts = A @ np.array([F, M[0], M[1]]) # Not using moment about Z-axis for limits
    prop_thrusts_clamped = np.clip(prop_thrusts, a_min=(model_drone.minF / 4), a_max=(model_drone.maxF / 4))

    B = np.array([[                      1,                      1,                       1,                       1],
                  [                      0, model_drone.arm_length,                       0, -model_drone.arm_length],
                  [-model_drone.arm_length,                       0, model_drone.arm_length,                       0]])
    F = B[0, :] @ prop_thrusts_clamped
    M = np.append(B[1:3, :] @ prop_thrusts_clamped, M[2]) # VertStack M = [B(2:3,:)*prop_thrusts_clamped; M(3)];

    # Assign states
    x = s[0]
    y = s[1]
    z = s[2]
    xdot = s[3]
    ydot = s[4]
    zdot = s[5]
    qW = s[6]
    qX = s[7]
    qY = s[8]
    qZ = s[9]
    p = s[10]
    q = s[11]
    r = s[12]

    quat = np.array([qW, qX, qY, qZ])
    bRw = QuatToRot(quat)
    wRb = bRw.T

    # Acceleration
    accel = 1 / model_drone.mass * (wRb @ np.array([0, 0, F]) - np.array([0, 0, model_drone.mass * model_drone.grav]))

    # Angular velocity
    K_quat = 2  #this enforces the magnitude 1 constraint for the quaternion
    quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2)
    qdot = -1 / 2 * np.array([[0, -p, -q, -r],
                              [p,  0, -r,  q],
                              [q,  r,  0, -p],
                              [r, -q,  p,  0]]) @ quat + K_quat * quaterror * quat

    # Angular acceleration
    omega = np.array([p, q, r])
    pqrdot = model_drone.invI @ (M - np.cross(omega, (model_drone.I @ omega).flatten()))

    # Assemble sdot
    sdot = np.zeros(13)
    sdot[0]  = xdot
    sdot[1]  = ydot
    sdot[2]  = zdot
    sdot[3]  = accel[0]
    sdot[4]  = accel[1]
    sdot[5]  = accel[2]
    sdot[6]  = qdot[0]
    sdot[7]  = qdot[1]
    sdot[8]  = qdot[2]
    sdot[9] = qdot[3]
    sdot[10] = pqrdot[0]
    sdot[11] = pqrdot[1]
    sdot[12] = pqrdot[2]

    return sdot


def quadEOM(t, s, controlhandle, trajhandle, model_drone):
    """
    QUADEOM Wrapper function for solving quadrotor equation of motion
    quadEOM takes in time, state vector, controller, trajectory generator
    and parameters and output the derivative of the state vector, the
    actual calculation is done in quadEOM_readonly.

    INPUTS:
    t             - 1 x 1, time
    s             - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
    qn            - quad number (used for multi-robot simulations)
    controlhandle - function handle of your controller
    trajhandle    - function handle of your trajectory generator
    params        - struct, output from crazyflie() and whatever parameters you want to pass in

    OUTPUTS:
    sdot          - 13 x 1, derivative of state vector s
    """
    
    # convert state to quad stuct for control
    qd = stateToQd(s)
    
    # Get desired_state
    s_p, s_v, s_a, s_j = trajhandle(t)
    
    # The desired_state is set in the trajectory generator
    qd.pos_des      = s_p
    qd.vel_des      = s_v
    qd.acc_des      = s_a
    qd.yaw_des      = 0
    qd.yawdot_des   = 0
    
    # get control outputs
    [F, M, trpy, drpy] = controlhandle(qd, t, model_drone)
    
    # compute derivative
    sdot = quadEOM_readonly(t, s, F, M, model_drone)
    
    return sdot


def eulerzxy_to_rot(arr_euler):
    """
    Converts three Euler angles (roll, pitch, yaw) to a rotation matrix
    """
    phi = arr_euler[0]
    phi_s = np.sin(phi)
    phi_c = np.cos(phi)

    theta = arr_euler[1]
    theta_s = np.sin(theta)
    theta_c = np.cos(theta)

    psi = arr_euler[2]
    psi_s = np.sin(psi)
    psi_c = np.cos(psi)

    R = np.array([[psi_c * theta_c - phi_s * psi_s * theta_s, phi_c * psi_s, psi_c * theta_s + theta_c * phi_s * psi_s],
                  [theta_c * psi_s + psi_c * phi_s * theta_s, phi_c * psi_c, psi_s * theta_s - psi_c * theta_c * phi_s],
                  [-phi_c * theta_s, phi_s, phi_c * theta_c]])

    return R


def rot_to_eulerzxy(R):
    """
    Converts a rotation matrix to three Euler angles (roll, pitch, yaw)
    """
    if R[2, 1] < 1:
        if R[2, 1] > -1:
            thetaX = np.arcsin(R[2, 1])
            thetaZ = np.arctan2(-R[0, 1], R[1, 1])
            thetaY = np.arctan2(-R[2, 0], R[2, 2])
        else:
            thetaX = -np.pi / 2
            thetaZ = -np.arctan2(R[0, 2], R[0, 0])
            thetaY = 0

    else:
        thetaX = np.pi / 2
        thetaZ = np.arctan2(R[0, 2], R[0, 0])
        thetaY = 0

    arr_euler = np.array([thetaX, thetaY, thetaZ])

    return arr_euler


def vee_map(R):
    """
    Performs the vee mapping from a rotation matrix to a vector
    """
    arr_out = np.zeros(3)
    arr_out[0] = -R[1, 2]
    arr_out[1] = R[0, 2]
    arr_out[2] = -R[0, 1]

    return arr_out


def controller_linear(qd, t, model_drone):
    """
    controller_linear is a controller based on linearization around the hover point

    INPUT
    qd: state and desired state information for quadrotor
    
        qd.pos, qd.vel   position and velocity
        qd.euler = [roll; pitch; yaw]
        qd.omega     angular velocity in body frame

        qd.pos_des, qd.vel_des, qd.acc_des  desired position, velocity, accel
        qd.yaw_des, qd.yawdot_des
    
    t: current time

    model_drone: various parameters
        params.I     moment of inertia
        params.grav  gravitational constant g (9.8...m/s^2)
        params.mass  mass of robot
    
    OUTPUT
    F: total thrust commanded (sum of forces from all rotors)
    M: total torque commanded
    trpy: thrust, roll, pitch, yaw (attitude you want to command!)
    drpy: time derivative of trpy
    """
    
    k_pi = model_drone.k_pi
    k_di = model_drone.k_di
    
    k_p = model_drone.k_p
    k_d = model_drone.k_d
    
    u = np.zeros(4)

    # Compute error in world frame where error = current - desired
    e_pos = (qd.pos - qd.pos_des)
    e_vel = (qd.vel - qd.vel_des)

    r_acc_des = qd.acc_des - k_di * e_vel - k_pi * e_pos
    r_acc_total = r_acc_des + np.array([0, 0, 1]) * model_drone.grav

    # Limit max tilt angle
    tiltangle = np.arccos(r_acc_total[2] / np.sqrt(np.sum(r_acc_total**2)))
    if tiltangle > model_drone.maxangle:
        xy_mag = np.sqrt(np.sum(r_acc_total[:2]**2))
        xy_mag_max = r_acc_total[2] * np.tan(model_drone.maxangle)
        r_acc_total[:2] = r_acc_total[:2] / xy_mag * xy_mag_max

    # Compute desired rotations and Euler error
    psi_des = qd.yaw_des
    theta_des = (np.cos(psi_des) * r_acc_total[0] + np.sin(psi_des) * r_acc_total[1]) / model_drone.grav
    phi_des = (-np.cos(psi_des) * r_acc_total[1] + np.sin(psi_des) * r_acc_total[0]) / model_drone.grav
    euler_des = np.array([phi_des, theta_des, psi_des])
    
    e_euler = qd.euler - euler_des

    # Assume that drone is around hover point
    u[0] = r_acc_total[2] * model_drone.mass
    u[1:] = model_drone.I @ (- k_p * e_euler - k_d * qd.omega)

    # Thrust
    F = u[0]

    # print('F = {0:2f}'.format(F))
    
    # Moment
    M = u[1:]    # note: params.I has the moment of inertia
    
    # Output trpy and drpy as in hardware
    trpy = np.array([F, phi_des, theta_des, psi_des])
    drpy = np.array([0, 0, 0, 0])
    
    return F, M, trpy, drpy


def controller_lee(qd, t, model_drone):
    """
    controller_lee is a controller based on work by Lee et al. (2010)

    INPUT
    qd: state and desired state information for quadrotor

        qd.pos, qd.vel   position and velocity
        qd.euler = [roll; pitch; yaw]
        qd.omega     angular velocity in body frame

        qd.pos_des, qd.vel_des, qd.acc_des  desired position, velocity, accel
        qd.yaw_des, qd.yawdot_des

    t: current time

    model_drone: various parameters
        params.I     moment of inertia
        params.grav  gravitational constant g (9.8...m/s^2)
        params.mass  mass of robot

    OUTPUT
    F: total thrust commanded (sum of forces from all rotors)
    M: total torque commanded
    trpy: thrust, roll, pitch, yaw (attitude you want to command!)
    drpy: time derivative of trpy
    """

    k_pi = model_drone.k_pi
    k_di = model_drone.k_di

    k_p = model_drone.k_p
    k_d = model_drone.k_d

    u = np.zeros(4)

    # Compute error in world frame where error = current - desired
    e_pos = (qd.pos - qd.pos_des)
    e_vel = (qd.vel - qd.vel_des)

    r_acc_des = qd.acc_des - k_di * e_vel - k_pi * e_pos
    r_acc_total = r_acc_des + np.array([0, 0, 1]) * model_drone.grav

    r_acc_mag = np.sqrt(np.sum(r_acc_total**2))
    r_acc_xymag = np.sqrt(np.sum(r_acc_total[:2]**2))

    # If drone is falling, emergency recover by limiting XY movement and raising Z
    if e_pos[-1] < -5:
        r_acc_total[:2] *= model_drone.maxXYaccel / r_acc_xymag

    # Limit max tilt angle
    tiltangle = np.arccos(r_acc_total[2] / r_acc_mag)
    scale_acc = 1
    if tiltangle > model_drone.maxangle:
        xy_mag_max = r_acc_total[2] * np.tan(model_drone.maxangle)
        scale_acc = xy_mag_max / r_acc_xymag
        r_acc_total[:2] = r_acc_total[:2] * scale_acc

    # Compute desired rotations
    a_psi = np.array([np.cos(qd.yaw_des), np.sin(qd.yaw_des), 0])
    b3_des = np.array(r_acc_total)
    b3_des /= np.sqrt(np.sum(b3_des**2))
    b2_des = np.cross(b3_des, a_psi)
    b2_des /= np.sqrt(np.sum(b2_des**2))
    b1_des = np.cross(b2_des, b3_des)
    b1_des /= np.sqrt(np.sum(b1_des**2))

    f_dot = model_drone.mass * scale_acc * k_pi * (-e_vel) # + qd.jrk_des
    f_mag = model_drone.mass * r_acc_mag
    b3_dot = np.cross(np.cross(b3_des, f_dot / f_mag), b3_des)
    a_psi_dot = np.array([-np.cos(qd.yaw_des) * qd.yawdot_des, -np.sin(qd.yaw_des) * qd.yawdot_des, 0])
    b1_dot = np.cross(np.cross(b1_des, (np.cross(a_psi_dot, b3_des) + np.cross(a_psi, b3_dot)) / np.sqrt(np.sum(np.cross(a_psi, b3_des)**2))), b1_des)
    b2_dot = np.cross(b3_dot, b1_des) + np.cross(b3_des, b1_dot)

    # Form rotation matrices
    R_des = np.vstack((b1_des, b2_des, b3_des)).T
    R_desdot = np.vstack((b1_dot, b2_dot, b3_dot)).T

    omega_hat = R_des.T @ R_desdot
    omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

    # Calculate desired Euler angles
    euler_des = rot_to_eulerzxy(R_des)

    R = eulerzxy_to_rot(qd.euler)

    e_euler = 0.5 * vee_map(R_des.T @ R - R.T @ R_des)

    u[0] = model_drone.mass * np.sum(R[:, 2] * r_acc_total)
    u[1:] = model_drone.I @ (- k_p * e_euler - k_d * qd.omega)

    # Thrust
    F = model_drone.mass * np.sum(R[:, 2] * r_acc_total)

    # print('F = {0:2f}'.format(F))

    # Moment
    M = u[1:]  # note: params.I has the moment of inertia

    # Output trpy and drpy as in hardware
    trpy = np.array([F, euler_des[0], euler_des[1], euler_des[2]])
    drpy = np.array([0, 0, 0, 0])

    # print("F: {0}  XY: {1}".format(F, r_acc_xymag))

    return F, M, trpy, drpy
