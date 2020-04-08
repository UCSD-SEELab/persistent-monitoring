import numpy as np

class qd_object:
    """
    Struct to hold qd information
    """
    def __init__(self):
        self.pos = 0
        self.vel = 0
        self.euler = 0
        self.omega = 0

class state_object:
    """
    Struct to hold state information
    """
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.yaw = 0
        self.yawdot = 0

def init_state(s_p, s_v, s_a, s_j, yaw):
    #INIT_STATE Initialize 13 x 1 state vector
    s     = np.zeros(13)
    phi0   = 0.0
    theta0 = 0.0
    psi0   = yaw
    Rot0   = RPYtoRot_ZXY(phi0, theta0, psi0)
    Quat0  = RotToQuat(Rot0)
    s[0]  = s_p[0] #x
    s[1]  = s_p[1] #y
    s[2]  = s_p[2] #z
    s[3]  = s_v[0] #xdot
    s[4]  = s_v[1] #ydot
    s[5]  = s_v[2] #zdot
    s[6]  = Quat0[0] #qw
    s[7]  = Quat0[1] #qx
    s[8]  = Quat0[2] #qy
    s[9] =  Quat0[3] #qz
    s[10] = 0        #p
    s[11] = 0        #q
    s[12] = 0        #r

    return s

def QuatToRot(q):
    """
    QuatToRot Converts a Quaternion to Rotation matrix
       written by Daniel Mellinger
    """
    # normalize q
    q = q / np.sqrt(np.sum(q**2))

    qahat = np.zeros([3, 3] )
    qahat[0, 1]  = -q[3]
    qahat[0, 2]  = q[2]
    qahat[1, 2]  = -q[1]
    qahat[1, 0]  = q[3]
    qahat[2, 0]  = -q[2]
    qahat[2, 1]  = q[1]

    R = np.identity(3) + 2 * qahat @ qahat + 2 * q[0] * qahat

    return R

def RotToQuat(R):
    """
ROTTOQUAT Converts a Rotation matrix into a Quaternion
   written by Daniel Mellinger
   from the following website, deals with the case when tr<0
   http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    """
    #takes in W_R_B rotation matrix

    tr = np.sum(np.trace(R))

    if (tr > 0):
      S = np.sqrt(tr + 1.0) * 2 # S=4*qw
      qw = 0.25 * S
      qx = (R[2, 1] - R[1, 2]) / S
      qy = (R[0, 2] - R[2, 0]) / S
      qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      S = np.sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2 # S=4*qx
      qw = (R[2, 1] - R[1, 2]) / S
      qx = 0.25 * S
      qy = (R[0, 1] + R[1, 0]) / S
      qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1]  > R[2, 2] :
      S = np.sqrt(1.0 + R[1, 1]  - R[0, 0]  - R[2, 2] ) * 2 # S=4*qy
      qw = (R[0, 2]  - R[2, 0] ) / S
      qx = (R[0, 1]  + R[1, 0] ) / S
      qy = 0.25 * S
      qz = (R[1, 2]  + R[2, 1] ) / S
    else:
      S = np.sqrt(1.0 + R[2, 2]  - R[0, 0]  - R[1, 1] ) * 2 # S=4*qz
      qw = (R[1, 0]  - R[0, 1] ) / S
      qx = (R[0, 2]  + R[2, 0] ) / S
      qy = (R[1, 2]  + R[2, 1] ) / S
      qz = 0.25 * S

    q = np.array([[qw], [qx], [qy], [qz]])
    q = q * np.sign(qw)

    return q

def RPYtoRot_ZXY(phi, theta, psi):
    """
    RPYtoRot_ZXY Converts roll, pitch, yaw to a body-to-world Rotation matrix
       The rotation matrix in this function is world to body [bRw] you will
       need to transpose this matrix to get the body to world [wRb] such that
       [wP] = [wRb] * [bP], where [bP] is a point in the body frame and [wP]
       is a point in the world frame
       written by Daniel Mellinger
    """
    R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                   np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta)],
                  [-np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi), np.sin(phi)],
                  [np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi),
                   np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi), np.cos(phi)*np.cos(theta)]])

    return R

def RotToRPY_ZXY(R):
    """
    RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
       The rotation matrix in this function is world to body [bRw] you will
       need to transpose the matrix if you have a body to world [wRb] such
       that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and
       [wP] is a point in the world frame
       written by Daniel Mellinger
       bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
               cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
              -cos(phi)*sin(theta)]
             [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
             [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
               sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
               cos(phi)*cos(theta)]
    """
    phi = np.arcsin(R[1, 2])
    psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))

    return phi, theta, psi


def qdToState(qd):
    """
     Converts state vector for simulation to qd struct used in hardware.
     x is 1 x 13 vector of state variables [pos vel quat omega]
     qd is a struct including the fields pos, vel, euler, and omega
    """
    x = np.zeros(13) #initialize dimensions

    x[0:3] = qd.pos
    x[3:6] = qd.vel

    Rot = RPYtoRot_ZXY(qd.euler[0], qd.euler[1], qd.euler[2])
    quat = RotToQuat(Rot)

    x[6:10] = quat
    x[11:13] = qd.omega

    return x

def stateToQd(x):
    """
    Converts qd struct used in hardware to x vector used in simulation
     x is 1 x 13 vector of state variables [pos vel quat omega]
     qd is a struct including the fields pos, vel, euler, and omega
    """
    qd = qd_object()

    # current state
    qd.pos = x[0:3]
    qd.vel = x[3:6]

    Rot = QuatToRot(x[6:10])
    [phi, theta, yaw] = RotToRPY_ZXY(Rot)

    qd.euler = np.array([phi, theta, yaw])
    qd.omega = x[10:13]

    return qd