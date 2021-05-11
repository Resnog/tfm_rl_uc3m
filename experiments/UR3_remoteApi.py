import sim
import numpy as np
from time import sleep
# ----------------------------
# UR3 Control for localization
# ----------------------------

def get_arm_joints(clientID,arm_name):

    joints = []

    for i in range(0,7):
        _, j = sim.simxGetObjectHandle(clientID, arm_name + '_joint' + str(i), sim.simx_opmode_blocking) 
        joints.append(j)

    return joints

def set_arm_joints_position(clientID,arm_joints, config):
    
    for i in range(len(arm_joints)):
       sim.simxSetJointTargetPosition(clientID, arm_joints[i], config[i], sim.simx_opmode_oneshot )


def deg2grad(config):
    
    joint_positions = []

    for deg in config:
        rad = deg*np.pi/180  
        joint_positions.append(rad)
    
    return joint_positions
    