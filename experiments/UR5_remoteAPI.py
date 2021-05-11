import sim 
from time import sleep

def get_arm_joints(clientID, arm_name):

    j1 = sim.simxGetObjectHandle(clientID, arm_name + '_joint1', sim.simx_opmode_blocking)
    j2 = sim.simxGetObjectHandle(clientID, arm_name + '_joint2', sim.simx_opmode_blocking)
    j3 = sim.simxGetObjectHandle(clientID, arm_name + '_joint3', sim.simx_opmode_blocking)
    j4 = sim.simxGetObjectHandle(clientID, arm_name + '_joint4', sim.simx_opmode_blocking)
    j5 = sim.simxGetObjectHandle(clientID, arm_name + '_joint5', sim.simx_opmode_blocking)
    j6 = sim.simxGetObjectHandle(clientID, arm_name + '_joint6', sim.simx_opmode_blocking)


    return j1,j2,j3,j4,j5,j6

