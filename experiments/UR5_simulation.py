import sim
import numpy as np 
from time import sleep
from UR5_remoteAPI import *

# Close all previous communications with COPSIM
sim.simxFinish(-1)
#Start COPSIM API
clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)


# Main Sim Loop
if clientID!=-1:
    print("Connected to COPSIM.")
    sim.simxAddStatusbarMessage(clientID, 'Hello world', sim.simx_opmode_oneshot)

    #Get arm joint's handles
    j1,j2,j3,j4,j5,j6 = get_arm_joints(clientID, "UR5")

    # Make sure the last message was sent before 
    sim.simxGetPingTime(clientID)
    # Termination
    sim.simxFinish(clientID)