import sim
from UR3_remoteApi import *


# --------------
# Initialization
# --------------
print("Program started.")
# Close previous connections in case one is open
sim.simxFinish(-1)                                
# Start CopSim connection          
clientID =sim.simxStart("127.0.0.1",19999,True,True,5000,5) 
# UR3 arm configuration
config = [0,-90,-100,-15,-45,30]
config = deg2grad(config)
# ----------
# Connection
# ----------
if clientID != -1:
    # Show that the connection was succesful
    print("Connection to RemoteApi server in CopSim engaged")
    # Send a message to the Status Bar
    sim.simxAddStatusbarMessage(clientID,'Hello CopSim', sim.simx_opmode_oneshot)
    # Get arm joints
    left_arm_joints = get_arm_joints(clientID,'UR3_left')
    # Make the arm change positions
    set_arm_joints_position(clientID, left_arm_joints, config)
    #move_robot_joint(clientID, left_arm_joints[6], -30)
    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)
    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
