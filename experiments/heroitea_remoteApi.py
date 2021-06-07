from fluids import spawn_particles
import sim 
from time import sleep
from UR3_remoteApi import *
# Close all previous communications with COPSIM
sim.simxFinish(-1)
#Start COPSIM API, connect to continuous remote API server
clientID = sim.simxStart("127.0.0.1", 19997, True, True, 5000, 5)

# Main Sim Loop
if clientID!=-1:
    print("Connected to COPSIM.")
    sleep(1)
    #Open Heroitea scene
    _ = sim.simxLoadScene(clientID, './scenes/heroitea_remoteApi_python.ttt',0, 
        sim.simx_opmode_blocking)
    
    sim.simxAddStatusbarMessage(clientID, 'Main HEROITEA scene loaded', sim.simx_opmode_oneshot)

    # Initialize required variables
    left_arm_joints = get_arm_joints(clientID, 'UR3_left')

    # UR3 arm configuration
    config = [0,-70,-100,-30,45,45,10]
    config = deg2grad(config)

    #for i in range(3):

    # Start simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    
    # Do Stuff
    # set_arm_joints_target_position(clientID,left_arm_joints,config)
    particle_handle, aFloat, someStrings, StringBuffer = spawn_particles(clientID,10)
    sleep(2.0)
                                                                        
    # Stop simulation
    sim.simxStopSimulation(clientID,
                        sim.simx_opmode_oneshot)

# Termination
    sim.simxGetPingTime(clientID)
    sim.simxFinish(clientID)
    print("Terminated. Client disconnected.")