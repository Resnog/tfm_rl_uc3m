import sim 
from time import sleep
# Close all previous communications with COPSIM
sim.simxFinish(-1)
#Start COPSIM API
clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)


# Main Sim Loop
if clientID!=-1:
    print("Connected to COPSIM.")
    sleep(1)
    sim.simxAddStatusbarMessage(clientID, 'Hello world', sim.simx_opmode_oneshot)
    sim.simxGetPingTime(clientID)

# Termination
    sim.simxFinish(clientID)