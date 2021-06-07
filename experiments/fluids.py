import sim


def spawn_particles(clientID, p_num):

    rCode, outInts, outFloats, outStrings, outBuffer= sim.simxCallScriptFunction(
        clientID,
        'joint_left_arm',
        1,
        'spawn_particles',
        [p_num],
        [],
        [],
        '',
        sim.simx_opmode_blocking
    )

    return outInts, outFloats, outStrings, outBuffer

def fill_cup(clientID,cup_name):

    # Get the cup handle to spawn the particles
    cup_handle = sim.simxGetObjectHandle(clientID,
                                        cup_name,
                sim.simx_opmode_blocking)
    
    # Get the cup position
    cup_pose = sim.simxGetObjectPosition(clientID,
                                    cup_handle,-1,
                sim.simx_opmode_streaming)

    #
    