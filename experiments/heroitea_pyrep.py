from pyrep import PyRep
from time import sleep
from pyrep_functions import *
from heroitea_robot import Heroitea
from pyrep.robots.arms import arm
from pyrep.backend import sim
from pyrep.objects.shape import Shape, Object

scene_path = "/home/greatceph/myRepos/tfm_rl_uc3m/scenes/heroitea_pyrep.ttt"
delta = 0.01

# Init COPSIM
pr = PyRep()
pr.launch(scene_path,headless=False) # Run COPSIM

# Init Heroitea robot
heroitea = Heroitea()

# Create the cup object to meassure articulation status and get spawn point for particles
hand_cup = Object.get_object("Cup") 
spawn_particle_position = hand_cup.get_position()
spawn_particle_position[2] += 0.01

pr.start()  # Start simulation

# Move arm to train position
train_arm_pose = [-70,-100,-30,45,45,10] 
train_arm_pose = deg2grad(train_arm_pose)
heroitea.left_arm.set_joint_target_positions(train_arm_pose)

# Fill cup to pour liquid

particles = [] # The list that holds all particles
num_par = 10
for i in range(1000):
    
    if(i%2 == 0 and len(particles) != num_par):
        # Spawn each particle and add the object to the list
        particles.append(spawn_liquid_particle(spawn_particle_position))
    
    if len(particles)>0:
        for par in particles:
            print(par.get_position())

    if len(particles) == num_par:
        break

    pr.step()

# Agent episode 
for i in range(150):
    # Here we make the agent do stuff in each episode
    pr.step()

pr.stop()       # Stop simulation


pr.shutdown()   # Close COPSIM