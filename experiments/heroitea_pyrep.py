from agent import agent
from pyrep import PyRep
from time import sleep
from pyrep_functions import *
from heroitea_robot import Heroitea
from pyrep.robots.arms import arm
from pyrep.backend import sim
from pyrep.objects.shape import Shape, Object
import numpy as np

scene_path = "/home/resnog/myRepos/tfm_rl_uc3m/scenes/heroitea_pyrep.ttt"
delta = 0.01

# Init COPSIM
pr = PyRep()
pr.launch(scene_path,headless=False) # Run COPSIM

# Init Heroitea robot
heroitea = Heroitea()

# Init Agent to control the robot

agent_init = {
    "n_states":180,     # For each degree the agent will have one state
    "n_actions":3,      # The agent will only move right, left or not move at all
    "epsilon" : 0.015,
    "discount" : 0.01,
    "step_size" : 0.015
}
"""
    Since this is the start of the RL development on Heroitea, the agent will be name Romulus,
as in the founder of Rome, the first Roman king of the Roman Kingdom, in the pre-Republican era.
"""
romulus = agent(agent_init)




pr.start()  # Start simulation

# Move arm to train position
train_arm_pose = [-70,-100,-30,45,45,45] 
train_arm_pose = deg2grad(train_arm_pose)
heroitea.left_arm.set_joint_target_positions(train_arm_pose)

# Wait for arm to arrive
pr.step()
vel = 0
while True:
    print("BLYAT")
    pr.step()
    vel = np.linalg.norm( np.array(heroitea.left_arm.get_joint_velocities() ) ) 

    if vel < 0.01:
        break

# Create the cup object to meassure articulation status and get spawn point for particles
hand_cup = Object.get_object("Cup") 
spawn_particle_position = hand_cup.get_position()
spawn_particle_position[2] += 0.01
# Fill cup to pour liquid
particles = [] # The list that holds all particles
num_par = 100

# Filling loop
for i in range(1000):

    if(i%2 == 0 and len(particles) != num_par):
        # Spawn each particle and add the object to the list
        particles.append(spawn_liquid_particle(spawn_particle_position))

    if len(particles) == num_par:
        print(len(particles))
        break

    pr.step()

# Agent episode 
for i in range(150):
    # Here we make the agent do stuff in each episode
    pr.step()

pr.stop()       # Stop simulation


pr.shutdown()   # Close COPSIM