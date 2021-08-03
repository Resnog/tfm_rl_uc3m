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
    "n_states":120,     # For each degree the agent will have one state
    "n_actions":3,      # The agent will only move right, left or not move at all
    "epsilon" : 0.015,
    "discount" : 0.01,
    "step_size" : 0.015,
    "seed": None
}
"""
    Since this is the start of the RL development on Heroitea, the agent will be name Romulus,
as in the founder of Rome, the first Roman king of the Roman Kingdom, in the pre-Republican era.
"""
romulus = agent(agent_init)




pr.start()  # Start simulation


# Move arm to train position
heroitea.sef_train_position(pr)

# Create the get spawn point for particles
hand_cup = Object.get_object("Cup") 
# Fill cup to pour liquid
particles = [] # The list that holds all particles
n_particles = 100
particles = fill_cup(hand_cup, n_particles, liquids, pr)

# Init episode



# Agent episode 
for i in range(150):
    # Here we make the agent do stuff in each episode
    
    # 1.- Make observation of the environment
    # 2.- Take action based on observation
    heroitea.move_end_effector(2)
    # 3.- Update Q-table
    # 4.- Take step within simulator
    pr.step()



pr.stop()       # Stop simulation


pr.shutdown()   # Close COPSIM