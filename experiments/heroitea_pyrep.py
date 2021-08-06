from agent import agent
from pyrep import PyRep
from time import sleep
from pyrep_functions import *
from heroitea_robot import Heroitea
from pyrep.robots.arms import arm
from pyrep.backend import sim
from pyrep.objects.shape import Shape, Object
from pyrep.objects.proximity_sensor import ProximitySensor
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
    "step_size" : 0.015,
    "seed": None
}
"""
    Since this is the start of the RL development on Heroitea, the agent will be name Romulus,
as in the founder of Rome, the first Roman king of the Roman Kingdom, in the pre-Republican era.
"""
romulus = agent(agent_init)
print(type(romulus.q_values))



# ------------
# Init episode
# ------------
pr.start()  # Start simulation

# Move arm to train position
heroitea.set_train_position(pr)

# Get the effector cup object for particle location
hand_cup = Object.get_object("hand_cup") 

# Get the table cup object for particle state calculation
table_cup = Object.get_object("table_cup")
# Get the cup sensor
cup_sensor = ProximitySensor("cup_sensor")

# Fill cup to pour liquid
particles = [] # The list that holds all particles
n_particles = 100
particles, par_visit = fill_cup(hand_cup, n_particles, big_solids, pr)

# Make first observation of the environment
state = heroitea.get_end_effector_state()
# Agent select action
action = romulus.agent_start(state) 
# Assing reward of -1 to push the agent for idleness
reward = -1
# ------------
# Episode Loop
# ------------
for i in range(1000):
    # Here we make the agent do stuff in each episode
    

    # 1.- Make observation of the environment
    state = heroitea.get_end_effector_state()
    print(state)
    # 2.- Check if state is terminal 
    if state < 0 or state > 180:
        # Give 100 penalty to the agent for going out of action range
        romulus.agent_end(-100)
        # Terminate episode if the end effector is outside of the action range
        break

    # 3.- Reward calculations
    # Check particle states   
    check_particle_states(particles,hand_cup, cup_sensor)
    # Calculate rewards
    reward = calculate_rewards(particles, par_visit)

    # 4.- Take action based on observation, updating Q(s,a) values
    action = romulus.agent_step(reward,state)
    heroitea.move_end_effector(action)

    # 5.- Take step within simulator
    pr.step()


pr.stop()       # Stop simulation

pr.shutdown()   # Close COPSIM