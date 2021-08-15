from agent import agentQL
from numpy.core.numeric import zeros_like
from numpy.lib.npyio import save
from pyrep import PyRep
from time import sleep
from pyrep_functions import *
from heroitea_robot import Heroitea
from pyrep.robots.arms import arm
from pyrep.backend import sim
from pyrep.objects.shape import Shape, Object
from pyrep.objects.proximity_sensor import ProximitySensor
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Init COPSIM
main_path = "/home/greatceph/myRepos/tfm_rl_uc3m/"
scene_path = main_path + "scenes/heroitea_pyrep.ttt"
save_path = save_path = main_path + "experiments/results/"

pr = PyRep()                            
pr.launch(scene_path,headless=False) # Run COPSIM

n_particles = 200                               # Particle number
particle_type = small_solids                    # Particle type (liquids, small solids, big solids)

agent_init = {
    "n_states":180,     # For each degree the agent will have one state
    "n_actions":3,      # The agent will only move right, left or not move at all
    "epsilon" : 0.03,  # e-greedy epsilon value
    "discount" : 0.9,  # Discount constante
    "step_size" : 0.03,# Steep size
    "seed": None        # Take random seed from clock
}

#----------------------
# Load agent's q_table
#----------------------
romulus_test = agentQL(agent_init)                   # Agent declaration
saved_q_table = np.load(save_path + "agent_ql_values.npy")
romulus_test.q_values = saved_q_table

# ------------
# Init test
# ------------
pr.start()  # Start simulation

# Init Heroitea robot
heroitea = Heroitea()

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
particles, par_visit = fill_cup(hand_cup, n_particles, particle_type, pr)

# Make first observation of the environment
state = heroitea.get_end_effector_state()
# Agent select action
action = romulus_test.agent_start(state) 
# Assing reward of -1 to push the agent for idleness
reward = -1
# Number of particles that reached the destination in the episode
p_in_goal = 0

print("------------------------")
print("Testing agent")
print("------------------------")
# ------------
# Episode Loop
# ------------    
# TIME LIMITED
# Each step in the simulation are 25ms, with 1500 timesteps we have  
# 37.5s of simulation for each episode
#for i in range(1,episode_len): 

# TIME UNLIMITED
while True:
    # Here we make the agent do stuff in each episode
    # 1.- Make observation of the environment
    state = heroitea.get_end_effector_state()
    # 2.- Check if state is terminal 
    
    # Check particle status for terminal conditions
    is_terminal,is_success, g_count = check_particle_terminal(particles) 

    # Update p_in_goal 
    if p_in_goal < g_count:
        p_in_goal = g_count
    # There is a terminal condition then
    if (is_terminal):
        # Check if the agent scored all particles
        if(is_success):
            print("ALL IN!")
            break
        # If some or all particles were lost
        else:
            # Reset streak count
            streaks = 0
        break

    # If within action range
    if state < 0 or state > 180:
        # Give proportional penalty to the agent for going out of action range
        # Consider all particles outside of goal, as lost and give appropriate reward
        romulus_test.agent_end(-1000)
        # Terminate episode if the end effector is outside of the action range
        break
    # If all the particles are outside the hand cup

    # 3.- Reward calculations
    # Check particle states   
    check_particle_states(particles,hand_cup, cup_sensor)
    # Calculate rewards
    reward = calculate_rewards(particles, par_visit)

    # 4.- Take action based on observation, updating Q(s,a) values
    action = romulus_test.solve_step(state)
    heroitea.move_end_effector(action)

    # 5.- Take step within simulator
    pr.step()

print_episode_data(1,0, p_in_goal, n_particles)

# Once the episode ends, stop simulation
pr.stop()