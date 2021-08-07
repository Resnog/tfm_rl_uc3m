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
import matplotlib.pyplot as plt

# Init COPSIM
scene_path = "/home/resnog/myRepos/tfm_rl_uc3m/scenes/heroitea_pyrep.ttt"
pr = PyRep()                            
pr.launch(scene_path,headless=False) # Run COPSIM


# Init Agent to control the robot
agent_init = {
    "n_states":180,     # For each degree the agent will have one state
    "n_actions":3,      # The agent will only move right, left or not move at all
    "epsilon" : 0.015,  # e-greedy epsilon value
    "discount" : 0.01,  # Discount constante
    "step_size" : 0.015,# Steep size
    "seed": None        # Take random seed from clock
}
"""
    Since this is the start of the RL development on Heroitea, the agent will be name Romulus,
as in the founder of Rome, the first Roman king of the Roman Kingdom, in the pre-Republican era.
"""

# ------------
# Init training
# ------------
romulus = agent(agent_init) # Agent declaration
n_episodes = 5            # Episode number
episode_len = 500           # Episode length
n_particles = 20            # Particle number
particle_type = big_solids  # Particle type (liquids, small solids, big solids)
reward_curve = []           # Curve to plot the reward per episode
max_streak_curve = []        # Maximum number of streaks per episode
# -------------
# Training loop
# -------------
for episode in range(n_episodes):

    # ------------
    # Init episode
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
    n_particles = 20
    particles, par_visit = fill_cup(hand_cup, n_particles, particle_type, pr)

    # Make first observation of the environment
    state = heroitea.get_end_effector_state()
    # Agent select action
    action = romulus.agent_start(state) 
    # Assing reward of -1 to push the agent for idleness
    reward = -1
    # Reward sum to add to reward curve
    reward_sum = 0
    # Number to calculate the consecutive successes
    streaks = 0                 
    # Max number of consecutive streaks
    max_streaks = 0
    
    # ------------
    # Episode Loop
    # ------------
    # Each episode has 500 steps at 25ms the step, so 12.5s in total
    for i in range(episode_len): 


        # Here we make the agent do stuff in each episode
        # 1.- Make observation of the environment
        state = heroitea.get_end_effector_state()
        # 2.- Check if state is terminal 
        
        # Add past reward to reward_sum
        reward_sum += reward
        # If within action range
        if state < 0 or state > 180:
            # Give 100 penalty to the agent for going out of action range
            romulus.agent_end(-100)
            # Terminate episode if the end effector is outside of the action range
            break
        # If all the particles are outside the hand cup
        
        # Check particle status for terminal conditions
        is_terminal,is_success = check_particle_terminal(particles) 
        # There is a terminal condition then
        if (is_terminal):
            # Check if the agent scored all particles
            if(is_success):
                # Add one to streak count
                streaks += 1
                # Check if current number of streaks is greater than current max number of streaks
                if streaks > max_streaks:
                    # Assing current streak number as new maximum
                    max_streaks = streaks
            # If some or all particles were lost
            else:
                # Reset streak count
                streaks = 0
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
    # Once the episode ends, stop simulation
    pr.stop()
    
    # Print episode data
    print_episode_data(episode,max_streaks)
    # Add last episode reward sum to the reward curve
    reward_curve.append(reward_sum)
    # Add max number of streaks in episode to streak curve
    max_streak_curve.append(max_streaks)
    # ----------------
    # Episode Loop end
    # ----------------

# -----------------
# Training loop end
# -----------------
    

# -----------------------
# Statistics and graphics
# -----------------------

plt.plot(n_episodes,reward_curve)
plt.ylabel("Sumatoria de recompensas por episodio")
plt.xlabel("Episodios")
plt.savefig("reward_sum_curve.png")
plt.show()

pr.shutdown()   # Close COPSIM