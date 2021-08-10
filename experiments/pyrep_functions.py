import os
import pyrep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape, Object
from pyrep.objects.proximity_sensor import ProximitySensor
import numpy as np 

def fill_cup(hand_cup, n_particles, particle_type, pr):

    particles = []
    par_visit = []
    # Filling loop
    for i in range(1000):

        if(i%2 == 0 and len(particles) != n_particles):

            # Get current cup position
            spawn_position = hand_cup.get_position()
            spawn_position[2] += 0.02

            # Spawn each particle and add the object to the list
            particles.append(spawn_liquid_particle(spawn_position, particle_type))
            par_visit.append(False)
        if len(particles) == n_particles:
            #print(len(particles))
            break

        pr.step()

    return particles, par_visit

def spawn_liquid_particle(cup_position, p_tuple):

    particle = Shape.create(type=PrimitiveShape.SPHERE,
                            color=YELLOW, size=[p_tuple[1],p_tuple[1],p_tuple[1]],
                            position=cup_position)
    
    particle.set_dynamic(True)                  # Make the particle dynamic
    particle.set_mass(p_tuple[0])                    # Set the particle's mass
    particle.set_collidable(True)               # Make the particle collidable so it can bounce     
    #particle.set_name("par"+str(item_number))   # Set the particle's name
    particle.get_color()
    return particle # Return the particle object 

def check_particle_states(particles, hand_cup, cup_sensor):#table_cup):


    hand_cup_position = hand_cup.get_position()
    #table_cup_position = table_cup.get_position()

    # Check distance between hand_cup and table_cup
    for p in particles:
        
        p_position = p.get_position()

        p_hand_dist = np.linalg.norm (p_position - hand_cup_position)
        #p_table_dist = np.linalg.norm (p_position - table_cup_position)

        # If within 9cm of the hand_cup effector
        if cup_sensor.is_detected(p):
            # Color GREEN
            p.set_color(GREEN)
        # If within 15cm of the hand_cup effector
        elif p_hand_dist < 0.15:
            # Color YELLOW
            p.set_color(YELLOW)
        # If neither 
        else:
            # Color RED
            p.set_color(RED)
        
def calculate_rewards(particles, par_visit):
    """
    Check all particles to see if they are visited and if not, check status
    (color) and return respective reward
    """

    # Standar reward for not getting particles into the glass
    reward = 0
    # Score
    count = 0

    # Iterate for each particle and particle visit state
    for i in range(len(particles)):
        
        # If particle is not visited
        if par_visit[i] == False:
            
            # If the particle is inside the glass
            if particles[i].get_color() == GREEN:
                # Give reward
                reward += 100
                # Set status of the particle as "visited"
                par_visit[i] = True
                count += 1
            # If the particle is lost 
            elif particles[i].get_color() == RED:
                # Give penalty
                reward -= 100
                # Set status of the particle as "visited"
                par_visit[i] = True
                # Add particle to visited count
                count += 1
            # If the particle hasn't reached any state
            else:
                # DO NOTHING!
                pass

    if count == 0:
        reward = -10

    return reward

def check_particle_terminal(particles):

    count = 0           # Count variable for different states than YELLOW
    g_count = 0         # Green particles counter
    terminal = False    # Terminal state bool
    success = False     # Success state bool

    # Main search loop
    for p in particles:
        # If the particle is either lost or reached their destination        
        if p.get_color() != YELLOW:
            count += 1
        # If the particle is GREEN
        elif p.get_color() == GREEN:
            g_count += 1

    # If all the particles are either lost or reached their destination 
    if count == len(particles):
        # Set terminal state
        terminal = True
        # Check if the count is equal to g_count
        if count == g_count:
            success = True

    return terminal, success, g_count

def print_episode_data(episode_number, streaks):

    print("------------------------")
    print("Episode number: {}".format(episode_number))
    print("Streaks:        {}".format(streaks))
    print("------------------------")
    
    
# COLOR VARIABLES
YELLOW = [205,205,0]
GREEN = [0,139,0]
RED = [139,0,0]

# Mass and radious of every test particle
# TUPLE = (MASS, RADIUS)
liquids = (0.0045,0.008)

small_solids = (0.009,0.012)

big_solids = (0.018,0.024)