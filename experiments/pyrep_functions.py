import os
import pyrep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape, Object
import numpy as np 

def fill_cup(hand_cup, n_particles, particle_type, pr):

    particles = []

    # Filling loop
    for i in range(1000):

        if(i%2 == 0 and len(particles) != n_particles):

            # Get current cup position
            spawn_position = hand_cup.get_position()
            spawn_position[2] += 0.02

            # Spawn each particle and add the object to the list
            particles.append(spawn_liquid_particle(spawn_position, particle_type))

        if len(particles) == n_particles:
            print(len(particles))
            break

        pr.step()

    return particles

def spawn_liquid_particle(cup_position, p_tuple):

    particle = Shape.create(type=PrimitiveShape.SPHERE,
                            color=[139,0,0], size=[p_tuple[1],p_tuple[1],p_tuple[1]],
                            position=cup_position)
    
    particle.set_dynamic(True)                  # Make the particle dynamic
    particle.set_mass(p_tuple[0])                    # Set the particle's mass
    particle.set_collidable(True)               # Make the particle collidable so it can bounce     
    #particle.set_name("par"+str(item_number))   # Set the particle's name
    

    return particle # Return the particle object 

# Mass and radious of every test particle
# TUPLE = (MASS, RADIUS)
liquids = (0.0045,0.008)

small_solids = (0.009,0.012)

big_solids = (0.018,0.024)