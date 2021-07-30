import pyrep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape, Object
import numpy as np 

def move_arm(config):
    pass    

def fill_cup(n_particles, spawn_position, pr):

    particles = []

    # Filling loop
    for i in range(1000):

        if(i%2 == 0 and len(particles) != n_particles):
            # Spawn each particle and add the object to the list
            particles.append(spawn_liquid_particle(spawn_position))

        if len(particles) == n_particles:
            print(len(particles))
            break

        pr.step()

    return particles

def deg2grad(config):
    
    joint_positions = []

    for deg in config:
        rad = deg*np.pi/180  
        joint_positions.append(rad)
    
    return joint_positions

def spawn_liquid_particle(cup_position):

    particle = Shape.create(type=PrimitiveShape.SPHERE,
                            color=[139,0,0], size=[0.015,0.015,0.015],
                            position=cup_position)
    
    particle.set_dynamic(True)                  # Make the particle dynamic
    particle.set_mass(0.005)                    # Set the particle's mass
    particle.set_collidable(True)               # Make the particle collidable so it can bounce     
    #particle.set_name("par"+str(item_number))   # Set the particle's name
    

    return particle # Return the particle object 

