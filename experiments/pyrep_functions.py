import pyrep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape, Object
import numpy as np 

def move_arm(config):
    pass    

def fill_cup(num_obj, cup_position):

    shape_handles = []

    for n in range(num_obj):    
        obj = Shape.create(type=PrimitiveShape.SPHERE,
                            size=[0.01],
                            color=[255,0,0],
                            position=[cup_position[0],cup_position[1],cup_position[2]+0.5])
        shape_handles.append(obj)

    return shape_handles

def deg2grad(config):
    
    joint_positions = []

    for deg in config:
        rad = deg*np.pi/180  
        joint_positions.append(rad)
    
    return joint_positions

def spawn_liquid_particle(cup_position):

    particle = Shape.create(type=PrimitiveShape.SPHERE,
                            color=[139,0,0], size=[0.01,0.01,0.01],
                            position=cup_position)
    
    particle.set_dynamic(True)                  # Make the particle dynamic
    particle.set_mass(0.005)                    # Set the particle's mass
    particle.set_collidable(True)               # Make the particle collidable so it can bounce     
    #particle.set_name("par"+str(item_number))   # Set the particle's name
    

    return particle # Return the particle object 

