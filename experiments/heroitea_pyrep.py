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
left_arm = arm.Arm(0, "UR3_left", 6)
right_arm = arm.Arm(0, "UR3_right", 6)
robot = Heroitea(left_arm,right_arm)

hand_cup = Object.get_object("Cup")
spawn_particle_position = hand_cup.get_position()
spawn_particle_position[2] += 0.01
print(spawn_particle_position)

pr.start()  # Start simulation

# Move arm to train position
train_arm_pose = [-70,-100,-30,45,45,10] 
train_arm_pose = deg2grad(train_arm_pose)
robot.left_arm.set_joint_target_positions(train_arm_pose)

# Fill cup to pour liquid
for i in range(1000):
    
    if(i%2 == 0):
        par_handle = spawn_liquid_particle(spawn_particle_position)
    pr.step()



pr.stop()       # Stop simulation


pr.shutdown()   # Close COPSIM