from pyrep.robots.arms import arm
import numpy as np 

class Heroitea(object):
    def __init__(self):
        self.left_arm = arm.Arm(0, "UR3_left", 6)
        self.right_arm = arm.Arm(0, "UR3_right", 6)
        self.init_arm_pose = [-70,-100,-30,45,45,45]    
        self.train_arm_pose = self.init_arm_pose
      
    def get_end_effector_state(self):

        # Get the current angle of the end effector and convert it to an int variable
        pose = np.rint(np.rad2deg( self.left_arm.get_joint_positions() ) )
        # Substract 30 so that the effector can only cage 15 degrees to the wrong position
        return int(pose[5]) - 30

    def set_train_position(self, pr):
        """
        This method sets the arm into the initial position for training the 
        RL agent
        """
        # Turn pose in degrees to radians
        arm_pose = np.deg2rad(self.init_arm_pose)
        # Set arm pose 
        self.left_arm.set_joint_target_positions(arm_pose)
        self.left_arm.set_joint_target_velocities([0,0,0,0,0,0])
        # Wait for arm to reach train position
        pr.step()
        
        vel = 0
        # Wait and check loop
        while True:
            pr.step()
            # Check for the nominal velocity of the arm to be near 0 
            vel = np.linalg.norm( np.array(self.left_arm.get_joint_velocities() ) ) 
            
            if vel < 0.01:
                break


    def move_end_effector(self, action):
        """
        This method moves the arm one degree clockwise, counter-clockwise
        or chooses to not move the effector at all. 

        Action: 0 - No movement
                1 - Left
                2 - Right
        """        
        if(action == 1):
            # Add one degree to current end effector pose
            self.train_arm_pose[5] -= 1
            #Move clockwise
            self.left_arm.set_joint_target_positions(np.deg2rad(self.train_arm_pose))
        elif( action == 2):
            # Add one degree to current end effector pose
            self.train_arm_pose[5] += 1 
            #Move counter-clockwise
            self.left_arm.set_joint_target_positions(np.deg2rad(self.train_arm_pose))
        else:
            #Do not move effector
            pass

    