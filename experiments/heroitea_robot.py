from pyrep.robots.arms import arm

class Heroitea(object):
    def __init__(self):
        self.left_arm = arm.Arm(0, "UR3_left", 6)
        self.right_arm = arm.Arm(0, "UR3_right", 6)

    

