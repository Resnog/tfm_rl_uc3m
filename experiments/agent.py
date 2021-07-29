import numpy as np

class agent():


    def __init__(self, agent_init):
        # Init class with a
        self.n_actions = agent_init["n_actions"]
        self.n_states = agent_init["n_states"]
        
        # Metaparameters
        self.epsilon = agent_init["epsilon"]
        self.discount = agent_init["discount"]
        self.step_size = agent_init["step_size"]
        # Previous state and action 
        self.prev_state = 0     # Define a unidimensional integer variable for each state since the agent controls 1DOF, an articulation
        self.prev_action = 0    # Ergo, from 0 to num_states - 1 
        # Q table
        self.q_values = np.zeros((self.n_states, self.n_actions))