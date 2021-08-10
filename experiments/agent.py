import numpy as np

class agentQL():


    def __init__(self, agent_init):
        # Init class with a
        self.n_actions = agent_init["n_actions"]
        self.n_states = agent_init["n_states"]
        
        # Metaparameters
        self.epsilon = agent_init["epsilon"]
        self.discount = agent_init["discount"]
        self.step_size = agent_init["step_size"]
        # Previous state and action 
        self.prev_state = 30     # Define a unidimensional integer variable for each state since the agent controls 1DOF, an articulation
        self.prev_action = 0    # Ergo, from 0 to num_states - 1 
        # Q table
        self.q_values = np.zeros((self.n_states, self.n_actions))
        # Random number generator for selections 
        self.rand_generator = np.random.RandomState(agent_init["seed"])

    def agent_start(self,observation):
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q_values[state,:]
      
        # Random number selection
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.n_actions)
        else:
            
            action = self.argmax(current_q)
        
        self.prev_state = state
        self.prev_action = action
        
        return action
    
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q_values[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.n_actions)
        else:
            action = self.argmax(current_q)
        
        # Select the previous Q-value
        prev_q = self.q_values[self.prev_state, self.prev_action]
        # Reward, Max Q-value and step-size calculation
        alpha_max_prev_q = self.step_size*(reward + np.max(current_q)- prev_q)
        # Update prev Q-value
        self.q_values[self.prev_state, self.prev_action] = prev_q + alpha_max_prev_q
        
        # Update previous state and action variables
        self.prev_state = state
        self.prev_action = action
        
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        
        # Create last_q var for the sake of brevity
        last_q = self.q_values[self.prev_state, self.prev_action]
        #
        self.q_values[self.prev_state, self.prev_action] = last_q + self.step_size*(reward- last_q)

        
        
        
    def argmax(self, q):
        """argmax with random tie-breaking
        Args:
            q (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q)):
            if q[i] > top:
                top = q[i]
                ties = []

            if q[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)