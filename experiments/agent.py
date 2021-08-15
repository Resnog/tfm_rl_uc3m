import numpy as np

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### Q-LEARNING AGENT 
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
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
        self.prev_state = 0     # Define a unidimensional integer variable for each state since the agent controls 1DOF, an articulation
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

    def solve_step(self,observation):
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q_values[state, :]
        # Go full greedy on solution
        action = self.argmax(current_q)

        # Update previous state and action variables
        self.prev_state = state
        self.prev_action = action
        
        return action
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### EXPECTED SARSA AGENT 
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
class agentExpectedSarsa():

    def __init__(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["n_actions"]
        self.num_states = agent_init_info["n_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.prev_state = 0
        self.prev_action = 0

        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, observation):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
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
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
     # Perform an update
        # --------------------------
        # your code here
        
        # Policy distribuition vector
        pi = np.zeros(current_q.shape)
        
        for i in range(self.num_actions):
                if( i == np.argmax(current_q)):
                    pi[i] = (1 - self.epsilon) + (self.epsilon/(self.num_actions))
                else:
                    pi[i] = self.epsilon/(self.num_actions)
        
        # Calculate expected Q-value for current state
        expected_q=np.sum(pi*current_q)
        # Last Q-value
        last_q = self.q[self.prev_state,self.prev_action]
        
        # Apply step size
        q_step = self.step_size*(reward + self.discount*expected_q - last_q)
        
        # Update previus Q value
        self.q[self.prev_state,self.prev_action] = last_q + q_step 
        
        #print(self.q)
        # --------------------------
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode
        # --------------------------
        # your code here
        
        # Create last_q var for the sake of brevity
        last_q = self.q[self.prev_state, self.prev_action]
        # Update Q-value
        self.q[self.prev_state, self.prev_action] = last_q + self.step_size*(reward- last_q)

        #print(self.q[self.prev_state,:])
        # --------------------------
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### Dyna-Q AGENT 
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------

class agentDynaQ():

    def __init__(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """

        # First, we get the relevant information from agent_info 
        # NOTE: we use np.random.RandomState(seed) to set the two different RNGs
        # for the planner and the rest of the code
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info["discount"]
        self.step_size = agent_info["step_size"]
        self.epsilon = agent_info["epsilon"]
        self.planning_steps = agent_info["planning_steps"]

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        # A simple way to implement the model is to have a dictionary of dictionaries, 
        #        mapping each state to a dictionary which maps actions to (reward, next state) tuples.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = 0
        self.past_state = 0
        self.model = {} # model is a dictionary of dictionaries, which maps states to actions to 
                        # (reward, next_state) tuples

def agent_start(self, state):
    """The first method called when the experiment starts, 
    called after the environment starts.
    Args:
        state (Numpy array): the state from the
            environment's env_start function.
    Returns:
        (int) the first action the agent takes.
    """
    action = self.choose_action_egreedy(state)
    
    self.past_state = state
    self.past_action = action
    
    return self.past_action

def agent_step(self, reward, state):
    """A step taken by the agent.

    Args:
        reward (float): the reward received for taking the last action taken
        state (Numpy array): the state from the
            environment's step based on where the agent ended up after the
            last step
    Returns:
        (int) The action the agent takes given this state.
    """
    
    # - Direct-RL step (~1-3 lines)
    # - Model Update step (~1 line)
    # - `planning_step` (~1 line)
    # - Action Selection step (~1 line)
    # Save the current state and action before returning the action to be performed. (~2 lines)

    # Direct-RL step
    q_max = np.max(self.q_values[state])
    self.q_values[self.past_state, self.past_action] += self.step_size * (reward + self.gamma * q_max
                                                                - self.q_values[self.past_state, self.past_action])
    # Model Update step
    self.update_model(self.past_state, self.past_action, state, reward)
    # Planning
    self.planning_step()
    # Choose Action
    action = self.choose_action_egreedy(state)
    # Save the current state and action
    self.past_state = state
    self.past_action = action
    
    return self.past_action

def agent_end(self, reward):
    """Called when the agent terminates.

    Args:
        reward (float): the reward the agent received for entering the
            terminal state.
    """
    
    # - Direct RL update with this final transition (1~2 lines)
    # - Model Update step with this final transition (~1 line)
    # - One final `planning_step` (~1 line)
    #
    # Note: the final transition needs to be handled carefully. Since there is no next state, 
    #       you will have to pass a dummy state (like -1), which you will be using in the planning_step() to 
    #       differentiate between updates with usual terminal and non-terminal transitions.

    # Direct-RL step
    self.q_values[self.past_state, self.past_action] += self.step_size * (reward
                                                                          - self.q_values[self.past_state, self.past_action])
    # Model Update step
    self.update_model(self.past_state, self.past_action, -1, reward)
    # Planning
    self.planning_step()

def argmax(self, q_values):
    """argmax with random tie-breaking
    Args:
        q_values (Numpy array): the array of action values
    Returns:
        action (int): an action with the highest value
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return self.rand_generator.choice(ties)

def choose_action_egreedy(self, state):
    """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

    Important: assume you have a random number generator 'rand_generator' as a part of the class
                which you can use as self.rand_generator.choice() or self.rand_generator.rand()

    Args:
        state (List): coordinates of the agent (two elements)
    Returns:
        The action taken w.r.t. the aforementioned epsilon-greedy policy
    """

    if self.rand_generator.rand() < self.epsilon:
        action = self.rand_generator.choice(self.actions)
    else:
        values = self.q_values[state,:]
        action = self.argmax(values)

    return action

def planning_step(self):
    """performs planning, i.e. indirect RL.

    Args:
        None
    Returns:
        Nothing
    """

    for i in range(self.planning_steps):
        past_state = self.planning_rand_generator.choice(list(self.model.keys()))
        past_action = self.planning_rand_generator.choice(list(self.model[past_state].keys()))
        next_state, reward = self.model[past_state][past_action]
    
        if next_state == -1:
            q_max = 0
        else:
            q_max = np.max(self.q_values[next_state])
    
        self.q_values[past_state, past_action] += self.step_size * (reward + self.gamma * q_max
                                                                    - self.q_values[past_state, past_action])

def update_model(self, past_state, past_action, state, reward):
    """updates the model 
    
    Args:
        past_state       (int): s
        past_action      (int): a
        state            (int): s'
        reward           (int): r
    Returns:
        Nothing
    """
    # Update the model with the (s,a,s',r) tuple (1~4 lines)    
    self.model[past_state] = self.model.get(past_state, {}) 
    self.model[past_state][past_action] = state, reward
