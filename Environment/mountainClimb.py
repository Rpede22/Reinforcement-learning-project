import numpy as np

class MountainClimb():
    """
    Mountain climb MDP:
    - States: 0,1,...,N-1 (state N-1 is terminal summit)
    - Actions: "Climb", "Camp"
    - Transitions:
            Climb: with prob p_up -> s+1; else -> 0 (start)
            Camp: stay in same state, reward = r_camp
    - Rewards:
            Climb: r_goal if you reach summit; else 0
            Camp: r_camp
    - Episode ends when you reach the summit
    """

    def __init__(self, num_states=10, p_up=0.5, r_camp=1.0, r_goal=10.0):
        self.num_states = num_states
        self.terminal_state = num_states - 1 
        self.p_up       = p_up
        self.r_camp     = r_camp
        self.r_goal     = r_goal
        self.num_actions = 2 

        self.transition_probs = self._init_transitions()

        # start out at state 0
        self.reset()

    def _init_transitions(self):
        """
        Build a dict mapping (state, action) → list of (next_state, prob).
        """
        probs = {}
        for s in range(self.num_states):
            # CAMP: stay with prob 1
            probs[(s, 1)] = [(s, 1.0)]

            # CLIMB: either succeed to s+1 or slip back to 0
            if s < self.terminal_state:
                probs[(s, 0)] = [(s + 1, self.p_up), (0, 1 - self.p_up)]

            else:
                # at summit, climbing keeps you there
                probs[(s, 0)] = [(s, 1.0)]
        return probs

    def step(self, action):
        """
        action ∈ {"CLIMB", "CAMP"}
        Returns: (next_state, reward)
        """
        trans = self.transition_probs[(self.state, action)]
        next_states, probs = zip(*trans)
        self.state = np.random.choice(next_states, p=probs)
        done = False

        # reaching summit
        if(action == 0 and self.state == self.terminal_state):
            reward = self.r_goal
            done = True
        # CAMP
        elif(action == 1):
            reward = self.r_camp
        # CLIMB
        else:
            reward = 0

        return self.state, reward, done

    def reset(self):
        #resets state to 0 
        self.state = 0
        return self.state
    