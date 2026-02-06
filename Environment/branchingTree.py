import numpy as np

class BranchingTree:
    """
    Full binary-tree MDP of depth D:
      - States:  0,1,...,2^(D+1)-2
            Internal nodes: 0..(2^D-2)
            Leaves:         2^D-1..(2^(D+1)-2)
      - Actions: "LEFT", "RIGHT", "STAY"
      - Transitions:
            Internal node s:
                LEFT  → child 2*s+1 w.p. p_move; else stay
                RIGHT → child 2*s+2 w.p. p_move; else stay
                STAY  → stays at s
      - Rewards:
            Internal moves: 0
            STAY at non-terminal: r_stay
            On first arrival at a leaf:  r ~ Uniform(mu_leaf, sigma_leaf^2)
      - Episode ends when you reach a leaf
    """

    def __init__(self, depth=10, mu_leaf=0.0, sigma_leaf=1.0, r_stay=0.5, p_move=0.7):
        self.depth = depth
        self.mu_leaf = mu_leaf
        self.sigma_leaf = sigma_leaf
        self.r_stay = r_stay
        self.p_move = p_move

        self.num_actions = 3

        # total states = 2^(D+1)-1
        self.num_states = 2**(depth + 1) - 1
        # first and set of leaves
        self.first_leaf = 2**depth - 1
        self.terminal_states = set(range(self.first_leaf, self.num_states))

        self.transition_probs = self._init_transitions()

        # start out at state 0
        self.reset()

    def _init_transitions(self):
        """
        Build dict mapping (state, action) → list of (next_state, prob).
        """
        probs = {}
        for s in range(self.num_states):
            # all states: STAY loops to self
            probs[(s, 0)] = [(s, 1.0)]

            if s in self.terminal_states:
                # leaf: LEFT/RIGHT also loop
                probs[(s, 1)] = [(s, 1.0)]
                probs[(s, 2)] = [(s, 1.0)]

            else:
                # internal node: probabilistic left/right
                left_child = 2*s + 1
                right_child = 2*s + 2
                probs[(s, 1)] = [(left_child, self.p_move), (s, 1-self.p_move)]
                probs[(s, 2)] = [(right_child, self.p_move), (s, 1-self.p_move)]
        return probs

    def step(self, action):
        """
        action ∈ {"LEFT", "RIGHT", "STAY"}
        Returns (next_state, reward, done)
        """
        trans = self.transition_probs[(self.state, action)]
        next_states, probs = zip(*trans)
        self.state = np.random.choice(next_states, p=probs)
        done = False

        # did we land on a leaf?
        is_leaf = (self.state in self.terminal_states)

        # reaching leaf
        if is_leaf and action in (1, 2):
            reward = np.random.uniform(self.mu_leaf, self.sigma_leaf)
            done = True
        # stay
        elif (action == 0):
            reward = self.r_stay
        # left or right
        else:
            reward = 0

        return self.state, reward, done

    def reset(self):
        #resets state to 0 
        self.state = 0
        return self.state